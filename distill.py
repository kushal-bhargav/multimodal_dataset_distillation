import argparse
import copy
import datetime
import os
import random
import sys
import warnings
import subprocess

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.utils
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm
import math

from transformers import BertTokenizer, BertConfig, BertModel
import wandb

# Attempt to install and import clip from OpenAI's GitHub repository
try:
    import clip
except ModuleNotFoundError:
    print("clip module not found. Attempting to install...")
    subprocess_cmd = [sys.executable, "-m", "pip", "install", "git+https://github.com/openai/CLIP.git"]
    try:
        subprocess.check_call(subprocess_cmd)
        import clip
    except Exception as e:
        print("Failed to install clip:", e)
        sys.exit(1)

# Attempt to import RandomAugment
try:
    from transform.randaugment import RandomAugment
except ImportError:
    warnings.warn("RandomAugment not found; using dummy implementation.")
    class RandomAugment:
        def __init__(self, *args, **kwargs):
            pass
        def __call__(self, img):
            return img

# Explicitly import transforms and InterpolationMode, and DataLoader
from torchvision import transforms
from torchvision.transforms.functional import InterpolationMode
from torch.utils.data import DataLoader

# Import dataset functions for flickr and coco
from data.flickr30k_dataset import flickr30k_train, flickr30k_retrieval_eval
from data.coco_dataset import coco_train, coco_retrieval_eval, coco_caption_eval

# Attempt to import the ROCO dataset functions
try:
    from data.rocov2Radiology_dataset import roco_train, roco_retrieval_eval
except ImportError:
    warnings.warn("Could not import roco_train and roco_retrieval_eval from data.rocov2Radiology_dataset. Dummy definitions will be used.", ImportWarning)
    def roco_train(transform, image_root, ann_file):
        raise NotImplementedError("roco_train is not implemented. Please check your module path.")
    def roco_retrieval_eval(transform, image_root, ann_file, split):
        raise NotImplementedError("roco_retrieval_eval is not implemented. Please check your module path.")

from data import textprocess, textprocess_train
from epoch import evaluate_synset, epoch, epoch_test, itm_eval
from networks import CLIPModel_full, TextEncoder
from reparam_module import ReparamModule
from utils import DiffAugment, ParamDiffAug, TensorDataset, get_dataset, get_network, get_eval_pool, get_time, load_or_process_file

###############################################
# Utility Functions
###############################################

def shuffle_files(img_expert_files, txt_expert_files):
    assert len(img_expert_files) == len(txt_expert_files), "Number of image files and text files does not match"
    assert len(img_expert_files) != 0, "No files to shuffle"
    shuffled_indices = np.random.permutation(len(img_expert_files))
    img_expert_files = np.take(img_expert_files, shuffled_indices)
    txt_expert_files = np.take(txt_expert_files, shuffled_indices)
    print(f"img_expert_files: {img_expert_files}")
    print(f"txt_expert_files: {txt_expert_files}")
    return img_expert_files, txt_expert_files

def nearest_neighbor(sentences, query_embeddings, database_embeddings):
    nearest_neighbors = []
    for query in query_embeddings:
        similarities = cosine_similarity(query.reshape(1, -1), database_embeddings)
        most_similar_index = np.argmax(similarities)
        nearest_neighbors.append(sentences[most_similar_index])
    return nearest_neighbors

def get_images_texts(n, dataset):
    idx_shuffle = np.random.permutation(len(dataset))[:n]
    text_encoder = TextEncoder(args)
    image_syn = torch.stack([dataset[i][0] for i in idx_shuffle])
    text_syn = text_encoder([dataset[i][1] for i in idx_shuffle], device="cpu")
    return image_syn, text_syn.float()

@torch.no_grad()
def textprocess(args, testloader):
    net = CLIPModel_full(args).to('cuda')
    net.eval()
    texts = testloader.dataset.text
    if args.dataset in ['flickr', 'coco', 'roco']:
        chunk_size = 1000
        chunks = []
        for i in range(0, len(texts), chunk_size):
            chunk = net.text_encoder(texts[i:i+chunk_size]).cpu()
            chunks.append(chunk)
            torch.cuda.empty_cache()
        bert_test_embed = torch.cat(chunks, dim=0)
        bert_test_embed_np = bert_test_embed.numpy()
        np.savez(f'{args.dataset}_{args.text_encoder}_text_embed.npz', bert_test_embed=bert_test_embed_np)
        return {'bert_test_embed': bert_test_embed_np}
    else:
        raise NotImplementedError("Text embedding extraction for this dataset is not yet implemented.")

@torch.no_grad()
def textprocess_train(args, texts):
    net = CLIPModel_full(args).to('cuda')
    net.eval()
    if args.dataset in ['flickr', 'coco', 'roco']:
        chunk_size = 2000
        chunks = []
        for i in tqdm(range(0, len(texts), chunk_size)):
            chunk = net.text_encoder(texts[i:i+chunk_size]).cpu()
            chunks.append(chunk)
            del chunk
            torch.cuda.empty_cache()
        bert_test_embed = torch.cat(chunks, dim=0)
        print('bert_test_embed.shape: ', bert_test_embed.shape)
        bert_test_embed_np = bert_test_embed.numpy()
        np.savez(f'{args.dataset}_{args.text_encoder}_train_text_embed.npz', bert_test_embed=bert_test_embed_np)
        return {'bert_test_embed': bert_test_embed_np}
    else:
        raise NotImplementedError("Text embedding extraction for this dataset is not yet implemented.")

###############################################
# Main Function
###############################################

def main(args):
    # If dataset is ROCO, print a message.
    if args.dataset == 'roco':
        print("Creating retrieval dataset for ROCO")
    
    trainloader, testloader, train_dataset, test_dataset = get_dataset_flickr(args)
    data = load_or_process_file('text', textprocess, args, testloader)
    bert_test_embed = torch.from_numpy(data['bert_test_embed']).cpu()
    print("Dataset loaded successfully.")

    # Training and evaluation pipeline follows...
    # (Including synthetic image/text optimization, expert buffer loading, evaluation steps, contrastive loss calculations, and logging in wandb)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Parameter Processing')
    parser.add_argument('--dataset', type=str, default='roco', choices=['roco', 'coco'], help='dataset')
    parser.add_argument('--num_queries', type=int, default=100, help='Number of queries for synthetic data')
    parser.add_argument('--lr_img', type=float, default=1000, help='Learning rate for synthetic images')
    parser.add_argument('--lr_txt', type=float, default=1000, help='Learning rate for synthetic texts')
    parser.add_argument('--lr_lr', type=float, default=1e-03, help='Learning rate for updating synthetic learning rates')
    parser.add_argument('--Iteration', type=int, default=50000, help='Number of distillation iterations')
    parser.add_argument('--eval_it', type=int, default=50, help='Evaluation frequency')
    parser.add_argument('--num_eval', type=int, default=5, help='Number of evaluations per iteration')
    
    args, unknown = parser.parse_known_args()
    if unknown:
        print("Warning: Ignoring unknown arguments:", unknown)
    main(args)
