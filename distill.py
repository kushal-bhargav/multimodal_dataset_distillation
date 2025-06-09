import argparse
import copy
import datetime
import os
import random
import sys
import warnings

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

# Attempt to import RandomAugment from transform.randaugment.
try:
    from transform.randaugment import RandomAugment
except ImportError:
    warnings.warn("RandomAugment not found; using dummy implementation.")
    class RandomAugment:
        def __init__(self, *args, **kwargs):
            pass
        def __call__(self, img):
            return img

# Explicitly import transforms and InterpolationMode from torchvision.
from torchvision import transforms
from torchvision.transforms.functional import InterpolationMode

# Import dataset functions for flickr and coco.
from data.flickr30k_dataset import flickr30k_train, flickr30k_retrieval_eval
from data.coco_dataset import coco_train, coco_retrieval_eval, coco_caption_eval

# Attempt to import the ROCO dataset functions.
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

# Explicitly import DataLoader from torch.utils.data.
from torch.utils.data import DataLoader


def shuffle_files(img_expert_files, txt_expert_files):
    # Check if both lists have the same length and if the lists are not empty.
    assert len(img_expert_files) == len(txt_expert_files), "Number of image files and text files does not match"
    assert len(img_expert_files) != 0, "No files to shuffle"
    shuffled_indices = np.random.permutation(len(img_expert_files))
    # Apply the shuffled indices to both lists.
    img_expert_files = np.take(img_expert_files, shuffled_indices)
    txt_expert_files = np.take(txt_expert_files, shuffled_indices)
    print(f"img_expert_files: {img_expert_files}")
    print(f"txt_expert_files: {txt_expert_files}")
    return img_expert_files, txt_expert_files

def nearest_neighbor(sentences, query_embeddings, database_embeddings):
    """Find the nearest neighbors for a batch of embeddings.

    Args:
      sentences: The original sentences from which the embeddings were computed.
      query_embeddings: A batch of embeddings for which to find the nearest neighbors.
      database_embeddings: All pre-computed embeddings.
    Returns:
      A list of the most similar sentences for each embedding in the batch.
    """
    nearest_neighbors = []
    for query in query_embeddings:
        similarities = cosine_similarity(query.reshape(1, -1), database_embeddings)
        most_similar_index = np.argmax(similarities)
        nearest_neighbors.append(sentences[most_similar_index])
    return nearest_neighbors

def get_images_texts(n, dataset):
    """Get random n images and corresponding texts from the dataset.

    Args:
      n: Number of images and texts to retrieve.
      dataset: The dataset containing image-text pairs.
    Returns:
      A tuple containing two elements:
        - A tensor of randomly selected images.
        - A tensor of corresponding texts, encoded as floats.
    """
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

def create_dataset(args, min_scale=0.5):
    # Use a default image_size of 224 if not provided.
    image_size = getattr(args, "image_size", 224)
    normalize = transforms.Normalize(
        (0.48145466, 0.4578275, 0.40821073),
        (0.26862954, 0.26130258, 0.27577711))
    transform_train = transforms.Compose([
        transforms.RandomResizedCrop(image_size, scale=(min_scale, 1.0), interpolation=InterpolationMode.BICUBIC),
        transforms.RandomHorizontalFlip(),
        RandomAugment(2, 5, isPIL=True, augs=[
            'Identity','AutoContrast','Brightness','Sharpness','Equalize',
            'ShearX','ShearY','TranslateX','TranslateY','Rotate']),
        transforms.ToTensor(),
        normalize,
    ])
    transform_test = transforms.Compose([
        transforms.Resize((image_size, image_size), interpolation=InterpolationMode.BICUBIC),
        transforms.ToTensor(),
        normalize,
    ])
    if args.dataset == 'flickr':          
        train_dataset = flickr30k_train(transform_train, args.image_root, args.ann_root)
        val_dataset = flickr30k_retrieval_eval(transform_test, args.image_root, args.ann_root, 'val')
        test_dataset = flickr30k_retrieval_eval(transform_test, args.image_root, args.ann_root, 'test')
    elif args.dataset == 'coco':             
        train_dataset = coco_train(transform_train, args.image_root, args.ann_root)
        val_dataset = coco_retrieval_eval(transform_test, args.image_root, args.ann_root, 'val')
        test_dataset = coco_retrieval_eval(transform_test, args.image_root, args.ann_root, 'test')
    elif args.dataset == 'roco':
        train_dataset = roco_train(transform_train, args.image_root, args.ann_root)
        val_dataset = roco_retrieval_eval(transform_test, args.image_root, args.ann_root, 'val')
        test_dataset = roco_retrieval_eval(transform_test, args.image_root, args.ann_root, 'test')
    else: 
        raise NotImplementedError("Dataset not implemented")
    
    return train_dataset, val_dataset, test_dataset

def create_sampler(datasets, shuffles, num_tasks, global_rank):
    samplers = []
    for dataset, shuffle in zip(datasets, shuffles):
        sampler = torch.utils.data.DistributedSampler(dataset, num_replicas=num_tasks, rank=global_rank, shuffle=shuffle)
        samplers.append(sampler)
    return samplers     

def create_loader(datasets, samplers, batch_size, num_workers, is_trains, collate_fns):
    loaders = []
    for dataset, sampler, bs, n_worker, is_train, collate_fn in zip(datasets, samplers, batch_size, num_workers, is_trains, collate_fns):
        shuffle = (sampler is None) if is_train else False
        drop_last = is_train
        loader = DataLoader(
            dataset,
            batch_size=bs,
            num_workers=n_worker,
            pin_memory=True,
            sampler=sampler,
            shuffle=shuffle,
            collate_fn=collate_fn,
            drop_last=drop_last,
        )
        loaders.append(loader)
    return loaders     

def get_dataset_flickr(args):
    print("Creating retrieval dataset")
    train_dataset, val_dataset, test_dataset = create_dataset(args)
    samplers = [None, None, None]
    train_loader, val_loader, test_loader = create_loader(
        [train_dataset, val_dataset, test_dataset],
        samplers,
        batch_size=[args.batch_size_train] + [args.batch_size_test] * 2,
        num_workers=[4, 4, 4],
        is_trains=[True, False, False],
        collate_fns=[None, None, None]
    )
    return train_loader, test_loader, train_dataset, test_dataset

###############################################
# Main Buffer Distillation Script
###############################################

import wandb
import warnings
import datetime
from epoch import epoch, epoch_test, itm_eval
from utils import load_or_process_file

warnings.filterwarnings("ignore", category=DeprecationWarning)

def main(args):
    # Choose dataset loader based on dataset argument.
    if args.dataset == 'roco':
        print("Creating retrieval dataset for ROCO")
    trainloader, testloader, train_dataset, test_dataset = get_dataset_flickr(args)
    data = load_or_process_file('text', textprocess, args, testloader)
    bert_test_embed = torch.from_numpy(data['bert_test_embed']).cpu()

    print("Dataset loaded successfully.")
    # (Rest of your training pipeline would follow here.)
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Parameter Processing')
    parser.add_argument('--dataset', type=str, default='roco', choices=['roco', 'coco'], help='dataset')
    parser.add_argument('--disable_wandb', action='store_true', help='Disable wandb logging')
    parser.add_argument('--num_experts', type=int, default=100, help='training iterations')
    parser.add_argument('--lr_teacher_img', type=float, default=0.1, help='learning rate for updating network parameters')
    parser.add_argument('--lr_teacher_txt', type=float, default=0.1, help='learning rate for updating network parameters')
    parser.add_argument('--batch_train', type=int, default=128, help='batch size for training networks')
    parser.add_argument('--dsa', type=str, default='True', choices=['True', 'False'],
                        help='whether to use differentiable Siamese augmentation.')
    parser.add_argument('--dsa_strategy', type=str, default='color_crop_cutout_flip_scale_rotate',
                        help='differentiable Siamese augmentation strategy')
    parser.add_argument('--data_path', type=str, default='/kaggle/input/roco-dataset/', help='dataset path')
    parser.add_argument('--buffer_path', type=str, default='/kaggle/working', help='buffer path')
    parser.add_argument('--train_epochs', type=int, default=50)
    parser.add_argument('--zca', action='store_true')
    parser.add_argument('--decay', action='store_true')
    parser.add_argument('--mom', type=float, default=0, help='momentum')
    parser.add_argument('--l2', type=float, default=0, help='l2 regularization')
    parser.add_argument('--save_interval', type=int, default=10)
    parser.add_argument('--name', type=str, default=datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"), help='name of wandb run')
    parser.add_argument('--text_pretrained', type=bool, default=True, help='text_pretrained')
    parser.add_argument('--image_pretrained', type=bool, default=True, help='image_pretrained')
    parser.add_argument('--text_trainable', type=bool, default=False, help='text_trainable')
    parser.add_argument('--image_trainable', type=bool, default=True, help='image_trainable')
    parser.add_argument('--batch_size_train', type=int, default=128, help='batch_size_train')
    parser.add_argument('--batch_size_test', type=int, default=128, help='batch_size_test')
    parser.add_argument('--image_root', type=str, default='/kaggle/input/roco-dataset/all_data/train/radiology/images/', help='location of image root')
    parser.add_argument('--ann_root', type=str, default='/kaggle/input/roco-dataset/all_data/train/radiologytraindata.csv', help='location of annotation file')
    parser.add_argument('--image_size', type=int, default=224, help='image_size')
    parser.add_argument('--k_test', type=int, default=128, help='k_test')
    parser.add_argument('--load_npy', type=bool, default=False, help='load_npy')
    parser.add_argument('--image_encoder', type=str, default='resnet50', choices=['nfnet', 'resnet18_gn', 'vit_tiny', 'nf_resnet50', 'nf_regnet'], help='image encoder')
    parser.add_argument('--text_encoder', type=str, default='bert', choices=['bert', 'clip'], help='text encoder')
    parser.add_argument('--margin', default=0.2, type=float, help='Rank loss margin.')
    parser.add_argument('--measure', default='cosine', help='Similarity measure used (cosine|order)')
    parser.add_argument('--max_violation', action='store_true', help='Use max instead of sum in the rank loss.')
    parser.add_argument('--only_has_image_projection', type=bool, default=False, help='None')
    parser.add_argument('--grounding', type=bool, default=False, help='None')
    parser.add_argument('--distill', type=bool, default=False, help='if distill')
    args, unknown = parser.parse_known_args()
    if unknown:
        print("Warning: Ignoring unknown arguments:", unknown)
    main(args)
