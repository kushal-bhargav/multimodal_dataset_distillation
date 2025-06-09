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

# Explicitly import transforms and InterpolationMode from torchvision
from torchvision import transforms
from torchvision.transforms.functional import InterpolationMode

# Import dataset functions for flickr and coco
from data.flickr30k_dataset import flickr30k_train, flickr30k_retrieval_eval
from data.coco_dataset import coco_train, coco_retrieval_eval, coco_caption_eval

# Attempt to import the ROCO dataset functions. If unsuccessful, define dummy functions.
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

# Fix for NameError: Explicitly import DataLoader from torch.utils.data
from torch.utils.data import DataLoader


def create_dataset(args, min_scale=0.5):
    normalize = transforms.Normalize(
        (0.48145466, 0.4578275, 0.40821073),
        (0.26862954, 0.26130258, 0.27577711))
    transform_train = transforms.Compose([
        transforms.RandomResizedCrop(args.image_size, scale=(min_scale, 1.0), interpolation=InterpolationMode.BICUBIC),
        transforms.RandomHorizontalFlip(),
        RandomAugment(2, 5, isPIL=True, augs=[
            'Identity','AutoContrast','Brightness','Sharpness','Equalize',
            'ShearX','ShearY','TranslateX','TranslateY','Rotate']),
        transforms.ToTensor(),
        normalize,
    ])
    transform_test = transforms.Compose([
        transforms.Resize((args.image_size, args.image_size), interpolation=InterpolationMode.BICUBIC),
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


def get_dataset(args):
    """Choose dataset loading method based on `--dataset` argument."""
    if args.dataset == 'roco':
        return get_dataset_roco(args)
    else:
        return get_dataset_flickr(args)


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


def get_dataset_roco(args):
    print("Creating retrieval dataset for ROCO")
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


def main(args):
    trainloader, testloader, train_dataset, test_dataset = get_dataset(args)
    data = load_or_process_file('text', textprocess, args, testloader)

    bert_test_embed = torch.from_numpy(data['bert_test_embed']).cpu()

    print("Dataset loaded successfully.")

    # Continue with training and evaluation processes...
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Parameter Processing')
    parser.add_argument('--dataset', type=str, default='roco', choices=['roco', 'coco'], help='dataset')
    args, unknown = parser.parse_known_args()
    if unknown:
        print("Warning: Ignoring unknown arguments:", unknown)
    
    main(args)
