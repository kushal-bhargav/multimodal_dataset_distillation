import os
import torch
from tqdm import tqdm
import numpy as np
import yaml
from transformers import BertTokenizer, BertModel
from torchvision import transforms as T
from torchvision.transforms.functional import InterpolationMode
from torch.utils.data import Dataset
from torchvision.datasets.utils import download_url
import argparse
import re
import json
import pandas as pd
from PIL import Image

def pre_caption(caption, max_words=50):
    """
    Cleans and truncates captions for consistency.
    """
    caption = re.sub(r"([.!\"()*#:;~])", ' ', caption.lower())
    caption = re.sub(r"\s{2,}", ' ', caption)
    caption = caption.rstrip('\n').strip(' ')
    caption_words = caption.split(' ')
    if len(caption_words) > max_words:
        caption = ' '.join(caption_words[:max_words])
    return caption

class roco_train(Dataset):
    def __init__(self, transform, image_root, ann_file, max_words=30, prompt=''):
        """
        Args:
            transform (callable): Image transformations.
            image_root (str): Root directory of images (e.g. /kaggle/input/roco-dataset/all_data/train/radiology/images)
            ann_file (str): CSV file with annotations (should contain 'id', 'name', and 'caption' columns)
            max_words (int): Maximum words allowed in captions.
            prompt (str): A string to be added as a prefix to each caption.
        """
        self.df = pd.read_csv(ann_file)
        self.transform = transform
        self.image_root = image_root
        self.max_words = max_words
        self.prompt = prompt

        # Create a list of (filename, caption) pairs.
        self.annotations = list(zip(self.df['name'], self.df['caption']))

        # Build an image ID mapping (from unique 'id' values) to a zero-based index.
        self.img_ids = {img_id: idx for idx, img_id in enumerate(self.df['id'])}

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, index):
        img_name, caption = self.annotations[index]
        image_path = os.path.join(self.image_root, img_name)

        # Load and preprocess image
        image = Image.open(image_path).convert('RGB')
        if self.transform:
            image = self.transform(image)

        caption_final = self.prompt + pre_caption(caption, self.max_words)
        return image, caption_final, self.img_ids[self.df.iloc[index]['id']]

    def get_all_captions(self):
        """
        Returns all captions as a list.
        """
        return [self.prompt + pre_caption(cap, self.max_words) for cap in self.df['caption']]

class roco_retrieval_eval(Dataset):
    def __init__(self, transform, image_root, ann_file, split, max_words=30):
        """
        Args:
            transform (callable): Image transformations.
            image_root (str): Root directory for evaluation images.
                (e.g. /kaggle/input/roco-dataset/all_data/validation/radiology/images)
            ann_file (str): CSV file with evaluation annotations.
            split (str): "val" or "test".
            max_words (int): Maximum words allowed in captions.
        """
        self.df = pd.read_csv(ann_file)
        self.transform = transform
        self.image_root = image_root
        self.max_words = max_words

        self.text = []     # List of preprocessed captions.
        self.image = []    # List of image filenames.
        self.txt2img = {}  # Map from text index to image index.
        self.img2txt = {}  # Map from image index to list of text indices.

        txt_id = 0
        for img_idx, row in self.df.iterrows():
            self.image.append(row['name'])
            self.img2txt[img_idx] = []
            proc_caption = pre_caption(row['caption'], max_words)
            self.text.append(proc_caption)
            self.img2txt[img_idx].append(txt_id)
            self.txt2img[txt_id] = img_idx
            txt_id += 1

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        # Returns the processed image and its index.
        img_name = self.df.iloc[index]['name']
        image_path = os.path.join(self.image_root, img_name)
        image = Image.open(image_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        return image, index
