#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
import copy
import datetime
import os
import random
import sys
import warnings
import subprocess

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.utils
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm
import math

from transformers import BertTokenizer, BertConfig, BertModel
import wandb

# ----- Automatic Installation for clip -----
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

# ----- Try to import RandomAugment -----
try:
    from transform.randaugment import RandomAugment
except ImportError:
    warnings.warn("RandomAugment not found; using dummy implementation.")
    class RandomAugment:
        def __init__(self, *args, **kwargs):
            pass
        def __call__(self, img):
            return img

# ----- Explicit imports for torchvision modules and DataLoader -----
from torchvision import transforms
from torchvision.transforms.functional import InterpolationMode
from torch.utils.data import DataLoader

# ----- Import dataset functions for flickr and coco -----
from data.flickr30k_dataset import flickr30k_train, flickr30k_retrieval_eval
from data.coco_dataset import coco_train, coco_retrieval_eval, coco_caption_eval

# ----- Attempt to import ROCO dataset functions -----
try:
    from data.rocov2Radiology_dataset import roco_train, roco_retrieval_eval
except ImportError:
    warnings.warn("Could not import roco_train and roco_retrieval_eval from data.rocov2Radiology_dataset. Dummy definitions will be used.", ImportWarning)
    def roco_train(transform, image_root, ann_file):
        raise NotImplementedError("roco_train is not implemented. Check your module path.")
    def roco_retrieval_eval(transform, image_root, ann_file, split):
        raise NotImplementedError("roco_retrieval_eval is not implemented. Check your module path.")

# ----- Import helper functions -----
from data import textprocess, textprocess_train
from epoch import evaluate_synset, epoch, epoch_test, itm_eval
from networks import CLIPModel_full, TextEncoder
from reparam_module import ReparamModule
from utils import DiffAugment, ParamDiffAug, TensorDataset, get_dataset, get_network, get_eval_pool, get_time, load_or_process_file

###############################################
# Utility Functions
###############################################

def shuffle_files(img_expert_files, txt_expert_files):
    assert len(img_expert_files) == len(txt_expert_files), "Mismatch between image and text file counts"
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
    # Get synthetic texts and immediately move them to args.device.
    text_syn = text_encoder([dataset[i][1] for i in idx_shuffle], device=args.device).to(args.device)
    return image_syn, text_syn.float()

@torch.no_grad()
def textprocess(args, testloader):
    net = CLIPModel_full(args).to("cuda")
    net.eval()
    texts = testloader.dataset.text
    if args.dataset in ["flickr", "coco", "roco"]:
        chunk_size = 1000
        chunks = []
        for i in range(0, len(texts), chunk_size):
            chunk = net.text_encoder(texts[i:i+chunk_size]).cpu()
            chunks.append(chunk)
            torch.cuda.empty_cache()
        bert_test_embed = torch.cat(chunks, dim=0)
        bert_test_embed_np = bert_test_embed.numpy()
        np.savez(f"{args.dataset}_{args.text_encoder}_text_embed.npz", bert_test_embed=bert_test_embed_np)
        return {"bert_test_embed": bert_test_embed_np}
    else:
        raise NotImplementedError("Text embedding extraction not implemented for this dataset.")

@torch.no_grad()
def textprocess_train(args, texts):
    net = CLIPModel_full(args).to("cuda")
    net.eval()
    if args.dataset in ["flickr", "coco", "roco"]:
        chunk_size = 2000
        chunks = []
        for i in tqdm(range(0, len(texts), chunk_size)):
            chunk = net.text_encoder(texts[i:i+chunk_size]).cpu()
            chunks.append(chunk)
            del chunk
            torch.cuda.empty_cache()
        bert_test_embed = torch.cat(chunks, dim=0)
        print("bert_test_embed.shape: ", bert_test_embed.shape)
        bert_test_embed_np = bert_test_embed.numpy()
        np.savez(f"{args.dataset}_{args.text_encoder}_train_text_embed.npz", bert_test_embed=bert_test_embed_np)
        return {"bert_test_embed": bert_test_embed_np}
    else:
        raise NotImplementedError("Text embedding extraction not implemented for this dataset.")

def create_dataset(args, min_scale=0.5):
    image_size = getattr(args, "image_size", 224)
    normalize = transforms.Normalize((0.48145466, 0.4578275, 0.40821073),
                                     (0.26862954, 0.26130258, 0.27577711))
    transform_train = transforms.Compose([
        transforms.RandomResizedCrop(image_size, scale=(min_scale, 1.0), interpolation=InterpolationMode.BICUBIC),
        transforms.RandomHorizontalFlip(),
        RandomAugment(2, 5, isPIL=True, augs=["Identity","AutoContrast","Brightness","Sharpness","Equalize",
                                               "ShearX","ShearY","TranslateX","TranslateY","Rotate"]),
        transforms.ToTensor(),
        normalize,
    ])
    transform_test = transforms.Compose([
        transforms.Resize((image_size, image_size), interpolation=InterpolationMode.BICUBIC),
        transforms.ToTensor(),
        normalize,
    ])
    if args.dataset == "flickr":
        train_dataset = flickr30k_train(transform_train, args.image_root, args.ann_root)
        val_dataset = flickr30k_retrieval_eval(transform_test, args.image_root, args.ann_root, "val")
        test_dataset = flickr30k_retrieval_eval(transform_test, args.image_root, args.ann_root, "test")
    elif args.dataset == "coco":
        train_dataset = coco_train(transform_train, args.image_root, args.ann_root)
        val_dataset = coco_retrieval_eval(transform_test, args.image_root, args.ann_root, "val")
        test_dataset = coco_retrieval_eval(transform_test, args.image_root, args.ann_root, "test")
    elif args.dataset == "roco":
        train_dataset = roco_train(transform_train, args.image_root, args.ann_root)
        val_dataset = roco_retrieval_eval(transform_test, args.image_root, args.ann_root, "val")
        test_dataset = roco_retrieval_eval(transform_test, args.image_root, args.ann_root, "test")
    else:
        raise NotImplementedError("Dataset not implemented")
    return train_dataset, val_dataset, test_dataset

def create_loader(datasets, samplers, batch_size, num_workers, is_trains, collate_fns):
    loaders = []
    for dataset, sampler, bs, n_worker, is_train, collate_fn in zip(datasets, samplers, batch_size, num_workers, is_trains, collate_fns):
        shuffle = (sampler is None) if is_train else False
        drop_last = is_train
        loader = DataLoader(dataset, batch_size=bs, num_workers=n_worker, pin_memory=True,
                            sampler=sampler, shuffle=shuffle, collate_fn=collate_fn, drop_last=drop_last)
        loaders.append(loader)
    return loaders

def get_dataset_flickr(args):
    print("Creating retrieval dataset")
    train_dataset, val_dataset, test_dataset = create_dataset(args)
    samplers = [None, None, None]
    train_loader, val_loader, test_loader = create_loader(
        [train_dataset, val_dataset, test_dataset],
        samplers,
        batch_size=[args.batch_size_train] + [args.batch_size_test]*2,
        num_workers=[4, 4, 4],
        is_trains=[True, False, False],
        collate_fns=[None, None, None]
    )
    return train_loader, test_loader, train_dataset, test_dataset

###############################################
# Main Distillation & Training Pipeline
###############################################

import wandb
import datetime
from epoch import epoch, epoch_test, itm_eval
from utils import load_or_process_file, get_time

def main(args):
    # Set device in args.
    args.device = "cuda" if torch.cuda.is_available() else "cpu"
    
    if args.dataset == "roco":
        print("Creating retrieval dataset for ROCO")
    trainloader, testloader, train_dataset, test_dataset = get_dataset_flickr(args)
    data = load_or_process_file("text", textprocess, args, testloader)
    bert_test_embed = torch.from_numpy(data["bert_test_embed"]).cpu()
    print("Dataset loaded successfully.")
    
    # --- Training and Evaluation Pipeline ---
    train_sentences = train_dataset.get_all_captions()
    train_caption = load_or_process_file("train_text", textprocess_train, args, train_sentences)
    train_caption_embed = torch.from_numpy(train_caption["bert_test_embed"]).cpu()
    
    image_syn, text_syn = get_images_texts(args.num_queries, train_dataset)
    image_syn = image_syn.detach().to(args.device).requires_grad_(True)
    optimizer_img = torch.optim.SGD([image_syn], lr=args.lr_img, momentum=0.5)
    optimizer_img.zero_grad()
    
    syn_lr_img = torch.tensor(args.lr_teacher_img).to(args.device).requires_grad_(True)
    syn_lr_txt = torch.tensor(args.lr_teacher_txt).to(args.device).requires_grad_(True)
    optimizer_lr = torch.optim.SGD([syn_lr_img, syn_lr_txt], lr=args.lr_lr, momentum=0.5)
    
    text_syn = text_syn.detach().to(args.device).requires_grad_(True)
    optimizer_txt = torch.optim.SGD([text_syn], lr=args.lr_txt, momentum=0.5)
    optimizer_txt.zero_grad()
    
    sentence_list = nearest_neighbor(train_sentences, text_syn.detach().cpu(), train_caption_embed)
    if args.draw:
        wandb.log({"original_sentence_list": wandb.Html("<br>".join(sentence_list))})
        wandb.log({"original_synthetic_images": wandb.Image(torch.nan_to_num(image_syn.detach().cpu()))})
    
    criterion = nn.CrossEntropyLoss().to(args.device)
    print("%s training begins" % get_time())
    
    expert_dir = args.buffer_path   # Using buffer_path as expert directory.
    print("Expert Dir: {}".format(expert_dir))
    
    img_expert_files = []
    txt_expert_files = []
    n = 0
    while os.path.exists(os.path.join(expert_dir, f"img_replay_buffer_{n}.pt")):
        img_expert_files.append(os.path.join(expert_dir, f"img_replay_buffer_{n}.pt"))
        txt_expert_files.append(os.path.join(expert_dir, f"txt_replay_buffer_{n}.pt"))
        n += 1
    # If no expert buffers are detected, create dummy buffers.
    if n == 0:
        print("No buffers detected at {}. Creating dummy replay buffers...".format(expert_dir))
        dummy_model = CLIPModel_full(args).to(args.device)
        dummy_img_traj = [[p.detach().cpu() for p in dummy_model.image_encoder.parameters()]]
        dummy_txt_traj = [[p.detach().cpu() for p in dummy_model.text_projection.parameters()]]
        img_buffer_path = os.path.join(expert_dir, "img_replay_buffer_0.pt")
        txt_buffer_path = os.path.join(expert_dir, "txt_replay_buffer_0.pt")
        torch.save(dummy_img_traj, img_buffer_path)
        torch.save(dummy_txt_traj, txt_buffer_path)
        img_expert_files.append(img_buffer_path)
        txt_expert_files.append(txt_buffer_path)
        n = 1
    
    img_expert_files, txt_expert_files = shuffle_files(img_expert_files, txt_expert_files)
    
    file_idx = 0
    expert_idx = 0
    print("loading file {}".format(img_expert_files[file_idx]))
    print("loading file {}".format(txt_expert_files[file_idx]))
    img_buffer = torch.load(img_expert_files[file_idx])
    txt_buffer = torch.load(txt_expert_files[file_idx])
    
    eval_it_pool = np.arange(0, args.Iteration + 1, args.eval_it).tolist()
    
    # ----- Main Distillation Loop -----
    for it in tqdm(range(args.Iteration + 1)):
        save_this_it = True
        wandb.log({"Progress": it}, step=it)
        
        # Evaluation block
        if it in eval_it_pool:
            print("-------------------------")
            print("Evaluation")
            print("image_model_train = %s, text_model_train = %s, iteration = %d" %
                  (args.image_encoder, args.text_encoder, it))
            if args.dsa:
                print("DSA augmentation strategy: ", args.dsa_strategy)
                print("DSA augmentation parameters: ", args.dsa_param.__dict__ if hasattr(args, "dsa_param") else "N/A")
            else:
                print("DC augmentation parameters: ", args.dc_aug_param if hasattr(args, "dc_aug_param") else "N/A")
            img_r1s, img_r5s, img_r10s, img_r_means = [], [], [], []
            txt_r1s, txt_r5s, txt_r10s, txt_r_means = [], [], [], []
            r_means = []
            for it_eval in range(args.num_eval):
                net_eval = CLIPModel_full(args, eval_stage=args.transfer)
                with torch.no_grad():
                    image_save = image_syn
                    text_save = text_syn
                image_syn_eval, text_syn_eval = copy.deepcopy(image_save.detach()), copy.deepcopy(text_save.detach())
                args.lr_net = syn_lr_img.item()
                print(image_syn_eval.shape)
                _, acc_train, val_result = evaluate_synset(it_eval, net_eval, image_syn_eval, text_syn_eval, testloader, args, bert_test_embed)
                print("Evaluate_%02d: Img R@1 = %.4f, Img R@5 = %.4f, Img R@10 = %.4f, Img R@Mean = %.4f, Txt R@1 = %.4f, Txt R@5 = %.4f, Txt R@10 = %.4f, Txt R@Mean = %.4f, R@Mean = %.4f" %
                      (it_eval,
                       val_result["img_r1"], val_result["img_r5"], val_result["img_r10"], val_result["img_r_mean"],
                       val_result["txt_r1"], val_result["txt_r5"], val_result["txt_r10"], val_result["txt_r_mean"],
                       val_result["r_mean"]))
                img_r1s.append(val_result["img_r1"])
                img_r5s.append(val_result["img_r5"])
                img_r10s.append(val_result["img_r10"])
                img_r_means.append(val_result["img_r_mean"])
                txt_r1s.append(val_result["txt_r1"])
                txt_r5s.append(val_result["txt_r5"])
                txt_r10s.append(val_result["txt_r10"])
                txt_r_means.append(val_result["txt_r_mean"])
                r_means.append(val_result["r_mean"])
                if not args.std:
                    wandb.log({"txt_r1": val_result["txt_r1"]})
                    wandb.log({"txt_r5": val_result["txt_r5"]})
                    wandb.log({"txt_r10": val_result["txt_r10"]})
                    wandb.log({"txt_r_mean": val_result["txt_r_mean"]})
                    wandb.log({"img_r1": val_result["img_r1"]})
                    wandb.log({"img_r5": val_result["img_r5"]})
                    wandb.log({"img_r10": val_result["img_r10"]})
                    wandb.log({"img_r_mean": val_result["img_r_mean"]})
                    wandb.log({"r_mean": val_result["r_mean"]})
            if args.std:
                img_r1_mean, img_r1_std = np.mean(img_r1s), np.std(img_r1s)
                img_r5_mean, img_r5_std = np.mean(img_r5s), np.std(img_r5s)
                img_r10_mean, img_r10_std = np.mean(img_r10s), np.std(img_r10s)
                img_r_mean_mean, img_r_mean_std = np.mean(img_r_means), np.std(img_r_means)
                txt_r1_mean, txt_r1_std = np.mean(txt_r1s), np.std(txt_r1s)
                txt_r5_mean, txt_r5_std = np.mean(txt_r5s), np.std(txt_r5s)
                txt_r10_mean, txt_r10_std = np.mean(txt_r10s), np.std(txt_r10s)
                txt_r_mean_mean, txt_r_mean_std = np.mean(txt_r_means), np.std(txt_r_means)
                r_mean_mean, r_mean_std = np.mean(r_means), np.std(r_means)
                wandb.log({"Mean/txt_r1": txt_r1_mean, "Std/txt_r1": txt_r1_std})
                wandb.log({"Mean/txt_r5": txt_r5_mean, "Std/txt_r5": txt_r5_std})
                wandb.log({"Mean/txt_r10": txt_r10_mean, "Std/txt_r10": txt_r10_std})
                wandb.log({"Mean/txt_r_mean": txt_r_mean_mean, "Std/txt_r_mean": txt_r_mean_std})
                wandb.log({"Mean/img_r1": img_r1_mean, "Std/img_r1": img_r1_std})
                wandb.log({"Mean/img_r5": img_r5_mean, "Std/img_r5": img_r5_std})
                wandb.log({"Mean/img_r10": img_r10_mean, "Std/img_r10": img_r10_std})
                wandb.log({"Mean/img_r_mean": img_r_mean_mean, "Std/img_r_mean": img_r_mean_std})
                wandb.log({"Mean/r_mean": r_mean_mean, "Std/r_mean": r_mean_std})
        if it in eval_it_pool and (save_this_it or it % 1000 == 0):
            if args.draw:
                with torch.no_grad():
                    image_save = image_syn_eval.cuda()
                    text_save = text_syn_eval.cuda()
                    save_dir = os.path.join(".", "logged_files", args.dataset, wandb.run.name)
                    print("Saving to {}".format(save_dir))
                    if not os.path.exists(save_dir):
                        os.makedirs(save_dir)
                    wandb.log({"Pixels": wandb.Histogram(torch.nan_to_num(image_syn.detach().cpu()))}, step=it)
                    if args.ipc < 50 or args.force_save:
                        upsampled = image_save[:90]
                        if args.dataset != "ImageNet":
                            upsampled = torch.repeat_interleave(upsampled, repeats=4, dim=2)
                            upsampled = torch.repeat_interleave(upsampled, repeats=4, dim=3)
                        grid = torchvision.utils.make_grid(upsampled, nrow=10, normalize=True, scale_each=True)
                        sentence_list = nearest_neighbor(train_sentences, text_syn.cpu(), train_caption_embed)
                        sentence_list = sentence_list[:90]
                        torchvision.utils.save_image(grid, os.path.join(save_dir, f"synthetic_images_{it}.png"))
                        with open(os.path.join(save_dir, f"synthetic_sentences_{it}.txt"), "w") as file:
                            file.write("\n".join(sentence_list))
                        wandb.log({"Synthetic_Images": wandb.Image(torch.nan_to_num(grid.detach().cpu()))}, step=it)
                        wandb.log({"Synthetic_Pixels": wandb.Histogram(torch.nan_to_num(image_save.detach().cpu()))}, step=it)
                        wandb.log({"Synthetic_Sentences": wandb.Html("<br>".join(sentence_list))}, step=it)
                        print("finish saving images")
                        for clip_val in [2.5]:
                            std = torch.std(image_save)
                            mean = torch.mean(image_save)
                            upsampled = torch.clip(image_save, min=mean-clip_val*std, max=mean+clip_val*std).cpu()
                            if args.dataset != "ImageNet":
                                upsampled = torch.repeat_interleave(upsampled, repeats=4, dim=2)
                                upsampled = torch.repeat_interleave(upsampled, repeats=4, dim=3)
                            grid = torchvision.utils.make_grid(upsampled[:90], nrow=10, normalize=True, scale_each=True)
                            wandb.log({f"Clipped_Synthetic_Images/std_{clip_val}": wandb.Image(torch.nan_to_num(grid))}, step=it)
                            torchvision.utils.save_image(grid, os.path.join(save_dir, f"clipped_synthetic_images_{it}_std_{clip_val}.png"))
                    if args.zca:
                        image_save = image_save.to(args.device)
                        image_save = args.zca_trans.inverse_transform(image_save.cpu())
                        torch.save(image_save, os.path.join(save_dir, f"images_zca_{it}.pt"))
                        upsampled = image_save
                        if args.dataset != "ImageNet":
                            upsampled = torch.repeat_interleave(upsampled, repeats=4, dim=2)
                            upsampled = torch.repeat_interleave(upsampled, repeats=4, dim=3)
                        grid = torchvision.utils.make_grid(upsampled, nrow=10, normalize=True, scale_each=True)
                        wandb.log({"Reconstructed_Images": wandb.Image(torch.nan_to_num(grid))}, step=it)
                        wandb.log({"Reconstructed_Pixels": wandb.Histogram(torch.nan_to_num(image_save.detach().cpu()))}, step=it)
                        for clip_val in [2.5]:
                            std = torch.std(image_save)
                            mean = torch.mean(image_save)
                            upsampled = torch.clip(image_save, min=mean-clip_val*std, max=mean+clip_val*std)
                            if args.dataset != "ImageNet":
                                upsampled = torch.repeat_interleave(upsampled, repeats=4, dim=2)
                                upsampled = torch.repeat_interleave(upsampled, repeats=4, dim=3)
                            grid = torchvision.utils.make_grid(upsampled, nrow=10, normalize=True, scale_each=True)
                            wandb.log({f"Clipped_Reconstructed_Images/std_{clip_val}": wandb.Image(torch.nan_to_num(grid.detach().cpu()))}, step=it)
                    if args.draw:
                        print("finish saving draw")
        wandb.log({"Synthetic_LR_Image": syn_lr_img.detach().cpu()}, step=it)
        wandb.log({"Synthetic_LR_Text": syn_lr_txt.detach().cpu()}, step=it)
        torch.cuda.empty_cache()
        student_net = CLIPModel_full(args)
        img_student_net = ReparamModule(student_net.image_encoder.to("cpu")).to("cuda")
        txt_student_net = ReparamModule(student_net.text_projection.to("cpu")).to("cuda")
        if args.distributed:
            img_student_net = torch.nn.DataParallel(img_student_net)
            txt_student_net = torch.nn.DataParallel(txt_student_net)
        img_student_net.train()
        txt_student_net.train()
        img_num_params = sum([np.prod(p.size()) for p in img_student_net.parameters()])
        txt_num_params = sum([np.prod(p.size()) for p in txt_student_net.parameters()])
        img_expert_trajectory = img_buffer[expert_idx]
        txt_expert_trajectory = txt_buffer[expert_idx]
        expert_idx += 1
        if expert_idx == len(img_buffer):
            expert_idx = 0
            file_idx += 1
            if file_idx == len(img_expert_files):
                file_idx = 0
                img_expert_files, txt_expert_files = shuffle_files(img_expert_files, txt_expert_files)
            print("loading file {}".format(img_expert_files[file_idx]))
            print("loading file {}".format(txt_expert_files[file_idx]))
            if args.max_files != 1:
                del img_buffer
                del txt_buffer
                img_buffer = torch.load(img_expert_files[file_idx])
                txt_buffer = torch.load(txt_expert_files[file_idx])
        start_epoch = np.random.randint(0, args.max_start_epoch)
        img_starting_params = img_expert_trajectory[start_epoch]
        txt_starting_params = txt_expert_trajectory[start_epoch]
        img_target_params = img_expert_trajectory[start_epoch + args.expert_epochs]
        txt_target_params = txt_expert_trajectory[start_epoch + args.expert_epochs]
        img_target_params = torch.cat([p.data.to(args.device).reshape(-1) for p in img_target_params], 0)
        txt_target_params = torch.cat([p.data.to(args.device).reshape(-1) for p in txt_target_params], 0)
        img_student_params = [torch.cat([p.data.to(args.device).reshape(-1) for p in img_starting_params], 0).requires_grad_(True)]
        txt_student_params = [torch.cat([p.data.to(args.device).reshape(-1) for p in txt_starting_params], 0).requires_grad_(True)]
        img_starting_params = torch.cat([p.data.to(args.device).reshape(-1) for p in img_starting_params], 0)
        txt_starting_params = torch.cat([p.data.to(args.device).reshape(-1) for p in txt_starting_params], 0)
        syn_images = image_syn
        syn_texts = text_syn

        img_param_loss_list = []
        txt_param_loss_list = []
        img_param_dist_list = []
        txt_param_dist_list = []

        for step in range(args.syn_steps):
            indices = torch.randperm(len(syn_images))
            these_indices = indices[:args.mini_batch_size]
            x = syn_images[these_indices]
            this_y = syn_texts[these_indices]
            if args.distributed:
                img_forward_params = img_student_params[-1].unsqueeze(0).expand(torch.cuda.device_count(), -1)
                txt_forward_params = txt_student_params[-1].unsqueeze(0).expand(torch.cuda.device_count(), -1)
            else:
                img_forward_params = img_student_params[-1]
                txt_forward_params = txt_student_params[-1]
            x = img_student_net(x, flat_param=img_forward_params)
            x = x / x.norm(dim=1, keepdim=True)
            this_y = txt_student_net(this_y, flat_param=txt_forward_params)
            this_y = this_y / this_y.norm(dim=1, keepdim=True)
            image_logits = syn_lr_img * x.float() @ this_y.float().t()  # Note: using syn_lr_img as logit scale.
            ground_truth = torch.arange(len(image_logits)).type_as(image_logits).long()
            contrastive_loss = (F.cross_entropy(image_logits, ground_truth) + F.cross_entropy(image_logits.t(), ground_truth)) / 2
            img_grad = torch.autograd.grad(contrastive_loss, img_student_params[-1], create_graph=True)[0]
            txt_grad = torch.autograd.grad(contrastive_loss, txt_student_params[-1], create_graph=True)[0]
            print(contrastive_loss)
            img_student_params.append(img_student_params[-1] - syn_lr_img * img_grad)
            txt_student_params.append(txt_student_params[-1] - syn_lr_txt * txt_grad)
        img_param_loss = torch.tensor(0.0).to(args.device)
        img_param_dist = torch.tensor(0.0).to(args.device)
        txt_param_loss = torch.tensor(0.0).to(args.device)
        txt_param_dist = torch.tensor(0.0).to(args.device)
        img_param_loss += torch.nn.functional.mse_loss(img_student_params[-1], img_target_params, reduction="sum")
        img_param_dist += torch.nn.functional.mse_loss(img_starting_params, img_target_params, reduction="sum")
        txt_param_loss += torch.nn.functional.mse_loss(txt_student_params[-1], txt_target_params, reduction="sum")
        txt_param_dist += torch.nn.functional.mse_loss(txt_starting_params, txt_target_params, reduction="sum")
        img_param_loss_list.append(img_param_loss)
        img_param_dist_list.append(img_param_dist)
        txt_param_loss_list.append(txt_param_loss)
        txt_param_dist_list.append(txt_param_dist)
        img_param_loss /= img_param_dist
        txt_param_loss /= txt_param_dist
        grand_loss = img_param_loss + txt_param_loss
        if math.isnan(img_param_loss):
            break
        print("img_param_loss: {}".format(img_param_loss))
        print("txt_param_loss: {}".format(txt_param_loss))
        optimizer_lr.zero_grad()
        optimizer_img.zero_grad()
        optimizer_txt.zero_grad()
        grand_loss.backward()
        print("syn_lr_img: {}".format(syn_lr_img.grad))
        print("syn_lr_txt: {}".format(syn_lr_txt.grad))
        wandb.log({"Synthetic_LR_Image": syn_lr_img.grad.detach().cpu()}, step=it)
        wandb.log({"Synthetic_LR_Text": syn_lr_txt.grad.detach().cpu()}, step=it)
        optimizer_lr.step()
        optimizer_img.step()
        optimizer_txt.step()
        wandb.log({"Grand_Loss": grand_loss.detach().cpu(), "Start_Epoch": start_epoch}, step=it)
        for param in img_student_params:
            del param
        for param in txt_student_params:
            del param
        if it % 10 == 0:
            print("%s iter = %04d, loss = %.4f" % (get_time(), it, grand_loss.item()))
    wandb.finish()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Parameter Processing")
    parser.add_argument("--dataset", type=str, default="roco", choices=["roco", "coco"], help="dataset")
    parser.add_argument("--num_queries", type=int, default=100, help="Number of queries for synthetic data")
    parser.add_argument("--lr_img", type=float, default=1000, help="Learning rate for synthetic images")
    parser.add_argument("--lr_txt", type=float, default=1000, help="Learning rate for synthetic texts")
    parser.add_argument("--lr_lr", type=float, default=1e-03, help="Learning rate for updating synthetic LRs")
    parser.add_argument("--Iteration", type=int, default=50000, help="Number of distillation iterations")
    parser.add_argument("--eval_it", type=int, default=50, help="Evaluation frequency (iterations)")
    parser.add_argument("--num_eval", type=int, default=5, help="Number of evaluations per eval iteration")
    parser.add_argument("--syn_steps", type=int, default=20, help="Number of synthetic training steps per iteration")
    parser.add_argument("--mini_batch_size", type=int, default=100, help="Mini batch size for synthetic update")
    parser.add_argument("--max_start_epoch", type=int, default=25, help="Maximum starting epoch for expert trajectory")
    parser.add_argument("--expert_epochs", type=int, default=3, help="Number of expert epochs to jump for targets")
    parser.add_argument("--ipc", type=int, default=1, help="Images per class (IPC)")
    parser.add_argument("--force_save", action="store_true", help="Force saving of synthetic images")
    parser.add_argument("--draw", type=bool, default=True, help="Enable saving drawn images")
    parser.add_argument("--transfer", type=bool, default=False, help="For transfer evaluation")
    parser.add_argument("--std", type=bool, default=False, help="Log std of metrics")
    parser.add_argument("--disable_wandb", action="store_true", help="Disable wandb logging")
    parser.add_argument("--num_experts", type=int, default=100, help="Number of expert iterations")
    parser.add_argument("--lr_teacher_img", type=float, default=0.1, help="Learning rate for teacher image model")
    parser.add_argument("--lr_teacher_txt", type=float, default=0.1, help="Learning rate for teacher text model")
    parser.add_argument("--batch_train", type=int, default=128, help="Batch size for training networks")
    parser.add_argument("--dsa", type=str, default="True", choices=["True", "False"], help="Use differentiable Siamese augmentation")
    parser.add_argument("--dsa_strategy", type=str, default="color_crop_cutout_flip_scale_rotate", help="Differentiable Siamese augmentation strategy")
    parser.add_argument("--data_path", type=str, default="/kaggle/input/roco-dataset/", help="Dataset path")
    parser.add_argument("--buffer_path", type=str, default="/kaggle/working", help="Buffer path")
    parser.add_argument("--train_epochs", type=int, default=50, help="Number of training epochs (for teacher update)")
    parser.add_argument("--zca", action="store_true", help="Use ZCA whitening")
    parser.add_argument("--decay", action="store_true", help="Enable LR decay")
    parser.add_argument("--mom", type=float, default=0, help="Momentum")
    parser.add_argument("--l2", type=float, default=0, help="L2 regularization")
    parser.add_argument("--save_interval", type=int, default=10, help="Save interval")
    parser.add_argument("--name", type=str, default=datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"), help="Name of wandb run")
    parser.add_argument("--text_pretrained", type=bool, default=True, help="Text pretrained")
    parser.add_argument("--image_pretrained", type=bool, default=True, help="Image pretrained")
    parser.add_argument("--text_trainable", type=bool, default=False, help="Text trainable")
    parser.add_argument("--image_trainable", type=bool, default=True, help="Image trainable")
    parser.add_argument("--batch_size_train", type=int, default=128, help="Training batch size")
    parser.add_argument("--batch_size_test", type=int, default=128, help="Testing batch size")
    parser.add_argument("--image_root", type=str, default="/kaggle/input/roco-dataset/all_data/train/radiology/images/", help="Location of image root")
    parser.add_argument("--ann_root", type=str, default="/kaggle/input/roco-dataset/all_data/train/radiologytraindata.csv", help="Annotation file path")
    parser.add_argument("--image_size", type=int, default=224, help="Image size")
    parser.add_argument("--k_test", type=int, default=128, help="k_test")
    parser.add_argument("--load_npy", type=bool, default=False, help="Load npy flag")
    parser.add_argument("--image_encoder", type=str, default="resnet50", choices=["nfnet", "resnet18_gn", "vit_tiny", "nf_resnet50", "nf_regnet"], help="Image encoder")
    parser.add_argument("--text_encoder", type=str, default="bert", choices=["bert", "clip"], help="Text encoder")
    parser.add_argument("--margin", default=0.2, type=float, help="Rank loss margin")
    parser.add_argument("--measure", default="cosine", help="Similarity measure (cosine|order)")
    parser.add_argument("--max_violation", action="store_true", help="Use max instead of sum in rank loss")
    parser.add_argument("--only_has_image_projection", type=bool, default=False, help="Not used")
    parser.add_argument("--grounding", type=bool, default=False, help="Not used")
    parser.add_argument("--distill", type=bool, default=False, help="If distill")
    args, unknown = parser.parse_known_args()
    if unknown:
        print("Warning: Ignoring unknown arguments:", unknown)
    main(args)
