'''
 * part of the code (i.e. def epoch_test() and itm_eval()) is from:
   https://github.com/salesforce/BLIP/blob/main/train_retrieval.py#L69
 * Copyright (c) 2022, salesforce.com, inc.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 * For full license text, see LICENSE.txt file in the repo root or 
   https://opensource.org/licenses/BSD-3-Clause
 * By Junnan Li
'''

import os
# Enable expandable segments to help reduce fragmentation
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import numpy as np
import torch
import time
import datetime
import torch.nn.functional as F
from tqdm import tqdm
import torch.nn as nn
from utils import *
from torch.utils.data import TensorDataset
import gc

def epoch(e, dataloader, net, optimizer_img, optimizer_txt, args):
    """
    Perform a training epoch on the given dataloader.

    Args:
        dataloader (torch.utils.data.DataLoader): The dataloader for iterating over the dataset.
        net: The model.
        optimizer_img: The optimizer for image parameters.
        optimizer_txt: The optimizer for text parameters.
        args (object): The arguments specifying the training configuration.

    Returns:
        Tuple of average loss and average accuracy.
    """
    net = net.to(args.device)
    net.train()
    loss_avg, acc_avg, num_exp = 0, 0, 0

    for i, data in tqdm(enumerate(dataloader), desc=f"Training Epoch {e}"):
        if args.distill:
            image, caption = data[:2]
        else:
            image, caption, index = data[:3]

        image = image.to(args.device)
        n_b = image.shape[0]

        loss, acc = net(image, caption, e)

        loss_avg += loss.item() * n_b
        acc_avg += acc
        num_exp += n_b

        optimizer_img.zero_grad()
        optimizer_txt.zero_grad()
        loss.backward()
        optimizer_img.step()
        optimizer_txt.step()

    loss_avg /= num_exp
    acc_avg /= num_exp

    return loss_avg, acc_avg


@torch.no_grad()
def epoch_test(dataloader, model, device, bert_test_embed):
    """
    Evaluate the model on the retrieval task.
    This function is adapted from BLIP's code (by Junnan Li) with modifications
    to compute the similarity matrix in very small chunks on the CPU rather than GPU.
    This avoids OOM errors.

    Args:
        dataloader (torch.utils.data.DataLoader): DataLoader for images.
        model: The model.
        device: Device string ('cuda' or 'cpu') for feature extraction.
        bert_test_embed: Pre-computed text embeddings (BERT features)

    Returns:
        A tuple of two numpy arrays representing the image-to-text and text-to-image scores.
    """
    model.eval()
    # Set logit scale as used in BLIP
    logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
    print('Computing features for evaluation...')
    start_time = time.time()

    # --- Compute text embeddings ---
    # First, compute and normalize text embeddings on GPU
    txt_embed = model.text_projection(bert_test_embed.float().to(device))
    text_embeds = txt_embed / txt_embed.norm(dim=1, keepdim=True)
    # Now move text embeddings to CPU to free GPU memory
    text_embeds = text_embeds.cpu()
    torch.cuda.empty_cache()
    gc.collect()

    # --- Compute image embeddings ---
    image_embeds_list = []
    for image, img_id in tqdm(dataloader, desc="Extracting image embeddings"):
        image_feat = model.image_encoder(image.to(device))
        im_embed = image_feat / image_feat.norm(dim=1, keepdim=True)
        image_embeds_list.append(im_embed)
    image_embeds = torch.cat(image_embeds_list, dim=0)
    # Optionally use an image projection if your model supports it.
    use_image_projection = False
    if use_image_projection:
        im_proj = model.image_projection(image_embeds.float())
        image_embeds = im_proj / im_proj.norm(dim=1, keepdim=True)
    else:
        image_embeds = image_embeds / image_embeds.norm(dim=1, keepdim=True)
    # Move image embeddings to CPU
    image_embeds = image_embeds.cpu()
    torch.cuda.empty_cache()
    gc.collect()

    # Bring logit scale (its exponential) to CPU as a scalar
    logit_scale_val = logit_scale.exp().item()

    # --- Compute similarity matrix in very small chunks on CPU ---
    B = image_embeds.size(0)
    N = text_embeds.size(0)
    # Lower chunk size further to reduce instantaneous memory requirements.
    chunk_size = 32
    sims_list = []
    for i in range(0, N, chunk_size):
        text_chunk = text_embeds[i:min(i+chunk_size, N)]
        # Compute similarity on CPU without further GPU usage.
        sims_chunk = logit_scale_val * (image_embeds @ text_chunk.t())
        sims_list.append(sims_chunk)
        gc.collect()
    sims_matrix = torch.cat(sims_list, dim=1)  # Final shape: (B, N)

    # Optionally log GPU memory usage if device is CUDA
    if device.lower().startswith("cuda"):
        print(torch.cuda.memory_summary(device=device, abbreviated=True))

    # --- Build retrieval score matrices ---
    # Build image-to-text score matrix (for each image, record top-k similarities)
    score_matrix_i2t = torch.full((B, N), -100.0)
    k = 128
    for i, sims in enumerate(sims_matrix):
        topk_sim, topk_idx = sims.topk(k=k, dim=0)
        score_matrix_i2t[i, topk_idx] = topk_sim

    # For text-to-image retrieval, work with the transpose of the similarity matrix.
    sims_matrix_t = sims_matrix.t()  # shape: (N, B)
    score_matrix_t2i = torch.full((N, B), -100.0)
    for i, sims in enumerate(sims_matrix_t):
        topk_sim, topk_idx = sims.topk(k=k, dim=0)
        score_matrix_t2i[i, topk_idx] = topk_sim

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Evaluation time {}'.format(total_time_str))

    return score_matrix_i2t.numpy(), score_matrix_t2i.numpy()


@torch.no_grad()
def itm_eval(scores_i2t, scores_t2i, txt2img, img2txt):
    """
    Evaluate image-text matching. Computes retrieval metrics for both image-to-text and text-to-image.
    This function is adapted from the BLIP repository.

    Args:
        scores_i2t (np.array): Similarity scores computed for image-to-text retrieval.
        scores_t2i (np.array): Similarity scores computed for text-to-image retrieval.
        txt2img (list or np.array): Ground-truth mapping from text captions to image indices.
        img2txt (list or np.array): Ground-truth mapping from images to text indices.

    Returns:
        Dictionary containing retrieval metrics.
    """
    # --- Image-to-Text Retrieval ---
    ranks = np.zeros(scores_i2t.shape[0])
    print("TR: ", len(ranks))
    for index, score in enumerate(scores_i2t):
        inds = np.argsort(score)[::-1]
        rank = 1e20  # start with a very large value
        for i in img2txt[index]:
            tmp = np.where(inds == i)[0][0]
            if tmp < rank:
                rank = tmp
        ranks[index] = rank

    tr1 = 100.0 * len(np.where(ranks < 1)[0]) / len(ranks)
    tr5 = 100.0 * len(np.where(ranks < 5)[0]) / len(ranks)
    tr10 = 100.0 * len(np.where(ranks < 10)[0]) / len(ranks)
  
    # --- Text-to-Image Retrieval ---
    ranks = np.zeros(scores_t2i.shape[0])
    print("IR: ", len(ranks))
    for index, score in enumerate(scores_t2i):
        inds = np.argsort(score)[::-1]
        ranks[index] = np.where(inds == txt2img[index])[0][0]

    ir1 = 100.0 * len(np.where(ranks < 1)[0]) / len(ranks)
    ir5 = 100.0 * len(np.where(ranks < 5)[0]) / len(ranks)
    ir10 = 100.0 * len(np.where(ranks < 10)[0]) / len(ranks)
    
    tr_mean = (tr1 + tr5 + tr10) / 3
    ir_mean = (ir1 + ir5 + ir10) / 3
    r_mean = (tr_mean + ir_mean) / 2

    eval_result = {
        'txt_r1': tr1,
        'txt_r5': tr5,
        'txt_r10': tr10,
        'txt_r_mean': tr_mean,
        'img_r1': ir1,
        'img_r5': ir5,
        'img_r10': ir10,
        'img_r_mean': ir_mean,
        'r_mean': r_mean
    }
    return eval_result


def evaluate_synset(it_eval, net, images_train, labels_train, testloader, args, bert_test_embed, return_loss=False):
    """
    Additional evaluation function to assess performance on a training subset (the "synset").
    This function trains the network on the given training data and evaluates using epoch_test() and itm_eval().

    Args:
        it_eval: Evaluation iteration.
        net: The model.
        images_train: Training images.
        labels_train: Training labels.
        testloader: Test DataLoader.
        args: Arguments.
        bert_test_embed: Pre-computed text embeddings.
        return_loss: Whether or not to return training loss.

    Returns:
        A tuple (net, acc_train_list, val_result)
    """
    net = net.to(args.device)
    images_train = images_train.to(args.device)
    labels_train = labels_train.to(args.device)
    lr = float(args.lr_net)
    Epoch = int(args.epoch_eval_train)
    optimizer_img = torch.optim.SGD(net.image_encoder.parameters(), lr=lr, momentum=0.9, weight_decay=0.0005)
    optimizer_txt = torch.optim.SGD(net.text_projection.parameters(), lr=lr, momentum=0.9, weight_decay=0.0005)

    dst_train = TensorDataset(images_train, labels_train)
    trainloader = torch.utils.data.DataLoader(dst_train, batch_size=args.batch_train, shuffle=True, num_workers=0)

    start = time.time()
    acc_train_list = []
    loss_train_list = []

    for ep in tqdm(range(Epoch+1), desc="Synset Evaluation"):
        loss_train, acc_train = epoch(ep, trainloader, net, optimizer_img, optimizer_txt, args)
        acc_train_list.append(acc_train)
        loss_train_list.append(loss_train)
        if ep == Epoch:
            with torch.no_grad():
                score_val_i2t, score_val_t2i = epoch_test(testloader, net, args.device, bert_test_embed)
                print(f"score_val_i2t: {score_val_i2t}, score_val_t2i: {score_val_t2i}")
                val_result = itm_eval(score_val_i2t, score_val_t2i,
                                      testloader.dataset.txt2img,
                                      testloader.dataset.img2txt)
                print(f"val_result: {val_result}")
            lr *= 0.1
            optimizer_img = torch.optim.SGD(net.image_encoder.parameters(), lr=lr, momentum=0.9, weight_decay=0.0005)
            optimizer_txt = torch.optim.SGD(net.text_projection.parameters(), lr=lr, momentum=0.9, weight_decay=0.0005)

    time_train = time.time() - start
    print("Synset Evaluation Time:", str(datetime.timedelta(seconds=int(time_train))))
    return net, acc_train_list, val_result
