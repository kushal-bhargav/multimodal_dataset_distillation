'''
 * part of the code (i.e. def epoch_test() and itm_eval()) is from: https://github.com/salesforce/BLIP/blob/main/train_retrieval.py#L69
 * Copyright (c) 2022, salesforce.com, inc.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 * For full license text, see LICENSE.txt file in the repo root or https://opensource.org/licenses/BSD-3-Clause
 * By Junnan Li
'''

import numpy as np
import torch
import time
import datetime
import torch.nn.functional as F
from tqdm import tqdm
import torch.nn as nn
from utils import *
from torch.utils.data import TensorDataset

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
    to compute the similarity matrix in smaller chunks to avoid CUDA OOM errors.

    Args:
        dataloader (torch.utils.data.DataLoader): DataLoader for images.
        model: The model.
        device: Device string ('cuda' or 'cpu')
        bert_test_embed: Pre-computed text embeddings (BERT features)

    Returns:
        A tuple of two numpy arrays representing the image-to-text and text-to-image scores.
    """
    model.eval() 
    # Use fixed logit scale as in BLIP
    logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
    metric_logger = MetricLogger(delimiter="  ")
    header = 'Evaluation:'    
    print('Computing features for evaluation...')
    start_time = time.time()  

    # Compute normalized text embeddings
    txt_embed = model.text_projection(bert_test_embed.float().to('cuda'))
    text_embeds = txt_embed / txt_embed.norm(dim=1, keepdim=True)  # e.g. shape: (5000, 768)
    text_embeds = text_embeds.to(device)

    # Compute normalized image embeddings from dataloader
    image_embeds = []
    for image, img_id in dataloader: 
        image_feat = model.image_encoder(image.to(device))
        im_embed = image_feat / image_feat.norm(dim=1, keepdim=True)
        image_embeds.append(im_embed)
    image_embeds = torch.cat(image_embeds, dim=0)
    
    use_image_projection = False
    if use_image_projection:
        im_proj = model.image_projection(image_embeds.float())
        image_embeds = im_proj / im_proj.norm(dim=1, keepdim=True)
    else:
        image_embeds = image_embeds / image_embeds.norm(dim=1, keepdim=True)
        
    # --- Chunked computation of similarity matrix ---
    B = image_embeds.size(0)
    N = text_embeds.size(0)
    chunk_size = 256  # Adjust this value based on available GPU memory
    sims_list = []
    for i in range(0, N, chunk_size):
        text_chunk = text_embeds[i:min(i+chunk_size, N)]
        # Each chunk: (B, chunk_size)
        sims_chunk = logit_scale.exp() * (image_embeds @ text_chunk.t())
        sims_list.append(sims_chunk)
    sims_matrix = torch.cat(sims_list, dim=1)  # Final shape: (B, N)

    # Compute image-to-text scores using top-k selection
    score_matrix_i2t = torch.full((B, N), -100.0).to(device)
    for i, sims in enumerate(sims_matrix):
        topk_sim, topk_idx = sims.topk(k=128, dim=0)
        score_matrix_i2t[i, topk_idx] = topk_sim 

    # For text-to-image retrieval, transpose the similarity matrix and compute top-k again
    sims_matrix = sims_matrix.t()  # shape now: (N, B)
    score_matrix_t2i = torch.full((N, B), -100.0).to(device)
    for i, sims in enumerate(sims_matrix):
        topk_sim, topk_idx = sims.topk(k=128, dim=0)
        score_matrix_t2i[i, topk_idx] = topk_sim

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Evaluation time {}'.format(total_time_str)) 

    return score_matrix_i2t.cpu().numpy(), score_matrix_t2i.cpu().numpy()


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
    # Image-to-Text Evaluation
    ranks = np.zeros(scores_i2t.shape[0])
    print("TR: ", len(ranks))
    for index, score in enumerate(scores_i2t):
        inds = np.argsort(score)[::-1]
        rank = 1e20  # large initial value
        for i in img2txt[index]:
            tmp = np.where(inds == i)[0][0]
            if tmp < rank:
                rank = tmp
        ranks[index] = rank

    tr1 = 100.0 * len(np.where(ranks < 1)[0]) / len(ranks)
    tr5 = 100.0 * len(np.where(ranks < 5)[0]) / len(ranks)
    tr10 = 100.0 * len(np.where(ranks < 10)[0]) / len(ranks)
  
    # Text-to-Image Evaluation
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
    lr_schedule = [Epoch//2+1]
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
                val_result = itm_eval(score_val_i2t, score_val_t2i, testloader.dataset.txt2img, testloader.dataset.img2txt)  
            lr *= 0.1 
            optimizer_img = torch.optim.SGD(net.image_encoder.parameters(), lr=lr, momentum=0.9, weight_decay=0.0005)
            optimizer_txt = torch.optim.SGD(net.text_projection.parameters(), lr=lr, momentum=0.9, weight_decay=0.0005)

    time_train = time.time() - start
    return net, acc_train_list, val_result
