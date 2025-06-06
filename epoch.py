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
    """
    print(f"DEBUG: Starting training epoch {e}...")
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
    print(f"DEBUG: Finished training epoch {e}: Average Loss = {loss_avg}, Average Accuracy = {acc_avg}")
    return loss_avg, acc_avg


@torch.no_grad()
def epoch_test(dataloader, model, device, bert_test_embed):
    """
    Evaluate the model on the retrieval task.
    Computes the similarity matrix in very small chunks on the CPU.
    """
    print("DEBUG: Starting epoch_test evaluation...")
    model.eval()
    # Set logit scale as used in BLIP
    logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
    print("DEBUG: Logit scale parameter set.")
    start_time = time.time()

    # --- Compute text embeddings ---
    print("DEBUG: Starting computation of text embeddings on GPU...")
    txt_embed = model.text_projection(bert_test_embed.float().to(device))
    text_embeds = txt_embed / txt_embed.norm(dim=1, keepdim=True)
    print(f"DEBUG: Text embeddings computed. Shape: {text_embeds.shape}")
    print("DEBUG: Moving text embeddings to CPU...")
    text_embeds = text_embeds.cpu()
    torch.cuda.empty_cache()
    gc.collect()
    print("DEBUG: GPU memory cleared after text embeddings.")

    # --- Compute image embeddings ---
    print("DEBUG: Starting extraction of image embeddings...")
    image_embeds_list = []
    for image, img_id in tqdm(dataloader, desc="Extracting image embeddings"):
        image_feat = model.image_encoder(image.to(device))
        im_embed = image_feat / image_feat.norm(dim=1, keepdim=True)
        image_embeds_list.append(im_embed)
    image_embeds = torch.cat(image_embeds_list, dim=0)
    print(f"DEBUG: Finished extracting image embeddings. Total images: {image_embeds.size(0)}")
    # Optionally use image projection if your model supports it.
    use_image_projection = False
    if use_image_projection:
        print("DEBUG: Applying image projection...")
        im_proj = model.image_projection(image_embeds.float())
        image_embeds = im_proj / im_proj.norm(dim=1, keepdim=True)
    else:
        image_embeds = image_embeds / image_embeds.norm(dim=1, keepdim=True)
    print("DEBUG: Normalizing and moving image embeddings to CPU...")
    image_embeds = image_embeds.cpu()
    torch.cuda.empty_cache()
    gc.collect()
    print("DEBUG: Image embeddings moved to CPU and memory cleared.")

    # Bring logit scale (its exponential) to CPU as a scalar
    logit_scale_val = logit_scale.exp().item()
    print(f"DEBUG: Logit scale value computed: {logit_scale_val}")

    # --- Compute similarity matrix in very small chunks on CPU ---
    print("DEBUG: Starting similarity matrix computation on CPU...")
    B = image_embeds.size(0)
    N = text_embeds.size(0)
    chunk_size = 32  # If needed, lower further to 16 or 8.
    sims_list = []
    total_chunks = (N + chunk_size - 1) // chunk_size
    print(f"DEBUG: Total text embedding chunks to process: {total_chunks}")
    for i in range(0, N, chunk_size):
        try:
            text_chunk = text_embeds[i:min(i + chunk_size, N)]
            print(f"DEBUG: Processing chunk: shape {text_chunk.shape} for indices {i} to {min(i + chunk_size, N)}")
            sims_chunk = logit_scale_val * (image_embeds @ text_chunk.t())
            sims_list.append(sims_chunk)
            print(f"DEBUG: Processed text embeddings chunk from index {i} to {min(i + chunk_size, N)}")
            if device.lower().startswith("cuda"):
                allocated = torch.cuda.memory_allocated(device)
                reserved = torch.cuda.memory_reserved(device)
                print(f"DEBUG: After chunk {i}-{min(i+chunk_size, N)}: Allocated {allocated/1e6:.2f} MB, Reserved {reserved/1e6:.2f} MB")
            else:
                print("DEBUG: Running on CPU.")
            gc.collect()
        except Exception as e:
            print(f"ERROR: Processing text chunk starting at index {i} failed: {e}")
            raise
    sims_matrix = torch.cat(sims_list, dim=1)  # Final shape: (B, N)
    print(f"DEBUG: Completed similarity matrix computation. Shape of similarity matrix: {sims_matrix.shape}")

    if device.lower().startswith("cuda"):
        print("DEBUG: GPU Memory Summary after similarity computation:")
        print(torch.cuda.memory_summary(device=device, abbreviated=True))

    # --- Build retrieval score matrices ---
    print("DEBUG: Building image-to-text score matrix...")
    score_matrix_i2t = torch.full((B, N), -100.0)
    k = 128
    for i, sims in enumerate(sims_matrix):
        topk_sim, topk_idx = sims.topk(k=k, dim=0)
        score_matrix_i2t[i, topk_idx] = topk_sim
    print("DEBUG: Finished building image-to-text score matrix.")

    print("DEBUG: Building text-to-image score matrix...")
    sims_matrix_t = sims_matrix.t()  # shape: (N, B)
    score_matrix_t2i = torch.full((N, B), -100.0)
    for i, sims in enumerate(sims_matrix_t):
        topk_sim, topk_idx = sims.topk(k=k, dim=0)
        score_matrix_t2i[i, topk_idx] = topk_sim
    print("DEBUG: Finished building text-to-image score matrix.")

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print("DEBUG: Total evaluation time:", total_time_str)

    return score_matrix_i2t.numpy(), score_matrix_t2i.numpy()


@torch.no_grad()
def itm_eval(scores_i2t, scores_t2i, txt2img, img2txt):
    """
    Evaluate image-text matching. Computes retrieval metrics.
    """
    print("DEBUG: Starting retrieval evaluation (itm_eval)...")
    # --- Image-to-Text Retrieval ---
    ranks = np.zeros(scores_i2t.shape[0])
    print(f"DEBUG: Number of image queries (TR): {len(ranks)}")
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
    print("DEBUG: Image-to-text retrieval metrics computed.")

    # --- Text-to-Image Retrieval ---
    ranks = np.zeros(scores_t2i.shape[0])
    print(f"DEBUG: Number of text queries (IR): {len(ranks)}")
    for index, score in enumerate(scores_t2i):
        inds = np.argsort(score)[::-1]
        ranks[index] = np.where(inds == txt2img[index])[0][0]
    ir1 = 100.0 * len(np.where(ranks < 1)[0]) / len(ranks)
    ir5 = 100.0 * len(np.where(ranks < 5)[0]) / len(ranks)
    ir10 = 100.0 * len(np.where(ranks < 10)[0]) / len(ranks)
    print("DEBUG: Text-to-image retrieval metrics computed.")

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
    print("DEBUG: Retrieval evaluation (itm_eval) complete. Result:", eval_result)
    return eval_result


def evaluate_synset(it_eval, net, images_train, labels_train, testloader, args, bert_test_embed, return_loss=False):
    """
    Additional evaluation function to assess performance on a training subset ("synset").
    """
    print("DEBUG: Starting evaluate_synset()...")
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

    for ep in tqdm(range(Epoch + 1), desc="Synset Evaluation"):
        print(f"DEBUG: Beginning epoch {ep} of synset training...")
        loss_train, acc_train = epoch(ep, trainloader, net, optimizer_img, optimizer_txt, args)
        acc_train_list.append(acc_train)
        loss_train_list.append(loss_train)
        print(f"DEBUG: Finished epoch {ep} of synset training: Loss = {loss_train}, Accuracy = {acc_train}")
        if ep == Epoch:
            with torch.no_grad():
                print("DEBUG: Starting evaluation of synset with epoch_test()...")
                score_val_i2t, score_val_t2i = epoch_test(testloader, net, args.device, bert_test_embed)
                print(f"DEBUG: score_val_i2t (first 2 rows): {score_val_i2t[:2]}")
                print(f"DEBUG: score_val_t2i (first 2 rows): {score_val_t2i[:2]}")
                val_result = itm_eval(score_val_i2t, score_val_t2i,
                                      testloader.dataset.txt2img,
                                      testloader.dataset.img2txt)
                print(f"DEBUG: Synset evaluation result: {val_result}")
            lr *= 0.1
            optimizer_img = torch.optim.SGD(net.image_encoder.parameters(), lr=lr, momentum=0.9, weight_decay=0.0005)
            optimizer_txt = torch.optim.SGD(net.text_projection.parameters(), lr=lr, momentum=0.9, weight_decay=0.0005)

    time_train = time.time() - start
    print("DEBUG: Synset Evaluation Time:", str(datetime.timedelta(seconds=int(time_train))))
    return net, acc_train_list, val_result
