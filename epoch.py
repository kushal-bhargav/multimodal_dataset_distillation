import os
# Make PyTorch’s allocator more flexible (reduce fragmentation)
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import numpy as np
import torch
import time
import datetime
import torch.nn.functional as F
from tqdm import tqdm
import torch.nn as nn
from utils import *             # Your MetricLogger, etc.
from torch.utils.data import TensorDataset
import gc

def epoch(e, dataloader, net, optimizer_img, optimizer_txt, args):
    """
    Perform a training epoch on the given dataloader.
    """
    print(f"DEBUG: Starting training epoch {e}...")
    net = net.to(args.device)
    net.train()
    loss_avg, acc_avg, num_exp = 0.0, 0.0, 0

    for i, data in tqdm(enumerate(dataloader), desc=f"Training Epoch {e}"):
        if args.distill:
            image, caption = data[:2]
        else:
            image, caption, index = data[:3]

        image = image.to(args.device)
        n_b = image.size(0)

        loss, acc = net(image, caption, e)
        loss_avg += loss.item() * n_b
        acc_avg += acc * n_b
        num_exp += n_b

        optimizer_img.zero_grad()
        optimizer_txt.zero_grad()
        loss.backward()
        optimizer_img.step()
        optimizer_txt.step()

    loss_avg /= num_exp
    acc_avg  /= num_exp
    print(f"DEBUG: Finished training epoch {e}: Loss={loss_avg:.4f}, Acc={acc_avg:.4f}")
    return loss_avg, acc_avg


@torch.no_grad()
def epoch_test(dataloader, model, device, bert_test_embed):
    """
    Evaluate the model on the retrieval task by streaming similarity
    computation in small chunks onto CPU to bound memory usage.
    """
    print("DEBUG: Starting epoch_test evaluation...")
    model.eval()

    # BLIP‐style logit scale
    logit_scale = nn.Parameter(torch.ones([]) * np.log(1/0.07))
    logit_scale_val = logit_scale.exp().item()
    print(f"DEBUG: logit_scale = {logit_scale_val:.4f}")

    start_time = time.time()

    # ---- 1) Text embeddings (GPU→CPU) ----
    print("DEBUG: Computing text embeddings on GPU...")
    txt_embed = model.text_projection(bert_test_embed.float().to(device))
    text_embeds = txt_embed / txt_embed.norm(dim=1, keepdim=True)
    print(f"DEBUG: text_embeds.shape = {tuple(text_embeds.shape)}")
    print("DEBUG: Moving text embeddings to CPU and clearing GPU memory...")
    text_embeds = text_embeds.cpu()
    torch.cuda.empty_cache(); gc.collect()

    # ---- 2) Image embeddings (GPU→CPU) ----
    print("DEBUG: Extracting image embeddings...")
    image_list = []
    for image, img_id in tqdm(dataloader, desc="Extracting image embeddings"):
        feat = model.image_encoder(image.to(device))
        normed = feat / feat.norm(dim=1, keepdim=True)
        image_list.append(normed.cpu())
    image_embeds = torch.cat(image_list, dim=0)  # CPU tensor (B, D)
    del image_list
    torch.cuda.empty_cache(); gc.collect()
    print(f"DEBUG: image_embeds.shape = {tuple(image_embeds.shape)}")

    # ---- 3) Pre‐allocate score matrices on CPU ----
    B = image_embeds.size(0)
    N = text_embeds.size(0)
    print(f"DEBUG: Pre-allocating score matrices: i2t=({B},{N}), t2i=({N},{B})")
    score_i2t = np.full((B, N), -100.0, dtype=np.float32)
    score_t2i = np.full((N, B), -100.0, dtype=np.float32)

    # ---- 4) Stream through text embeddings in small chunks ----
    chunk_size = 16   # reduce further if necessary
    total_chunks = (N + chunk_size - 1) // chunk_size
    print(f"DEBUG: Will process {total_chunks} chunks of size {chunk_size} each.")

    for chunk_idx, start in enumerate(range(0, N, chunk_size), 1):
        end = min(start + chunk_size, N)
        text_chunk = text_embeds[start:end]         # CPU tensor (c, D)
        # Compute similarity on CPU: (B, D) @ (D, c) → (B, c)
        sims_chunk = logit_scale_val * (image_embeds @ text_chunk.t())
        sims_np = sims_chunk.numpy()                # (B, c) float32

        # 4a) Fill image→text portion
        score_i2t[:, start:end] = sims_np

        # 4b) For each text in this chunk, compute top-k images
        k = 128
        for j in range(end - start):
            col = sims_np[:, j]                     # shape (B,)
            idx = np.argpartition(-col, k-1)[:k]    # unsorted top k
            vals = col[idx]
            score_t2i[start+j, idx] = vals

        # Debug prints
        allocated = torch.cuda.memory_allocated(device) / (1024**2)
        reserved  = torch.cuda.memory_reserved(device)  / (1024**2)
        print(f"DEBUG: Chunk {chunk_idx}/{total_chunks} [{start}:{end}]  "
              f"-> sims_np.shape={sims_np.shape}  "
              f"GPU mem allocated={allocated:.1f} MB reserved={reserved:.1f} MB")
        del text_chunk, sims_chunk, sims_np
        gc.collect()

    # ---- 5) Final timing & return ----
    elapsed = time.time() - start_time
    print(f"DEBUG: epoch_test completed in {str(datetime.timedelta(seconds=int(elapsed)))}")
    return score_i2t, score_t2i


@torch.no_grad()
def itm_eval(scores_i2t, scores_t2i, txt2img, img2txt):
    """
    Compute retrieval metrics from numpy score matrices.
    """
    print("DEBUG: Starting itm_eval...")
    # Image→Text
    ranks = np.zeros(scores_i2t.shape[0], dtype=np.int32)
    for i, row in enumerate(scores_i2t):
        inds = np.argsort(row)[::-1]
        best = min(np.where(np.isin(inds, img2txt[i]))[0])
        ranks[i] = best
    tr1 = 100*(ranks<1).sum()/len(ranks)
    tr5 = 100*(ranks<5).sum()/len(ranks)
    tr10= 100*(ranks<10).sum()/len(ranks)

    # Text→Image
    ranks = np.zeros(scores_t2i.shape[0], dtype=np.int32)
    for i, row in enumerate(scores_t2i):
        inds = np.argsort(row)[::-1]
        ranks[i] = np.where(inds==txt2img[i])[0][0]
    ir1 = 100*(ranks<1).sum()/len(ranks)
    ir5 = 100*(ranks<5).sum()/len(ranks)
    ir10= 100*(ranks<10).sum()/len(ranks)

    result = {
        'txt_r1': tr1, 'txt_r5': tr5, 'txt_r10': tr10, 
        'txt_r_mean': (tr1+tr5+tr10)/3,
        'img_r1': ir1, 'img_r5': ir5, 'img_r10': ir10,
        'img_r_mean': (ir1+ir5+ir10)/3,
        'r_mean': (tr1+tr5+tr10+ir1+ir5+ir10)/6
    }
    print("DEBUG: itm_eval results:", result)
    return result


def evaluate_synset(it_eval, net, images_train, labels_train, testloader, args, bert_test_embed, return_loss=False):
    """
    Runs training on the “synset” then calls epoch_test + itm_eval.
    """
    print("DEBUG: Starting evaluate_synset...")
    net = net.to(args.device)
    images_train = images_train.to(args.device)
    labels_train = labels_train.to(args.device)

    lr = float(args.lr_net)
    Epoch = int(args.epoch_eval_train)
    opt_img = torch.optim.SGD(net.image_encoder.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4)
    opt_txt = torch.optim.SGD(net.text_projection.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4)

    dst = TensorDataset(images_train, labels_train)
    loader = torch.utils.data.DataLoader(dst, batch_size=args.batch_train, shuffle=True)

    accs, losses = [], []
    start = time.time()
    for ep in range(Epoch+1):
        l, a = epoch(ep, loader, net, opt_img, opt_txt, args)
        accs.append(a); losses.append(l)
        print(f"DEBUG: Synset train epoch {ep} → loss {l:.4f} acc {a:.4f}")
        if ep == Epoch:
            print("DEBUG: Final synset evaluation with epoch_test()...")
            i2t, t2i = epoch_test(testloader, net, args.device, bert_test_embed)
            res   = itm_eval(i2t, t2i, testloader.dataset.txt2img, testloader.dataset.img2txt)
    total = time.time() - start
    print("DEBUG: evaluate_synset done in", str(datetime.timedelta(seconds=int(total))))
    return net, accs, res
