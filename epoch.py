import os
# Help reduce CUDA fragmentation
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import numpy as np
import torch
import time
import datetime
import torch.nn as nn
from tqdm import tqdm
from utils import MetricLogger          # your existing helper
from torch.utils.data import TensorDataset
import gc

def epoch(e, dataloader, net, optimizer_img, optimizer_txt, args):
    """
    Perform a training epoch on the given dataloader.
    (Unchanged from your original code.)
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
        acc_avg  += acc * n_b
        num_exp  += n_b

        optimizer_img.zero_grad()
        optimizer_txt.zero_grad()
        loss.backward()
        optimizer_img.step()
        optimizer_txt.step()

    loss_avg /= num_exp
    acc_avg  /= num_exp
    print(f"DEBUG: Finished epoch {e}: Loss={loss_avg:.4f}, Acc={acc_avg:.4f}")
    return loss_avg, acc_avg


@torch.no_grad()
def epoch_test(dataloader, model, device, text_embed_path):
    """
    Evaluate retrieval by streaming text embeddings via NumPy memmap
    and using the original (GPU‐based) image embedding pipeline.
    """

    print("DEBUG: Starting epoch_test evaluation...")
    model.eval()
    # BLIP‐style logit scale
    logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
    logit_scale_val = logit_scale.exp().item()
    print(f"DEBUG: logit_scale = {logit_scale_val:.4f}")

    start_time = time.time()

    # ----- 1) STREAM-IN TEXT EMBEDDINGS VIA MEMMAP -----
    print(f"DEBUG: Loading text embeddings memmap from {text_embed_path}")
    data = np.load(text_embed_path, mmap_mode="r")
    bert_np = data["bert_test_embed"]       # shape (N, D)
    N, D = bert_np.shape
    print(f"DEBUG: bert_np memmap shape = ({N}, {D})")

    # ----- 2) ORIGINAL IMAGE EMBEDDING PIPELINE (GPU) -----
    print("DEBUG: Extracting image embeddings on GPU...")
    image_embeds_gpu = []
    for image, img_id in tqdm(dataloader, desc="Extracting image embeddings"):
        feat = model.image_encoder(image.to(device))
        normalized = feat / feat.norm(dim=1, keepdim=True)
        image_embeds_gpu.append(normalized)
    image_embeds_gpu = torch.cat(image_embeds_gpu, dim=0)  # shape (B, D)
    print(f"DEBUG: image_embeds_gpu shape = {tuple(image_embeds_gpu.shape)}")

    # If your model has a separate image_projection, apply it:
    if hasattr(model, "image_projection"):
        print("DEBUG: Applying image_projection...")
        proj = model.image_projection(image_embeds_gpu)
        image_embeds_gpu = proj / proj.norm(dim=1, keepdim=True)

    # ----- 3) PREALLOCATE CPU-SIDE SCORE MATRICES -----
    B = image_embeds_gpu.size(0)
    print(f"DEBUG: Pre‐allocating score arrays: i2t({B},{N}), t2i({N},{B})")
    score_i2t = np.full((B, N), -100.0, dtype=np.float32)
    score_t2i = np.full((N, B), -100.0, dtype=np.float32)

    # ----- 4) CHUNKED SIMILARITY COMPUTATION -----
    chunk_size = 32   # adjust lower if still OOM
    total_chunks = (N + chunk_size - 1) // chunk_size
    print(f"DEBUG: Will process {total_chunks} text chunks (size {chunk_size})")

    for chunk_idx, start in enumerate(range(0, N, chunk_size), start=1):
        end = min(start + chunk_size, N)
        print(f"DEBUG: Processing text chunk {chunk_idx}/{total_chunks}: indices [{start}:{end}]")

        # 4a) load that slice from memmap, move to GPU
        bert_chunk = torch.from_numpy(bert_np[start:end]).float().to(device)  # shape (c, D)

        # 4b) project & normalize
        txt_proj = model.text_projection(bert_chunk)
        txt_proj = txt_proj / txt_proj.norm(dim=1, keepdim=True)            # (c, D)
        del bert_chunk
        torch.cuda.empty_cache()

        # 4c) compute similarity on GPU, bring to CPU
        sims_gpu = logit_scale_val * (image_embeds_gpu @ txt_proj.t())       # (B, c)
        sims_chunk = sims_gpu.cpu().numpy()                                 # (B, c)
        del sims_gpu, txt_proj
        torch.cuda.empty_cache()

        # 4d) fill image→text
        score_i2t[:, start:end] = sims_chunk

        # 4e) fill text→image (top-k)
        k = 128
        for j in range(end - start):
            col = sims_chunk[:, j]
            topk_idx = np.argpartition(-col, k - 1)[:k]
            score_t2i[start + j, topk_idx] = col[topk_idx]

        # 4f) debug memory
        if device.startswith("cuda"):
            alloc = torch.cuda.memory_allocated(device) / (1024 ** 2)
            resv  = torch.cuda.memory_reserved(device)  / (1024 ** 2)
            print(f"DEBUG: After chunk {chunk_idx}: GPU alloc={alloc:.1f}MB, reserved={resv:.1f}MB")
        gc.collect()

    # ----- 5) FINISH -----
    elapsed = time.time() - start_time
    print("DEBUG: epoch_test completed in", str(datetime.timedelta(seconds=int(elapsed))))
    return score_i2t, score_t2i


@torch.no_grad()
def itm_eval(scores_i2t, scores_t2i, txt2img, img2txt):
    """
    Compute retrieval metrics from CPU numpy arrays.
    (Unchanged from your original code.)
    """
    print("DEBUG: Starting itm_eval...")
    # Image→Text
    ranks = np.zeros(scores_i2t.shape[0], dtype=np.int32)
    for i, row in enumerate(scores_i2t):
        inds = np.argsort(row)[::-1]
        best = min(np.where(np.isin(inds, img2txt[i]))[0])
        ranks[i] = best
    tr1 = 100.0 * (ranks < 1).sum() / len(ranks)
    tr5 = 100.0 * (ranks < 5).sum() / len(ranks)
    tr10= 100.0 * (ranks < 10).sum()/ len(ranks)

    # Text→Image
    ranks = np.zeros(scores_t2i.shape[0], dtype=np.int32)
    for i, row in enumerate(scores_t2i):
        inds = np.argsort(row)[::-1]
        ranks[i] = np.where(inds == txt2img[i])[0][0]
    ir1 = 100.0 * (ranks < 1).sum() / len(ranks)
    ir5 = 100.0 * (ranks < 5).sum() / len(ranks)
    ir10= 100.0 * (ranks < 10).sum()/ len(ranks)

    results = {
        'txt_r1': tr1, 'txt_r5': tr5, 'txt_r10': tr10, 'txt_r_mean': (tr1+tr5+tr10)/3,
        'img_r1': ir1, 'img_r5': ir5, 'img_r10': ir10, 'img_r_mean': (ir1+ir5+ir10)/3,
        'r_mean': (tr1+tr5+tr10+ir1+ir5+ir10)/6
    }
    print("DEBUG: itm_eval results:", results)
    return results


def evaluate_synset(it_eval, net, images_train, labels_train, testloader, args, bert_test_embed, return_loss=False):
    """
    Synset evaluation combining epoch_test + itm_eval.
    (Unchanged from your original code.)
    """
    print("DEBUG: Starting evaluate_synset()...")
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
    t0 = time.time()
    for ep in range(Epoch + 1):
        l, a = epoch(ep, loader, net, opt_img, opt_txt, args)
        losses.append(l); accs.append(a)
        print(f"DEBUG: Synset train epoch {ep}: Loss={l:.4f}, Acc={a:.4f}")
        if ep == Epoch:
            print("DEBUG: Final synset evaluation...")
            i2t, t2i = epoch_test(testloader, net, args.device, bert_test_embed)
            res    = itm_eval(i2t, t2i, testloader.dataset.txt2img, testloader.dataset.img2txt)
    t1 = time.time() - t0
    print("DEBUG: evaluate_synset total time:", str(datetime.timedelta(seconds=int(t1))))
    return net, accs, res
