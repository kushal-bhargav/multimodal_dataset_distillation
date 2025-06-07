import os
# Help reduce CUDA fragmentation
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import numpy as np
import torch
import time
import datetime
import torch.nn as nn
from tqdm import tqdm
from utils import MetricLogger  # your existing helper, if any
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
def epoch_test(dataloader, model, device, text_embed_input):
    """
    Evaluate the model on the retrieval task.
    Computes the similarity matrix in very small chunks on the CPU.
    
    The argument 'text_embed_input' can be either:
      - a string representing a file path (to a saved NPZ file with key "bert_test_embed")
      - or a torch.Tensor containing the text embeddings.
    """
    print("DEBUG: Starting epoch_test evaluation...")
    model.eval()
    # BLIP-style logit scale
    logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
    logit_scale_val = logit_scale.exp().item()
    print(f"DEBUG: logit_scale = {logit_scale_val:.4f}")

    start_time = time.time()

    # ---- Process text embeddings ----
    if isinstance(text_embed_input, str):
        # Load from file using memmap
        print(f"DEBUG: Loading text embeddings memmap from {text_embed_input}")
        data = np.load(text_embed_input, mmap_mode="r")
        bert_np = data["bert_test_embed"]  # shape: (N, D)
    elif isinstance(text_embed_input, torch.Tensor):
        print("DEBUG: Received text embeddings as a Tensor; moving to CPU and converting to numpy.")
        bert_np = text_embed_input.cpu().numpy()
    else:
        raise ValueError("text_embed_input must be either a file path (str) or a torch.Tensor")

    N, D = bert_np.shape
    print(f"DEBUG: Loaded text embeddings shape: ({N}, {D})")

    # ---- Image embeddings (as before) ----
    print("DEBUG: Extracting image embeddings on GPU...")
    image_embeds_list = []
    for image, img_id in tqdm(dataloader, desc="Extracting image embeddings"):
        feat = model.image_encoder(image.to(device))
        normalized = feat / feat.norm(dim=1, keepdim=True)
        image_embeds_list.append(normalized)
    image_embeds = torch.cat(image_embeds_list, dim=0)  # shape (B, D)
    print(f"DEBUG: image_embeds.shape = {tuple(image_embeds.shape)}")
    if hasattr(model, "image_projection"):
        print("DEBUG: Applying image_projection...")
        proj = model.image_projection(image_embeds)
        image_embeds = proj / proj.norm(dim=1, keepdim=True)
    image_embeds = image_embeds.cpu()
    torch.cuda.empty_cache()
    gc.collect()
    print("DEBUG: Image embeddings moved to CPU.")

    # ---- Preallocate score matrices on CPU ----
    B = image_embeds.size(0)
    print(f"DEBUG: Preallocating score matrices: image-to-text ({B}, {N}) and text-to-image ({N}, {B})")
    score_i2t = np.full((B, N), -100.0, dtype=np.float32)
    score_t2i = np.full((N, B), -100.0, dtype=np.float32)

    # ---- Process text embeddings in chunks ----
    chunk_size = 32  # adjust if necessary
    total_chunks = (N + chunk_size - 1) // chunk_size
    print(f"DEBUG: Will process {total_chunks} text chunks (chunk size: {chunk_size}).")
    for chunk_idx, start in enumerate(range(0, N, chunk_size), start=1):
        end = min(start + chunk_size, N)
        print(f"DEBUG: Processing chunk {chunk_idx}/{total_chunks}: indices [{start}:{end}]")
        # Load the current chunk's text embeddings and move to GPU for projection.
        text_chunk = torch.from_numpy(bert_np[start:end]).float().to(device)  # (c, D)
        txt_proj = model.text_projection(text_chunk)      # (c, D)
        txt_proj = txt_proj / txt_proj.norm(dim=1, keepdim=True)
        del text_chunk
        torch.cuda.empty_cache()
        # Similarity: (B, D) @ (D, c) → (B, c)
        sims_gpu = logit_scale_val * (image_embeds @ txt_proj.t())
        sims_chunk = sims_gpu.cpu().numpy()  # (B, c)
        del sims_gpu, txt_proj
        torch.cuda.empty_cache(); gc.collect()

        score_i2t[:, start:end] = sims_chunk
        # For text->image: for each text in the chunk, get top k indices.
        k = 128
        for j in range(end - start):
            col = sims_chunk[:, j]
            topk_idx = np.argpartition(-col, k - 1)[:k]
            score_t2i[start + j, topk_idx] = col[topk_idx]
        if device.lower().startswith("cuda"):
            allocated = torch.cuda.memory_allocated(device) / 1e6
            reserved  = torch.cuda.memory_reserved(device) / 1e6
            print(f"DEBUG: After chunk {chunk_idx}: GPU allocated={allocated:.1f} MB, reserved={reserved:.1f} MB")
        gc.collect()

    elapsed = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(elapsed)))
    print("DEBUG: epoch_test completed in", total_time_str)
    return score_i2t, score_t2i


@torch.no_grad()
def itm_eval(scores_i2t, scores_t2i, txt2img, img2txt):
    """
    Evaluate image-text matching. Computes retrieval metrics.
    """
    print("DEBUG: Starting itm_eval...")
    # Image->Text
    ranks = np.zeros(scores_i2t.shape[0], dtype=np.int32)
    for i, row in enumerate(scores_i2t):
        inds = np.argsort(row)[::-1]
        best = min(np.where(np.isin(inds, img2txt[i]))[0])
        ranks[i] = best
    tr1 = 100.0 * (ranks < 1).sum() / len(ranks)
    tr5 = 100.0 * (ranks < 5).sum() / len(ranks)
    tr10 = 100.0 * (ranks < 10).sum() / len(ranks)

    # Text->Image
    ranks = np.zeros(scores_t2i.shape[0], dtype=np.int32)
    for i, row in enumerate(scores_t2i):
        inds = np.argsort(row)[::-1]
        ranks[i] = np.where(inds == txt2img[i])[0][0]
    ir1 = 100.0 * (ranks < 1).sum() / len(ranks)
    ir5 = 100.0 * (ranks < 5).sum() / len(ranks)
    ir10 = 100.0 * (ranks < 10).sum() / len(ranks)

    result = {
        'txt_r1': tr1, 'txt_r5': tr5, 'txt_r10': tr10, 'txt_r_mean': (tr1+tr5+tr10)/3,
        'img_r1': ir1, 'img_r5': ir5, 'img_r10': ir10, 'img_r_mean': (ir1+ir5+ir10)/3,
        'r_mean': (tr1+tr5+tr10+ir1+ir5+ir10)/6
    }
    print("DEBUG: itm_eval results:", result)
    return result


def evaluate_synset(it_eval, net, images_train, labels_train, testloader, args, text_embed_input, return_loss=False):
    """
    Additional evaluation function for a training subset (synset).
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
        losses.append(l)
        accs.append(a)
        print(f"DEBUG: Synset train epoch {ep} → loss {l:.4f}, acc {a:.4f}")
        if ep == Epoch:
            print("DEBUG: Final synset evaluation with epoch_test()...")
            i2t, t2i = epoch_test(testloader, net, args.device, text_embed_input)
            res = itm_eval(i2t, t2i, testloader.dataset.txt2img, testloader.dataset.img2txt)
            print("DEBUG: Synset evaluation result:", res)
    t1 = time.time() - t0
    print("DEBUG: evaluate_synset total time:", str(datetime.timedelta(seconds=int(t1))))
    return net, accs, res
