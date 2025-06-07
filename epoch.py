import os
# Enable expandable segments for more flexible GPU memory management
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import numpy as np
import torch
import time
import datetime
import torch.nn as nn
from tqdm import tqdm
from torch.utils.data import TensorDataset
import gc

def epoch(e, dataloader, net, optimizer_img, optimizer_txt, args):
    """
    Perform a training epoch on the given dataloader.
    (Unmodified.)
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
    Evaluate retrieval metrics in a streaming manner without pre-allocating a huge similarity matrix.
    
    The argument 'text_embed_input' can be:
       - a string: a path to an NPZ file containing the key "bert_test_embed", or
       - a torch.Tensor containing the text embeddings.
    
    The test dataset (accessed via dataloader.dataset) must have:
       - 'img2txt': a list of correct text index lists for each image.
       - 'txt2img': a list/array with one correct image index per text.
    
    Returns:
       A dictionary of retrieval metrics.
    """
    print("DEBUG: Starting epoch_test_metrics evaluation...")
    model.eval()
    # BLIP-style logit scale
    logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
    logit_scale_val = logit_scale.exp().item()
    print(f"DEBUG: logit_scale = {logit_scale_val:.4f}")
    start_time = time.time()

    # --- Process Text Embeddings ---
    if isinstance(text_embed_input, str):
        print(f"DEBUG: Loading text embeddings memmap from {text_embed_input}")
        data = np.load(text_embed_input, mmap_mode="r")
        bert_np = data["bert_test_embed"]  # shape (N, D)
    elif isinstance(text_embed_input, torch.Tensor):
        print("DEBUG: Received text embeddings as a Tensor; converting to numpy.")
        bert_np = text_embed_input.cpu().numpy()
    else:
        raise ValueError("text_embed_input must be a file path or a torch.Tensor")
    N, D = bert_np.shape
    print(f"DEBUG: Loaded text embeddings: shape=({N},{D})")
    # Keep text embeddings on CPU
    text_embeds = torch.from_numpy(bert_np).float()  # shape: (N, D)

    # --- Extract Image Embeddings ---
    print("DEBUG: Extracting image embeddings on GPU...")
    image_embeds_list = []
    for image, img_id in tqdm(dataloader, desc="Extracting image embeddings"):
        feat = model.image_encoder(image.to(device))
        normed = feat / feat.norm(dim=1, keepdim=True)
        image_embeds_list.append(normed.cpu())
    image_embeds = torch.cat(image_embeds_list, dim=0)  # shape: (B, D_im)
    B, D_im = image_embeds.shape
    print(f"DEBUG: Extracted image embeddings: shape=({B},{D_im})")
    if hasattr(model, "image_projection"):
        print("DEBUG: Applying image_projection...")
        proj = model.image_projection(image_embeds)
        image_embeds = proj / proj.norm(dim=1, keepdim=True)
    image_embeds = image_embeds.cpu()
    torch.cuda.empty_cache(); gc.collect()
    print("DEBUG: Image embeddings moved to CPU.")

    # --- Compute Image-to-Text Retrieval Metrics Streaming ---
    print("DEBUG: Computing image-to-text retrieval metrics...")
    img_ranks = np.empty(B, dtype=np.int32)
    for i in range(B):
        sim_row = logit_scale_val * (image_embeds[i].unsqueeze(0) @ text_embeds.t())
        sim_row = sim_row.squeeze(0).numpy()  # shape: (N,)
        sorted_idx = np.argsort(sim_row)[::-1]
        # ground truth list for image i (assumed available as dataloader.dataset.img2txt)
        gt = dataloader.dataset.img2txt[i]
        rank = int(np.min(np.where(np.isin(sorted_idx, gt))[0]))
        img_ranks[i] = rank
        if i % 1000 == 0:
            print(f"DEBUG: Processed image {i}/{B}, current avg rank (img->txt): {img_ranks[:i+1].mean():.2f}")
    tr1 = 100.0 * np.sum(img_ranks < 1) / B
    tr5 = 100.0 * np.sum(img_ranks < 5) / B
    tr10 = 100.0 * np.sum(img_ranks < 10) / B
    print("DEBUG: Image-to-text metrics computed.")

    # --- Compute Text-to-Image Retrieval Metrics Streaming ---
    print("DEBUG: Computing text-to-image retrieval metrics...")
    txt_ranks = np.empty(N, dtype=np.int32)
    for j in range(N):
        sim_col = logit_scale_val * (text_embeds[j].unsqueeze(0) @ image_embeds.t())
        sim_col = sim_col.squeeze(0).numpy()  # shape: (B,)
        sorted_idx = np.argsort(sim_col)[::-1]
        gt = dataloader.dataset.txt2img[j]
        rank = int(np.where(sorted_idx == gt)[0][0])
        txt_ranks[j] = rank
        if j % 1000 == 0:
            print(f"DEBUG: Processed text {j}/{N}, current avg rank (txt->img): {txt_ranks[:j+1].mean():.2f}")
    ir1 = 100.0 * np.sum(txt_ranks < 1) / N
    ir5 = 100.0 * np.sum(txt_ranks < 5) / N
    ir10 = 100.0 * np.sum(txt_ranks < 10) / N
    print("DEBUG: Text-to-image metrics computed.")

    res = {
        'txt_r1': tr1, 'txt_r5': tr5, 'txt_r10': tr10, 'txt_r_mean': (tr1 + tr5 + tr10) / 3,
        'img_r1': ir1, 'img_r5': ir5, 'img_r10': ir10, 'img_r_mean': (ir1 + ir5 + ir10) / 3,
        'r_mean': (tr1 + tr5 + tr10 + ir1 + ir5 + ir10) / 6
    }
    elapsed = time.time() - start_time
    print("DEBUG: epoch_test_metrics completed in", str(datetime.timedelta(seconds=int(elapsed))))
    return res


@torch.no_grad()
def itm_eval(scores_i2t, scores_t2i, txt2img, img2txt):
    """
    Evaluate image-text matching given full score matrices.
    (This version is kept for compatibility, but you are now using the streaming version.)
    """
    print("DEBUG: Starting itm_eval...")
    # Image->Text
    ranks = np.zeros(scores_i2t.shape[0], dtype=np.int32)
    for i, row in enumerate(scores_i2t):
        inds = np.argsort(row)[::-1]
        best = int(np.min(np.where(np.isin(inds, img2txt[i]))[0]))
        ranks[i] = best
    tr1 = 100.0 * (ranks < 1).sum() / len(ranks)
    tr5 = 100.0 * (ranks < 5).sum() / len(ranks)
    tr10 = 100.0 * (ranks < 10).sum() / len(ranks)

    # Text->Image
    ranks = np.zeros(scores_t2i.shape[0], dtype=np.int32)
    for i, row in enumerate(scores_t2i):
        inds = np.argsort(row)[::-1]
        ranks[i] = int(np.where(inds == txt2img[i])[0][0])
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
    Evaluate the synset by training on the subset and then computing retrieval metrics via streaming.
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
    loader = torch.utils.data.DataLoader(dst, batch_size=args.batch_train, shuffle=True, num_workers=0)

    accs, losses = [], []
    t0 = time.time()
    for ep in range(Epoch + 1):
        l, a = epoch(ep, loader, net, opt_img, opt_txt, args)
        losses.append(l)
        accs.append(a)
        print(f"DEBUG: Synset train epoch {ep} → Loss={l:.4f}, Acc={a:.4f}")
        if ep == Epoch:
            print("DEBUG: Starting final synset evaluation with epoch_test_metrics()...")
            res = epoch_test_metrics(testloader, net, args.device, text_embed_input)
            print("DEBUG: Synset evaluation result:", res)
    t1 = time.time() - t0
    print("DEBUG: evaluate_synset total time:", str(datetime.timedelta(seconds=int(t1))))
    return net, accs, res
