# import os
# # Enable expandable segments for flexible GPU memory management.
# os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

# import numpy as np
# import torch
# import time
# import datetime
# import torch.nn as nn
# from tqdm import tqdm
# from torch.utils.data import TensorDataset
# import gc

# # Standard functions remain unchanged.

# def epoch(e, dataloader, net, optimizer_img, optimizer_txt, args):
#     """
#     Perform a training epoch on the given dataloader.
#     """
#     print(f"DEBUG: Starting training epoch {e}...")
#     net = net.to(args.device)
#     net.train()
#     loss_avg, acc_avg, num_exp = 0.0, 0.0, 0

#     for i, data in tqdm(enumerate(dataloader), desc=f"Training Epoch {e}"):
#         if args.distill:
#             image, caption = data[:2]
#         else:
#             image, caption, index = data[:3]

#         image = image.to(args.device)
#         n_b = image.size(0)

#         loss, acc = net(image, caption, e)
#         loss_avg += loss.item() * n_b
#         acc_avg  += acc * n_b
#         num_exp  += n_b

#         optimizer_img.zero_grad()
#         optimizer_txt.zero_grad()
#         loss.backward()
#         optimizer_img.step()
#         optimizer_txt.step()

#     loss_avg /= num_exp
#     acc_avg  /= num_exp
#     print(f"DEBUG: Finished epoch {e}: Loss={loss_avg:.4f}, Acc={acc_avg:.4f}")
#     return loss_avg, acc_avg


# @torch.no_grad()
# def epoch_test(dataloader, model, device, text_embed_input):
#     """
#     Compute retrieval metrics in a streaming manner without preallocating 
#     a huge similarity matrix.
    
#     The argument 'text_embed_input' may be:
#       - A file path (string) to an NPZ file containing the key "bert_test_embed", or
#       - A torch.Tensor containing the text embeddings.
      
#     The test dataset (accessed via dataloader.dataset) must provide:
#       - 'img2txt': for each image, a list of matching text indices.
#       - 'txt2img': for each text, the ground-truth image index.
      
#     Returns a dictionary of retrieval metrics.
#     """
#     print("DEBUG: Starting epoch_test evaluation...")
#     model.eval()
#     # BLIP-style logit scale.
#     logit_scale = nn.Parameter(torch.ones([]) * np.log(1/0.07))
#     logit_scale_val = logit_scale.exp().item()
#     print(f"DEBUG: logit_scale = {logit_scale_val:.4f}")
#     start_time = time.time()
    
#     # --- Process text embeddings ---
#     if isinstance(text_embed_input, str):
#         print(f"DEBUG: Loading text embeddings memmap from {text_embed_input}")
#         data = np.load(text_embed_input, mmap_mode="r")
#         bert_np = data["bert_test_embed"]  # shape (N, D)
#     elif isinstance(text_embed_input, torch.Tensor):
#         print("DEBUG: Received text embeddings as a Tensor; converting to numpy.")
#         bert_np = text_embed_input.cpu().numpy()
#     else:
#         raise ValueError("text_embed_input must be a file path (str) or a torch.Tensor")
#     N, D = bert_np.shape
#     print(f"DEBUG: Loaded text embeddings: shape=({N},{D})")
#     # Keep text embeddings on CPU.
#     # (They will be processed in chunks later.)
#     # --- Extract image embeddings (unchanged) ---
#     print("DEBUG: Extracting image embeddings on GPU...")
#     image_embeds_list = []
#     for image, img_id in tqdm(dataloader, desc="Extracting image embeddings"):
#         feat = model.image_encoder(image.to(device))
#         normed = feat / feat.norm(dim=1, keepdim=True)
#         image_embeds_list.append(normed.cpu())
#     image_embeds = torch.cat(image_embeds_list, dim=0)  # shape: (B, D_im)
#     B, D_im = image_embeds.shape
#     print(f"DEBUG: Extracted image embeddings: shape=({B},{D_im})")
#     if hasattr(model, "image_projection"):
#         print("DEBUG: Applying image_projection...")
#         proj = model.image_projection(image_embeds)
#         image_embeds = proj / proj.norm(dim=1, keepdim=True)
#     image_embeds = image_embeds.cpu()
#     torch.cuda.empty_cache(); gc.collect()
#     print("DEBUG: Image embeddings moved to CPU.")
    
#     # --- Compute image-to-text retrieval metrics ---
#     print("DEBUG: Computing image-to-text retrieval metrics (streaming per image)...")
#     img_ranks = np.empty(B, dtype=np.int32)
#     # Use a chunk size for faster processing (adjust as needed)
#     chunk_size = 1024  
#     for i in range(B):
#         sim_row = np.empty(N, dtype=np.float32)
#         for start_idx in range(0, N, chunk_size):
#             end_idx = min(start_idx + chunk_size, N)
#             # Process a chunk of raw text embeddings and project them.
#             text_chunk = torch.from_numpy(bert_np[start_idx:end_idx]).float().to(device)
#             txt_proj = model.text_projection(text_chunk)
#             txt_proj = txt_proj / txt_proj.norm(dim=1, keepdim=True)
#             chunk_sim = logit_scale_val * (image_embeds[i].unsqueeze(0).to(device) @ txt_proj.t())
#             sim_row[start_idx:end_idx] = chunk_sim.squeeze(0).cpu().numpy()
#             del text_chunk, txt_proj, chunk_sim
#             torch.cuda.empty_cache()
#             gc.collect()
#         sorted_idx = np.argsort(sim_row)[::-1]
#         gt = dataloader.dataset.img2txt[i]  # list of correct text indices for image i
#         img_ranks[i] = int(np.min(np.where(np.isin(sorted_idx, gt))[0]))
#         if i % 2 == 0:
#             print(f"DEBUG: Processed image {i}/{B}, current avg rank (img->txt): {img_ranks[:i+1].mean():.2f}")
#     tr1 = 100.0 * np.sum(img_ranks < 1) / B
#     tr5 = 100.0 * np.sum(img_ranks < 5) / B
#     tr10 = 100.0 * np.sum(img_ranks < 10) / B
#     print("DEBUG: Image-to-text metrics computed.")
    
#     # --- Compute text-to-image retrieval metrics asynchronously ---
#     print("DEBUG: Computing text-to-image retrieval metrics (streaming per text using asynchronous chunks)...")
#     # Import nest_asyncio and asyncio.
#     import nest_asyncio, asyncio
#     nest_asyncio.apply()
    
#     async def process_text(j):
#         """Process one text sample asynchronously."""
#         text_tensor = torch.from_numpy(bert_np[j:j+1]).float().to(device)
#         txt_proj = model.text_projection(text_tensor)
#         txt_proj = txt_proj / txt_proj.norm(dim=1, keepdim=True)
#         sim_vector = logit_scale_val * (txt_proj @ image_embeds.to(device).t())
#         sim_vector = sim_vector.squeeze(0).cpu().numpy()  # shape: (B,)
#         sorted_idx = np.argsort(sim_vector)[::-1]
#         gt = dataloader.dataset.txt2img[j]
#         rank = int(np.where(sorted_idx == gt)[0][0])
#         del text_tensor, txt_proj, sim_vector
#         torch.cuda.empty_cache()
#         gc.collect()
#         return j, rank

#     # We will run tasks in batches to avoid creating 65k tasks at once.
#     txt_ranks = np.empty(N, dtype=np.int32)
#     async_batch_size = 1000
#     async_tasks = []
#     loop = asyncio.get_event_loop()
#     for j in range(N):
#         async_tasks.append(process_text(j))
#         if len(async_tasks) >= async_batch_size:
#             results = loop.run_until_complete(asyncio.gather(*async_tasks))
#             for r in results:
#                 txt_ranks[r[0]] = r[1]
#             async_tasks = []
#             print(f"DEBUG: Processed text {j}/{N}, current avg rank (txt->img): {txt_ranks[:j+1].mean():.2f}")
#     if async_tasks:
#         results = loop.run_until_complete(asyncio.gather(*async_tasks))
#         for r in results:
#             txt_ranks[r[0]] = r[1]
#         print(f"DEBUG: Processed text {N}/{N}, current avg rank (txt->img): {txt_ranks.mean():.2f}")
    
#     ir1 = 100.0 * np.sum(txt_ranks < 1) / N
#     ir5 = 100.0 * np.sum(txt_ranks < 5) / N
#     ir10 = 100.0 * np.sum(txt_ranks < 10) / N
#     print("DEBUG: Text-to-image metrics computed.")
    
#     res = {
#         'txt_r1': tr1, 
#         'txt_r5': tr5, 
#         'txt_r10': tr10, 
#         'txt_r_mean': (tr1+tr5+tr10)/3,
#         'img_r1': ir1, 
#         'img_r5': ir5, 
#         'img_r10': ir10, 
#         'img_r_mean': (ir1+ir5+ir10)/3,
#         'r_mean': (tr1+tr5+tr10+ir1+ir5+ir10)/6
#     }
#     elapsed = time.time() - start_time
#     print("DEBUG: epoch_test completed in", str(datetime.timedelta(seconds=int(elapsed))))
#     return res


# @torch.no_grad()
# def itm_eval(scores_i2t, scores_t2i, txt2img, img2txt):
#     """
#     Evaluate image-text matching given full score matrices.
#     (Retained for compatibility; not used in the asynchronous streaming version.)
#     """
#     print("DEBUG: Starting itm_eval...")
#     # Image→Text
#     ranks = np.zeros(scores_i2t.shape[0], dtype=np.int32)
#     for i, row in enumerate(scores_i2t):
#         inds = np.argsort(row)[::-1]
#         best = int(np.min(np.where(np.isin(inds, img2txt[i]))[0]))
#         ranks[i] = best
#     tr1 = 100.0 * (ranks < 1).sum() / len(ranks)
#     tr5 = 100.0 * (ranks < 5).sum() / len(ranks)
#     tr10 = 100.0 * (ranks < 10).sum() / len(ranks)
    
#     # Text→Image
#     ranks = np.zeros(scores_t2i.shape[0], dtype=np.int32)
#     for i, row in enumerate(scores_t2i):
#         inds = np.argsort(row)[::-1]
#         ranks[i] = int(np.where(inds == txt2img[i])[0][0])
#     ir1 = 100.0 * (ranks < 1).sum() / len(ranks)
#     ir5 = 100.0 * (ranks < 5).sum() / len(ranks)
#     ir10 = 100.0 * (ranks < 10).sum() / len(ranks)
    
#     result = {
#         'txt_r1': tr1, 
#         'txt_r5': tr5, 
#         'txt_r10': tr10, 
#         'txt_r_mean': (tr1+tr5+tr10)/3,
#         'img_r1': ir1, 
#         'img_r5': ir5, 
#         'img_r10': ir10, 
#         'img_r_mean': (ir1+ir5+ir10)/3,
#         'r_mean': (tr1+tr5+tr10+ir1+ir5+ir10)/6
#     }
#     print("DEBUG: itm_eval results:", result)
#     return result


# def evaluate_synset(it_eval, net, images_train, labels_train, testloader, args, text_embed_input, return_loss=False):
#     """
#     Evaluate the synset by training on a subset and then computing retrieval metrics via streaming.
#     """
#     print("DEBUG: Starting evaluate_synset()...")
#     net = net.to(args.device)
#     images_train = images_train.to(args.device)
#     labels_train = labels_train.to(args.device)

#     lr = float(args.lr_net)
#     Epoch = int(args.epoch_eval_train)
#     opt_img = torch.optim.SGD(net.image_encoder.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4)
#     opt_txt = torch.optim.SGD(net.text_projection.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4)

#     dst = TensorDataset(images_train, labels_train)
#     loader = torch.utils.data.DataLoader(dst, batch_size=args.batch_train, shuffle=True, num_workers=0)

#     accs, losses = [], []
#     t0 = time.time()
#     for ep in range(Epoch + 1):
#         l, a = epoch(ep, loader, net, opt_img, opt_txt, args)
#         losses.append(l)
#         accs.append(a)
#         print(f"DEBUG: Synset train epoch {ep} → Loss={l:.4f}, Acc={a:.4f}")
#         if ep == Epoch:
#             print("DEBUG: Starting final synset evaluation with epoch_test()...")
#             res = epoch_test(testloader, net, args.device, text_embed_input)
#             print("DEBUG: Synset evaluation result:", res)
#     t1 = time.time() - t0
#     print("DEBUG: evaluate_synset total time:", str(datetime.timedelta(seconds=int(t1))))
#     return net, accs, res






import os
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

        if i % 10 == 0:
            print(f"DEBUG: Batch {i} | Loss: {loss.item():.4f}, Acc: {acc:.4f}")

    loss_avg /= num_exp
    acc_avg /= num_exp
    print(f"DEBUG: Finished epoch {e}: Loss={loss_avg:.4f}, Acc={acc_avg:.4f}")
    return loss_avg, acc_avg


@torch.no_grad()
def epoch_test_metrics(dataloader, model, device, text_embed_input):
    print("DEBUG: Starting epoch_test_metrics evaluation...")
    model.eval()
    logit_scale = nn.Parameter(torch.ones([]) * np.log(1/0.07))
    logit_scale_val = logit_scale.exp().item()
    print(f"DEBUG: logit_scale = {logit_scale_val:.4f}")
    start_time = time.time()

    if isinstance(text_embed_input, str):
        print(f"DEBUG: Loading text embeddings from file: {text_embed_input}")
        data = np.load(text_embed_input, mmap_mode="r")
        bert_np = data["bert_test_embed"]
    elif isinstance(text_embed_input, torch.Tensor):
        print("DEBUG: Received text embeddings as tensor, converting to numpy")
        bert_np = text_embed_input.cpu().numpy()
    else:
        raise ValueError("Invalid text_embed_input")

    N, D = bert_np.shape
    print(f"DEBUG: Loaded text embeddings: shape=({N},{D})")

    print("DEBUG: Extracting image embeddings...")
    image_embeds_list = []
    for image, img_id in tqdm(dataloader, desc="Extracting image embeddings"):
        feat = model.image_encoder(image.to(device))
        normed = feat / feat.norm(dim=1, keepdim=True)
        image_embeds_list.append(normed.cpu())
    image_embeds = torch.cat(image_embeds_list, dim=0)
    B, D_im = image_embeds.shape
    print(f"DEBUG: Extracted image embeddings: shape=({B},{D_im})")

    if hasattr(model, "image_projection"):
        print("DEBUG: Applying image projection...")
        image_embeds = model.image_projection(image_embeds)
        image_embeds = image_embeds / image_embeds.norm(dim=1, keepdim=True)

    image_embeds = image_embeds.cpu()
    torch.cuda.empty_cache(); gc.collect()
    print("DEBUG: Image embeddings moved to CPU")

    print("DEBUG: Computing image-to-text retrieval metrics...")
    img_ranks = np.empty(B, dtype=np.int32)
    chunk_size = 1024
    for i in range(B):
        sim_row = np.empty(N, dtype=np.float32)
        for start_idx in range(0, N, chunk_size):
            end_idx = min(start_idx + chunk_size, N)
            text_chunk = torch.from_numpy(bert_np[start_idx:end_idx]).float().to(device)
            txt_proj = model.text_projection(text_chunk)
            txt_proj = txt_proj / txt_proj.norm(dim=1, keepdim=True)
            chunk_sim = logit_scale_val * (image_embeds[i].unsqueeze(0).to(device) @ txt_proj.t())
            sim_row[start_idx:end_idx] = chunk_sim.squeeze(0).cpu().numpy()
            del text_chunk, txt_proj, chunk_sim
            torch.cuda.empty_cache(); gc.collect()
        sorted_idx = np.argsort(sim_row)[::-1]
        gt = dataloader.dataset.img2txt[i]
        img_ranks[i] = int(np.min(np.where(np.isin(sorted_idx, gt))[0]))
        if i % 100 == 0:
            print(f"DEBUG: Processed image {i}/{B} | Avg Rank: {img_ranks[:i+1].mean():.2f}")
    tr1 = 100.0 * np.sum(img_ranks < 1) / B
    tr5 = 100.0 * np.sum(img_ranks < 5) / B
    tr10 = 100.0 * np.sum(img_ranks < 10) / B
    print("DEBUG: Completed image-to-text ranking")

    print("DEBUG: Computing text-to-image retrieval metrics (async)...")
    import nest_asyncio, asyncio
    nest_asyncio.apply()

    async def process_text(j):
        text_tensor = torch.from_numpy(bert_np[j:j+1]).float().to(device)
        txt_proj = model.text_projection(text_tensor)
        txt_proj = txt_proj / txt_proj.norm(dim=1, keepdim=True)
        sim_vector = logit_scale_val * (txt_proj @ image_embeds.to(device).t())
        sim_vector = sim_vector.squeeze(0).cpu().numpy()
        sorted_idx = np.argsort(sim_vector)[::-1]
        gt = dataloader.dataset.txt2img[j]
        rank = int(np.where(sorted_idx == gt)[0][0])
        del text_tensor, txt_proj, sim_vector
        torch.cuda.empty_cache(); gc.collect()
        return j, rank

    txt_ranks = np.empty(N, dtype=np.int32)
    async_batch_size = 500
    async_tasks = []
    loop = asyncio.get_event_loop()

    for j in range(N):
        async_tasks.append(process_text(j))
        if len(async_tasks) >= async_batch_size:
            results = loop.run_until_complete(asyncio.gather(*async_tasks))
            for r in results:
                txt_ranks[r[0]] = r[1]
            async_tasks = []
            print(f"DEBUG: Processed texts up to {j}/{N} | Avg Rank: {txt_ranks[:j+1].mean():.2f}")
    if async_tasks:
        results = loop.run_until_complete(asyncio.gather(*async_tasks))
        for r in results:
            txt_ranks[r[0]] = r[1]
        print(f"DEBUG: Processed all texts | Final Avg Rank: {txt_ranks.mean():.2f}")

    ir1 = 100.0 * np.sum(txt_ranks < 1) / N
    ir5 = 100.0 * np.sum(txt_ranks < 5) / N
    ir10 = 100.0 * np.sum(txt_ranks < 10) / N
    print("DEBUG: Completed text-to-image ranking")

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
    print("DEBUG: Starting itm_eval...")
    ranks = np.zeros(scores_i2t.shape[0], dtype=np.int32)
    for i, row in enumerate(scores_i2t):
        inds = np.argsort(row)[::-1]
        best = int(np.min(np.where(np.isin(inds, img2txt[i]))[0]))
        ranks[i] = best
    tr1 = 100.0 * (ranks < 1).sum() / len(ranks)
    tr5 = 100.0 * (ranks < 5).sum() / len(ranks)
    tr10 = 100.0 * (ranks < 10).sum() / len(ranks)

    ranks = np.zeros(scores_t2i.shape[0], dtype=np.int32)
    for i, row in enumerate(scores_t2i):
        inds = np.argsort(row)[::-1]
        best = int(np.where(inds == txt2img[i])[0][0])
        ranks[i] = best
    ir1 = 100.0 * (ranks < 1).sum() / len(ranks)
    ir5 = 100.0 * (ranks < 5).sum() / len(ranks)
    ir10 = 100.0 * (ranks < 10).sum() / len(ranks)

    print("DEBUG: itm_eval completed.")
    return {
        'txt_r1': tr1, 'txt_r5': tr5, 'txt_r10': tr10, 'txt_r_mean': (tr1 + tr5 + tr10) / 3,
        'img_r1': ir1, 'img_r5': ir5, 'img_r10': ir10, 'img_r_mean': (ir1 + ir5 + ir10) / 3,
        'r_mean': (tr1 + tr5 + tr10 + ir1 + ir5 + ir10) / 6
    }
