import argparse
import copy
import datetime
import os
import random
import sys
import warnings

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.utils
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm
import math

from transformers import BertTokenizer, BertConfig, BertModel
import wandb

from data import get_dataset_flickr, textprocess, textprocess_train
from epoch import evaluate_synset, epoch, epoch_test, itm_eval
from networks import CLIPModel_full, TextEncoder
from reparam_module import ReparamModule
from utils import DiffAugment, ParamDiffAug, TensorDataset, get_dataset, get_network, get_eval_pool, get_time, load_or_process_file


def shuffle_files(img_expert_files, txt_expert_files):
    # Check if both lists have the same length and if the lists are not empty
    assert len(img_expert_files) == len(txt_expert_files), "Number of image files and text files does not match"
    assert len(img_expert_files) != 0, "No files to shuffle"
    shuffled_indices = np.random.permutation(len(img_expert_files))

    # Apply the shuffled indices to both lists
    img_expert_files = np.take(img_expert_files, shuffled_indices)
    txt_expert_files = np.take(txt_expert_files, shuffled_indices)
    print(f"img_expert_files: {img_expert_files}")
    print(f"txt_expert_files: {txt_expert_files}")
    return img_expert_files, txt_expert_files

def nearest_neighbor(sentences, query_embeddings, database_embeddings):
    """Find the nearest neighbors for a batch of embeddings.

    Args:
      sentences: The original sentences from which the embeddings were computed.
      query_embeddings: A batch of embeddings for which to find the nearest neighbors.
      database_embeddings: All pre-computed embeddings.

    Returns:
      A list of the most similar sentences for each embedding in the batch.
    """
    nearest_neighbors = []
    for query in query_embeddings:
        similarities = cosine_similarity(query.reshape(1, -1), database_embeddings)
        most_similar_index = np.argmax(similarities)
        nearest_neighbors.append(sentences[most_similar_index])
    return nearest_neighbors

def get_images_texts(n, dataset):
    """Get random n images and corresponding texts from the dataset.

    Args:
      n: Number of images and texts to retrieve.
      dataset: The dataset containing image-text pairs.

    Returns:
      A tuple containing two elements:
        - A tensor of randomly selected images.
        - A tensor of the corresponding texts, encoded as floats.
    """
    # Generate n unique random indices
    idx_shuffle = np.random.permutation(len(dataset))[:n]
    # Initialize the text encoder
    text_encoder = TextEncoder(args)
    image_syn = torch.stack([dataset[i][0] for i in idx_shuffle])
    text_syn = text_encoder([dataset[i][1] for i in idx_shuffle], device="cpu")
    return image_syn, text_syn.float()


@torch.no_grad()
def textprocess(args, testloader):
    net = CLIPModel_full(args).to('cuda')
    net.eval()
    texts = testloader.dataset.text
    # Process text embeddings for flickr, coco, and roco
    if args.dataset in ['flickr', 'coco', 'roco']:
        chunk_size = 1000  # Reduced chunk size for test texts
        chunks = []
        for i in range(0, len(texts), chunk_size):
            chunk = net.text_encoder(texts[i:i + chunk_size]).cpu()  # Store output on CPU
            chunks.append(chunk)
            torch.cuda.empty_cache()  # Free unused memory
        bert_test_embed = torch.cat(chunks, dim=0)
        bert_test_embed_np = bert_test_embed.numpy()
        np.savez(f'{args.dataset}_{args.text_encoder}_text_embed.npz', bert_test_embed=bert_test_embed_np)
        return {'bert_test_embed': bert_test_embed_np}
    else:
        raise NotImplementedError("Text embedding extraction for this dataset is not yet implemented.")


@torch.no_grad()
def textprocess_train(args, texts):
    net = CLIPModel_full(args).to('cuda')
    net.eval()
    if args.dataset in ['flickr', 'coco', 'roco']:
        chunk_size = 2000  # Bigger chunk size for training texts
        chunks = []
        for i in tqdm(range(0, len(texts), chunk_size)):
            chunk = net.text_encoder(texts[i:i + chunk_size]).cpu()
            chunks.append(chunk)
            del chunk
            torch.cuda.empty_cache()  # Free memory
        bert_test_embed = torch.cat(chunks, dim=0)
        print('bert_test_embed.shape: ', bert_test_embed.shape)
        bert_test_embed_np = bert_test_embed.numpy()
        np.savez(f'{args.dataset}_{args.text_encoder}_train_text_embed.npz', bert_test_embed=bert_test_embed_np)
        return {'bert_test_embed': bert_test_embed_np}
    else:
        raise NotImplementedError("Text embedding extraction for this dataset is not yet implemented.")


def create_dataset(args, min_scale=0.5):
    normalize = transforms.Normalize(
        (0.48145466, 0.4578275, 0.40821073),
        (0.26862954, 0.26130258, 0.27577711))
    transform_train = transforms.Compose([
        transforms.RandomResizedCrop(args.image_size, scale=(min_scale, 1.0), interpolation=InterpolationMode.BICUBIC),
        transforms.RandomHorizontalFlip(),
        RandomAugment(2, 5, isPIL=True, augs=[
            'Identity', 'AutoContrast', 'Brightness', 'Sharpness', 'Equalize',
            'ShearX', 'ShearY', 'TranslateX', 'TranslateY', 'Rotate'
        ]),
        transforms.ToTensor(),
        normalize,
    ])
    transform_test = transforms.Compose([
        transforms.Resize((args.image_size, args.image_size), interpolation=InterpolationMode.BICUBIC),
        transforms.ToTensor(),
        normalize,
    ])

    if args.dataset == 'flickr':
        train_dataset = flickr30k_train(transform_train, args.image_root, args.ann_root)
        val_dataset = flickr30k_retrieval_eval(transform_test, args.image_root, args.ann_root, 'val')
        test_dataset = flickr30k_retrieval_eval(transform_test, args.image_root, args.ann_root, 'test')
        return train_dataset, val_dataset, test_dataset
    elif args.dataset == 'coco':
        train_dataset = coco_train(transform_train, args.image_root, args.ann_root)
        val_dataset = coco_retrieval_eval(transform_test, args.image_root, args.ann_root, 'val')
        test_dataset = coco_retrieval_eval(transform_test, args.image_root, args.ann_root, 'test')
        return train_dataset, val_dataset, test_dataset
    elif args.dataset == 'roco':
        train_dataset = roco_train(transform_train, args.image_root, args.ann_root)
        val_dataset = roco_retrieval_eval(transform_test, args.image_root, args.ann_root, 'val')
        test_dataset = roco_retrieval_eval(transform_test, args.image_root, args.ann_root, 'test')
        return train_dataset, val_dataset, test_dataset
    else:
        raise NotImplementedError("Dataset not implemented")


def create_sampler(datasets, shuffles, num_tasks, global_rank):
    samplers = []
    for dataset, shuffle in zip(datasets, shuffles):
        sampler = torch.utils.data.DistributedSampler(dataset, num_replicas=num_tasks, rank=global_rank, shuffle=shuffle)
        samplers.append(sampler)
    return samplers

def create_loader(datasets, samplers, batch_size, num_workers, is_trains, collate_fns):
    loaders = []
    for dataset, sampler, bs, n_worker, is_train, collate_fn in zip(datasets, samplers, batch_size, num_workers, is_trains, collate_fns):
        if is_train:
            shuffle = (sampler is None)
            drop_last = True
        else:
            shuffle = False
            drop_last = False
        loader = DataLoader(
            dataset,
            batch_size=bs,
            num_workers=n_worker,
            pin_memory=True,
            sampler=sampler,
            shuffle=shuffle,
            collate_fn=collate_fn,
            drop_last=drop_last,
        )
        loaders.append(loader)
    return loaders

def get_dataset_flickr(args):
    print("Creating retrieval dataset")
    train_dataset, val_dataset, test_dataset = create_dataset(args)
    samplers = [None, None, None]
    train_shuffle = True
    train_loader, val_loader, test_loader = create_loader(
        [train_dataset, val_dataset, test_dataset],
        samplers,
        batch_size=[args.batch_size_train] + [args.batch_size_test] * 2,
        num_workers=[4, 4, 4],
        is_trains=[train_shuffle, False, False],
        collate_fns=[None, None, None])
    return train_loader, test_loader, train_dataset, test_dataset

###############################################
# Main Buffer Distillation Script
###############################################

import wandb
import warnings
import datetime
from epoch import epoch, epoch_test, itm_eval
from utils import load_or_process_file

warnings.filterwarnings("ignore", category=DeprecationWarning)

def main(args):
    # Initialize wandb
    if args.disable_wandb:
        import os
        os.environ["WANDB_MODE"] = "disabled"
        import wandb
        wandb.init(mode="disabled")
        print("wandb logging disabled.")
    else:
        import wandb
        wandb.init(project='DatasetDistillation', entity='dataset_distillation', config=args, name=args.name)
   
    args.dsa = True if args.dsa == 'True' else False
    args.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    args.distributed = torch.cuda.device_count() > 1

    print('Hyper-parameters: \n', args.__dict__)

    save_dir = os.path.join(args.buffer_path, args.dataset)
    if args.dataset in ["CIFAR10", "CIFAR100"] and not args.zca:
        save_dir += "_NO_ZCA"
    save_dir = os.path.join(save_dir, args.image_encoder, args.text_encoder)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    ''' Organize the datasets '''
    if args.dataset == 'flickr':
        trainloader, testloader, train_dataset, test_dataset = get_dataset_flickr(args)
        data = load_or_process_file('text', textprocess, args, testloader)
    elif args.dataset == 'roco':
        trainloader, testloader, train_dataset, test_dataset = get_dataset_flickr(args)
        data = load_or_process_file('text', textprocess, args, testloader)
    elif args.dataset == 'coco':
        trainloader, testloader, train_dataset, test_dataset = get_dataset_flickr(args)
        data = load_or_process_file('text', textprocess, args, testloader)
    else:
        raise NotImplementedError("Dataset not implemented")
    
    bert_test_embed = torch.from_numpy(data['bert_test_embed']).cpu()

    img_trajectories = []
    txt_trajectories = []

    for it in range(0, args.num_experts):
        ''' Train synthetic data '''
        teacher_net = CLIPModel_full(args)
        img_teacher_net = teacher_net.image_encoder.to(args.device)
        txt_teacher_net = teacher_net.text_projection.to(args.device)
        if args.text_trainable:
            txt_teacher_net = teacher_net.text_encoder.to(args.device)
        if args.distributed:
            img_teacher_net = torch.nn.DataParallel(img_teacher_net)
            txt_teacher_net = torch.nn.DataParallel(txt_teacher_net)
        img_teacher_net.train()
        txt_teacher_net.train()
        lr_img = args.lr_teacher_img
        lr_txt = args.lr_teacher_txt

        teacher_optim_img = torch.optim.SGD(img_teacher_net.parameters(), lr=lr_img, momentum=args.mom, weight_decay=args.l2)
        teacher_optim_txt = torch.optim.SGD(txt_teacher_net.parameters(), lr=lr_txt, momentum=args.mom, weight_decay=args.l2)
        teacher_optim_img.zero_grad()
        teacher_optim_txt.zero_grad()

        img_timestamps = []
        txt_timestamps = []

        img_timestamps.append([p.detach().cpu() for p in img_teacher_net.parameters()])
        txt_timestamps.append([p.detach().cpu() for p in txt_teacher_net.parameters()])

        lr_schedule = [args.train_epochs // 2 + 1]
        lr = lr_img

        for e in range(args.train_epochs):
            train_loss, train_acc = epoch(e, trainloader, teacher_net, teacher_optim_img, teacher_optim_txt, args)
            # Modified unpacking to ignore extra returned values.
            score_val_i2t, score_val_t2i, *rest = epoch_test(testloader, teacher_net, args.device, bert_test_embed)
            score_val_i2t = np.asarray(score_val_i2t)
            score_val_t2i = np.asarray(score_val_t2i)
            if score_val_i2t.ndim == 0:
                score_val_i2t = np.expand_dims(score_val_i2t, axis=0)
            if score_val_t2i.ndim == 0:
                score_val_t2i = np.expand_dims(score_val_t2i, axis=0)
            val_result = itm_eval(score_val_i2t, score_val_t2i, testloader.dataset.txt2img, testloader.dataset.img2txt)

            wandb.log({"train_loss": train_loss})
            wandb.log({"train_acc": train_acc})
            wandb.log({"txt_r1": val_result['txt_r1']})
            wandb.log({"txt_r5": val_result['txt_r5']})
            wandb.log({"txt_r10": val_result['txt_r10']})
            wandb.log({"txt_r_mean": val_result['txt_r_mean']})
            wandb.log({"img_r1": val_result['img_r1']})
            wandb.log({"img_r5": val_result['img_r5']})
            wandb.log({"img_r10": val_result['img_r10']})
            wandb.log({"img_r_mean": val_result['img_r_mean']})
            wandb.log({"r_mean": val_result['r_mean']})

            print("Itr: {}\tEpoch: {}\tTrain Acc: {}\tImg R@1: {}\tR@5: {}\tR@10: {}\tR@Mean: {}\tTxt R@1: {}\tR@5: {}\tR@10: {}\tR@Mean: {}".format(
                it, e, train_acc,
                val_result['img_r1'], val_result['img_r5'], val_result['img_r10'], val_result['img_r_mean'],
                val_result['txt_r1'], val_result['txt_r5'], val_result['txt_r10'], val_result['txt_r_mean']))
            img_timestamps.append([p.detach().cpu() for p in img_teacher_net.parameters()])
            txt_timestamps.append([p.detach().cpu() for p in txt_teacher_net.parameters()])

            if e in lr_schedule and args.decay:
                lr *= 0.1
                teacher_optim_img = torch.optim.SGD(img_teacher_net.parameters(), lr=lr, momentum=args.mom, weight_decay=args.l2)
                teacher_optim_txt = torch.optim.SGD(txt_teacher_net.parameters(), lr=lr, momentum=args.mom, weight_decay=args.l2)
                teacher_optim_img.zero_grad()
                teacher_optim_txt.zero_grad()

        img_trajectories.append(img_timestamps)
        txt_trajectories.append(txt_timestamps)
        n = 0
        while os.path.exists(os.path.join(save_dir, "img_replay_buffer_{}.pt".format(n))):
            n += 1
        print("Saving {}".format(os.path.join(save_dir, "img_replay_buffer_{}.pt".format(n))))
        torch.save(img_trajectories, os.path.join(save_dir, "img_replay_buffer_{}.pt".format(n)))
        print("Saving {}".format(os.path.join(save_dir, "txt_replay_buffer_{}.pt".format(n))))
        torch.save(txt_trajectories, os.path.join(save_dir, "txt_replay_buffer_{}.pt".format(n)))
        img_trajectories = []
        txt_trajectories = []

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Parameter Processing')
    parser.add_argument('--dataset', type=str, default='roco', choices=['roco', 'coco'], help='dataset')
    parser.add_argument('--disable_wandb', action='store_true', help='Disable wandb logging')
    parser.add_argument('--num_experts', type=int, default=100, help='training iterations')
    parser.add_argument('--lr_teacher_img', type=float, default=0.1, help='learning rate for updating network parameters')
    parser.add_argument('--lr_teacher_txt', type=float, default=0.1, help='learning rate for updating network parameters')
    parser.add_argument('--batch_train', type=int, default=128, help='batch size for training networks')
    parser.add_argument('--dsa', type=str, default='True', choices=['True', 'False'],
                        help='whether to use differentiable Siamese augmentation.')
    parser.add_argument('--dsa_strategy', type=str, default='color_crop_cutout_flip_scale_rotate',
                        help='differentiable Siamese augmentation strategy')
    parser.add_argument('--data_path', type=str, default='/kaggle/input/roco-dataset/', help='dataset path')
    parser.add_argument('--buffer_path', type=str, default='/kaggle/working', help='buffer path')
    parser.add_argument('--train_epochs', type=int, default=50)
    parser.add_argument('--zca', action='store_true')
    parser.add_argument('--decay', action='store_true')
    parser.add_argument('--mom', type=float, default=0, help='momentum')
    parser.add_argument('--l2', type=float, default=0, help='l2 regularization')
    parser.add_argument('--save_interval', type=int, default=10)
    import datetime
    current_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    parser.add_argument('--name', type=str, default=current_time, help='name of wandb run')
    parser.add_argument('--text_pretrained', type=bool, default=True, help='text_pretrained')
    parser.add_argument('--image_pretrained', type=bool, default=True, help='image_pretrained')
    parser.add_argument('--text_trainable', type=bool, default=False, help='text_trainable')
    parser.add_argument('--image_trainable', type=bool, default=True, help='image_trainable')
    parser.add_argument('--batch_size_train', type=int, default=128, help='batch_size_train')
    parser.add_argument('--batch_size_test', type=int, default=128, help='batch_size_test')
    parser.add_argument('--image_root', type=str, default='/kaggle/input/roco-dataset/all_data/train/radiology/images/', help='location of image root')
    parser.add_argument('--ann_root', type=str, default='/kaggle/input/roco-dataset/all_data/train/radiologytraindata.csv', help='location of annotation file')
    parser.add_argument('--image_size', type=int, default=224, help='image_size')
    parser.add_argument('--k_test', type=int, default=128, help='k_test')
    parser.add_argument('--load_npy', type=bool, default=False, help='load_npy')
    parser.add_argument('--image_encoder', type=str, default='resnet50', choices=['nfnet', 'resnet18_gn', 'vit_tiny', 'nf_resnet50', 'nf_regnet'], help='image encoder')
    parser.add_argument('--text_encoder', type=str, default='bert', choices=['bert', 'clip'], help='text encoder')
    parser.add_argument('--margin', default=0.2, type=float, help='Rank loss margin.')
    parser.add_argument('--measure', default='cosine', help='Similarity measure used (cosine|order)')
    parser.add_argument('--max_violation', action='store_true', help='Use max instead of sum in the rank loss.')
    parser.add_argument('--only_has_image_projection', type=bool, default=False, help='None')
    parser.add_argument('--grounding', type=bool, default=False, help='None')
    parser.add_argument('--distill', type=bool, default=False, help='if distill')
    args = parser.parse_args()
    main(args)
