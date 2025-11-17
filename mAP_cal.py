# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import wandb
import torch
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
import argparse
from efficient_sam.build_efficient_sam import build_efficient_sam_vitt, build_efficient_sam_vits
import torch
from sklearn.metrics import average_precision_score
from dataloader import get_dataset
import os
import cv2
os.environ['CUDA_VISIBLE_DEVICES'] = "3" 


def load_model(args, train_root, sam):=
    model_list = os.listdir(os.path.join(train_root, 'model'))
    latest_num = 0
    best_model = model_list[0]
    for model_name in model_list:
        ns = model_name.split('_')
        if ns[1] == 'best':
            if int(ns[0]) >= latest_num:
                best_model = model_name
                latest_num = int(ns[0])
    print('best_model: ',best_model)
    state_dicts = torch.load(os.path.join(train_root, 'model', best_model))

    for ind_blk, blk in enumerate(sam.image_encoder.blocks):
        if ind_blk >= args.RouGE_start_index:
            blk.RouGE.load_state_dict(state_dicts['model_state_dict'][ind_blk][0]) 
    return sam


def show_mask(mask, ax, random_color=False):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        color = np.array([30/255, 144/255, 255/255, 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)

def show_points(coords, labels, ax, marker_size=375):
    pos_points = coords[labels == 1]
    neg_points = coords[labels == 0]
    ax.scatter(
        pos_points[:, 0],
        pos_points[:, 1],
        color="green",
        marker="*",
        s=marker_size,
        edgecolor="white",
        linewidth=1.25,
    )
    ax.scatter(
        neg_points[:, 0],
        neg_points[:, 1],
        color="red",
        marker="*",
        s=marker_size,
        edgecolor="white",
        linewidth=1.25,
    )

def draw_masks_and_point_on_image(name, image, masks, point_coords, colors=None, alpha=0.3):
    plt.clf()
    fig, ax = plt.subplots(1, 1, figsize=(10,10))
    plt.tight_layout()
    image = image.cpu()
    image = image[0, [0, 1, 2], :, :].permute(1,2,0)
    masks = masks.unsqueeze(1).cpu()
    masks[masks>0] = 1
    masks[masks<0] = 0
    point_coords = point_coords.squeeze(0).squeeze(0).cpu()
    show_points(point_coords, np.array([1]), ax)
    ax.imshow(image)
    show_mask(masks, ax)
    ax.axis('off')
    name = name+'.png'
    if not os.path.exists( os.path.dirname(name)):
        os.makedirs(os.path.dirname(name))
    plt.savefig(name)
    

def mask_mAP(args, train_root):
    print("Loading model...")
    
    sam = build_efficient_sam_vitt().cuda()
    sam_image_encoder = sam.image_encoder
    sam = load_model(args, train_root, sam)
    print('SAM encoder loaded')
    for p in sam_image_encoder.parameters():
        p.requires_grad = False

    print('number of adapter:', len(sam_image_encoder.blocks) - args.RouGE_start_index)
    
    
    
    if args.datatype == 'APcal':
        setnames = ['Rain200L', 'DDN', 'GoPro', 'LIS', 'snow100k']
    trainloaders, testloaders = get_dataset(args, sam)
    for i, testloader in enumerate(testloaders):
        ori_deg_AP = []
        ori_normal_AP = []
        RouGE_deg_AP = []
        RouGE_normal_AP = []
        ori_normal_IoU = []
        ori_deg_IoU = []
        RouGE_deg_IoU = []
        RouGE_normal_IoU = []
        j = 0
        for deg_img, normal_img, deg_clipfeat, normal_clipfeat, gt_type, points, gt_masks in tqdm(testloader):
            j += 1
            deg_img = deg_img.cuda()
            normal_img = normal_img.cuda()
            deg_clipfeat = deg_clipfeat.cuda()
            normal_clipfeat = normal_clipfeat.cuda()
            labels = torch.tensor([[[1]] for _ in range(len(points))]).cuda()
            points = points.cuda()

            predicted_logits, predicted_iou = sam(
                batched_images = normal_img,
                batched_points = points,
                batched_point_labels = labels,
                RouGE_start_index = -1
            )
            sorted_ids = torch.argsort(predicted_iou, dim=-1, descending=True)
            predicted_iou = torch.take_along_dim(predicted_iou, sorted_ids, dim=2)
            predicted_logits = torch.take_along_dim(
                predicted_logits, sorted_ids[..., None, None], dim=2
            )
            predicted_logits = predicted_logits.detach().cpu()
            mask = predicted_logits[:, 0, 0, :, :]
            ori_normal_AP.append(average_precision_score(gt_masks.flatten(), mask.flatten()))
         
            if args.save_mask:
                draw_masks_and_point_on_image(os.path.join('./img',setnames[i],'ori_normal',str(j)), normal_img, mask, points)

            predicted_logits, predicted_iou = sam(
                batched_images = deg_img,
                batched_points = points,
                batched_point_labels = labels,
                RouGE_start_index = -1
            )
            sorted_ids = torch.argsort(predicted_iou, dim=-1, descending=True)
            predicted_iou = torch.take_along_dim(predicted_iou, sorted_ids, dim=2)
            predicted_logits = torch.take_along_dim(
                predicted_logits, sorted_ids[..., None, None], dim=2
            )
            predicted_logits = predicted_logits.detach().cpu()
            mask = predicted_logits[:, 0, 0, :, :]
            ori_deg_AP.append(average_precision_score(gt_masks.flatten(), mask.flatten()))
            
            if args.save_mask:
                draw_masks_and_point_on_image(os.path.join('./img',setnames[i],'ori_deg',str(j)), deg_img, mask, points)

            predicted_logits, predicted_iou = sam(
                batched_images = deg_img,
                batched_points = points,
                batched_point_labels = labels,
                RouGE_start_index = args.RouGE_start_index,
                clip_feat = deg_clipfeat,
                gt_type = gt_type,
            )
            sorted_ids = torch.argsort(predicted_iou, dim=-1, descending=True)
            predicted_iou = torch.take_along_dim(predicted_iou, sorted_ids, dim=2)
            predicted_logits = torch.take_along_dim(
                predicted_logits, sorted_ids[..., None, None], dim=2
            )
            predicted_logits = predicted_logits.detach().cpu()
            mask = predicted_logits[:, 0, 0, :, :]
            RouGE_deg_AP.append(average_precision_score(gt_masks.flatten(), mask.flatten()))
            
            if args.save_mask:
                draw_masks_and_point_on_image(os.path.join('./img',setnames[i],'ASG_deg',str(j)), deg_img, mask, points)


            predicted_logits, predicted_iou = sam(
                batched_images = normal_img,
                batched_points = points,
                batched_point_labels = labels,
                RouGE_start_index = args.RouGE_start_index,
                clip_feat = normal_clipfeat,
                gt_type = 0,
            )
            sorted_ids = torch.argsort(predicted_iou, dim=-1, descending=True)
            predicted_iou = torch.take_along_dim(predicted_iou, sorted_ids, dim=2)
            predicted_logits = torch.take_along_dim(
                predicted_logits, sorted_ids[..., None, None], dim=2
            )
            predicted_logits = predicted_logits.detach().cpu()
            mask = predicted_logits[:, 0, 0, :, :]
            RouGE_normal_AP.append(average_precision_score(gt_masks.flatten(), mask.flatten()))
            
            
            if args.save_mask:
                draw_masks_and_point_on_image(os.path.join('./img',setnames[i],'ASG_normal',str(j)), normal_img, mask, points)
        
        print('------------------------------------------')
        print('dataset:', setnames[i])
        print('ori_deg_AP:', sum(ori_deg_AP) / len(ori_deg_AP))
        print('ori_normal_AP:', sum(ori_normal_AP) / len(ori_normal_AP))
        print('RouGE_deg_AP:', sum(RouGE_deg_AP) / len(RouGE_deg_AP))
        print('RouGE_normal_AP:', sum(RouGE_normal_AP) / len(RouGE_normal_AP))
        wandb.log({'ori_deg_AP_'+setnames[i]:sum(ori_deg_AP) / len(ori_deg_AP), 'ori_normal_AP_'+setnames[i]:sum(ori_normal_AP) / len(ori_normal_AP),'RouGE_deg_AP_'+setnames[i]:sum(RouGE_deg_AP) / len(RouGE_deg_AP),'RouGE_normal_AP_' +setnames[i]:sum(RouGE_normal_AP) / len(RouGE_normal_AP)})



if __name__ == '__main__':
    wandb.login()
    
    parser = argparse.ArgumentParser()
    parser.add_argument('-bz', '--batch_size', type = int, default = 1, help = 'batch size')
    parser.add_argument("--RouGE_start_index", type=int, default = 10, help="The starting index of block that use RouGE to finetune.")
    parser.add_argument("--datatype", type=str, default='APcal')
    parser.add_argument('--save_mask', type=bool, default=False)
        
    args = parser.parse_args()
    run = wandb.init(
        project='additive gated mAP',
        config={
            'type':'const prob'
        },
    )
    train_root = './ckpt/train/all/RouGE_from_10/Weights_for_RouGE'

    
    mask_mAP(args, train_root)
