# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import wandb
import torch
import time
import argparse
from efficient_sam.build_efficient_sam import build_efficient_sam_vitt, build_efficient_sam_vits
from torch.optim.lr_scheduler import LinearLR
import torch
from Trainer import train_epoch_full, test_epoch_full, test_mAP
from dataloader import get_dataset
import os
import torch.nn.functional as F
os.environ['CUDA_VISIBLE_DEVICES'] = "0" 

def dice_loss(targets, inputs, num_boxes=1):
    """
    Compute the DICE loss, similar to generalized IOU for masks
    Args:
        inputs: A float tensor of arbitrary shape.
                The predictions for each example.
        targets: A float tensor with the same shape as inputs. Stores the binary
                 classification label for each element in inputs
                (0 for the negative class and 1 for the positive class).
    """
    inputs = inputs.sigmoid()
    inputs = inputs.flatten(1)
    targets = targets.flatten(1)
    numerator = 2 * (inputs * targets).sum(1)
    denominator = inputs.sum(-1) + targets.sum(-1)
    loss = 1 - (numerator + 1) / (denominator + 1)
    # print(loss.sum() / num_boxes)
    return loss.sum() / num_boxes


def sigmoid_focal_loss(targets, inputs, num_boxes=1, alpha: float = 0.25, gamma: float = 2):
    """
    Loss used in RetinaNet for dense detection: https://arxiv.org/abs/1708.02002.
    Args:
        inputs: A float tensor of arbitrary shape.
                The predictions for each example.
        targets: A float tensor with the same shape as inputs. Stores the binary
                 classification label for each element in inputs
                (0 for the negative class and 1 for the positive class).
        alpha: (optional) Weighting factor in range (0,1) to balance
                positive vs negative examples. Default = -1 (no weighting).
        gamma: Exponent of the modulating factor (1 - p_t) to
               balance easy vs hard examples.
    Returns:
        Loss tensor
    """
    prob = inputs.sigmoid()
    ce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none")
    p_t = prob * targets + (1 - prob) * (1 - targets)
    loss = ce_loss * ((1 - p_t) ** gamma)

    if alpha >= 0:
        alpha_t = alpha * targets + (1 - alpha) * (1 - targets)
        loss = alpha_t * loss
    # print(loss.mean((2,3)).sum() / num_boxes)
    return loss.mean((2,3)).sum() / num_boxes

def dice_sigfocal(targets, inputs, num_boxes = 1):
    return dice_loss(targets, inputs, num_boxes) + sigmoid_focal_loss(targets, inputs, num_boxes)

def save_model(args, save_path, epoch, sam, optimizer, epoch_test_loss):
    if args.train_LN:
        torch.save({
        'epoch': epoch,
        'model_state_dict':sam.state_dict(), 
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': epoch_test_loss,
        },save_path)
    else:
        RouGEs_dict = {}
        for ind_blk, blk in enumerate(sam.image_encoder.blocks):
            if ind_blk >= args.RouGE_start_index:
                RouGEs_dict[ind_blk] = [blk.RouGE.state_dict()]
        torch.save({
            'epoch': epoch,
            'model_state_dict':RouGEs_dict, 
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': epoch_test_loss,
            },save_path)

def main(args, train_root):
    print("Loading model...")
    
    sam = build_efficient_sam_vitt().cuda()
    sam_image_encoder = sam.image_encoder
    print('SAM encoder loaded')
    for p in sam.parameters():
        p.requires_grad = False
    
    print('number of RouGE:', len(sam_image_encoder.blocks) - args.RouGE_start_index)

    start_epoch = 0

    RouGE_list = []
    for i, blk in enumerate(sam_image_encoder.blocks):
        if i >= args.RouGE_start_index :
            RouGE_list.append({'params':blk.RouGE.parameters()})
        else:
            del blk.RouGE 
    if args.optim == 'AdamW':
        optimizer = torch.optim.AdamW(
            RouGE_list,
            lr=args.lr, 
            weight_decay=args.wd) 
    elif args.optim == 'SGD':
        optimizer = torch.optim.SGD(
            RouGE_list, 
            lr=args.lr, 
            momentum=0.0
            )
        
    if args.scheduler:
        scheduler = LinearLR(optimizer, total_iters=args.num_epoch)
    print('model RouGE loaded')

    if args.resume == True:
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
        for ind_blk, blk in enumerate(sam_image_encoder.blocks):
            if ind_blk >= args.RouGE_start_index:
                blk.RouGE.load_state_dict(state_dicts['model_state_dict'][ind_blk][0]) 

        optimizer.load_state_dict(state_dicts['optimizer_state_dict']) 
        start_epoch = state_dicts['epoch']

    min_test_loss = 1000000
    epoch_test_loss = min_test_loss
    max_mAP = 0.0
    if args.loss == 'mse':
        loss_f = torch.nn.MSELoss()
    elif args.loss == 'L1':
        loss_f = torch.nn.L1Loss()
    elif args.loss == 'dice_sigfocal':
        loss_f = dice_sigfocal
    elif args.loss == 'dice_loss':
        loss_f = dice_loss
    elif args.loss == 'sigmoid_focal_loss':
        loss_f = sigmoid_focal_loss

    trainloaders, testloaders = get_dataset(args, sam)
    args.datatype = 'APcal'
    _, mAP_testloaders = get_dataset(args, sam)

    for epoch in range(start_epoch, args.num_epoch):
        print('current epoch: ', epoch)
        # train
        epoch_train_losses, NaN_count = train_epoch_full(
            dataloaders = trainloaders, 
            sam = sam, 
            loss_f = loss_f, 
            optimizer = optimizer, 
            args = args
            )
        if args.scheduler:
            scheduler.step()
        print('epoch_train_loss: ',sum(epoch_train_losses)/len(epoch_train_losses))

        if args.local_rank == 0:
            wandb.log({'train loss': sum(epoch_train_losses)/len(epoch_train_losses),'train NaN count': NaN_count})

        if epoch % 5 == 0:
            # test
            epoch_test_losses, NaN_count = test_epoch_full(
                dataloaders = testloaders, 
                sam = sam,
                loss_f = loss_f, 
                optimizer = optimizer, 
                args = args
            )            
            print('epoch_test_loss: ',epoch_test_losses)
            epoch_test_loss = sum(epoch_test_losses)/len(epoch_test_losses)
            wandb.log({'test loss': epoch_test_loss,'test NaN count': NaN_count})

        # AP cal
        deg_mAPs, normal_mAPs = test_mAP(
            dataloaders = mAP_testloaders, 
            sam = sam,
            args = args
        )     
        print('deg_mAPs: ',deg_mAPs)
        print('normal_mAPs: ',normal_mAPs)
        mAP = sum(deg_mAPs)/len(deg_mAPs) + sum(normal_mAPs)/len(normal_mAPs)
        
        save_path = os.path.join(train_root, 'model',str(epoch)+'_epoch.pth')
        save_model(args, save_path, epoch, sam, optimizer, epoch_test_loss)

        if epoch_test_loss < min_test_loss or mAP >= max_mAP:
            min_test_loss = epoch_test_loss
            max_mAP = mAP
            print('new min test loss:', min_test_loss)
            save_path = os.path.join(train_root, 'model',str(epoch)+'_best_model.pth')
            save_model(args, save_path, epoch, sam, optimizer, epoch_test_loss)

        print("epoch:" + str(epoch) + "Done!")
    return

if __name__ == '__main__':
    wandb.login()
    
    parser = argparse.ArgumentParser()

    parser.add_argument('-e', '--num_epoch', type = int, default = 30, help = 'num of training epoch')
    parser.add_argument('-r', '--lr', type = float, default = 1e-4, help = 'initial learning rate')
    parser.add_argument('-bz', '--batch_size', type = int, default = 1, help = 'batch size')
    parser.add_argument('--wd', type = float, default = 5e-2, help = 'weight decay')
    parser.add_argument("--RouGE_start_index", type=int, help="The starting index of block that use RouGE to finetune.")
    parser.add_argument("--local_rank", type=int)
    parser.add_argument("--scheduler", type=bool, default = False)
    parser.add_argument("--loss", type=str, default='L1')
    parser.add_argument("--optim", type=str, default='AdamW')
    parser.add_argument("--datatype", type=str, default='all')
    parser.add_argument("--resume",type=bool,default= False)
    parser.add_argument("--train_LN",type=bool,default= False)
        
    args = parser.parse_args()
    run = wandb.init(
        project='additive gated test',
        config={
            "learning_rate": args.lr,
            'weight decay': args.wd,
            "epochs": args.num_epoch,
            'RouGE_start_index':args.RouGE_start_index,
            'scheduler': args.scheduler,
            'loss': args.loss,
            'optim': args.optim,
        },
    )
    root = './ckpt/train/all'
    if args.resume == False:
        time_stamp = time.strftime("%Y-%m-%d-%H-%M-%S",time.localtime())
        if args.train_LN:
            train_root = os.path.join(root, 'LN_GE','RouGE_from_' + str(args.RouGE_start_index), time_stamp)
        else:
            train_root = os.path.join(root, 'RouGE_from_' + str(args.RouGE_start_index), time_stamp)

        if args.local_rank == 0:
            os.makedirs(train_root)
            os.makedirs(os.path.join(train_root, 'loss'))
            os.makedirs(os.path.join(train_root, 'model'))
    
    else:
        if args.train_LN:
            old_root = os.path.join(root,'LN_GE', 'RouGE_from_' + str(args.RouGE_start_index))
        else:
            old_root = os.path.join(root, 'RouGE_from_' + str(args.RouGE_start_index))
        root_l = os.listdir(old_root)
        root = root_l[0]
        for _root in root_l:
            if _root > root:
                root = _root
        train_root = os.path.join(old_root, root)


    main(args, train_root)

    # python train.py -e 20 --local_rank 0 --RouGE_start_index 10 -bz 16 --loss dice_sigfocal --lr 1e-4