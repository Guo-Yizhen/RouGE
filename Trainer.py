import torch
import tqdm
import wandb
from sklearn.metrics import average_precision_score

def full_epoch_adaptor(dataloaders, sam, loss_f, optimizer, RouGE_start_index, train = True):
    epoch_losses = []
    NaN_count = 0
    for dataloader in dataloaders:
        epoch_loss = []
        for deg_img, normal_img, deg_clipfeat, normal_clipfeat, gt_type, points, masks in tqdm.tqdm(dataloader):
            deg_img = deg_img.cuda()
            normal_img = normal_img.cuda()
            deg_clipfeat = deg_clipfeat.cuda()
            normal_clipfeat = normal_clipfeat.cuda()
            masks = masks.cuda()
            labels = torch.tensor([[[1]] for _ in range(len(points))]).cuda()
            points = points.cuda()
            with torch.no_grad():
                target_predicted_logits, target_predicted_iou = sam(
                    batched_images = normal_img,
                    batched_points = points,
                    batched_point_labels = labels,
                    RouGE_start_index = -1
                )
                sorted_ids = torch.argsort(target_predicted_iou, dim=-1, descending=True)
                target_predicted_iou = torch.take_along_dim(target_predicted_iou, sorted_ids, dim=2)
                target_predicted_logits = torch.take_along_dim(
                    target_predicted_logits, sorted_ids[..., None, None], dim=2
                )
                target_predicted_mask = torch.ge(target_predicted_logits[:,:,0,:,:], 0).float()
                
            optimizer.zero_grad()
            deg_predicted_logits, predicted_iou = sam(
                batched_images = deg_img,
                batched_points = points,
                batched_point_labels = labels,
                RouGE_start_index = RouGE_start_index,
                clip_feat = deg_clipfeat,
                gt_type = gt_type,
            )
            sorted_ids = torch.argsort(predicted_iou, dim=-1, descending=True)
            predicted_iou = torch.take_along_dim(predicted_iou, sorted_ids, dim=2)
            deg_predicted_logits = torch.take_along_dim(
                deg_predicted_logits, sorted_ids[..., None, None], dim=2
            )

            loss_end_1 = loss_f(target_predicted_mask, deg_predicted_logits[:,:,0,:,:])  # self-supervised learning 
            # loss_end_1 = loss_f(masks, deg_predicted_logits[:,:,0,:,:])
            epoch_loss.append(loss_end_1.item()/points.shape[0])
            wandb.log({'loss dark': loss_end_1.item()})

            if train:
                loss_end_1.backward()
                optimizer.step()

            optimizer.zero_grad()
            normal_predicted_logits, predicted_iou = sam(
                batched_images = normal_img,
                batched_points = points,
                batched_point_labels = labels,
                RouGE_start_index = RouGE_start_index,
                clip_feat = normal_clipfeat,
                gt_type = 0,
            )
            sorted_ids = torch.argsort(predicted_iou, dim=-1, descending=True)
            predicted_iou = torch.take_along_dim(predicted_iou, sorted_ids, dim=2)
            normal_predicted_logits = torch.take_along_dim(
                normal_predicted_logits, sorted_ids[..., None, None], dim=2
            )
            loss_end_2 = loss_f(target_predicted_mask, normal_predicted_logits[:,:,0,:,:]) #self supervise
            # loss_end_2 = loss_f(masks, normal_predicted_logits[:,:,0,:,:])

            epoch_loss.append(loss_end_2.item()/points.shape[0])
            wandb.log({'loss normal': loss_end_2.item()})
           
            if train:
                loss_end_2.backward()
                optimizer.step()
        epoch_losses.append(sum(epoch_loss)/len(epoch_loss))
    return epoch_losses, NaN_count

def mAP_cal(dataloaders, sam, RouGE_start_index):
    
    setnames = ['Rain200L', 'DDN', 'GoPro', 'LIS', 'snow100k']
    AGS_degs = []
    AGS_normals = []
    for i, testloader in enumerate(dataloaders):
        AGS_deg_AP = []
        AGS_normal_AP = []
        for deg_img, normal_img, deg_clipfeat, normal_clipfeat, gt_type, points, gt_masks in tqdm.tqdm(testloader):
            deg_img = deg_img.cuda()
            normal_img = normal_img.cuda()
            deg_clipfeat = deg_clipfeat.cuda()
            normal_clipfeat = normal_clipfeat.cuda()
            labels = torch.tensor([[[1]] for _ in range(len(points))]).cuda()
            points = points.cuda()


            predicted_logits, predicted_iou = sam(
                batched_images = deg_img,
                batched_points = points,
                batched_point_labels = labels,
                RouGE_start_index = RouGE_start_index,
                clip_feat = deg_clipfeat,
                gt_type = gt_type,
            )

            sorted_ids = torch.argsort(predicted_iou, dim=-1, descending=True)
            predicted_iou = torch.take_along_dim(predicted_iou, sorted_ids, dim=2)
            predicted_logits = torch.take_along_dim(
                predicted_logits, sorted_ids[..., None, None], dim=2
            )
            predicted_logits = predicted_logits.detach().cpu().numpy()
            mask = predicted_logits[:, 0, 0, :, :]
            AGS_deg_AP.append(average_precision_score(gt_masks.flatten(), mask.flatten()))

            predicted_logits, predicted_iou = sam(
                batched_images = normal_img,
                batched_points = points,
                batched_point_labels = labels,
                RouGE_start_index = RouGE_start_index,
                clip_feat = normal_clipfeat,
                gt_type = gt_type,
            )
            sorted_ids = torch.argsort(predicted_iou, dim=-1, descending=True)
            predicted_iou = torch.take_along_dim(predicted_iou, sorted_ids, dim=2)
            predicted_logits = torch.take_along_dim(
                predicted_logits, sorted_ids[..., None, None], dim=2
            )
            predicted_logits = predicted_logits.detach().cpu().numpy()
            mask = predicted_logits[:, 0, 0, :, :]
            AGS_normal_AP.append(average_precision_score(gt_masks.flatten(), mask.flatten()))
        
        AGS_deg = sum(AGS_deg_AP) / len(AGS_deg_AP)
        AGS_normal = sum(AGS_normal_AP) / len(AGS_normal_AP)
        AGS_degs.append(AGS_deg)
        AGS_normals.append(AGS_normal)
        print('dataset:', setnames[i])
        print('AGS_deg_AP:', AGS_deg)
        print('AGS_normal_AP:', AGS_normal)
        wandb.log({'AGS_deg_'+setnames[i]:AGS_deg, 'AGS_normal_'+setnames[i]:AGS_normal})
    return AGS_degs, AGS_normals


def train_epoch_full(dataloaders, sam, loss_f, optimizer, args):
    total_params = 0
    for i, blk in enumerate(sam.image_encoder.blocks):
        if i >= args.RouGE_start_index:
            for p in blk.adaptor.parameters():  # train gates and exports
                p.requires_grad = True
                total_params += p.numel()
    if args.train_LN:
        for name, p in sam.mask_decoder.named_parameters(): 
            if 'norm' in name:
                p.requires_grad = True
                total_params += p.numel()
    

    print('total trainable parameters:', total_params)
    
    epoch_losses, NaN_count = full_epoch_adaptor(dataloaders, sam, loss_f, optimizer, args.RouGE_start_index)
    return epoch_losses, NaN_count 

def test_epoch_full(dataloaders, sam, loss_f, optimizer, args):
    with torch.no_grad():
        epoch_losses, NaN_count = full_epoch_adaptor(dataloaders, sam, loss_f, optimizer, args.RouGE_start_index, train = False)
    return epoch_losses, NaN_count

def test_mAP(dataloaders, sam, args):
    with torch.no_grad():
        AGS_degs, AGS_normals = mAP_cal(dataloaders, sam, args.RouGE_start_index)
    return AGS_degs, AGS_normals