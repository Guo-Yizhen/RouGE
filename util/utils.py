import json
import torch.nn.functional as F
import torch

def size_fix_all(x, p, mask, img_size):
    mask = mask.float()
    if (
            x.shape[1] != img_size
            or x.shape[2] != img_size
        ):
            p[0][0][0] = p[0][0][0] * img_size / x.shape[2]
            p[0][0][1] = p[0][0][1] * img_size / x.shape[1] 
            mask = F.interpolate(
                mask,
                (img_size, img_size),
                mode="nearest",
            ).squeeze(0)
            
            x = F.interpolate(
                x.unsqueeze(0),
                (img_size, img_size),
                mode="bilinear",
            ).squeeze(0)   

    return x, p, mask

def size_fix(x, img_size):
    if (
            x.shape[1] != img_size
            or x.shape[2] != img_size
        ):
            x = F.interpolate(
                x.unsqueeze(0),
                (img_size, img_size),
                mode="bilinear",
            ).squeeze(0)   

    return x

def load_list_from_json(filename):
    with open(filename, 'r') as file:
        loaded_list = json.load(file)
    return loaded_list

def load_list_from_txt(filename):
    lines = []
    with open(filename, 'r') as file:
        for line in file:
            lines.append(line.strip())
    return lines

def mask_fix(x, output_h, output_w):
    output_masks = F.interpolate(
        x, (output_h, output_w), mode="bicubic"
    )
    output_masks = torch.reshape(
        output_masks,
        (x.shape[0], 1, x.shape[1], output_h, output_w),
    )