import torch
from typing import tuple

def rfdiffusion_mse(
    target : tuple(torch.Tensor, torch.Tensor),
    pred:tuple(torch.Tensor, torch.Tensor),
    mask: torch.Tensor
):
    """
    Follow the vanila RF diffusion mse loss for frames
    target/pred: (translate, Rotation)
    translate : [Batch, N_res, 3] for CA atom coordinate
    rotation : [Batch, N_res, 3, 3]
    mask : [Batch, N_res]
    """
    length = torch.sum(mask) # [B,]
    ca_dis = (target[0]-pred[0]) * mask # [B, L, 3]
    ca_mse = torch.sum( torch.sum((ca_dis ** 2),dim=-1) / length , dim =- 1) # [B, ]
    rotation_dis = torch.eye(3).unsqueeze(0).expand(target[1].shape[0],3,3) - pred[1].transpose(-1,-2) @ target # [B, L, 3, 3]
    rotation_dis = rotation_dis * mask
    rotation_mse = torch.linalg.norm(rotation_dis,ord="fro",dim=(-1,-2)) # [B, L]
    rotation_mse = torch.sum(rotation_mse/length, dim = -1) # [B, ]
    mse = torch.mean(torch.sqrt(ca_mse+rotation_mse),dim = -1) 

    return mse
