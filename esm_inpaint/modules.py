import torch
import torch.nn as nn
from esm.esmfold.v1.trunk import FoldingTrunk
from esm.esmfold.v1.esmfold import ESMFold
import utils
import numpy as np

class ProteinFeatures(nn.Module):
    def __init__(self, embedding_dim, num_rbf=16, augment_eps=0.):
        """ Extract protein features """
        super(ProteinFeatures, self).__init__()
        self.embedding_dim = embedding_dim
        self.augment_eps = augment_eps 
        self.num_rbf = num_rbf
        self.edge_embedding = nn.Linear(num_rbf*25, embedding_dim, bias=False)

    def rbf(self,values, v_min=2., v_max=22.):
        """
        Returns RBF encodings in a new dimension at the end.
        """
        rbf_centers = torch.linspace(v_min, v_max, self.num_rbf, device=values.device)
        rbf_centers = rbf_centers.view([1] * len(values.shape) + [-1]) # view (*(1,)*len(values.shape),-1)
        rbf_std = (v_max - v_min) / self.num_rbf
        v_expand = torch.unsqueeze(values, -1)
        z = (values.unsqueeze(-1) - rbf_centers) / rbf_std
        return torch.exp(-z ** 2)

    def forward(self, X, mask):
        """
        input  - 
        X    : [B, L, 4, 3]  N,CA,C,O,Virtual CB
        mask : [B, L]
        output -
        [B, L, L, embedding_dim]

        """
        if self.augment_eps > 0:
            X = X + self.augment_eps * torch.randn_like(X)
        
        b = X[:,:,1,:] - X[:,:,0,:]
        c = X[:,:,2,:] - X[:,:,1,:]
        a = torch.cross(b, c, dim=-1)
        Cb = -0.58273431*a + 0.56802827*b - 0.54067466*c + X[:,:,1,:]
        Ca = X[:,:,1,:]
        N = X[:,:,0,:]
        C = X[:,:,2,:]
        O = X[:,:,3,:]
        atom_list = [N,Ca,C,Cb,O] # [B, L, 3] for some specific atoms
        RBF_all = [] # [B, L]
        for atom1 in atom_list: 
            for atom2 in atom_list:
                dist = torch.sqrt(torch.sum(torch.square(atom1[:,:,None,:]-atom2[:,None,:,:])) + 1e-6) # [B, L, L, 1]
                rbf_dist = self.rbf(dist) # [B, L, L, 16]
                RBF_all.append(rbf_dist)
        RBF_all = torch.cat(tuple(RBF_all), dim=-1) # [B, L, L, 16*25]
        print(RBF_all.dtype)
        E = self.edge_embedding(RBF_all)
        mask_2d = mask[:,:,None] * mask[:,None,:] # [B, L, L]
        mask_2d = mask_2d.unsqueeze(-1)
        return E * mask_2d

class esm_inpaint(nn.Module):
    def __init__(self,cfg):
        super().__init__()
        self.esmfold = ESMFold(cfg)
        self.ProteinFeatures = ProteinFeatures(cfg.trunk.pairwise_state_dim)
    
    def forward(self,coord,mask,S):
        coord = utils.nan_to_num(coord)
        dis_embed = self.ProteinFeatures(coord,mask)
        bb_frame_atom = coord[:,:,0:3,:]
        bb_rotation,bb_translation = utils.get_bb_frames(bb_frame_atom)
        mask_frame = mask.reshape(*mask.shape,1,1)
        mask_frame = mask_frame.expand(*mask_frame.shape[:-2],4,4)
        bb_frame = torch.zeros((*bb_rotation.shape[:-2],4,4),device=coord.device)
        bb_frame[...,:3,:3] = bb_rotation
        bb_frame[...,:3,3] = bb_translation # [B, L, 4, 4]
        bb_frame = bb_frame * mask_frame
        print(f"coord {coord.device}\n mask {coord.device}\n mask_frame {mask.device}\nembedding {dis_embed.device}\n bb_frame {bb_frame.device}")
        print(dis_embed.shape)
        structure = self.esmfold(dis_embed,bb_frame,S,mask)
        return structure
