import torch
import torch.nn as nn
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import numpy as np
import json,copy,time
import utils

restypes = [
    "A",
    "R",
    "N",
    "D",
    "C",
    "Q",
    "E",
    "G",
    "H",
    "I",
    "L",
    "K",
    "M",
    "F",
    "P",
    "S",
    "T",
    "W",
    "Y",
    "V",
]
restype_order = {restype: i for i, restype in enumerate(restypes)}

class StructureDataset(Dataset):
    def __init__(self,jsonl_file,max_length=500,low_fraction=0.3,high_fraction=0.6):
        dataset = utils.load_jsonl(jsonl_file)
        self.data = []
        self.discard = {"bad_chars":0,"too_long":0}
        for entry in dataset:
            seq = entry['seq']
            # Check if in alphabet
            bad_chars = set([s for s in seq]).difference(restypes)
            if len(bad_chars) == 0:
                if len(entry['seq']) <= max_length:
                    pass
                else:
                    self.discard['too_long'] += 1
                    continue
            else:
                # print(entry['name'], bad_chars, entry['seq'])
                self.discard['bad_chars'] += 1
                continue
            seq = torch.tensor([restype_order[i] for i in seq],dtype=torch.long)
            coord = torch.from_numpy(np.stack(list(entry['coords'].values()),axis=-2))
            coord = coord.to(torch.float32)
            coord = utils.nan_to_num(coord) # remove the nan value
            bert_mask_fraction = torch.tensor([np.random.uniform(low=low_fraction, high=high_fraction),],dtype=torch.float32)
            bert_mask = []
            for _ in range(len(seq)):
                if np.random.random() < bert_mask_fraction:
                    bert_mask.append(False)
                else:
                    bert_mask.append(True)
            bert_mask = torch.tensor(bert_mask,dtype=torch.bool) # 0.0, mask; 1 unmask
            mask_coord = copy.deepcopy(coord)
            mask_coord[~bert_mask] = 0.0 # zero the mask backbone
            mask_seq = copy.deepcopy(seq)
            mask_seq[~bert_mask] = 0

            self.data.append({
                "coord":coord,
                "seq":seq,
                "mask_coord":mask_coord,
                "mask_seq":mask_seq,
                "bert_mask_fraction":bert_mask_fraction,
                "bert_mask":bert_mask
            })
        print(f"UNK token:{self.discard['bad_chars']},too long:{self.discard['too_long']}")

    def __len__(self):
        return len(self.data)

    def __getitem__(self,idx):
        return self.data[idx]

def batch_collate_function(batch):
    """
    A customized wrap up collate function
    Args:
        batch: a list of structure objects
    Shape:
        Output:
            coord_batch [B, L, 4, 3] dtype=float32
            seq_batch   [B, L]       dtype=int64
            bert_mask_fraction_batch [B,] dtype=float32
            bert_mask_batch          [B, L]  dtype=torch.bool   0 represents mask, 1 no mask
            padding_mask_batch       [B, L]  dtype=torch.float32   0 represents mask, 1 no mask
    """
    coord_batch = utils.CoordBatchConverter.collate_dense_tensors([i['coord'] for i in batch],0.0)
    seq_batch = utils.CoordBatchConverter.collate_dense_tensors([i['seq'] for i in batch],20)
    mask_coord_batch = utils.CoordBatchConverter.collate_dense_tensors([i['mask_coord'] for i in batch],0.0)
    mask_seq_batch = utils.CoordBatchConverter.collate_dense_tensors([i['mask_seq'] for i in batch],20)
    bert_mask_fraction_batch = utils.CoordBatchConverter.collate_dense_tensors([i['bert_mask_fraction'] for i in batch],0.0)
    bert_mask_batch = utils.CoordBatchConverter.collate_dense_tensors([i['bert_mask'] for i in batch],0.0)
    padding_mask_batch = seq_batch!=20 #True not mask, False represents mask
    padding_mask_batch = padding_mask_batch.to(torch.float32)
    output = {
        "coord":coord_batch,
        "seq":seq_batch,
        "mask_coord":mask_coord_batch,
        "mask_seq":mask_seq_batch,
        "bert_mask_fraction":bert_mask_fraction_batch,
        "bert_mask":bert_mask_batch,
        "padding_mask":padding_mask_batch
    }
    return output
    
def StructureDataloader(dataset,batch_size,num_workers,shuffle=True):
    """
    A wrap up dataloader
    """
    return DataLoader(dataset,batch_size=batch_size,num_workers=num_workers,shuffle=shuffle,collate_fn=batch_collate_function)


class TokenDataloader():
    def __init__(self, dataset, batch_size=100, shuffle=True,
        collate_fn=lambda x:x, drop_last=False):
        self.dataset = dataset
        self.size = len(dataset)
        self.lengths = [len(dataset[i]['seq']) for i in range(self.size)]
        self.batch_size = batch_size
        sorted_ix = np.argsort(self.lengths)

        # Cluster into batches of similar sizes
        clusters, batch = [], []
        batch_max = 0
        for ix in sorted_ix:
            size = self.lengths[ix]
            if size * (len(batch) + 1) <= self.batch_size:
                batch.append(ix)
                batch_max = size
            else:
                clusters.append(batch)
                batch, batch_max = [], 0
        if len(batch) > 0:
            clusters.append(batch)
        self.clusters = clusters

    def __len__(self):
        return len(self.clusters)

    def __iter__(self):
        np.random.shuffle(self.clusters)
        for b_idx in self.clusters:
            batch = [self.dataset[i] for i in b_idx]
            yield batch
        