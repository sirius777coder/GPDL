# From https://github.com/aqlaboratory/openfold/blob/main/openfold/utils/validation_metrics.py

import torch


def drmsd(structure_1, structure_2, mask=None):
    """
    distance map rmsd (using ca atom distance map)
    structure_1 [B, L, 3] 
    structure_2 [B, L, 3]
    mask        [B, L]

    NOTE : we don't need superimpose two structures
    """

    def prep_d(structure):
        # [B, L, L, 3]
        d = structure[..., :, None, :] - structure[..., None, :, :]
        d = d ** 2
        # [B, L, L]
        d = torch.sqrt(torch.sum(d, dim=-1))
        return d

    d1 = prep_d(structure_1)
    d2 = prep_d(structure_2)

    drmsd = d1 - d2
    drmsd = drmsd ** 2
    if(mask is not None):
        # distance mask [B, L, L]
        # [B, L, L]
        drmsd = drmsd * (mask[..., None] * mask[..., None, :])
    # [B,]
    drmsd = torch.sum(drmsd, dim=(-1, -2))
    # n : total number of residue pairs [B,]
    n = d1.shape[-1] if mask is None else torch.sum(mask, dim=-1)
    drmsd = drmsd * (1 / (n * (n - 1))) if n > 1 else (drmsd * 0.)
    drmsd = torch.sqrt(drmsd)

    return drmsd


def drmsd_np(structure_1, structure_2, mask=None):
    structure_1 = torch.tensor(structure_1)
    structure_2 = torch.tensor(structure_2)
    if(mask is not None):
        mask = torch.tensor(mask)

    return drmsd(structure_1, structure_2, mask)


def gdt(p1, p2, mask, cutoffs):
    """
    p1 : [B, L, 3] or [L, 3]_float32
    p2 : [B, L, 3] or [L, 3]_float32
    mask : [B ,L] or [L,]
    cutoff : list of floats or a float
    """

    # [B,] or single value
    n = torch.sum(mask, dim=-1)
    p1 = p1.float()
    p2 = p2.float()
    # [B,L] or [L,]
    distances = torch.sqrt(torch.sum((p1 - p2)**2, dim=-1))
    scores = []
    for c in cutoffs:
        # [B,] or a float
        score = torch.sum((distances <= c) * mask, dim=-1) / n
        # float : if batch exists, average all the batch sample
        score = torch.mean(score)
        # add the score of this cutoff
        scores.append(score)

    return sum(scores) / len(scores)


def gdt_ts(p1, p2, mask):
    return gdt(p1, p2, mask, [1., 2., 4., 8.])


def gdt_ha(p1, p2, mask):
    return gdt(p1, p2, mask, [0.5, 1., 2., 4.])