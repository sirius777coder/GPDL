# A lot of functions are taken from
# 1. https://github.com/facebookresearch/esm
# 2. https://github.com/jingraham/neurips19-graph-protein-design
# 3. https://github.com/aqlaboratory/openfold

# PS : here we use biotite to build our code blocks rather than bipython
import json
import math
import sys

import biotite.structure
from biotite.structure.io import pdbx, pdb
from biotite.structure.residues import get_residues
from biotite.structure import get_chains
from biotite.sequence import ProteinSequence
import biotite.structure.io as strucio
import biotite.structure as struc
import numpy as np
from scipy.spatial import transform
from scipy.stats import special_ortho_group
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as data
from typing import Sequence, Tuple, List, Optional,Iterable
from openfold.utils.rigid_utils import Rotation, Rigid
from openfold.np import residue_constants
from openfold.utils.tensor_utils import (
    tree_map,
    tensor_tree_map,
    masked_mean,
    permute_final_dims,
    batched_gather,
)
from esm.data import BatchConverter



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
restype_num = len(restypes)  # := 20.
alphabet = {
    'ALA': 'A', 'VAL': 'V', 'PHE': 'F', 'PRO': 'P', 'MET': 'M',
    'ILE': 'I', 'LEU': 'L', 'ASP': 'D', 'GLU': 'E', 'LYS': 'K',
    'ARG': 'R', 'SER': 'S', 'THR': 'T', 'TYR': 'Y', 'HIS': 'H',
    'CYS': 'C', 'ASN': 'N', 'GLN': 'Q', 'TRP': 'W', 'GLY': 'G',
}
alphabet_map = {v:k for k,v in alphabet.items()}


def extract_seq(protein, chain_id=None):
    if isinstance(protein, str):
        atom_array = strucio.load_structure(protein, model=1)
    elif isinstance(protein, struc.AtomArrayStack):
        atom_array = protein[0]
    elif isinstance(protein, struc.AtomArray):
        atom_array = protein
    aa_mask = struc.filter_canonical_amino_acids(atom_array)
    atom_array = atom_array[aa_mask]
    all_chains = get_chains(atom_array)
    if len(all_chains) == 0:
        raise ValueError('No chains found in the input file.')
    if chain_id is None:
        chain_ids = all_chains
    elif isinstance(chain, list):
        chain_ids = chain
    else:
        chain_ids = [chain]
    for chain in chain_ids:
        if chain not in all_chains:
            raise ValueError(f'Chain {chain} not found in input file')
    chain_filter = [a.chain_id in chain_ids for a in atom_array]
    atom_array = atom_array[chain_filter]
    # mask canonical aa
    aa_mask = struc.filter_amino_acids(atom_array)
    atom_array = atom_array[aa_mask]
    residue_identities = get_residues(atom_array)[1]
    seq = ''.join([ProteinSequence.convert_letter_3to1(r)
                  for r in residue_identities])
    return seq


def extract_plddt(protein,chain_id=None):
    if isinstance(protein,str):
        # model = 1 to load a AtomArray object
        # extra_fields to load the b_factor column
        atom_array = strucio.load_structure(protein,model=1,extra_fields=["b_factor"])
    elif isinstance(protein, struc.AtomArrayStack):
        atom_array = protein[0]
    elif isinstance(protein, struc.AtomArray):
        atom_array = protein

    # add multiple chain sequence subtract function
    all_chains = get_chains(atom_array)
    if len(all_chains) == 0:
        raise ValueError('No chains found in the input file.')
    if chain_id is None:
        chain_ids = all_chains
    elif isinstance(chain_id, list):
        chain_ids = chain_id
    else:
        chain_ids = [chain_id] 
    for chain in chain_ids:
        if chain not in all_chains:
            raise ValueError(f'Chain {chain} not found in input file')
    chain_filter = [a.chain_id in chain_ids for a in atom_array]
    atom_array = atom_array[chain_filter]

    # mask canonical aa 
    aa_mask = struc.filter_canonical_amino_acids(atom_array)
    atom_array = atom_array[aa_mask]

    # ca atom only
    atom_array = atom_array[atom_array.atom_name == "CA"]

    plddt = np.array([i.b_factor for i in atom_array])

    return plddt, np.mean(plddt)

def output_to_pdb(positions:torch.Tensor, aatype:torch.Tensor, plddt:torch.Tensor = None, file_path = "result.pdb"):
    """
    Assume batch = 1 (B=1)
    positions [B, L, 3, 3]
    aatype [B, L]
    plddt [B, L, 3]
    """
    if plddt is None:
        plddt = torch.zeros((1,positions.shape[1],3))
    positions = torch.detach(positions)
    positions = positions.to("cpu").numpy()
    
    aatype = torch.detach(aatype)
    aatype = aatype.to("cpu").numpy()

    plddt = torch.detach(plddt)
    plddt = plddt.to("cpu").numpy()

    B,L = positions.shape[:2]
    atom_list = []
    for i in range(L):
        if aatype[0][i] < len(restypes):
            res_type_1c = restypes[aatype[0][i]]
            res_type_3c = alphabet_map[res_type_1c]
        else:
            res_type_3c = "UNK"
        atom_N = positions[0][i][0]
        plddt_N = plddt[0][i][0]

        atom_CA = positions[0][i][1]
        plddt_CA = plddt[0][i][1]

        atom_C = positions[0][i][2]
        plddt_C = plddt[0][i][2]

        atom1 = biotite.structure.Atom(atom_N, chain_id="A", res_id=i+1, res_name=res_type_3c,
                        atom_name="N", element="N", b_factor=plddt_N)
        atom2 = biotite.structure.Atom(atom_CA, chain_id="A", res_id=i+1, res_name=res_type_3c,
                        atom_name="CA", element="C", b_factor=plddt_CA)
        atom3 = biotite.structure.Atom(atom_C, chain_id="A", res_id=i+1, res_name=res_type_3c,
                        atom_name="C", element="C", b_factor=plddt_C)
        atom_list += [atom1,atom2,atom3]
    array = biotite.structure.array(atom_list)
    
    pdb_file = pdb.PDBFile()
    pdb_file.set_structure(array)
    pdb_file.write(file_path)


def trainable_parameters(parameters):
    num_para = 0
    for parameter in parameters():
        if parameter.grad is not None:
            num_para += parameter.numel()
    return num_para


def move_batch(batch,device=torch.device("cuda")):
    batch = {
        k: torch.as_tensor(v, device=device)
        for k, v in batch.items()
    }
    return batch

def recur_print(x):
    if isinstance(x, torch.Tensor) or isinstance(x, np.ndarray):
        return f"{x.shape}_{x.dtype}"
    elif isinstance(x, dict):
        return {k: recur_print(v) for k, v in x.items()}
    elif isinstance(x, Iterable):
        return [recur_print(v) for v in x]
    else:
        raise RuntimeError(x)

def load_jsonl(json_file: str) -> list:
    data = []
    with open(json_file, "r") as f:
        for line in f:
            try:
                # 每一行代表一个序列的字典字符串加一个'\n',对于字符串应该用json.loads()文件进行读取
                data.append(json.loads(line.replace("\n", "")))
            except ValueError:
                pass
    return data

def identity(a:str, b:str) -> float:
    """
    identity of two strings 
    """
    return np.mean([i==j for i,j in zip(a,b)])

def loss_nll(S, log_probs, mask):
    """ Negative log probabilities from https://github.com/jingraham/neurips19-graph-protein-design/blob/master/experiments/utils.py
    - input 
        - S         : [B, N_res]
        - log_probs : [B, N_res, 20]
        - mask      : [B, N_res] 1 means compute loss while 0 doesn't
    """
    criterion = torch.nn.NLLLoss(reduction='none')
    loss = criterion(
        log_probs.contiguous().view(-1, log_probs.size(-1)), S.contiguous().view(-1)
    ).view(S.size())
    loss_av = torch.sum(loss * mask) / torch.sum(mask)
    return loss, loss_av

def loss_smoothed(S, log_probs, mask, weight=0.1, vocab=22):
    """ Negative log probabilities """
    S_onehot = torch.nn.functional.one_hot(S).float()

    # Label smoothing
    # S_onehot = S_onehot + weight / float(S_onehot.size(-1))
    S_onehot = S_onehot + weight / vocab
    S_onehot = S_onehot / S_onehot.sum(-1, keepdim=True)

    loss = -(S_onehot * log_probs).sum(-1)
    loss_av = torch.sum(loss * mask) / torch.sum(mask)
    return loss, loss_av


# Biotite module forthe pdb/cif structures
def load_structure(fpath,chain=None):
    """
    loading atom from the fpath, from the given chain
    """
    structure = strucio.load_structure(fpath,model=1)
    aa_mask = struc.filter_amino_acids(structure)
    structure = structure[aa_mask]
    all_chains = get_chains(structure)
    if len(all_chains) == 0:
        raise ValueError('No chains found in the input file.')
    if chain is None:
        chain_ids = all_chains
    elif isinstance(chain, list):
        chain_ids = chain
    else:
        chain_ids = [chain] 
    for chain in chain_ids:
        if chain not in all_chains:
            raise ValueError(f'Chain {chain} not found in input file')
    chain_filter = [a.chain_id in chain_ids for a in structure]
    structure = structure[chain_filter]
    # filter the canonical amino acid
    aa_mask = struc.filter_amino_acids(structure)
    structure = structure[aa_mask]
    return structure


def extract_coords_from_structure(structure: biotite.structure.AtomArray,pattern="max"):
    """
    Args:
        structure: An instance of biotite AtomArray
    Returns:
        Tuple (coords, seq)
            - coords is an L x 3 x 3 array for N, CA, C coordinates or O atom for pattern = "max"
            - seq is the extracted sequence
    """
    if pattern == "min":
        coords = get_atom_coords_residuewise(["N", "CA", "C"], structure)
    elif pattern == "max":
        coords = get_atom_coords_residuewise(["N", "CA", "C", "O"], structure)
    residue_identities = get_residues(structure)[1]
    seq = ''.join([ProteinSequence.convert_letter_3to1(r)
                  for r in residue_identities])
    return coords, seq


def load_coords(fpath, chain):
    """
    Args:
        fpath: filepath to either pdb or cif file
        chain: the chain id
    Returns:
        Tuple (coords, seq)
            - coords is an L x 3 x 3 array for N, CA, C coordinates
            - seq is the extracted sequence
    """
    structure = load_structure(fpath, chain)
    return extract_coords_from_structure(structure)


def get_atom_coords_residuewise(atoms: List[str], struct: biotite.structure.AtomArray):
    """
    Example for atoms argument: ["N", "CA", "C"]
    """
    def filterfn(s, axis=None):
        filters = np.stack([s.atom_name == name for name in atoms], axis=1)
        sum = filters.sum(0)
        if not np.all(sum <= np.ones(filters.shape[1])):
            raise RuntimeError("structure has multiple atoms with same name")
        index = filters.argmax(0)
        coords = s[index].coord
        coords[sum == 0] = float("nan")
        return coords

    return biotite.structure.apply_residue_wise(struct, struct, filterfn)


# Rotation and translation by tensor computation
def transform(v, R, t=None):
    """
    Rotates a vector by a rotation matrix.

    Args:
        v: 3D vector, tensor of shape (length x batch_size x channels x 3)
        R: rotation matrix, tensor of shape (length x batch_size x 3 x 3)
        t: translation vector, (length x batch_size x 3)
    Returns:
        Rotated version of v by rotation matrix R.
    """
    R = R.unsqueeze(-3)  # [B, L, 3, 3]        -> [B, L, 1, 3, 3]
    v = v.unsqueeze(-1)  # [B, L, channels, 3] -> [B, L, channels, 3 ,1 ]
    v_rotation = (R @ v).squeeze(-1)
    if t is not None:
        v_translation = v_rotation + t
    return v_translation


def get_bb_frames(coords):
    """
    Returns a local rotation frame defined by N, CA, C positions.
    Args:
        coords: coordinates, tensor of shape (batch_size x length x 3 x 3)
        where the third dimension is in order of N, CA, C
    Returns:
        Local relative rotation frames in shape (batch_size x length x 3 x 3)
        Local translation in shape (batch_size x length x 3)
    """
    v1 = coords[:, :, 2] - coords[:, :, 1]
    v2 = coords[:, :, 0] - coords[:, :, 1]
    e1 = normalize(v1, dim=-1)
    u2 = v2 - e1 * torch.sum(e1 * v2, dim=-1, keepdim=True)
    e2 = normalize(u2, dim=-1)
    e3 = torch.cross(e1, e2, dim = -1)
    R = torch.stack([e1, e2, e3], dim=-1) # note dim = -1 !!!
    t = coords[:, :, 1]  # translation is just the CA atom coordinate
    return R, t



def rot_to_quat(
    rot: torch.Tensor,
):
    if(rot.shape[-2:] != (3, 3)):
        raise ValueError("Input rotation is incorrectly shaped")

    rot = [[rot[..., i, j] for j in range(3)] for i in range(3)]
    [[xx, xy, xz], [yx, yy, yz], [zx, zy, zz]] = rot 

    k = [
        [ xx + yy + zz,      zy - yz,      xz - zx,      yx - xy,],
        [      zy - yz, xx - yy - zz,      xy + yx,      xz + zx,],
        [      xz - zx,      xy + yx, yy - xx - zz,      yz + zy,],
        [      yx - xy,      xz + zx,      yz + zy, zz - xx - yy,]
    ]

    k = (1./3.) * torch.stack([torch.stack(t, dim=-1) for t in k], dim=-2)

    _, vectors = torch.linalg.eigh(k)
    return vectors[..., -1]

def norm(tensor, dim, eps=1e-8, keepdim=False):
    """
    Returns L2 norm along a dimension.
    """
    return torch.sqrt(
        torch.sum(torch.square(tensor), dim=dim, keepdim=keepdim) + eps)


def normalize(tensor, dim=-1):
    """
    Normalizes a tensor along a dimension after removing nans.
    """
    return nan_to_num(
        torch.div(tensor, norm(tensor, dim=dim, keepdim=True))
    )


def nan_to_num(ts, val=0.0):
    """
    Replaces nans in tensor with a fixed value.    
    """
    val = torch.tensor(val, dtype=ts.dtype, device=ts.device)
    return torch.where(~torch.isfinite(ts), val, ts)




# Rigid apply
# rot_vec_mul : r[*, 3, 3] + point[*, 3] --> 最终会将point和frame的broadcast到相同维度

def compute_fape(
    pred_frames: Rigid,
    target_frames: Rigid,
    frames_mask: torch.Tensor,
    pred_positions: torch.Tensor,
    target_positions: torch.Tensor,
    positions_mask: torch.Tensor,
    length_scale: 10.0,
    l1_clamp_distance: Optional[float] = None,
    eps=1e-8,
) -> torch.Tensor:
    """
        Computes FAPE loss.
        Args:
            pred_frames:
                [*, N_frames] Rigid object of predicted frames
            target_frames:
                [*, N_frames] Rigid object of ground truth frames
            frames_mask:
                [*, N_frames] binary mask for the frames
            pred_positions: ---注意此时的N_pts是将所有每个序列里所有氨基酸落到一起 -- N_res * N_points
                [*, N_pts, 3] predicted atom positions
            target_positions:
                [*, N_pts, 3] ground truth positions
            positions_mask:
                [*, N_pts] positions mask
            length_scale:
                Length scale by which the loss is divided
            l1_clamp_distance:
                Cutoff above which distance errors are disregarded
            eps:
                Small value used to regularize denominators
        Returns:
            [*] loss tensor
    """

    # [*, N_frames, N_pts, 3]
    local_pred_pos = pred_frames.invert()[..., None].apply(
        pred_positions[..., None, :, :],
    ) # frame [B, N_frames, 1, 3, 3] apply points [B, 1, N_points, 3] -> SE(3) opertaion points: [B, N_frames, N_points, 3]

    local_target_pos = target_frames.invert()[..., None].apply(
        target_positions[..., None, :, :],
    )

    # [*, N_frames, N_pts]
    error_dist = torch.sqrt(
        torch.sum((local_pred_pos - local_target_pos) ** 2, dim=-1) + eps
    )

    if l1_clamp_distance is not None:
        error_dist = torch.clamp(error_dist, min=0, max=l1_clamp_distance)

    normed_error = error_dist / length_scale

    # [*, N_frames, N_pts]
    # [B, N_frames, N_pts] * [B, N_frames, 1] --- mask the padding frames
    normed_error = normed_error * frames_mask[..., None]

    # [*, N_frames, N_pts]
    # [*, N_frames, N_pts] * [*, 1, N_pts] 
    normed_error = normed_error * positions_mask[..., None, :]

    # FP16-friendly averaging. Roughly equivalent to:
    #
    # norm_factor = (
    #     torch.sum(frames_mask, dim=-1) *
    #     torch.sum(positions_mask, dim=-1)
    # )
    # normed_error = torch.sum(normed_error, dim=(-1, -2)) / (eps + norm_factor)
    #
    # ("roughly" because eps is necessarily duplicated in the latter)

    # summation over all the points for a single relative frame transformation,and divide the atom numbers
    # [B, N_frames]
    # 1. sum the atoms [B, N_frames, N_points] -> [B, N_frames]
    # 2. divide the frame numbers [B, N_frames]/[B,1] ?
    normed_error = torch.sum(normed_error, dim=-1)
    normed_error = (
        normed_error / (eps + torch.sum(frames_mask, dim=-1))[..., None]
    )

    # summation over all the relative transofmations
    # [B,]
    # 3. sum the relative frames [B,]
    # 4. divide the atom positions ?
    normed_error = torch.sum(normed_error, dim=-1)
    normed_error = normed_error / (eps + torch.sum(positions_mask, dim=-1))

    return normed_error


def backbone_loss(
    backbone_rigid_tensor: torch.Tensor,
    backbone_rigid_mask: torch.Tensor,
    traj: torch.Tensor,
    use_clamped_fape: Optional[torch.Tensor] = None,
    clamp_distance: float = 10.0,
    loss_unit_distance: float = 10.0,
    eps: float = 1e-4,
    **kwargs,
) -> torch.Tensor:
    pred_aff = Rigid.from_tensor_7(traj)
    pred_aff = Rigid(
        Rotation(rot_mats=pred_aff.get_rots().get_rot_mats(), quats=None),
        pred_aff.get_trans(),
    )

    # DISCREPANCY: DeepMind somehow gets a hold of a tensor_7 version of
    # backbone tensor, normalizes it, and then turns it back to a rotation
    # matrix. To avoid a potentially numerically unstable rotation matrix
    # to quaternion conversion, we just use the original rotation matrix
    # outright. This one hasn't been composed a bunch of times, though, so
    # it might be fine.
    gt_aff = Rigid.from_tensor_4x4(backbone_rigid_tensor)

    fape_loss = compute_fape(
        pred_aff,
        gt_aff[None],
        backbone_rigid_mask[None],
        pred_aff.get_trans(),
        gt_aff[None].get_trans(),
        backbone_rigid_mask[None],
        l1_clamp_distance=clamp_distance,
        length_scale=loss_unit_distance,
        eps=eps,
    )
    if use_clamped_fape is not None:
        unclamped_fape_loss = compute_fape(
            pred_aff,
            gt_aff[None],
            backbone_rigid_mask[None],
            pred_aff.get_trans(),
            gt_aff[None].get_trans(),
            backbone_rigid_mask[None],
            l1_clamp_distance=None,
            length_scale=loss_unit_distance,
            eps=eps,
        )

        fape_loss = fape_loss * use_clamped_fape + unclamped_fape_loss * (
            1 - use_clamped_fape
        )

    # Average over the batch dimension
    fape_loss = torch.mean(fape_loss)

    return fape_loss

def lddt(
    all_atom_pred_pos: torch.Tensor,
    all_atom_positions: torch.Tensor,
    all_atom_mask: torch.Tensor,
    cutoff: float = 15.0,
    eps: float = 1e-10,
    per_residue: bool = True,
) -> torch.Tensor:
    n = all_atom_mask.shape[-2]
    dmat_true = torch.sqrt(
        eps
        + torch.sum(
            (
                all_atom_positions[..., None, :]
                - all_atom_positions[..., None, :, :]
            )
            ** 2,
            dim=-1,
        )
    ) # [B, L, L] 

    dmat_pred = torch.sqrt(
        eps
        + torch.sum(
            (
                all_atom_pred_pos[..., None, :]
                - all_atom_pred_pos[..., None, :, :]
            )
            ** 2,
            dim=-1,
        )
    )
    dists_to_score = (
        (dmat_true < cutoff)
        * all_atom_mask
        * permute_final_dims(all_atom_mask, (1, 0))
        * (1.0 - torch.eye(n, device=all_atom_mask.device))
    )

    dist_l1 = torch.abs(dmat_true - dmat_pred)

    score = (
        (dist_l1 < 0.5).type(dist_l1.dtype)
        + (dist_l1 < 1.0).type(dist_l1.dtype)
        + (dist_l1 < 2.0).type(dist_l1.dtype)
        + (dist_l1 < 4.0).type(dist_l1.dtype)
    )
    score = score * 0.25

    dims = (-1,) if per_residue else (-2, -1)
    norm = 1.0 / (eps + torch.sum(dists_to_score, dim=dims))
    score = norm * (eps + torch.sum(dists_to_score * score, dim=dims))

    return score

def lddt_ca(
    all_atom_pred_pos: torch.Tensor,
    all_atom_positions: torch.Tensor,
    all_atom_mask: torch.Tensor,
    cutoff: float = 15.0,
    eps: float = 1e-10,
    per_residue: bool = True,
) -> torch.Tensor:
    ca_pos = residue_constants.atom_order["CA"]
    all_atom_pred_pos = all_atom_pred_pos[..., ca_pos, :]
    all_atom_positions = all_atom_positions[..., ca_pos, :]
    # all_atom_mask = all_atom_mask[..., ca_pos : (ca_pos + 1)]  # keep dim

    return lddt(
        all_atom_pred_pos,
        all_atom_positions,
        all_atom_mask,
        cutoff=cutoff,
        eps=eps,
        per_residue=per_residue,
    )

class CoordBatchConverter(BatchConverter):
    def __init__(self, alphabet, truncation_seq_length: int = None):
        self.alphabet = alphabet
        self.truncation_seq_length = truncation_seq_length

    def __call__(self, raw_batch: Sequence[Tuple[Sequence, str]], device=None):
        """
        Args:
            raw_batch: List of tuples (coords, confidence, seq)
            In each tuple,
                coords: list of floats, shape L x 3 x 3
                confidence: list of floats, shape L; or scalar float; or None
                seq: string of length L
        Returns:
            coords: Tensor of shape batch_size x L x 3 x 3
            confidence: Tensor of shape batch_size x L
            strs: list of strings
            tokens: LongTensor of shape batch_size x L
            padding_mask: ByteTensor of shape batch_size x L
        """
        self.alphabet.cls_idx = self.alphabet.get_idx("<cath>")
        batch = []
        for coords, confidence, seq in raw_batch:
            if confidence is None:
                confidence = 1.
            if isinstance(confidence, float) or isinstance(confidence, int):
                confidence = [float(confidence)] * len(coords)
            if seq is None:
                seq = 'X' * len(coords)
            batch.append(((coords, confidence), seq))

        coords_and_confidence, strs, tokens = super().__call__(batch)

        # pad beginning and end of each protein due to legacy reasons
        coords = [
            F.pad(torch.tensor(cd), (0, 0, 0, 0, 1, 1), value=np.inf)
            for cd, _ in coords_and_confidence
        ]
        confidence = [
            F.pad(torch.tensor(cf), (1, 1), value=-1.)
            for _, cf in coords_and_confidence
        ]
        coords = self.collate_dense_tensors(coords, pad_v=np.nan)
        confidence = self.collate_dense_tensors(confidence, pad_v=-1.)
        if device is not None:
            coords = coords.to(device)
            confidence = confidence.to(device)
            tokens = tokens.to(device)
        padding_mask = torch.isnan(coords[:, :, 0, 0])
        coord_mask = torch.isfinite(coords.sum(-2).sum(-1))
        confidence = confidence * coord_mask + (-1.) * padding_mask
        return coords, confidence, strs, tokens, padding_mask

    def from_lists(self, coords_list, confidence_list=None, seq_list=None, device=None):
        """
        Args:
            coords_list: list of length batch_size, each item is a list of
            floats in shape L x 3 x 3 to describe a backbone
            confidence_list: one of
                - None, default to highest confidence
                - list of length batch_size, each item is a scalar
                - list of length batch_size, each item is a list of floats of
                    length L to describe the confidence scores for the backbone
                    with values between 0. and 1.
            seq_list: either None or a list of strings
        Returns:
            coords: Tensor of shape batch_size x L x 3 x 3
            confidence: Tensor of shape batch_size x L
            strs: list of strings
            tokens: LongTensor of shape batch_size x L
            padding_mask: ByteTensor of shape batch_size x L
        """
        batch_size = len(coords_list)
        if confidence_list is None:
            confidence_list = [None] * batch_size
        if seq_list is None:
            seq_list = [None] * batch_size
        raw_batch = zip(coords_list, confidence_list, seq_list)
        return self.__call__(raw_batch, device)

    @staticmethod
    def collate_dense_tensors(samples, pad_v):
        """
        Takes a list of tensors with the following dimensions:
            [(d_11,       ...,           d_1K),
             (d_21,       ...,           d_2K),
             ...,
             (d_N1,       ...,           d_NK)]
        and stack + pads them into a single tensor of:
        (N, max_i=1,N { d_i1 }, ..., max_i=1,N {diK})
        """
        if len(samples) == 0:
            return torch.Tensor()
        if len(set(x.dim() for x in samples)) != 1:
            raise RuntimeError(
                f"Samples has varying dimensions: {[x.dim() for x in samples]}"
            )
        # assumes all on same device
        (device,) = tuple(set(x.device for x in samples))
        max_shape = [max(lst) for lst in zip(
            *[x.shape for x in samples])]  # 必须要zip打包 zip (*)
        result = torch.empty(
            len(samples), *max_shape, dtype=samples[0].dtype, device=device
        )
        result.fill_(pad_v)
        for i in range(len(samples)):
            result_i = result[i]  # 浅拷贝
            t = samples[i]
            result_i[tuple(slice(0, k) for k in t.shape)] = t
        return result

if __name__ == "__main__":
    print(f"{extract_seq(sys.argv[1])}")