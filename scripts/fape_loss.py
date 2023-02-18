import rigid_utils
import torch
import torch.nn as nn
import numpy as np

def fape(pred_coord:torch.Tensor,gt_coord:torch.Tensor, mask:torch.Tensor=None):
    """
    Compute backbone fape loss for the structure
    pred_coord [B, N_res, 3, 3]      prediction results
    gr_coord [B, N_res, 3, 3]        ground truth results
    mask [B, N_res]                  mask for batch data
    """

    B,N_res = pred_coord.shape[:2]

    if mask is None:
        mask = torch.ones((B,N_res))
    mask_atom_positions = mask.unsqueeze(-1).expand(B,N_res,3)

    # convert the pred coord to global frames
    bb_frame_atom = pred_coord[:,:,0:3,:]
    # rotation [B, L, 3, 3]
    # translation [B, L, 3]
    bb_rotation,bb_translation = get_bb_frames(bb_frame_atom)
    bb_frame = torch.zeros((*bb_rotation.shape[:-2],4,4),device=pred_coord.device)
    bb_frame[...,:3,:3] = bb_rotation
    bb_frame[...,:3,3] = bb_translation # [B, L, 4, 4]
    bb_frame = Rigid.from_tensor_4x4(bb_frame)

    gt_bb_frame_atom = gt_coord[:,:,0:3,:]
    gt_bb_rotation,gt_bb_translation = get_bb_frames(gt_bb_frame_atom)
    gt_bb_frame = torch.zeros((*gt_bb_rotation.shape[:-2],4,4),device=gt_coord.device)
    gt_bb_frame[...,:3,:3] = gt_bb_rotation
    gt_bb_frame[...,:3,3] = gt_bb_translation # [B, L, 4, 4]
    gt_bb_frame = Rigid.from_tensor_4x4(gt_bb_frame)
    

    return torch.mean(compute_fape(pred_frames=bb_frame,target_frames=gt_bb_frame,frames_mask=mask,pred_positions=pred_coord,target_positions=gt_coord,positions_mask=mask_atom_positions))

    
    




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

if __name__ == "__main__":
    # fape()
    pass