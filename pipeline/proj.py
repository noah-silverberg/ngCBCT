# Implements Notebook 1 functionality: load projection data, simulate non-gated and interpolate
import numpy as np
import torch
import scipy.io
import mat73
import matplotlib.pyplot as plt
import torch.nn as nn
from .config import DATA_DIR


def load_projection_mat(patient: str, scan: str, scan_type: str):
    """Load projection data from mat file. Returns odd_index (numpy), angles (torch.FloatTensor), prj (torch.FloatTensor)."""
    mat_path = (
        f"{DATA_DIR}/panc_prj/{scan_type}/mat/panc{patient}.{scan_type}{scan}.mat"
    )
    mat = mat73.loadmat(mat_path)
    odd_index = np.array(mat["odd_index"])
    angles = torch.from_numpy(np.array(mat["angles"])).float()
    prj = torch.from_numpy(np.array(mat["prj"])).float()
    return odd_index, angles, prj


def reformat_for_tigre(prj: torch.Tensor, angles: torch.Tensor):
    """Reformat prj and angles for TIGRE: flip and permute appropriately."""
    prj_gcbct = prj.detach().clone()
    prj_gcbct = torch.flip(prj_gcbct, (1,))
    prj_gcbct = prj_gcbct.permute(2, 1, 0)
    prj_gcbct = torch.flip(prj_gcbct, (2,))
    angles1 = -(angles + np.pi / 2)
    # Ensure monotonic
    if (angles1[-1:] - angles1[0]) < 0:
        angles1 = torch.flip(angles1, (0,))
        prj_gcbct = torch.flip(prj_gcbct, (0,))
    return prj_gcbct, angles1


def find_missing_indices(odd_index: np.ndarray):
    """Return sorted missing indices between first and last of odd_index."""
    first, last = odd_index[0], odd_index[-1]
    full_range = set(range(first, last + 1))
    present = set(odd_index)
    missing = sorted(full_range - present)
    return missing


def interpolate_projections(prj_gcbct: torch.Tensor, odd_index: np.ndarray):
    """Given prj_gcbct [num_angles, H, W] and acquired odd_index (1-based), zero out missing and linearly interpolate."""
    # Convert odd_index to zero-based
    ngcbct_idx = odd_index.astype(np.int64) - 1
    num_angles = prj_gcbct.shape[0]
    # Initialize prj_ngcbct with zeros
    prj_ngcbct = torch.zeros_like(prj_gcbct)
    # Place acquired
    prj_ngcbct[ngcbct_idx] = prj_gcbct[ngcbct_idx]
    # Prepare for interpolation: work in Tensor for math, but easier in numpy-like via torch
    tmp = prj_ngcbct.detach().clone().permute(1, 0, 2)  # [H, angles, W]
    miss_idx = find_missing_indices(odd_index)
    # Find gap boundaries
    sorted_idx = sorted(ngcbct_idx)
    gaps = []  # list of (start, end) indices of acquired where a gap begins and width
    for i in range(len(sorted_idx) - 1):
        if sorted_idx[i + 1] - sorted_idx[i] != 1:
            gaps.append((sorted_idx[i], sorted_idx[i + 1]))
    gap_pointer = 0
    # Interpolate each missing index
    # Note: this simple loop assumes no wrap-around interpolation
    for miss in miss_idx:
        # Determine which gap this miss falls into
        # Ensure gap_pointer correct such that miss is between sorted_idx[i] and sorted_idx[i+1]
        while gap_pointer < len(gaps) and not (
            gaps[gap_pointer][0] < miss < gaps[gap_pointer][1]
        ):
            gap_pointer += 1
        if gap_pointer >= len(gaps):
            # No gap found (e.g., at ends): skip or extrapolate between last and first? Here skip
            continue
        a0 = gaps[gap_pointer][0]
        a1 = gaps[gap_pointer][1]
        width = a1 - a0
        # get tensors at boundaries
        before = tmp[:, a0, :]
        after = tmp[:, a1, :]
        # linear interpolation fraction for index miss
        frac = (miss - a0) / width
        tmp[:, miss, :] = before + (after - before) * frac
    # Handle potential tail after last acquired until end: here we could extrapolate or leave zero; original code attempted extrapolation toward first angle
    # For simplicity: if last acquired < num_angles-1, we linearly interpolate between last acquired and first acquired
    last_idx = sorted_idx[-1]
    first_idx = sorted_idx[0]
    if last_idx < num_angles - 1:
        for j in range(last_idx + 1, num_angles):
            width = num_angles - last_idx
            before = tmp[:, last_idx, :]
            after = tmp[:, first_idx, :]
            frac = (j - last_idx) / width
            tmp[:, j, :] = before + (after - before) * frac
    prj_ngcbct_li = tmp.permute(1, 0, 2).clone()
    return prj_ngcbct_li


def pad_and_reshape(prj: torch.Tensor, v_dim: int):
    """Reflection pad 1d from 510 to 512, then select first and last v_dim angles to form [2*v_dim, v_dim, 512] tensor"""
    # prj: [angles, H, W] where W maybe 510
    # ReflectionPad1d applies to last dimension
    # First ensure prj shape [angles, H, W]
    # Pad width dimension from 510 to 512: pad 1 on each side if needed or appropriate
    # Use ReflectionPad1d: expects shape [*, L], so reshape to [angles*H, W]
    _, H, W = prj.shape
    # If W != 512, pad to 512
    if W != 512:
        pad = nn.ReflectionPad1d(1)
        # reshape to [angles*H, W]
        flat = prj.reshape(-1, W)
        flat_padded = pad(flat.unsqueeze(1)).squeeze(1)  # may need adjust dims
        # Actually ReflectionPad1d pads last dim: need shape [batch, channels, L]
        flat = prj.reshape(-1, W).unsqueeze(1)  # [angles*H,1,W]
        flat_padded = pad(flat)  # [angles*H,1,W+2]
        # Now reshape back
        newW = flat_padded.shape[-1]
        prj = flat_padded.squeeze(1).reshape(prj.shape[0], H, newW)
    # Now assemble first and last v_dim slices along angle axis
    num_angles = prj.shape[0]
    # If num_angles >= 2*v_dim: take [0:v_dim] and [num_angles-v_dim:num_angles]
    if num_angles >= 2 * v_dim:
        top = prj[:v_dim]
        bottom = prj[num_angles - v_dim :]
        combined = torch.cat([top, bottom], dim=0)  # shape [2*v_dim, H, W]
    else:
        raise ValueError(f"Not enough angles ({num_angles}) for v_dim {v_dim}")
    # Add channel dim: [2*v_dim, 1, H, W]
    combined = combined.unsqueeze(1)
    return combined
