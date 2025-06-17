# Implements Notebook 1 functionality: load projection data, simulate non-gated and interpolate
import numpy as np
import torch
import scipy.io
import mat73
import matplotlib.pyplot as plt
import torch.nn as nn
from .config import WORK_ROOT


def load_projection_mat(patient: str, scan: str, scan_type: str):
    """Load projection data from mat file.."""
    # Load projection mat file for a given scan
    mat_path = f"{WORK_ROOT}/data/panc_prj/{scan_type}/mat/panc{patient}.{scan_type}{scan}.mat"  # TODO change path as needed
    mat = mat73.loadmat(mat_path)

    odd_index = np.array(mat["odd_index"])  # angle indices to keep for nonstop gated
    angles = torch.from_numpy(np.array(mat["angles"])).float()  # angles acquired
    prj = torch.from_numpy(np.array(mat["prj"])).float()  # sinogram projections
    return odd_index, angles, prj


def reformat_sinogram(prj: torch.Tensor, angles: torch.Tensor):
    """Reformat sinogram and angles."""
    # Flips and permutations to match the expected format
    prj_gcbct = prj.detach().clone()
    prj_gcbct = torch.flip(prj_gcbct, (1,))
    prj_gcbct = prj_gcbct.permute(2, 1, 0)
    prj_gcbct = torch.flip(prj_gcbct, (2,))

    # Flips the angles if they are in the opposite order
    angles1 = -(angles + np.pi / 2)
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
    # NOTE: This function is a bit hard to read...but it works
    #       it also is not optimized for speed
    #       but this is not a bottleneck in the pipeline so we don't worry about it

    # Convert odd_index to zero-based
    ngcbct_idx = odd_index.astype(np.int64) - 1
    num_angles = prj_gcbct.shape[0]

    # Initialize a new tensor for nonstop-gated, and fill it with the acquired angles
    prj_ngcbct = torch.zeros_like(prj_gcbct)
    prj_ngcbct[ngcbct_idx] = prj_gcbct[ngcbct_idx]

    tmp = prj_ngcbct.detach().clone().permute(1, 0, 2)  # [H, angles, W]

    # Get the indices of the unacquired angles in nonstop-gated
    miss_idx = find_missing_indices(odd_index)

    # Now we find the start and end indices of gaps in the acquired angles
    # we do this by sorting the acquired indices and then seeing where
    # the difference between consecutive values is not 1
    sorted_idx = sorted(ngcbct_idx)
    gaps = []
    for i in range(len(sorted_idx) - 1):
        if sorted_idx[i + 1] - sorted_idx[i] != 1:
            # Missing angles are sorted_idx[i]+1 to sorted_idx[i+1]-1
            gaps.append((sorted_idx[i], sorted_idx[i + 1]))

    # Now we have a list of gaps where interpolation is needed
    # so we can go through the missing indices and fill them in
    # we do this by linearly interpolating between the start and end of each gap
    # NOTE: We do not need to worry about the case where the first angle is missing,
    #       since the first angle is always acquired
    gap_pointer = 0  # running pointer of which gap we are currently filling
    for miss in miss_idx:

        # If we go past the current gap, we need to move the pointer forward
        while gap_pointer < len(gaps) and not (
            gaps[gap_pointer][0] < miss < gaps[gap_pointer][1]
        ):
            gap_pointer += 1

        # If we are past the last gap, we skip
        # NOTE: This includes the case where the gap extends to the end of the angles
        #       This case is handled outside the 'for' loop, since it requires special handling
        if gap_pointer >= len(gaps):
            continue

        # Get the width of the gap
        a0 = gaps[gap_pointer][0]
        a1 = gaps[gap_pointer][1]
        width = a1 - a0

        # Get the sinogram values on the boundaries of the gap
        before = tmp[:, a0, :]
        after = tmp[:, a1, :]

        # Linearly interpolate between the two boundaries
        frac = (miss - a0) / width
        tmp[:, miss, :] = before + (after - before) * frac

    # Handle the case where the final gap extends to the end of the angles
    # In this case we "wrap around" to the first angle and linearly interpolate
    # (the same way as before)
    last_idx = sorted_idx[-1]
    first_idx = sorted_idx[0]
    num_angles = prj_gcbct.shape[0]
    if last_idx < num_angles - 1:
        for j in range(last_idx + 1, num_angles):
            width = num_angles - last_idx
            before = tmp[:, last_idx, :]
            after = tmp[:, first_idx, :]
            frac = (j - last_idx) / width
            tmp[:, j, :] = before + (after - before) * frac

    # Now we have fully interpolated the nonstop-gated missing angles
    # We just reshape to [angles, H, W]
    prj_ngcbct_li = tmp.permute(1, 0, 2).clone()

    return prj_ngcbct_li


def pad_and_reshape(prj: torch.Tensor):
    """Pad prj to [angles, 1, H, 512] tensor with reflection padding, and reshape."""
    # prj: [angles, H, 510]
    # Pads the final dimension from 510 to 512
    # Since we need this shape for the CNN (due to maxpooling)
    pad = nn.ReflectionPad1d(1)
    prj = pad(prj)

    # Reshape to [H, angles, 512]
    prj = prj.permute(1, 0, 2)
    return prj


def divide_sinogram(prj: torch.Tensor, v_dim: int):
    """Reflection pad last dimension from 510 to 512, then select first and last v_dim angles"""
    # prj: [H, angles, 512]
    # Now assemble first and last v_dim slices along H axis
    top = prj[:, :v_dim, :]
    bottom = prj[:, -v_dim:, :]
    combined = torch.cat([top, bottom], dim=0)
    combined = combined.unsqueeze(1)
    # combined: [2*H, 1, v_dim, 512]
    return combined
