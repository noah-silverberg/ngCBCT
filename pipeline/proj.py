import numpy as np
import torch
import mat73
import torch.nn as nn
from pipeline.paths import Files
import scipy.io as sio


def load_projection_mat(
    mat_path, exclude_prj=False
):
    """
    Load projection data from a .mat file.

    Args:
        mat_path (str): Path to the .mat file.
        exclude_prj (bool): If True, only load odd_index and angles, not prj.

    Returns:
        odd_index (np.ndarray): shape (K,), 1-based indices of nonstop-gated angles.
        angles (torch.Tensor): shape (A,), gated projection angles.
        prj (torch.Tensor): shape (W, H, A), gated projection data.
    """
    try:
        # Load projection mat file for a given scan
        if exclude_prj:
            mat = mat73.loadmat(mat_path, only_include=["odd_index", "angles"])
        else:
            mat = mat73.loadmat(mat_path)
    except TypeError:
        if exclude_prj:
            raise ValueError(f"exclude_prj=True is not supported for scipy.io.loadmat")
        mat = sio.loadmat(mat_path)

    odd_index = np.array(mat["odd_index"]).flatten()  # angle indices to keep for nonstop gated
    angles = torch.from_numpy(np.array(mat["angles"]).flatten()).float()  # angles acquired
    if exclude_prj:
        return odd_index, angles

    prj = torch.from_numpy(np.array(mat["prj"])).float()  # sinogram projections
    return odd_index, angles, prj


def reformat_sinogram(prj: torch.Tensor, angles: torch.Tensor):
    """
    Reformat sinogram tensor and adjust angles.

    Args:
        prj (torch.Tensor): shape (W, H, A), gated sinogram.
        angles (torch.Tensor): shape (A,), gated acquisition angles.

    Returns:
        prj (torch.Tensor): shape (A, H, W), flipped and permuted sinogram.
        angles1 (torch.Tensor): shape (A,), reformatted angles in radians.
        flipped (bool): whether angles were flipped.
    """
    # Flips and permutations to match the expected format
    prj = prj.detach().clone()
    prj = torch.flip(prj, (1,))
    prj = prj.permute(2, 1, 0)
    prj = torch.flip(prj, (2,))

    flipped = False

    # Flips the angles if they are in the opposite order
    angles1 = -(angles + np.pi / 2)
    if (angles1[-1:] - angles1[0]) < 0:
        angles1 = torch.flip(angles1, (0,))
        prj = torch.flip(prj, (0,))
        flipped = True
    return prj, angles1, flipped


def find_missing_indices(odd_index: np.ndarray):
    """
    Find missing angle indices in a sequence.

    Args:
        odd_index (np.ndarray): shape (K,), sorted 1-based acquired indices.

    Returns:
        List[int]: sorted missing integer indices between first and last.
    """
    first, last = odd_index[0], odd_index[-1]
    full_range = set(range(first, last + 1))
    present = set(odd_index)
    missing = sorted(full_range - present)
    return missing

def undersample_projections(prj_gcbct: torch.Tensor, odd_index: np.ndarray):
    """
    Undersample projections to simulate nonstop-gated scan.

    Args:
        prj_gcbct (torch.Tensor): shape (A, H, W), gated sinogram.
        odd_index (np.ndarray): shape (K,), 1-based indices of acquired angles.

    Returns:
        prj_ngcbct (torch.Tensor): shape (K, H, W), gated sinogram with only acquired angles.
        ngcbct_idx (np.ndarray): shape (K,), 0-based indices of acquired angles.
    """
    # Convert odd_index to zero-based
    ngcbct_idx = odd_index.astype(np.int64) - 1
    prj_ngcbct = torch.zeros_like(prj_gcbct)
    prj_ngcbct[ngcbct_idx] = prj_gcbct[ngcbct_idx]
    return prj_ngcbct, ngcbct_idx

def get_even_index(odd_index: np.ndarray, num_angles: int = None):
    """
    Get even indices from odd indices.

    Args:
        odd_index (np.ndarray): shape (K,), 1-based indices of acquired angles.

    Returns:
        np.ndarray: shape (K,), 1-based indices of even angles.
    """
    # Set even_index to all the angles that are not in odd_index
    even_index = np.setdiff1d(np.arange(1, num_angles + 1), odd_index, assume_unique=True)
    return even_index

def interpolate_projections(prj_gcbct: torch.Tensor, odd_index: np.ndarray, odd: bool):
    """
    Simulate nonstop-gated scan.
    Zero out missing angles in sinogram and linearly interpolate them.

    Args:
        prj_gcbct (torch.Tensor): shape (A, H, W), gated sinogram.
        odd_index (np.ndarray): shape (K,), 1-based indices of acquired angles.
        odd (bool): If True, use odd indices for interpolate. If False, use even indices.

    Returns:
        prj_ngcbct_li (torch.Tensor): shape (A, H, W), nonstop-gated sinogram with missing angles interpolated.
    """
    # NOTE: This function is a bit hard to read...but it works
    #       it also is not optimized for speed
    #       but this is not a bottleneck in the pipeline so we don't worry about it

    num_angles = prj_gcbct.shape[0]

    if not odd:
        even_index = get_even_index(odd_index, num_angles)

        # Now, we know that the first chunk is NOT acquired (since it IS acquired if we use odd incdices)
        # so we can just rotate the angles until we have the first angle acquired
        # and then interpolate
        # and then rotate back at the end

        # First we need to figure out the first angle that is acquired
        first_acquired = even_index[0]

        # Now we need to rotate the angles so that the first acquired angle is at index 0
        prj_gcbct = torch.roll(prj_gcbct, -first_acquired + 1, dims=0)

        # Now we need to update the odd_index to reflect the rotation
        odd_index = even_index - first_acquired + 1
        odd_index[odd_index <= 0] += num_angles

    # Initialize a new tensor for nonstop-gated, and fill it with the acquired angles
    prj_ngcbct, ngcbct_idx = undersample_projections(prj_gcbct, odd_index)
    tmp = prj_ngcbct.detach().clone().permute(1, 0, 2)  # [H, angles, W]

    # Get the indices of the unacquired angles in nonstop-gated
    miss_idx = find_missing_indices(ngcbct_idx)

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
    prj_ngcbct_li = tmp.permute(1, 0, 2)

    if not odd:
        # If we rotated the angles at the start, we need to rotate back
        prj_ngcbct_li = torch.roll(prj_ngcbct_li, first_acquired - 1, dims=0)

    return prj_ngcbct_li.clone()


def pad_and_reshape(prj: torch.Tensor):
    """
    Pad and reshape sinogram for CNN input.

    Args:
        prj (torch.Tensor): shape (A, H, 510), where A is number of angles.

    Returns:
        torch.Tensor: shape (H, A, 512), after reflection padding and permute.
    """
    # prj: [angles, H, 510]
    # Pads the final dimension from 510 to 512
    # Since we need this shape for the CNN (due to maxpooling)
    pad = nn.ReflectionPad1d(1)
    prj = pad(prj)

    # Reshape to [H, angles, 512]
    prj = prj.permute(1, 0, 2)
    return prj


def divide_sinogram(prj: torch.Tensor, v_dim: int, patches: int = 2):
    """
    Select top/bottom angle "halves" and combine.
    So each sinogram is divided into two halves, which are then concatenated.

    Args:
        prj (torch.Tensor): shape (H, A, 512).
        v_dim (int): number of angles to select from start and end.
        patches (int): number of patches to create (default: 2).

    Returns:
        torch.Tensor: shape (2*H, 1, v_dim, 512), concatenated "half"-sinograms.
    """
    # prj: [H, angles, 512]
    if patches == 2:
        # Now assemble first and last v_dim slices along H axis
        top = prj[:, :v_dim, :]
        bottom = prj[:, -v_dim:, :]
        combined = torch.cat([top, bottom], dim=0)
    elif patches == 3:
        # If we want to divide into 3 patches, we take the first, middle, and last v_dim slices
        top = prj[:, :v_dim, :]
        middle_idx = prj.shape[1] // 2 - v_dim // 2
        middle = prj[:, middle_idx:middle_idx + v_dim, :]
        bottom = prj[:, -v_dim:, :]
        combined = torch.cat([top, middle, bottom], dim=0)
    else:
        raise ValueError("Unsupported number of patches. Only 2 or 3 are supported.")
    combined = combined.unsqueeze(1)
    # combined: [2*H, 1, v_dim, 512] or [3*H, 1, v_dim, 512]
    return combined
