# Implements Notebook 4: apply trained model to projections to predict missing, visualize, save
import os
import numpy as np
import torch
import scipy.io
import mat73
from .config import (
    MODEL_DIR,
    SAVE_DIR,
    DATA_DIR,
    CLIP_LOW,
    CLIP_HIGH,
    CUDA_DEVICE,
    WORK_ROOT,
)
from .utils import plot_loss, ensure_dir, display_slices_grid, plot_single_slices
from network_instance import IResNet


def load_model(model_name: str, device=None):
    model = IResNet()
    state = torch.load(os.path.join(MODEL_DIR, f"{model_name}.pth"))
    model.load_state_dict(state)
    if device:
        model = model.to(device)
    model.eval()
    return model


def apply_model_to_projections(
    patient_id: int,
    scan_id: int,
    mode: str,
    data_ver: str,
    model_name: str,
    v_dim: int,
    odd_index: np.ndarray,
    angles: np.ndarray,
    prj_gcbct: torch.Tensor,
    prj_ngcbct_li: torch.Tensor,
    save=True,
):
    """Apply CNN model slice-wise to non-gated projections to predict missing projections and combine."""
    # prj_gcbct, prj_ngcbct_li: torch tensors [angles, 1, H, W] or [angles, H, W]
    device = torch.device(CUDA_DEVICE)
    model = load_model(model_name, device=device)
    # Prepare angles1
    angles1 = -(torch.from_numpy(np.array(angles)) + np.pi / 2)
    if (angles1[-1:] - angles1[0]) < 0:
        angles1 = torch.flip(angles1, (0,))
    angles1 = angles1.detach().cpu().numpy()
    odd_index0 = odd_index.astype(np.int64)
    # Initialize prj_ngcbct_cnn
    num_angles = len(angles1)
    # Assuming prj_ngcbct_li shape [angles, 1, H, W]
    # Remove channel dim for processing
    prj_ngcbct_li_noc = (
        prj_ngcbct_li.squeeze(1) if prj_ngcbct_li.ndim == 4 else prj_ngcbct_li
    )
    prj_ngcbct_cnn = torch.zeros_like(prj_ngcbct_li_noc)
    overlap = v_dim * 2 - num_angles
    # Loop over projection index (e.g., 382), similar to original code
    # Assuming the first dimension corresponds to half angles count; adapt as in notebook
    half = prj_ngcbct_li_noc.shape[0] // 2
    for i in range(half):
        # First block
        img1 = prj_ngcbct_li_noc[i].unsqueeze(0).unsqueeze(0).to(device)
        out1 = model(img1).detach().cpu()
        # Second block
        img2 = prj_ngcbct_li_noc[i + half].unsqueeze(0).unsqueeze(0).to(device)
        out2 = model(img2).detach().cpu()
        # Combine
        if overlap >= 0:
            prj_ngcbct_cnn[0 : (v_dim - overlap), i, :] = out1[
                0, 0, 0 : (v_dim - overlap), 1:511
            ]
            prj_ngcbct_cnn[(v_dim - overlap) : v_dim, i, :] = (
                out1[0, 0, (v_dim - overlap) : v_dim, 1:511]
                + out2[0, 0, 0:overlap, 1:511]
            ) / 2
            prj_ngcbct_cnn[v_dim:, i, :] = out2[0, 0, overlap:, 1:511]
        else:
            prj_ngcbct_cnn[0:v_dim, i, :] = out1[0, 0, :, 1:511]
            diff = (out2[0, 0, 0, 1:511] - out1[0, 0, -1, 1:511]) / (-overlap)
            for j in range(-overlap):
                prj_ngcbct_cnn[v_dim + j, i, :] = out1[0, 0, -1, 1:511] + (j + 1) * diff
            prj_ngcbct_cnn[(v_dim - overlap) :, i, :] = out2[0, 0, :, 1:511]
    # Assemble mixed sinogram
    prj_gcbct_noc = prj_gcbct.squeeze(1) if prj_gcbct.ndim == 4 else prj_gcbct
    prj_ngcbct_mix = torch.zeros_like(prj_gcbct_noc)
    ngcbct_idx0 = odd_index0 - 1
    # Fill real and predicted
    for idx in range(len(angles1)):
        if idx in ngcbct_idx0:
            prj_ngcbct_mix[idx] = prj_gcbct_noc[idx]
        else:
            prj_ngcbct_mix[idx] = prj_ngcbct_cnn[idx]
    # Visualization
    idx0, idy, idz = 7, 100, 100
    # Use utils display functions as needed
    # Save mixed sinogram to .mat
    if save:
        out_mat = {
            "angles": angles1,
            "odd_index": odd_index0,
            "prj": prj_ngcbct_mix.numpy(),
        }
        out_dir = os.path.join(WORK_ROOT, "output", "prj_mat", mode)
        ensure_dir(out_dir)
        scipy.io.savemat(
            os.path.join(out_dir, f"panc{patient_id}.{mode}_ns.mat"), out_mat
        )
    return prj_ngcbct_mix
