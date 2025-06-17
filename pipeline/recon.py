# Implements Notebook 0 functionality: loading recon volumes and plotting
import os
import numpy as np
import scipy.io
import matplotlib.pyplot as plt
import torch
from .config import DATA_DIR, SAVE_DIR, FLAGS, CLOUD_ROOT
from .utils import display_slices_grid, plot_single_slices, ensure_dir


def load_recon(
    patient_id,
    scan_id,
    clip_low=0.012,
    clip_high=0.028,
    full=True,
    nsFDK=False,
    nsPL=False,
    PD=False,
):
    """Load different reconstructions for given patient/scan. Return dict of numpy arrays."""
    save_name = f"panc{patient_id:02}.HF{scan_id:02}"
    results = {}
    if full:
        path = os.path.join(DATA_DIR, "panc_recon", f"recon_{save_name}.u_FDK_full.mat")
        mat = scipy.io.loadmat(path)
        arr = mat["u_FDK_full"]
        arr = np.clip(arr, 0, 0.04)
        results["full_FDK"] = arr
    if nsFDK:
        path = os.path.join(DATA_DIR, "panc_recon", f"recon_{save_name}.u_FDK.mat")
        mat = scipy.io.loadmat(path)
        arr = mat["u_FDK"]
        arr = np.clip(arr, 0, 0.04)
        results["ns_FDK"] = arr
    if nsPL:
        pl_mode = FLAGS.get("pl_mode", "b2.5")
        path = os.path.join(
            DATA_DIR, "panc_recon", f"recon_{save_name}.u_PL.{pl_mode}.mat"
        )
        mat = scipy.io.loadmat(path)
        arr = mat["u_PL"]
        arr = np.clip(arr, 0, 0.04)
        results["ns_PL"] = arr
    if PD:
        path = os.path.join(DATA_DIR, "DS12", f"reconFDK_{save_name}.HF_ns.mat")
        mat = scipy.io.loadmat(path)
        arr = mat["reconFDK"]
        arr = np.clip(arr, 0, 0.04)
        results["ns_PD"] = arr
    return results


def display_recons(
    recons: dict,
    patient_id,
    scan_id,
    tumor_location_tensor_path,
    clip_low=0.012,
    clip_high=0.028,
    save=False,
    save_dir=None,
):
    """Display grids and single slices for loaded reconstructions."""
    # Load tumor location
    tumor_location = torch.load(
        os.path.join(CLOUD_ROOT, "information", tumor_location_tensor_path)
    )
    row, col, index = tumor_location[patient_id][scan_id]
    for key, arr in recons.items():
        # grid displays
        display_slices_grid(arr, axis=0, clip_low=clip_low, clip_high=clip_high)
        if save and save_dir:
            ensure_dir(save_dir)
            plt.savefig(
                os.path.join(save_dir, f"{patient_id}_{scan_id}_{key}_grid0.png")
            )
        display_slices_grid(arr, axis=1, clip_low=clip_low, clip_high=clip_high)
        if save and save_dir:
            plt.savefig(
                os.path.join(save_dir, f"{patient_id}_{scan_id}_{key}_grid1.png")
            )
        display_slices_grid(arr, axis=2, clip_low=clip_low, clip_high=clip_high)
        if save and save_dir:
            plt.savefig(
                os.path.join(save_dir, f"{patient_id}_{scan_id}_{key}_grid2.png")
            )
        # single slices at tumor
        plot_single_slices(
            arr,
            row=row,
            col=col,
            index=index,
            clip_low=clip_low,
            clip_high=clip_high,
            save_prefix=f"p{patient_id}_{key}",
            save_dir=save_dir,
        )
