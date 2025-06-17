# Implements Notebook 7: inference on test set, assemble 3D outputs
import os
import numpy as np
import torch
import matplotlib.pyplot as plt
from .config import (
    DATA_DIR,
    SAVE_DIR,
    CUDA_DEVICE,
    CLIP_LOW,
    CLIP_HIGH,
    RESULT_DIR,
    FIGURE_DIR,
    CLOUD_ROOT,
)
from .utils import ensure_dir
from network_instance import IResNet
from dsets import TestScan


def load_test_recon(
    mode: str, data_ver: str, patient_id: int, scan_id: int, model_name: str
):
    """Load ground truth and ns reconstructions for a test scan."""
    save_name = f"panc{patient_id:02}.{mode}0{scan_id}"
    if mode == "HF":
        truth_path = os.path.join(
            DATA_DIR,
            f"DS{data_ver}/{model_name}/test/full/recon_{save_name}.u_FDK_full.mat",
        )
        ns_path = os.path.join(
            DATA_DIR,
            f"DS{data_ver}/{model_name}/test/ns/reconFDK_{save_name}.HF_ns.mat",
        )
    else:
        truth_path = os.path.join(
            DATA_DIR,
            f"DS{data_ver}/{model_name}/test/full/recon_{save_name}.u_FDK_ROI_fullView.mat",
        )
        ns_path = os.path.join(
            DATA_DIR,
            f"DS{data_ver}/{model_name}/test/ns/reconFDK_{save_name}.FF_ROI_ns.mat",
        )
    truth_sets = TestScan(truth_path)
    test_sets = TestScan(ns_path)
    truth_images = truth_sets[0]
    test_images = test_sets[0]
    return truth_images, test_images, save_name


def inference_3d(
    patient_id: int,
    scan_id: int,
    mode: str,
    data_ver: str,
    model_name: str,
    tumor_location_tensor_path: str,
    batch_size: int = 20,
    save=True,
):
    """Run inference on test_images slice-wise using trained model, assemble 3D volume, visualize tumor slices."""
    device = torch.device(CUDA_DEVICE)
    truth_images, test_images, save_name = load_test_recon(
        mode, data_ver, patient_id, scan_id, model_name
    )
    # Crop if FF
    if mode == "FF":
        truth_images = truth_images[:, :, 128:384, 128:384]
        test_images = test_images[:, :, 128:384, 128:384]
    model = IResNet()
    model.load_state_dict(
        torch.load(os.path.join(SAVE_DIR, "model", f"{model_name}.pth"))
    )
    model = model.to(device)
    model.eval()
    # Batch inference
    num_slices = test_images.shape[0]
    # Save per-batch outputs
    tmp_dir = os.path.join(RESULT_DIR, "3D_Recon", "tmp")
    ensure_dir(tmp_dir)
    for bstart in range(0, num_slices, batch_size):
        bend = min(bstart + batch_size, num_slices)
        batch = test_images[bstart:bend].to(device)
        with torch.no_grad():
            out = model(batch)
        # Save
        out_cpu = out.cpu()
        torch.save(
            out_cpu, os.path.join(tmp_dir, f"{patient_id}_{bstart//batch_size}.pt")
        )
    # Assemble full volume
    vol = None
    for bstart in range(0, num_slices, batch_size):
        part = torch.load(
            os.path.join(tmp_dir, f"{patient_id}_{bstart//batch_size}.pt")
        )
        # part shape [b,1,H,W]
        part = part.cpu().squeeze(1).permute(1, 2, 0)  # [H,W,b]
        if vol is None:
            vol = part
        else:
            vol = torch.cat((vol, part), dim=2)
    # Save full 3D
    out_path = os.path.join(RESULT_DIR, "3D_Recon", f"{save_name}_{model_name}_3D.pt")
    if save:
        ensure_dir(os.path.dirname(out_path))
        torch.save(vol, out_path)
    # Load tumor location
    tumor_location = torch.load(
        os.path.join(CLOUD_ROOT, "information", tumor_location_tensor_path)
    )
    row, col, index = tumor_location[patient_id][scan_id]
    if mode == "HF":
        index = index - 20
    else:
        row, col = row - 128, col - 128
    # Visualize slices
    fig_dir = os.path.join(FIGURE_DIR)
    ensure_dir(fig_dir)
    # Axial
    plt.figure()
    plt.axis("off")
    plt.imshow(vol[row, :, :].T, cmap=plt.cm.gray, vmin=CLIP_LOW, vmax=CLIP_HIGH)
    if save:
        plt.savefig(
            os.path.join(fig_dir, f"{save_name}_row{row}_{model_name}.png"),
            bbox_inches="tight",
            pad_inches=0,
        )
    plt.close()
    # Coronal
    plt.figure()
    plt.axis("off")
    plt.imshow(vol[:, col, :].T, cmap=plt.cm.gray, vmin=CLIP_LOW, vmax=CLIP_HIGH)
    if save:
        plt.savefig(
            os.path.join(fig_dir, f"{save_name}_col{col}_{model_name}.png"),
            bbox_inches="tight",
            pad_inches=0,
        )
    plt.close()
    # Index/sagittal
    plt.figure()
    plt.axis("off")
    plt.imshow(vol[:, :, index], cmap=plt.cm.gray, vmin=CLIP_LOW, vmax=CLIP_HIGH)
    if save:
        plt.savefig(
            os.path.join(fig_dir, f"{save_name}_index{index}_{model_name}.png"),
            bbox_inches="tight",
            pad_inches=0,
        )
    plt.close()
    return vol
