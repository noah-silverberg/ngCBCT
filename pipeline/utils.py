# Common utility functions: plotting slices, creating directories, logging
import os
import matplotlib.pyplot as plt
import torch
import tigre
import numpy as np

def read_scans_agg_file(path, list_=False):
    # Read the aggregation scans file and split into TRAIN/VALIDATION/TEST
    with open(path, "r") as f:
        lines = [l.strip() for l in f.readlines()]

    # The first line is the scan type (HF or FF)
    scan_type = lines[0]

    if list_:
        all_scans = []
        for line in lines[1:]:
            if line:
                patient, scan = line.split()
                all_scans.append((patient, scan, scan_type))
                
        return all_scans, scan_type
    else:
        # Assemble the blocks of scans
        blocks = []
        current = []
        for line in lines[1:]:
            if not line:
                if current:
                    blocks.append(current)
                    current = []
            else:
                current.append(line)
        if current:
            blocks.append(current)

        # Now we have blocks of scans, each block corresponds to a sample (TRAIN, VALIDATION, TEST)
        samples = ["TRAIN", "VALIDATION", "TEST"]
        AGG_SCANS = {sample: [] for sample in samples}
        for sample, block in zip(samples, blocks):
            for entry in block:
                patient, scan = entry.split()
                AGG_SCANS[sample].append((patient, scan, scan_type))

        return AGG_SCANS, scan_type


def ensure_dir(path):
    """Create directory if not exists."""
    os.makedirs(path, exist_ok=True)


def plot_loss(train_loss, val_loss, model_name, save_dir=None):
    """Plot and optionally save training and validation loss curves."""
    plt.figure()
    plt.plot(torch.tensor(train_loss).cpu().numpy(), "r", label="train_loss")
    plt.plot(torch.tensor(val_loss).cpu().numpy(), "g", label="val_loss")
    plt.title(model_name)
    plt.legend()
    if save_dir:
        ensure_dir(save_dir)
        plt.savefig(os.path.join(save_dir, f"{model_name}_loss.png"))
    plt.close()


def display_slices_grid(
    idata, axis, sstep=5, fcols=2, cmap=plt.cm.gray, clip_low=None, clip_high=None
):
    """Display a grid sampling slices along specified axis (0,1,2)."""
    # idata: numpy or torch array with shape [D0, D1, D2]
    # axis: which axis to sample (0: rows, 1: cols, 2: indices)
    arr = idata
    # Determine shape
    shape = arr.shape
    # Determine range for sampling
    if axis == 0:
        max_idx = shape[0]
    elif axis == 1:
        max_idx = shape[1]
    elif axis == 2:
        max_idx = shape[2]
    else:
        raise ValueError("axis must be 0,1,2")
    indices = list(range(0, max_idx, sstep))
    frows = len(indices) // fcols + 1
    plt.figure(figsize=(10, frows * 3))
    for i, idx in enumerate(indices):
        sub = plt.subplot(frows, fcols, i + 1)
        sub.set_title(f"Axis {axis} slice {idx}")
        if axis == 0:
            img = arr[idx, :, :]
        elif axis == 1:
            img = arr[:, idx, :]
        else:
            img = arr[:, :, idx]
        sub.imshow(
            img.T if axis != 2 else img, cmap=cmap, vmin=clip_low, vmax=clip_high
        )
        sub.axis("off")
    plt.tight_layout()


def plot_single_slices(
    idata,
    row=None,
    col=None,
    index=None,
    cmap=plt.cm.gray,
    clip_low=None,
    clip_high=None,
    save_prefix=None,
    save_dir=None,
):
    """Plot single slices at given row, col, index. idata shape [D0,D1,D2]."""
    # row: slice along axis 0; col: axis1; index: axis2
    if save_dir:
        ensure_dir(save_dir)
    if row is not None:
        plt.figure()
        img = idata[row, :, :].T
        plt.imshow(img, cmap=cmap, vmin=clip_low, vmax=clip_high)
        plt.axis("off")
        if save_dir and save_prefix:
            plt.savefig(
                os.path.join(save_dir, f"{save_prefix}_row{row}.png"),
                bbox_inches="tight",
                pad_inches=0,
            )
        plt.close()
    if col is not None:
        plt.figure()
        img = idata[:, col, :].T
        plt.imshow(img, cmap=cmap, vmin=clip_low, vmax=clip_high)
        plt.axis("off")
        if save_dir and save_prefix:
            plt.savefig(
                os.path.join(save_dir, f"{save_prefix}_col{col}.png"),
                bbox_inches="tight",
                pad_inches=0,
            )
        plt.close()
    if index is not None:
        plt.figure()
        img = idata[:, :, index]
        plt.imshow(img, cmap=cmap, vmin=clip_low, vmax=clip_high)
        plt.axis("off")
        if save_dir and save_prefix:
            plt.savefig(
                os.path.join(save_dir, f"{save_prefix}_index{index}.png"),
                bbox_inches="tight",
                pad_inches=0,
            )
        plt.close()

def get_geometry():
    geo = tigre.geometry()
    # VARIABLE                                   DESCRIPTION                    UNITS
    # -------------------------------------------------------------------------------------
    # Distances
    geo.DSD = 1500  # Distance Source Detector      (mm)
    geo.DSO = 1000  # Distance Source Origin        (mm)
    # Detector parameters
    PixelSize = 0.388  # in mm
    rebin = 2  # we did 2x2 rebinning to make 0.776x0.776 detector bins
    # number of pixels              (px)
    # geo.nDetector = np.array(prj.shape[1], prj.shape[0])
    geo.nDetector = np.array([382, 510])
    # size of each pixel            (mm)
    geo.dDetector = PixelSize * rebin * np.array([1, 1])
    # total size of the detector    (mm)
    geo.sDetector = geo.nDetector * geo.dDetector
    # Image parameters
    geo.nVoxel = np.array([200, 512, 512])  # number of voxels              (vx)
    geo.dVoxel = np.array([1.0, 1.0, 1.0])  # size of each voxel            (mm)
    geo.sVoxel = geo.nVoxel * geo.dVoxel  # total size of the image       (mm)

    # Offsets
    geo.offOrigin = np.array([0, 0, 0])  # Offset of image from origin   (mm)
    geo.offDetector = np.array([0, 160])  # Offset of Detector            (mm)
    # These two can be also defined
    # per angle

    # Auxiliary
    geo.accuracy = 0.5  # Variable to define accuracy of
    # 'interpolated' projection
    # It defines the amoutn of
    # samples per voxel.
    # Recommended <=0.5             (vx/sample)

    # Optional Parameters
    # There is no need to define these unless you actually need them in your
    # reconstruction


    geo.COR = 0  # y direction displacement for
    # centre of rotation
    # correction                   (mm)
    # This can also be defined per
    # angle

    geo.rotDetector = np.array([0, 0, 0])  # Rotation of the detector, by
    # X,Y and Z axis respectively. (rad)
    # This can also be defined per
    # angle

    geo.mode = "cone"  # Or 'parallel'. Geometry type.

    return geo