#!/usr/bin/env python3
import os
import glob
from scipy.io import loadmat
import torch
import numpy as np
import matplotlib.pyplot as plt

# -----------------------------------------------------------------------------
# CONFIG
# -----------------------------------------------------------------------------
BASE_DIR = "Data/3D_recon"
OUTPUT_DIR = "."
RUN = "1"

# the three scans to plot
SCANS = [
    ("HF", "20", "01"),
    ("HF", "14", "01"),
    ("FF", "22", "01"),
]

# the crops for each scan (y0, y1, x0, x1)
CROPS = {
    ("HF", "20", "01"): {
        "index": (70, 512 - 160, 75, 512 - 35),
        "width": (0, 160 - 0, 60, 512 - 170),
        "height": (0, 160 - 0, 70, 512 - 45),
    },
    ("HF", "14", "01"): {
        "index": (95, 512 - 190, 110, 512 - 105),
        "width": (0, 160 - 0, 80, 512 - 190),
        "height": (0, 160 - 0, 85, 512 - 95),
    },
    ("FF", "22", "01"): {
        "index": (5, 256 - 40, 0, 256 - 0),
        "width": (0, 160 - 0, 15, 256 - 53),
        "height": (0, 160 - 0, 8, 256 - 9),
    },
}

VIEWS = ["index", "width", "height"]
METHODS = [
    ("u_FDK", "FDK"),
    ("u_PL", "IR"),
    ("FBPCONVNet", "FBPCONVNet"),
    ("IResNet", "IResNet"),
    ("DDCNN", "DDCNN"),
]


def load_gt_and_recons(scan_type, pid, sid):
    """Load GT, FDK, IR, DDCNN, FBPCONVNet, IResNet volumes for a given scan."""
    mat_dir = os.path.join(BASE_DIR, scan_type)
    # GT filename
    if scan_type == "FF":
        gt_pattern = f"recon_p{pid}.{scan_type}{sid}.u_FDK_ROI_fullView.mat"
    else:
        gt_pattern = f"recon_p{pid}.{scan_type}{sid}.u_FDK_full.mat"
    gt_path = glob.glob(os.path.join(mat_dir, gt_pattern))[0]
    gt_mat = loadmat(gt_path)
    # pick GT key
    gt_key = "u_FDK_ROI_fullView" if scan_type == "FF" else "u_FDK_full"
    gt_vol = gt_mat[gt_key][..., 20:-20]
    if scan_type == "FF":
        # crop 256×256
        gt_vol = gt_vol[128:-128, 128:-128]
    # FDK
    fdk_key = "u_FDK_ROI" if scan_type == "FF" else "u_FDK"
    fdk_path = (
        gt_path.replace("FDK_ROI_fullView", "FDK_ROI")
        if scan_type == "FF"
        else gt_path.replace("FDK_full", "FDK")
    )
    fdk = loadmat(fdk_path)[fdk_key][..., 20:-20]
    if scan_type == "FF":
        fdk = fdk[128:-128, 128:-128]
    # IR (PL)
    pl_key = "u_PL_ROI" if scan_type == "FF" else "u_PL"
    pl_pat = "PL_ROI.b1" if scan_type == "FF" else "PL.b1.iter200"
    pl_path = (
        gt_path.replace("FDK_ROI_fullView", pl_pat)
        if scan_type == "FF"
        else gt_path.replace("FDK_full", pl_pat)
    )
    pl = loadmat(pl_path)[pl_key][..., 20:-20]
    if scan_type == "FF":
        pl = pl[128:-128, 128:-128]

    # clip and normalize GT, FDK, PL
    for v in [gt_vol, fdk, pl]:
        np.clip(v, 0, 0.04, out=v)
        v -= v.min()
        v /= v.max()

    # DDCNN
    ds = 14 if scan_type == "FF" else 13
    dd_path = os.path.join(
        mat_dir, f"p{pid}.{scan_type}{sid}_IResNet_MK6_DS{ds}.2_run{RUN}_3D.pt"
    )
    ddcnn = torch.load(dd_path, weights_only=False)
    # FBPCONVNet
    fbp_glob = glob.glob(
        os.path.join(mat_dir, f"p{pid}.{scan_type}{sid}_FBPCONVNet*_3D.pt")
    )
    fbpcnn = torch.load(fbp_glob[0], weights_only=False)
    # IResNet
    ire_glob = glob.glob(
        os.path.join(mat_dir, f"p{pid}.{scan_type}{sid}_IResNet*PL_3D.pt")
    )
    ire = torch.load(ire_glob[0], weights_only=False)
    # tumor loc
    tlocs = torch.load(
        os.path.join(
            BASE_DIR,
            "tumor_location_FF.pt" if scan_type == "FF" else "tumor_location.pt",
        ),
        weights_only=False,
    )
    t = tlocs[int(pid), int(sid)]
    if scan_type == "FF":
        t[:2] -= 128
    t[2] -= 20
    return gt_vol, fdk, pl, ddcnn, fbpcnn, ire, t


def extract_view(vols, tloc, view):
    """
    vols: list of 3D-arrays [gt, fdk, pl, dd, fbp, ire]
    tloc: (z,y,x) in index view
    view: one of "index","height","width"
    returns list of 2D-slices in given view order
    """
    gt, fdk, pl, dd, fbp, ire = vols
    if view == "index":
        sl = int(tloc[2])
        return [v[..., sl] for v in vols]
    elif view == "height":
        # swap axis 0<->2, then same as index
        vols2 = [np.swapaxes(v, 0, 2) for v in vols]
        sl = int(tloc[0])
        return [v[..., sl] for v in vols2]
    else:  # width
        # transpose to (z,y,x)
        vols2 = [np.transpose(v, (2, 0, 1)) for v in vols]
        sl = int(tloc[1])
        return [v[..., sl] for v in vols2]


def plot_scan(scan_type, pid, sid):
    gt, fdk, pl, ddcnn, fbpcnn, ire, tloc = load_gt_and_recons(scan_type, pid, sid)
    vols = [gt, fdk, pl, ddcnn, fbpcnn, ire]
    # rename METHODS to match vols order:
    names = ["FDK (Gated)", "FDK", "IR", "FBPCONVNet", "IResNet", "DDCNN"]
    ncols = len(names)
    nrows = len(VIEWS)
    fig, axes = plt.subplots(
        nrows, ncols, figsize=(3 * ncols, 3 * nrows), squeeze=False
    )
    for i, view in enumerate(VIEWS):
        slices = extract_view(vols, tloc, view)
        for j, sl in enumerate(slices):
            ax = axes[i, j]

            # apply crop
            crop = CROPS.get((scan_type, pid, sid), {}).get(view)
            if crop is not None:
                y0, y1, x0, x1 = crop
                sl = sl[y0:y1, x0:x1]

            ax.imshow(sl, cmap="gray", vmin=0, vmax=1)
            if i == 0:
                ax.set_title(names[j], fontsize=12)
            ax.axis("off")
    plt.tight_layout()
    outname = f"{scan_type}_p{pid}_{sid}.png"
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    fig.savefig(os.path.join(OUTPUT_DIR, outname), dpi=400)
    plt.close(fig)
    print("Saved", outname)


def main():
    for scan in SCANS:
        plot_scan(*scan)


if __name__ == "__main__":
    main()
