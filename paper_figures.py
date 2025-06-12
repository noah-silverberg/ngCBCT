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
        "index": (5, 256 - 45, 0, 256 - 0),
        "width": (0, 160 - 0, 15, 256 - 53),
        "height": (0, 160 - 0, 8, 256 - 9),
    },
}

SUBPLOTS = {
    ("HF", "20", "01"): {
        "top": 0.85,
        "hspace": 0.05,
        "height_factor": 3.7,
    },
    ("HF", "14", "01"): {
        "top": 0.88,
        "hspace": 0.01,
        "height_factor": 3.5,
    },
    ("FF", "22", "01"): {
        "top": 0.90,
        "hspace": 0.05,
        "height_factor": 3.4,
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
    names = ["FDK", "FDK", "IR", "FBPCONVNet", "IResNet", "DDCNN"]
    ncols = len(names)
    nrows = len(VIEWS)
    # Calculate heights so that each view fills the full horizontal space
    # use the CROPS dict
    heights = []
    for view in VIEWS:
        crop = CROPS.get((scan_type, pid, sid), {}).get(view)
        if crop is not None:
            y0, y1, x0, x1 = crop
            height = (y1 - y0) / (x1 - x0)
            heights.append(height)
        else:
            heights.append(1.0)

    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(
            3 * ncols,
            sum(heights)
            * SUBPLOTS.get((scan_type, pid, sid), {}).get("height_factor", 3.5),
        ),
        squeeze=False,
        facecolor="black",
        gridspec_kw={"height_ratios": heights},
    )
    fig.patch.set_facecolor("black")

    # add the two big titles in white, centered over the appropriate columns
    # cols are 0…5; col 0 is gated, cols 1–5 are nonstop gated
    fig.text(
        x=(0 + 0.5) / ncols,
        y=0.95,
        s="Gated CBCT",
        color="white",
        weight="bold",
        ha="center",
        fontsize=20,
    )
    fig.text(
        x=(1 + 5 + 1) / 2 / ncols,
        y=0.95,  # midpoint of cols 1–5
        s="Nonstop Gated CBCT",
        color="white",
        weight="bold",
        ha="center",
        fontsize=20,
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
                ax.set_title(names[j], fontsize=16, color="white", weight="bold")
            ax.axis("off")

    # tighten up margins so images butt right up against each other
    fig.subplots_adjust(
        left=0.01,
        right=0.99,
        top=SUBPLOTS.get((scan_type, pid, sid), {}).get("top", 0.9),
        bottom=0.02,
        wspace=0.01,
        hspace=SUBPLOTS.get((scan_type, pid, sid), {}).get("hspace", 0.05),
    )

    outname = f"{scan_type}_p{pid}_{sid}.png"
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    fig.savefig(os.path.join(OUTPUT_DIR, outname), dpi=200)  # TODO
    plt.close(fig)
    print("Saved", outname)


def main():
    for scan in SCANS:
        plot_scan(*scan)


if __name__ == "__main__":
    main()
