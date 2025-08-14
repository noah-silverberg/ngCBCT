#!/usr/bin/env python3
import os
import glob
from scipy.io import loadmat
import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches

# -----------------------------------------------------------------------------
# CONFIG
# -----------------------------------------------------------------------------
BASE_DIR = "Data/3D_recon"
OUTPUT_DIR = "DDCNN_paper_figures_full"
RUN = "1"

# the three scans to plot
SCANS = [
    ("HF", "20", "01"),
    ("HF", "14", "01"),
    ("FF", "22", "01"),
    ("FF", "18", "01"),
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
    ("FF", "18", "01"): {
        "index": (5, 256 - 55, 0, 256 - 0),
        "width": (0, 160 - 0, 15, 256 - 53),
        "height": (0, 160 - 0, 35, 256 - 9),
    },
}

# Scan times for Gated and Nonstop acquisitions
SCAN_TIMES = {
    ("HF", "20", "01"): {"gated": "6.3 min", "nonstop": "1 min"},
    ("HF", "14", "01"): {"gated": "3.2 min", "nonstop": "1 min"},
    ("FF", "22", "01"): {"gated": "5.4 min", "nonstop": "0.6 min"},
    ("FF", "18", "01"): {"gated": "2.1 min", "nonstop": "0.6 min"},
}

# per‐scan, per‐view arrow customization:
# keys are (scan_type, pid, sid), values are dicts mapping view→params
# params:
#   tail_dx, tail_dy : shifts in pixels from tumor to arrow tail
#   lw               : line width of arrow
#   tip_dx, tip_dy   : additional x/y translation of the head (if desired)
ARROW_PARAMS = {
    ("HF", "20", "01"): {
        "index": {"tail_dx": 50, "tail_dy": -50, "tip_dx": 7, "tip_dy": -7, "lw": 3},
        "width": {"tail_dx": 35, "tail_dy": -40, "tip_dx": 4, "tip_dy": -4, "lw": 3},
        "height": {"tail_dx": 35, "tail_dy": -45, "tip_dx": 3, "tip_dy": -3, "lw": 3},
    },
    ("HF", "14", "01"): {
        "index": {"tail_dx": 45, "tail_dy": -60, "tip_dx": 15, "tip_dy": -22, "lw": 3},
        "width": {"tail_dx": 30, "tail_dy": -56, "tip_dx": 10, "tip_dy": -30, "lw": 3},
        "height": {"tail_dx": 48, "tail_dy": -52, "tip_dx": 18, "tip_dy": -17, "lw": 3},
    },
    ("FF", "22", "01"): {
        "index": {"tail_dx": -32, "tail_dy": -30, "tip_dx": -7, "tip_dy": -2, "lw": 3},
        "width": {"tail_dx": -25, "tail_dy": -29, "tip_dx": -2, "tip_dy": -7, "lw": 3},
        "height": {
            "tail_dx": -30,
            "tail_dy": -34,
            "tip_dx": -5,
            "tip_dy": -10,
            "lw": 3,
        },
    },
    ("FF", "18", "01"): {
        "index": {"tail_dx": -40, "tail_dy": -53, "tip_dx": -12, "tip_dy": -22, "lw": 3},
        "width": {"tail_dx": -28, "tail_dy": -47, "tip_dx": -5, "tip_dy": -18, "lw": 3},
        "height": {
            "tail_dx": -30,
            "tail_dy": -40,
            "tip_dx": -5,
            "tip_dy": -10,
            "lw": 3,
        },
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
    vols = [gt, fdk, pl, fbpcnn, ire, ddcnn]
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

    # --- Automatic figsize calculation ---
    subplot_base_width_inches = 3.0
    top_padding_inches = 1.2

    fig_width = subplot_base_width_inches * ncols
    image_area_height = subplot_base_width_inches * sum(heights)
    fig_height = image_area_height + top_padding_inches
    # ------------------------------------

    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(fig_width, fig_height),
        squeeze=False,
        facecolor="black",
        gridspec_kw={"height_ratios": heights},
    )
    fig.patch.set_facecolor("black")

    # --- Add Titles ---
    top_of_plots = image_area_height / fig_height
    top_margin_height = 1.0 - top_of_plots
    main_title_y = top_of_plots + top_margin_height * 0.70
    scan_time_y = top_of_plots + top_margin_height * 0.5

    # Main titles (Gated/Nonstop)
    fig.text(
        x=(0 + 0.5) / ncols,
        y=main_title_y,
        s="Gated CBCT",
        color="white",
        weight="bold",
        ha="center",
        fontsize=24,
    )
    fig.text(
        x=(1 + 5 + 1) / 2 / ncols,
        y=main_title_y,
        s="Nonstop Gated CBCT",
        color="white",
        weight="bold",
        ha="center",
        fontsize=24,
    )

    # Scan time subtitles
    scan_key = (scan_type, pid, sid)
    times = SCAN_TIMES.get(scan_key)
    if times:
        fig.text(
            x=(0 + 0.5) / ncols,
            y=scan_time_y,
            s=f"Scan Time: {times['gated']}",
            color="white",
            ha="center",
            fontsize=16,
        )
        fig.text(
            x=(1 + 5 + 1) / 2 / ncols,
            y=scan_time_y,
            s=f"Scan Time: {times['nonstop']}",
            color="white",
            ha="center",
            fontsize=16,
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

            # ─── arrow pointing to tumor in the GT (first) column using ARROW_PARAMS ───────────
            if j == 0:
                # determine tumor pixel coords for this view
                if view == "index":
                    ty, tx = tloc[0], tloc[1]
                elif view == "width":
                    # for width‐view we transposed (2,0,1) so x=orig y, y=orig z
                    ty, tx = tloc[2], tloc[0]
                else:  # height
                    # for height‐view we swapaxes(0,2) so x=orig x, y=orig z
                    ty, tx = tloc[2], tloc[1]

                # adjust for crop
                crop = CROPS.get((scan_type, pid, sid), {}).get(view)
                if crop is not None:
                    y0, y1, x0, x1 = crop
                    ty -= y0
                    tx -= x0

                # pull params (or use defaults)
                cfg = ARROW_PARAMS.get((scan_type, pid, sid), {}).get(
                    view,
                    {"tail_dx": 30, "tail_dy": 30, "tip_dx": 0, "tip_dy": 0, "lw": 3},
                )
                tail_dx = cfg["tail_dx"]
                tail_dy = cfg["tail_dy"]
                tip_dx = cfg["tip_dx"]
                tip_dy = cfg["tip_dy"]
                lw = cfg["lw"]

                # draw arrow
                ax.annotate(
                    "",
                    xy=(tx + tip_dx, ty + tip_dy),  # arrow head at tumor (+ tip shift)
                    xytext=(tx + tail_dx, ty + tail_dy),  # tail position
                    arrowprops=dict(
                        color="white",
                        arrowstyle="->",
                        lw=lw,
                    ),
                )
            # ────────────────────────────────────────────────────────────────

            if i == 0:
                ax.set_title(names[j], fontsize=20, color="white", weight="bold", pad=15)
            ax.axis("off")

    # tighten up margins so images butt right up against each other
    fig.subplots_adjust(
        left=0.01,
        right=0.99,
        top=top_of_plots,
        bottom=0.02,
        wspace=0.01,
        hspace=0.01,
    )

    # # ─── add subplot letter labels a)–r) ────────────────────────────────────────
    # # total subplots = nrows * ncols
    # labels = [f"{chr(ord('a') + k)})" for k in range(nrows * ncols)]
    # k = 0
    # for i in range(nrows):
    #     for j in range(ncols):
    #         ax = axes[i, j]
    #         ax.text(
    #             0.02,
    #             0.97,  # small inset from top‐left
    #             labels[k],
    #             color="white",
    #             weight="bold",
    #             fontsize=16,
    #             va="top",
    #             ha="left",
    #             transform=ax.transAxes,
    #         )
    #         k += 1
    # # ──────────────────────────────────────────────────────────────────────────

    # ─── draw dashed box around the first (gated) column ─────────────────────────
    # halfway between title baseline and top of figure:
    y_top = 0.99

    # get bottom edge of the bottom gated subplot:
    ax_bl = axes[-1, 0]
    pos_bl = ax_bl.get_position()
    y_bot = pos_bl.y0

    # halfway between bottom subplot and bottom of figure:
    y_bottom = y_bot / 2.0

    # left edge and width of the gated column (unchanged):
    ax_tl = axes[0, 0]
    pos_tl = ax_tl.get_position()
    x0 = pos_tl.x0
    width = pos_tl.width

    # draw the dashed box from y_bottom up to y_top:
    rect = patches.Rectangle(
        (x0, y_bottom),
        width,
        y_top - y_bottom,
        transform=fig.transFigure,
        fill=False,
        edgecolor="white",
        linestyle="--",
        linewidth=2,
    )
    fig.patches.append(rect)
    # ─────────────────────────────────────────────────────────────────────────────

    outname_png = f"{scan_type}_p{pid}_{sid}.png"
    outname_pdf = f"{scan_type}_p{pid}_{sid}.pdf"
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    fig.savefig(os.path.join(OUTPUT_DIR, outname_png), dpi=600)
    fig.savefig(os.path.join(OUTPUT_DIR, outname_pdf), dpi=600)
    plt.close(fig)
    print("Saved", outname_png, "and", outname_pdf)


def main():
    for scan in SCANS:
        plot_scan(*scan)


if __name__ == "__main__":
    main()