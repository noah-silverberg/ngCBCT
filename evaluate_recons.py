#!/usr/bin/env python3
import os
import glob
from scipy.io import loadmat
import torch
import numpy as np
import matplotlib.pyplot as plt
from skimage.metrics import structural_similarity as ssim
import pandas as pd

# -----------------------------------------------------------------------------
# CONFIG
# -----------------------------------------------------------------------------
BASE_DIR = "Data/3D_recon"
OUTPUT_DIR = "outputs"
RUN = "1"
SSIM_KWARGS = {"K1": 0.03, "K2": 0.06, "win_size": 11}
VIEWS = ["index", "height", "width"]
SCAN_TYPES = ["FF", "HF"]


# -----------------------------------------------------------------------------
# UTILS: mask creation
# -----------------------------------------------------------------------------
def make_mask(gt, view):
    if view == "index":
        # build circular mask from first slice of gt[...,0]
        first = gt[..., 0]
        fg = first != 0
        H, W = first.shape
        cy, cx = H / 2, W / 2
        ys, xs = np.nonzero(fg)
        d = np.sqrt((ys - cy) ** 2 + (xs - cx) ** 2)
        r = np.percentile(d, 99)
        Y, X = np.ogrid[:H, :W]
        return (Y - cy) ** 2 + (X - cx) ** 2 <= r**2
    else:
        # no cropping for height/width
        return np.ones(gt.shape[:2], dtype=bool)


# -----------------------------------------------------------------------------
# UTILS: metrics
# -----------------------------------------------------------------------------
def psnr_per_slice(gt, rec, mask):
    mse = np.mean((gt - rec) ** 2, axis=(0, 1), where=mask[..., None])
    psnr = 20 * np.log10(np.max(gt * mask[..., None], axis=(0, 1))) - 10 * np.log10(mse)
    psnr[np.isinf(psnr)] = np.nan
    return psnr


def mssim_per_slice(gt, rec, mask):
    vals = []
    for i in range(gt.shape[2]):
        _, mp = ssim(
            gt[:, :, i], rec[:, :, i], full=True, data_range=1.0, **SSIM_KWARGS
        )
        vals.append(mp[mask])
    return np.array([v.mean() for v in vals])


# -----------------------------------------------------------------------------
# PLOTTING
# -----------------------------------------------------------------------------
def save_ssim_map(
    gt, rec_list, titles, mask, tumor_slice, tumor_xy, outpath, scan_type
):
    n = len(rec_list) + 1
    # two rows: top = actual scans, bottom = SSIM maps
    fig, axes = plt.subplots(2, n, figsize=(3 * n, 6))
    # determine arrow offset based on scan type
    arrow_offset = 40 if scan_type == "HF" else 20
    # top row: GT and reconstructions
    for i in range(n):
        ax = axes[0, i]
        if i == 0:
            ax.imshow(gt[..., tumor_slice] * mask, cmap="gray", vmin=0, vmax=1)
            ax.set_title("GT")
        else:
            ax.imshow(
                rec_list[i - 1][..., tumor_slice] * mask, cmap="gray", vmin=0, vmax=1
            )
            ax.set_title(titles[i - 1])

        # arrow pointing to tumor
        ax.annotate(
            "",
            xy=tumor_xy,
            xytext=(tumor_xy[0] + arrow_offset, tumor_xy[1] + arrow_offset),
            arrowprops=dict(color="red", arrowstyle="->", lw=1.5),
        )
        ax.axis("off")
    # bottom row: SSIM maps with arrow
    for i in range(n):
        ax = axes[1, i]
        if i == 0:
            ax.axis("off")
        else:
            _, smap = ssim(
                gt[..., tumor_slice],
                rec_list[i - 1][..., tumor_slice],
                full=True,
                data_range=1.0,
                **SSIM_KWARGS,
            )
            im = ax.imshow(smap * mask, cmap="viridis", vmin=0, vmax=1)
            # arrow pointing to tumor
            ax.annotate(
                "",
                xy=tumor_xy,
                xytext=(tumor_xy[0] + arrow_offset, tumor_xy[1] + arrow_offset),
                arrowprops=dict(color="red", arrowstyle="->", lw=1.5),
            )
            ax.set_title(f"SSIM {titles[i-1]}")
            ax.axis("off")
    # make room for colorbar
    fig.subplots_adjust(right=0.85)
    fig.colorbar(im, ax=axes[1, 1:].tolist(), fraction=0.02, pad=0.02, location="right")
    fig.savefig(outpath, dpi=300)
    plt.close(fig)


def save_psnr_mssim_plot(gt, fdk, pl, dd, mask, outpath):
    psnr_fdk = psnr_per_slice(gt, fdk, mask)
    psnr_pl = psnr_per_slice(gt, pl, mask)
    psnr_ddcnn = psnr_per_slice(gt, dd, mask)
    mssim_fdk = mssim_per_slice(gt, fdk, mask)
    mssim_pl = mssim_per_slice(gt, pl, mask)
    mssim_ddcnn = mssim_per_slice(gt, dd, mask)
    N = gt.shape[2]

    fig, ax = plt.subplots(2, 2, figsize=(10, 8))
    # MSSIM vs idx
    ax[0, 0].plot(range(N), mssim_fdk, label="FDK", color="blue")
    ax[0, 0].plot(range(N), mssim_pl, label="PL", color="orange")
    ax[0, 0].plot(range(N), mssim_ddcnn, label="DDCNN", color="green")
    ax[0, 0].set(title="MSSIM vs Slice", xlabel="Slice", ylabel="MSSIM")
    ax[0, 0].legend()

    ax[0, 1].hist(
        mssim_fdk,
        color="blue",
        bins=50,
        label="FDK",
        alpha=0.6,
    )
    ax[0, 1].hist(
        mssim_pl,
        color="orange",
        bins=50,
        label="PL",
        alpha=0.6,
    )
    ax[0, 1].hist(
        mssim_ddcnn,
        color="green",
        bins=50,
        label="DDCNN",
        alpha=0.6,
    )
    ax[0, 1].set(title="MSSIM Histogram", xlabel="MSSIM")
    ax[0, 1].legend()

    ax[1, 0].plot(range(N), psnr_fdk, label="FDK", color="blue")
    ax[1, 0].plot(range(N), psnr_pl, label="PL", color="orange")
    ax[1, 0].plot(range(N), psnr_ddcnn, label="DDCNN", color="green")
    ax[1, 0].set(title="PSNR vs Slice", xlabel="Slice", ylabel="PSNR (dB)")
    ax[1, 0].legend()

    ax[1, 1].hist(
        psnr_fdk,
        color="blue",
        bins=50,
        label="FDK",
        alpha=0.6,
    )
    ax[1, 1].hist(
        psnr_pl,
        color="orange",
        bins=50,
        label="PL",
        alpha=0.6,
    )
    ax[1, 1].hist(
        psnr_ddcnn,
        color="green",
        bins=50,
        label="DDCNN",
        alpha=0.6,
    )
    ax[1, 1].set(title="PSNR Histogram", xlabel="PSNR (dB)")
    ax[1, 1].legend()

    plt.tight_layout()
    fig.savefig(outpath, dpi=300)
    plt.close(fig)


# -----------------------------------------------------------------------------
# MAIN
# -----------------------------------------------------------------------------
def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    records = []

    for scan_type in SCAN_TYPES:
        mat_dir = os.path.join(BASE_DIR, scan_type)
        # choose correct FDK file pattern by type:
        if scan_type == "FF":
            patt = os.path.join(
                mat_dir, f"recon_p*.{scan_type}??.u_FDK_ROI_fullView.mat"
            )
        else:
            patt = os.path.join(mat_dir, f"recon_p*.{scan_type}??.u_FDK_full.mat")

        for gt_path in glob.glob(patt):
            # extract patient & scan IDs: recon_p{pid}.{type}{sid}.*
            base = os.path.basename(gt_path)
            pid = base.split(".")[0].split("p")[1]
            sid = base.split(".")[1][len(scan_type) :]

            # load all mats
            if scan_type == "FF":
                fdk_path = gt_path.replace("FDK_ROI_fullView", "FDK_ROI")
                pl_path = gt_path.replace("FDK_ROI_fullView", "PL_ROI.b1")
            else:
                fdk_path = gt_path.replace("FDK_full", "FDK")
                pl_path = gt_path.replace("FDK_full", "PL.b1.iter200")

            gt_mat = loadmat(gt_path)
            fdk_mat = loadmat(fdk_path)
            pl_mat = loadmat(pl_path)
            # pick the right variable
            if scan_type == "FF":
                gt_vol = gt_mat["u_FDK_ROI_fullView"]
                fdk = fdk_mat["u_FDK_ROI"]
                pl = pl_mat["u_PL_ROI"]
            else:
                gt_vol = gt_mat["u_FDK_full"]
                fdk = fdk_mat["u_FDK"]
                pl = pl_mat["u_PL"]

            # trim slices & crop
            gt_vol = gt_vol[..., 20:-20]
            fdk = fdk[..., 20:-20]
            pl = pl[..., 20:-20]
            if scan_type == "FF":
                gt_vol = gt_vol[128:-128, 128:-128]
                fdk = fdk[128:-128, 128:-128]
                pl = pl[128:-128, 128:-128]

            # load network output
            ds = 14 if scan_type == "FF" else 13
            ddcnn = torch.load(
                os.path.join(
                    mat_dir,
                    f"p{pid}.{scan_type}{sid}_IResNet_MK6_DS{ds}.2_run{RUN}_3D.pt",
                ),
                weights_only=False,
            )

            # tumor location
            tlocs = torch.load(
                os.path.join(
                    BASE_DIR,
                    (
                        f"tumor_location_FF.pt"
                        if scan_type == "FF"
                        else "tumor_location.pt"
                    ),
                ),
                weights_only=False,
            )
            tloc = tlocs[int(pid), int(sid)]
            if scan_type == "FF":
                tloc[:2] -= 128

            for view in VIEWS:
                # apply axis-swaps
                gt = gt_vol.copy()
                fdk_v = fdk.copy()
                pl_v = pl.copy()
                dd_v = ddcnn.copy()
                t = tloc.copy()
                if view == "height":
                    gt, fdk_v, pl_v, dd_v = [
                        np.swapaxes(x, 0, 2) for x in (gt, fdk_v, pl_v, dd_v)
                    ]
                    t = np.array([t[2], t[1], t[0]])
                elif view == "width":
                    gt, fdk_v, pl_v, dd_v = [
                        np.swapaxes(x, 1, 2) for x in (gt, fdk_v, pl_v, dd_v)
                    ]
                    t = np.array([t[0], t[2], t[1]])

                # use mask only for FF scans; otherwise full-true
                if scan_type == "FF":
                    mask = make_mask(gt, view)
                else:
                    mask = np.ones(gt.shape[:2], dtype=bool)
                tumor_slice = int(t[2])

                # normalize & clip
                # only clip/normalize GT, FDK, PL
                for arr in (gt, fdk_v, pl_v):
                    np.clip(arr, 0, 0.04, out=arr)
                    arr -= arr.min()
                    arr /= arr.max()
                # leave dd_v unchanged

                # make output subdir
                odir = os.path.join(OUTPUT_DIR, f"{scan_type}_p{pid}_{sid}_{view}")
                os.makedirs(odir, exist_ok=True)

                # # 1) SSIM map
                # save_ssim_map(
                #     gt,
                #     [fdk_v, pl_v, dd_v],
                #     ["FDK", "PL", "DDCNN"],
                #     mask,
                #     tumor_slice,
                #     (t[1], t[0]),
                #     os.path.join(odir, "SSIM_map.png"),
                #     scan_type,
                # )

                # # 2) PSNR/MSSIM curves
                # save_psnr_mssim_plot(
                #     gt, fdk_v, pl_v, dd_v, mask, os.path.join(odir, "PSNR_MSSIM.png")
                # )

                # 3) summary metrics
                ps_all = {}
                ms_all = {}
                for name, arr in zip(["FDK", "PL", "DDCNN"], [fdk_v, pl_v, dd_v]):
                    ps = psnr_per_slice(gt, arr, mask)
                    ms = mssim_per_slice(gt, arr, mask)
                    ps_all[name] = np.nanmean(ps)
                    ms_all[name] = np.nanmean(ms)
                # tumor slice only
                recs = dict(FDK=fdk_v, PL=pl_v, DDCNN=dd_v)
                for name, arr in recs.items():
                    ps_all[f"{name}_tumor"] = psnr_per_slice(gt, arr, mask)[tumor_slice]
                    ms_all[f"{name}_tumor"] = mssim_per_slice(gt, arr, mask)[
                        tumor_slice
                    ]

                rec = {
                    "scan_type": scan_type,
                    "patient_id": pid,
                    "scan_id": sid,
                    "view": view,
                    **{f"psnr_{k}": v for k, v in ps_all.items()},
                    **{f"mssim_{k}": v for k, v in ms_all.items()},
                }
                records.append(rec)

    # write summary
    df = pd.DataFrame(records)
    df.to_csv(os.path.join(OUTPUT_DIR, "summary.csv"), index=False)

    # LaTeX full‐document output
    tex_path = os.path.join(OUTPUT_DIR, "summary.tex")
    with open(tex_path, "w") as f:
        f.write(
            r"""\documentclass{article}
\usepackage{graphicx}
\usepackage{amsmath,amssymb}
\usepackage{amsthm}
\usepackage{hyperref}
\usepackage{underscore} % allow underscores in table text
\hypersetup{colorlinks=true, urlcolor=blue, linkcolor=blue, citecolor=red}
\usepackage{booktabs}

\title{Analysis of DDCNN Performance}
\author{Noah Silverberg}
\date{}

\begin{document}

\maketitle

\section{Summary table}
\begin{table}[ht]
\caption{Mean PSNR and MSSIM across all slices and tumor slice}
\label{tab:summary}
\resizebox{\textwidth}{!}{%
"""
        )
        # insert only the tabular itself
        f.write(
            df.to_latex(
                index=False,
                float_format="%.3f",
                escape=False,  # keep underscores
                header=True,
                longtable=False,
            )
        )
        # finish the document
        f.write(
            r"""%
} % end resizebox
\end{table}

\end{document}
"""
        )

    print("Done! Outputs in", OUTPUT_DIR)


if __name__ == "__main__":
    main()
