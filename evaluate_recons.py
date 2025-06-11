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


# ──────────────────────────────────────────────────────
# helper: collect per‐slice metrics across all patients/views
# ──────────────────────────────────────────────────────
def collect_per_slice(df, base_dir, scan_type, metric_fn):
    """
    df: the DataFrame of records
    base_dir: same as BASE_DIR
    scan_type: "FF" or "HF"
    metric_fn: one of psnr_per_slice or mssim_per_slice
    returns: dict view → list of arrays (one array per volume)
    """
    out = {v: [] for v in VIEWS}
    for _, row in df[df.scan_type == scan_type].iterrows():
        # reload volumes exactly as in main...
        pid, sid, view = row.patient_id, row.scan_id, row["view"]
        mat_dir = os.path.join(base_dir, scan_type)
        # reconstruct GT and DDCNN paths
        if scan_type == "FF":
            gt_path = os.path.join(
                mat_dir,
                f"recon_p{pid}.{scan_type}{sid}.u_FDK_ROI_fullView.mat",
            )
            ds = 14
        else:
            gt_path = os.path.join(
                mat_dir, f"recon_p{pid}.{scan_type}{sid}.u_FDK_full.mat"
            )
            ds = 13
        dd_path = os.path.join(
            mat_dir,
            f"p{pid}.{scan_type}{sid}_IResNet_MK6_DS{ds}.2_run{RUN}_3D.pt",
        )

        # load GT volume
        gt_mat = loadmat(gt_path)
        if scan_type == "FF":
            gt_vol = gt_mat["u_FDK_ROI_fullView"]
        else:
            gt_vol = gt_mat["u_FDK_full"]

        # crop slices & ROI for FF
        gt_vol = gt_vol[..., 20:-20]
        if scan_type == "FF":
            gt_vol = gt_vol[128:-128, 128:-128]

        # load DDCNN output
        ddcnn = torch.load(dd_path, weights_only=False)

        # apply view‐specific axis swap
        gt_v = gt_vol.copy()
        rec_v = ddcnn.copy()
        if view == "height":
            gt_v = np.swapaxes(gt_v, 0, 2)
            rec_v = np.swapaxes(rec_v, 0, 2)
        elif view == "width":
            gt_v = np.swapaxes(gt_v, 1, 2)
            rec_v = np.swapaxes(rec_v, 1, 2)

        # build mask
        if scan_type == "FF":
            mask = make_mask(gt_v, view)
        else:
            mask = np.ones(gt_v.shape[:2], dtype=bool)

        # normalize GT to [0,1]
        np.clip(gt_v, 0, 0.04, out=gt_v)
        gt_v -= gt_v.min()
        gt_v /= gt_v.max()
        # then:
        arr = metric_fn(gt_v, rec_v, mask)
        out[view].append(arr)
    return out


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

        for gt_path in glob.glob(patt)[:2]:  # TODO remove [:2]
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

                # # 1) SSIM map # TODO uncomment these plots
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

    # ──────────────────────────────────────────────────────
    # 1) write separate LaTeX tables for FF vs HF (grouped by view)
    # ──────────────────────────────────────────────────────
    for stype, sub in df.groupby("scan_type"):
        # aggregate mean ± std per view
        agg = sub.groupby("view")[["psnr_DDCNN", "mssim_DDCNN"]].agg(["mean", "std"])

        # format mean±std (PSNR: 1 decimal, MSSIM: 3 decimals)
        def fmt(val, sd, metric):
            if metric == "psnr_DDCNN":
                return f"{val:.1f}±{sd:.1f}"
            else:
                return f"{val:.3f}±{sd:.3f}"

        table = agg.reset_index().assign(
            PSNR=lambda d: [
                fmt(m, s, "psnr_DDCNN")
                for (m, s) in zip(d[("psnr_DDCNN", "mean")], d[("psnr_DDCNN", "std")])
            ],
            SSIM=lambda d: [
                fmt(m, s, "mssim_DDCNN")
                for (m, s) in zip(d[("mssim_DDCNN", "mean")], d[("mssim_DDCNN", "std")])
            ],
        )[["view", "PSNR", "SSIM"]]

        tex = table.to_latex(
            index=False,
            escape=False,
            caption=f"{stype} – Mean DDCNN PSNR & SSIM by view",
            label=f"tab:{stype.lower()}_by_view",
        )
        with open(os.path.join(OUTPUT_DIR, f"summary_{stype}.tex"), "w") as fh:
            fh.write(r"\begin{table}[ht]\centering" + "\n")
            fh.write(tex.replace("\\toprule", "\\toprule\n\\midrule"))
            fh.write(r"\end{table}")

        # ──────────────────────────────────────────────────────
    # 2) view-centric bar charts: mean±std for DDCNN only
    # ──────────────────────────────────────────────────────
    import matplotlib.pyplot as plt

    for metric, ylabel, fmt in [
        ("psnr_DDCNN", "PSNR (dB)", "{:.1f}"),
        ("mssim_DDCNN", "MSSIM", "{:.3f}"),
    ]:
        fig, ax = plt.subplots(figsize=(6, 4))
        summary = (
            df.groupby(["view", "scan_type"])[metric]
            .agg(["mean", "std"])
            .unstack("scan_type")
        )
        # bars for FF, HF side-by-side
        summary["mean"].plot.bar(
            y=["FF", "HF"], yerr=summary["std"][["FF", "HF"]], rot=0, ax=ax
        )
        ax.set_xlabel("View")
        ax.set_ylabel(ylabel)
        ax.set_title(f"DDCNN {ylabel} by View (mean±SD)")
        # annotate bar labels
        for p in ax.patches:
            h = p.get_height()
            if not np.isnan(h):
                ax.text(
                    p.get_x() + p.get_width() / 2,
                    h + 0.01 * h,
                    fmt.format(h),
                    ha="center",
                    va="bottom",
                    fontsize=8,
                )
        plt.tight_layout()
        fig.savefig(os.path.join(OUTPUT_DIR, f"bar_{metric}.png"), dpi=300)
        plt.close(fig)

        # ──────────────────────────────────────────────────────
    # 4) boxplots of slice‐wise PSNR & SSIM for each scan_type
    # ──────────────────────────────────────────────────────
    import matplotlib.pyplot as plt

    for stype in SCAN_TYPES:
        # PSNR boxplot
        ps_data = collect_per_slice(df, BASE_DIR, stype, psnr_per_slice)
        fig, ax = plt.subplots(1, 1, figsize=(6, 4))
        ax.boxplot(
            [np.concatenate(ps_data[v]) for v in VIEWS], labels=VIEWS, showfliers=False
        )
        ax.set_title(f"{stype} PSNR distribution across all slices")
        ax.set_ylabel("PSNR (dB)")
        plt.tight_layout()
        fig.savefig(os.path.join(OUTPUT_DIR, f"box_psnr_{stype}.png"), dpi=300)
        plt.close(fig)

        # SSIM boxplot
        ss_data = collect_per_slice(df, BASE_DIR, stype, mssim_per_slice)
        fig, ax = plt.subplots(1, 1, figsize=(6, 4))
        ax.boxplot(
            [np.concatenate(ss_data[v]) for v in VIEWS], labels=VIEWS, showfliers=False
        )
        ax.set_title(f"{stype} SSIM distribution across all slices")
        ax.set_ylabel("SSIM")
        plt.tight_layout()
        fig.savefig(os.path.join(OUTPUT_DIR, f"box_ssim_{stype}.png"), dpi=300)
        plt.close(fig)

        # ──────────────────────────────────────────────────────
    # 5) overall FF vs HF (averaged over view & patient)
    # ──────────────────────────────────────────────────────
    agg2 = df.groupby("scan_type")[["psnr_DDCNN", "mssim_DDCNN"]].agg(["mean", "std"])
    # format as “val±sd”
    summary2 = {
        stype: {
            "PSNR": f"{agg2.loc[stype,('psnr_DDCNN','mean')]:.1f}±{agg2.loc[stype,('psnr_DDCNN','std')]:.1f}",
            "SSIM": f"{agg2.loc[stype,('mssim_DDCNN','mean')]:.3f}±{agg2.loc[stype,('mssim_DDCNN','std')]:.3f}",
        }
        for stype in SCAN_TYPES
    }
    df2 = pd.DataFrame.from_dict(summary2, orient="index")[["PSNR", "SSIM"]]
    df2.to_csv(os.path.join(OUTPUT_DIR, "summary_overall.csv"))
    with open(os.path.join(OUTPUT_DIR, "summary_overall.tex"), "w") as fh:
        fh.write(
            df2.to_latex(
                escape=False,
                caption="Overall DDCNN PSNR & SSIM (FF vs HF)",
                label="tab:overall",
            )
        )

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
