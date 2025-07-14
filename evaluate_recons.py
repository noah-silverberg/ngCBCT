#!/usr/bin/env python3
import os
import glob
from scipy.io import loadmat
import torch
import numpy as np
import matplotlib.pyplot as plt
import torchmetrics.image
import pandas as pd

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "1" 
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE} named '{torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}'")

# -----------------------------------------------------------------------------
# CONFIG
# -----------------------------------------------------------------------------
RESULTS_DIR = "D:/NoahSilverberg/ngCBCT/phase7/DS13/results/images/MK7_07"
TRUTH_DIR = "D:/NoahSilverberg/ngCBCT/gated/fdk_recon"
TUMOR_DIR = "D:/NoahSilverberg/ngCBCT/3D_recon"
OUTPUT_DIR = "outputs"
RUN = "1"
SSIM_KWARGS = {"K1": 0.03, "K2": 0.06, "win_size": 11}
VIEWS = ["index", "height", "width"]
SCAN_TYPES = ["HF"]


# -----------------------------------------------------------------------------
# UTILS: metrics
# -----------------------------------------------------------------------------
def psnr_per_slice(gt, rec, mask, device):
    gt, rec, mask = gt.to(device), rec.to(device), mask.to(device)
    num_slices = gt.shape[2]
    psnrs = []
    
    # Init PSNR metric
    psnr_metric = torchmetrics.image.PeakSignalNoiseRatio(data_range=1.0, reduction='none').to(device)

    for i in range(num_slices):
        # Add batch and channel dims for torchmetrics
        gt_slice = gt[:, :, i].unsqueeze(0).unsqueeze(0)
        rec_slice = rec[:, :, i].unsqueeze(0).unsqueeze(0)
        
        # Apply mask
        gt_slice = gt_slice * mask.unsqueeze(0).unsqueeze(0)
        rec_slice = rec_slice * mask.unsqueeze(0).unsqueeze(0)

        val = psnr_metric(rec_slice, gt_slice)
        psnrs.append(val.item())
        
    psnrs_np = np.array(psnrs)
    psnrs_np[np.isinf(psnrs_np)] = np.nan
    return psnrs_np


def mssim_per_slice(gt, rec, mask, device):
    gt, rec, mask = gt.to(device), rec.to(device), mask.to(device)
    num_slices = gt.shape[2]
    ssims = []
    
    # Init SSIM metric
    ssim_metric = torchmetrics.image.StructuralSimilarityIndexMeasure(data_range=1.0, **SSIM_KWARGS).to(device)

    for i in range(num_slices):
        # Add batch and channel dims for torchmetrics
        gt_slice = gt[:, :, i].unsqueeze(0).unsqueeze(0)
        rec_slice = rec[:, :, i].unsqueeze(0).unsqueeze(0)
        
        # Apply mask
        gt_slice = gt_slice * mask.unsqueeze(0).unsqueeze(0)
        rec_slice = rec_slice * mask.unsqueeze(0).unsqueeze(0)
        
        ssims.append(ssim_metric(rec_slice, gt_slice).item())
        
    return np.array(ssims)


# -----------------------------------------------------------------------------
# PLOTTING
# -----------------------------------------------------------------------------
def save_ssim_map(
    gt, rec_list, titles, mask, tumor_slice, tumor_xy, outpath, scan_type, device
):
    n = len(rec_list) + 1
    # two rows: top = actual scans, bottom = empty for SSIM value text
    fig, axes = plt.subplots(2, n, figsize=(3 * n, 6))
    # determine arrow offset based on scan type
    arrow_offset = 40 if scan_type == "HF" else 20
    
    # init ssim metric
    ssim_metric = torchmetrics.image.StructuralSimilarityIndexMeasure(data_range=1.0, **SSIM_KWARGS).to(device)

    # top row: GT and reconstructions
    for i in range(n):
        ax = axes[0, i]
        if i == 0:
            ax.imshow(gt[..., tumor_slice].cpu().numpy() * mask.cpu().numpy(), cmap="gray", vmin=0, vmax=1)
            ax.set_title("GT")
        else:
            ax.imshow(
                rec_list[i - 1][..., tumor_slice].cpu().numpy() * mask.cpu().numpy(), cmap="gray", vmin=0, vmax=1
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
    
    # bottom row: SSIM values
    for i in range(n):
        ax = axes[1, i]
        ax.axis("off")
        if i > 0:
            gt_slice = gt[..., tumor_slice].unsqueeze(0).unsqueeze(0)
            rec_slice = rec_list[i-1][..., tumor_slice].unsqueeze(0).unsqueeze(0)
            
            val = ssim_metric(rec_slice, gt_slice)
            ax.text(0.5, 0.5, f"SSIM: {val.item():.4f}", ha='center', va='center', fontsize=12)

    fig.savefig(outpath, dpi=200)
    plt.close(fig)


def save_psnr_mssim_plot(gt, dd, mask, outpath, device):
    psnr_ddcnn = psnr_per_slice(gt, dd, mask, device)
    mssim_ddcnn = mssim_per_slice(gt, dd, mask, device)
    N = gt.shape[2]

    fig, ax = plt.subplots(2, 2, figsize=(10, 8))
    # MSSIM vs idx
    ax[0, 0].plot(range(N), mssim_ddcnn, label="DDCNN", color="green")
    ax[0, 0].set(title="MSSIM vs Slice", xlabel="Slice", ylabel="MSSIM")
    ax[0, 0].legend()

    ax[0, 1].hist(
        mssim_ddcnn,
        color="green",
        bins=50,
        label="DDCNN",
        alpha=0.6,
    )
    ax[0, 1].set(title="MSSIM Histogram", xlabel="MSSIM")
    ax[0, 1].legend()

    ax[1, 0].plot(range(N), psnr_ddcnn, label="DDCNN", color="green")
    ax[1, 0].set(title="PSNR vs Slice", xlabel="Slice", ylabel="PSNR (dB)")
    ax[1, 0].legend()

    ax[1, 1].hist(
        psnr_ddcnn,
        color="green",
        bins=50,
    )
    ax[1, 0].plot(range(N), psnr_ddcnn, label="DDCNN", color="green")
    ax[1, 0].set(title="PSNR vs Slice", xlabel="Slice", ylabel="PSNR (dB)")
    ax[1, 0].legend()

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
    fig.savefig(outpath, dpi=200)
    plt.close(fig)


# -----------------------------------------------------------------------------
# MAIN
# -----------------------------------------------------------------------------
def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    records = []

    for gt_path in os.listdir(TRUTH_DIR):
        base = os.path.basename(gt_path)
        pid = base.split("_")[1][1:]
        sid = base.split("_")[2]

        if not (pid, sid) in [
            ("08", "01"),
            ("10", "01"),
            ("14", "01"),
            ("14", "02"),
            ("15", "01"),
            ("20", "01"),
        ]:
            continue

        # load the ground truth volume
        gt_vol = torch.load(
            os.path.join(TRUTH_DIR, gt_path), weights_only=False
        ).cpu().numpy()

        # load network output
        ddcnn: np.ndarray = torch.load(
            os.path.join(
                RESULTS_DIR,
                f"p{pid}_{sid}.pt",
            ),
            weights_only=False,
        ).cpu().numpy()
        
        # squeeze axis 1 of ddcnn
        if ddcnn.ndim == 4:
            ddcnn = ddcnn.squeeze(1)


        # tumor location
        tlocs = torch.load(
            os.path.join(
                TUMOR_DIR,
                (
                    "tumor_location.pt"
                ),
            ),
            weights_only=False,
        )
        tloc = tlocs[int(pid), int(sid)]

        tloc[2] -= 20  # adjust for slice trimming

        # remove first and last 20 slices
        gt_vol = gt_vol[20:-20, ...]

        # swap final two axes
        gt_vol = np.swapaxes(gt_vol, 1, 2)

        # make the axes match the old format
        ddcnn = np.transpose(ddcnn, (1, 2, 0))
        gt_vol = np.transpose(gt_vol, (1, 2, 0))

        for view in VIEWS:
            # apply axis-swaps
            gt = gt_vol.copy()
            dd_v = ddcnn.copy()
            t = tloc.copy()
            if view == "height":
                gt, dd_v = [
                    np.swapaxes(x, 0, 2)
                    for x in (gt, dd_v)
                ]
                t = np.array([t[2], t[1], t[0]])
            elif view == "width":
                gt, dd_v = [
                    np.transpose(x, (2, 0, 1))
                    for x in (gt, dd_v)
                ]
                t = np.array([t[2], t[0], t[1]])

            # mask as a cylinder through the volume:
            #   - circular in the index view
            #   - straight band in height/width views
            if view == "index":
                mask = np.zeros(gt.shape[:2], dtype=bool)
                center0 = gt.shape[0] // 2
                center1 = gt.shape[1] // 2
                radius = 225
                y, x = np.ogrid[: gt.shape[0], : gt.shape[1]]
                mask[(y - center0) ** 2 + (x - center1) ** 2 <= radius ** 2] = True
            else:
                mask = np.zeros(gt.shape[:2], dtype=bool)
                center1 = gt.shape[1] // 2
                radius = 225
                # band across axis-1 (width in height view, height in width view)
                mask[:, center1 - radius : center1 + radius + 1] = True
            tumor_slice = int(t[2])

            # adjust SSIM window size per scan_type & view
            SSIM_KWARGS["win_size"] = 15 if view == "index" else 11

            # normalize & clip
            for arr in (gt):
                np.clip(arr, 0, 0.04, out=arr)
                arr *= 25.
            # leave dd_v unchanged

            gt_t = torch.from_numpy(gt).float().to(DEVICE)
            dd_v_t = torch.from_numpy(dd_v).float().to(DEVICE)
            mask_t = torch.from_numpy(mask).to(DEVICE)

            # make output subdir
            odir = os.path.join(OUTPUT_DIR, f"{'HF'}_p{pid}_{sid}_{view}")
            os.makedirs(odir, exist_ok=True)

            # 1) SSIM map
            save_ssim_map(
                gt,
                [dd_v],
                ["DDCNN"],
                mask,
                tumor_slice,
                (t[1], t[0]),
                os.path.join(odir, "SSIM_map.png"),
                'HF',
                DEVICE,
            )

            # 2) PSNR/MSSIM curves
            save_psnr_mssim_plot(
                gt,
                dd_v,
                mask,
                os.path.join(odir, "PSNR_MSSIM.png"),
                DEVICE,
            )

            # 3) summary metrics
            ps_all = {}
            ms_all = {}
            for name, arr in zip(
                ["DDCNN"],
                [dd_v],
            ):
                ps = psnr_per_slice(gt, arr, mask, DEVICE)
                ms = mssim_per_slice(gt, arr, mask, DEVICE)
                ps_all[name] = np.nanmean(ps)
                ms_all[name] = np.nanmean(ms)

                # compute mean metrics over ±10 and ±20 slices around tumor
                start_10 = max(0, tumor_slice - 10)
                end_10 = min(gt.shape[2], tumor_slice + 10 + 1)
                start_20 = max(0, tumor_slice - 20)
                end_20 = min(gt.shape[2], tumor_slice + 20 + 1)
                ps_all[f"{name}_around10"] = np.nanmean(ps[start_10:end_10])
                ps_all[f"{name}_around20"] = np.nanmean(ps[start_20:end_20])
                ms_all[f"{name}_around10"] = np.nanmean(ms[start_10:end_10])
                ms_all[f"{name}_around20"] = np.nanmean(ms[start_20:end_20])

            # (done) TODO subtract 20 from all index for tumor location (HF and FF)
            # (done) TODO transpose the "width" view images
            # (done) TODO exclude FF 16 scan 01
            # (done) TODO also make 3 more tables, with 20 slices around tumor (10 on each side)
            # (done) TODO also make 3 more tables, with 40 slices around tumor (20 on each side)
            # (done) TODO redo paper figures (black background)
            # FIGURE 5 -- HF P20 SCAN01
            # FIGURE 6 -- HF P14 SCAN01
            # FIGURE 7 -- FF P22 SCAN01

            rec = {
                "scan_type": 'HF',
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
    # Print summary tables and save to log file
    # ──────────────────────────────────────────────────────
    log_path = os.path.join(OUTPUT_DIR, "summary_tables.log")
    with open(log_path, "w") as f:
        methods = [
            ("DDCNN", "DDCNN"),
        ]
        # 1) full‐volume tables (existing)
        for view in VIEWS:
            sub = df[df["view"] == view]
            agg = sub.groupby("scan_type").agg(
                {
                    "psnr_DDCNN": "mean",
                    "mssim_DDCNN": "mean",
                }
            )
            table_str = f"\n{'='*40}\n{view.capitalize()} view: PSNR & SSIM by method and fan type\n{'='*40}\n"
            table_str += f"{'Method':<10} {'Fan':<10} {'PSNR':>8} {'SSIM':>8}\n"
            for key, label in methods:
                for fan in agg.index:
                    ps = agg.loc[fan, f"psnr_{key}"]
                    ss = agg.loc[fan, f"mssim_{key}"]
                    table_str += f"{label:<10} {fan:<10} {ps:8.2f} {ss:8.3f}\n"
            print(table_str)
            f.write(table_str)

        # 2) tables for ±10 and ±20 slices around tumor
        windows = [("±10", "around10"), ("±20", "around20")]
        for w_label, col_suffix in windows:
            for view in VIEWS:
                sub = df[df["view"] == view]
                agg = sub.groupby("scan_type").agg(
                    {
                        **{f"psnr_{key}_{col_suffix}": "mean" for key, _ in methods},
                        **{f"mssim_{key}_{col_suffix}": "mean" for key, _ in methods},
                    }
                )
                table_str = f"\n{'='*40}\n{view.capitalize()} view: PSNR & SSIM by method and fan type for tumor surrounding {w_label} slices\n{'='*40}\n"
                table_str += f"{'Method':<10} {'Fan':<10} {'PSNR':>8} {'SSIM':>8}\n"
                for key, label in methods:
                    for fan in agg.index:
                        ps = agg.loc[fan, f"psnr_{key}_{col_suffix}"]
                        ss = agg.loc[fan, f"mssim_{key}_{col_suffix}"]
                        table_str += f"{label:<10} {fan:<10} {ps:8.2f} {ss:8.3f}\n"
                print(table_str)
                f.write(table_str)


    print("Done! Outputs in", OUTPUT_DIR)


if __name__ == "__main__":
    main()
