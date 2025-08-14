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
BASE_DIR = "Data/3D_recon"
OUTPUT_DIR = "DDCNN_paper_figures_ssim_psnr"
RUN = "1"
SSIM_KWARGS = {"k1": 0.03, "k2": 0.06, "kernel_size": 11}
VIEWS = ["index"]
SCAN_TYPES = ["HF", 'FF']

# -----------------------------------------------------------------------------
# UTILS: mask creation
# -----------------------------------------------------------------------------
def make_mask(gt, view):
    if view == "index":
        # build circular mask
        H, W, _ = gt.shape
        cy, cx = H / 2, W / 2
        r = 120
        Y, X = np.ogrid[:H, :W]
        return (Y - cy) ** 2 + (X - cx) ** 2 <= r**2
    else:
        # no cropping for height/width
        return np.ones(gt.shape[:2], dtype=bool)


# -----------------------------------------------------------------------------
# UTILS: metrics
# -----------------------------------------------------------------------------
def psnr_per_slice(gt, rec, mask, device):
    gt, rec, mask = gt.to(device), rec.to(device), mask.to(device)

    # 1. Calculate Mean Squared Error per slice, inside the masked region
    num_pixels_in_mask = torch.sum(mask)
    squared_error = torch.pow(gt - rec, 2)
    sum_squared_error = torch.sum(squared_error * torch.unsqueeze(mask, 2), dim=(0, 1))

    # 2. Calculate PSNR from MSE for each slice, using a data_range of 1.0
    data_range = torch.tensor(1.0, device=device)
    psnr_vol = 2 * torch.log(data_range) - torch.log(sum_squared_error / num_pixels_in_mask) * (10 / torch.log(torch.tensor(10.0, device=device)))

    return psnr_vol.cpu().numpy()


def mssim_per_slice(gt, rec, mask, device):
    gt, rec, mask = gt.to(device), rec.to(device), mask.to(device)
    
    # Init SSIM metric
    ssim_metric = torchmetrics.image.StructuralSimilarityIndexMeasure(data_range=1.0, **SSIM_KWARGS, return_full_image=True).to(device)
        
    rec = torch.permute(rec, (2, 0, 1))
    gt = torch.permute(gt, (2, 0, 1))

    rec = torch.unsqueeze(rec, 1)
    gt = torch.unsqueeze(gt, 1)
    mask = torch.unsqueeze(mask, 0)
    mask = torch.unsqueeze(mask, 1)

    ssim_vol = ssim_metric(rec, gt)[1]
    ssim_vol = ssim_vol * mask

    # Average SSIM over all pixels in each slice
    ssim_vol = ssim_vol.mean(dim=(2, 3))
    ssim_vol = torch.squeeze(ssim_vol, 1)

    return ssim_vol.cpu().numpy()


# -----------------------------------------------------------------------------
# PLOTTING
# -----------------------------------------------------------------------------
def save_ssim_map(
    gt, rec_list, titles, mask, tumor_slice, tumor_xy, outpath, scan_type, device
):
    n = len(rec_list) + 1
    # two rows: top = actual scans, bottom = SSIM maps
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

    # bottom row: SSIM maps with arrow
    for i in range(n):
        ax = axes[1, i]
        if i == 0:
            ax.axis("off")
        else:
            _, smap = ssim_metric(
                gt[..., tumor_slice],
                rec_list[i - 1][..., tumor_slice],
                data_range=1.0,
                **SSIM_KWARGS,
            ).cpu().numpy()
            im = ax.imshow(smap * mask.cpu().numpy(), cmap="viridis", vmin=0, vmax=1)
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
    fig.savefig(outpath, dpi=600)
    plt.close(fig)


def generate_per_scan_plot(metric_results, metric_name, num_slices, outpath):
    """
    Generates and saves a publication-quality plot for a single metric (SSIM or PSNR)
    for a single scan.
    """
    plt.style.use('seaborn-v0_8-paper')
    colors = {"FDK": "#0072B2", "PL": "#E69F00", "DDCNN": "#009E73", "FBPCONVNet": "#D55E00", "IResNet": "#CC79A7"}
    font_config = {'fontname': 'Arial', 'fontsize': 12}
    title_config = {'fontname': 'Arial', 'fontsize': 14, 'weight': 'bold'}

    fig, ax = plt.subplots(figsize=(7, 4))

    for name, values in metric_results.items():
        # Clean data by replacing inf/-inf with NaN to avoid plotting issues
        clean_values = np.copy(values)
        clean_values[np.isinf(clean_values)] = np.nan
        ax.plot(range(num_slices), clean_values, label=name, color=colors[name], linewidth=1.5)

    ylabel = "SSIM" if metric_name == "SSIM" else "PSNR (dB)"
    ax.set_ylabel(ylabel, **font_config)
    ax.set_xlabel("Slice Number", **font_config)
    ax.set_title(f"{metric_name} vs. Slice Number", **title_config)
    ax.grid(True, which='both', linestyle='--', linewidth=0.5)

    # Place legend outside the plot
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.2), ncol=len(metric_results), fancybox=True)
    fig.savefig(outpath, dpi=600, bbox_inches='tight')
    plt.close(fig)


def generate_master_plot(all_scans_data, share_mode, outpath_hf, outpath_ff):
    """
    Generates separate master plots for HF and FF scans. Each plot has two rows:
    PSNR (top) and SSIM (bottom), with shared y-axes within each row.
    """
    # Separate HF and FF scans
    hf_scans = [s for s in all_scans_data if s['scan_type'] == 'HF']
    ff_scans = [s for s in all_scans_data if s['scan_type'] == 'FF']

    def create_plot(scans, outpath, st):
        if not scans:
            print(f"No data available to generate master plot for {outpath}.")
            return

        n_scans = len(scans)
        plt.style.use('seaborn-v0_8-paper')
        colors = {"FDK": "#0072B2", "PL": "#E69F00", "DDCNN": "#009E73", "FBPCONVNet": "#D55E00", "IResNet": "#CC79A7"}
        font_config = {'fontname': 'Arial', 'fontsize': 10}

        fig, axes = plt.subplots(2, n_scans, figsize=(4 * n_scans, 4.5), sharey='row')
        fig.subplots_adjust(wspace=0.05, hspace=0.07, top=.81)

        for i, scan_data in enumerate(scans):
            # Top row: PSNR
            ax_psnr = axes[0, i]
            psnr_results = scan_data['psnr']
            for name, values in psnr_results.items():
                clean_values = np.copy(values)
                clean_values[np.isinf(clean_values)] = np.nan
                ax_psnr.plot(range(scan_data['num_slices']), clean_values, label=name, color=colors[name], linewidth=1.2)
            ax_psnr.grid(True, which='both', linestyle='--', linewidth=0.4)
            ax_psnr.set_title(f"{scan_data['scan_type']} #{i+1}", fontsize=10)
            if i == 0:
                ax_psnr.set_ylabel("PSNR (dB)", **font_config)
            ax_psnr.tick_params(axis='x', which='both', bottom=False, labelbottom=False)  # Remove x-axis ticks for the top row

            # Bottom row: SSIM
            ax_ssim = axes[1, i]
            ssim_results = scan_data['ssim']
            for name, values in ssim_results.items():
                clean_values = np.copy(values)
                clean_values[np.isinf(clean_values)] = np.nan
                ax_ssim.plot(range(scan_data['num_slices']), clean_values, label=name, color=colors[name], linewidth=1.2)
            ax_ssim.grid(True, which='both', linestyle='--', linewidth=0.4)
            if i == 0:
                ax_ssim.set_ylabel("SSIM", **font_config)

            if i != 0:
                ax_psnr.tick_params(axis='y', which='both', left=False, labelleft=False)  # Remove y-axis ticks/labels for non-labeled subplots
                ax_ssim.tick_params(axis='y', which='both', left=False, labelleft=False)  # Remove y-axis ticks/labels for non-labeled subplots
            ax_ssim.set_xlabel("Transverse Slice", **font_config)

        # Add a single, shared legend
        handles, labels = ax_psnr.get_legend_handles_labels()
        fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, 0.94),
                   ncol=len(colors), fancybox=True, fontsize=10)

        # Add a figure title
        fig.suptitle(f"{st} Image Quality Across Methods", fontsize=12, fontweight='bold')

        fig.savefig(outpath, dpi=600, bbox_inches='tight')
        fig.savefig(outpath.replace('.png', '.pdf'), dpi=600, bbox_inches='tight')
        plt.close(fig)

    # Create plots for HF and FF
    create_plot(hf_scans, outpath_hf, "HF")
    create_plot(ff_scans, outpath_ff, "FF")


# -----------------------------------------------------------------------------
# MAIN
# -----------------------------------------------------------------------------
def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    all_scans_data = []
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

            if scan_type == "FF" and pid == "16" and sid == "01":
                print(f"Skipping FF scan 16.01")
                continue

            if (scan_type, pid, sid) not in [('HF', '20', '01'), ('HF', '14', '01'), ('FF', '22', '01'), ('FF', '18', '01')]: # TODO
                print (f"Skipping scan {scan_type} {pid}.{sid}")
                continue

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

            # also load FBPCONVNet and IResNet variants
            fbp_path = glob.glob(
                os.path.join(mat_dir, f"p{pid}.{scan_type}{sid}_FBPCONVNet*_3D.pt")
            )
            if len(fbp_path) > 1:
                raise
            else:
                fbp_path = fbp_path[0]
            fbpcnn = torch.load(fbp_path, weights_only=False)
            ire_path = glob.glob(
                os.path.join(mat_dir, f"p{pid}.{scan_type}{sid}_IResNet*PL_3D.pt")
            )
            if len(ire_path) > 1:
                raise
            else:
                ire_path = ire_path[0]
            iresnet = torch.load(ire_path, weights_only=False)

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

            tloc[2] -= 20  # adjust for slice trimming

            for view in VIEWS:
                # apply axis-swaps
                gt = gt_vol.copy()
                fdk_v = fdk.copy()
                pl_v = pl.copy()
                dd_v = ddcnn.copy()
                fbpcnn_v = fbpcnn.copy()
                iresnet_v = iresnet.copy()
                t = tloc.copy()
                if view == "height":
                    gt, fdk_v, pl_v, dd_v, fbpcnn_v, iresnet_v = [
                        np.swapaxes(x, 0, 2)
                        for x in (gt, fdk_v, pl_v, dd_v, fbpcnn_v, iresnet_v)
                    ]
                    t = np.array([t[2], t[1], t[0]])
                elif view == "width":
                    gt, fdk_v, pl_v, dd_v, fbpcnn_v, iresnet_v = [
                        np.transpose(x, (2, 0, 1))
                        for x in (gt, fdk_v, pl_v, dd_v, fbpcnn_v, iresnet_v)
                    ]
                    t = np.array([t[2], t[0], t[1]])

                # use mask only for FF scans; otherwise full-true
                if scan_type == "FF":
                    mask = make_mask(gt, view)
                else:
                    mask = np.ones(gt.shape[:2], dtype=bool)
                with torch.no_grad():
                    mask = torch.tensor(mask).to(DEVICE)
                tumor_slice = int(t[2])

                # adjust SSIM window size per scan_type & view
                if scan_type == "HF":
                    SSIM_KWARGS["kernel_size"] = 15 if view == "index" else 11
                else:  # FF
                    SSIM_KWARGS["kernel_size"] = 11 if view == "index" else 7

                # normalize & clip
                # only clip/normalize GT, FDK, PL
                for arr in (gt, fdk_v, pl_v):
                    np.clip(arr, 0, 0.04, out=arr)
                    arr *= 25.
                # leave dd_v unchanged

                with torch.no_grad():
                    gt = torch.tensor(gt).float().to(DEVICE)
                    fdk_v = torch.tensor(fdk_v).float().to(DEVICE)
                    pl_v = torch.tensor(pl_v).float().to(DEVICE)
                    dd_v = torch.tensor(dd_v).float().to(DEVICE)
                    fbpcnn_v = torch.tensor(fbpcnn_v).float().to(DEVICE)
                    iresnet_v = torch.tensor(iresnet_v).float().to(DEVICE)

                # make output subdir
                odir = os.path.join(OUTPUT_DIR, f"{scan_type}_p{pid}_{sid}_{view}")
                os.makedirs(odir, exist_ok=True)

                # # 1) SSIM map
                # save_ssim_map(
                #     gt,
                #     [fdk_v, pl_v, dd_v, fbpcnn_v, iresnet_v],
                #     ["FDK", "PL", "DDCNN", "FBPCONVNet", "IResNet"],
                #     mask,
                #     tumor_slice,
                #     (t[1], t[0]),
                #     os.path.join(odir, "SSIM_map.png"),
                #     scan_type,
                #     DEVICE,
                # )

                # --- Calculate metrics for all methods ---
                methods_data = {
                    "FDK": fdk_v, "PL": pl_v, "DDCNN": dd_v,
                    "FBPCONVNet": fbpcnn_v, "IResNet": iresnet_v
                }
                
                psnr_results = {name: psnr_per_slice(gt, data, mask, DEVICE) for name, data in methods_data.items()}
                ssim_results = {name: mssim_per_slice(gt, data, mask, DEVICE) for name, data in methods_data.items()}

                # --- 1. Generate and save the per-scan plots ---
                generate_per_scan_plot(ssim_results, "SSIM", gt.shape[2], os.path.join(odir, "SSIM_per_slice.png"))
                generate_per_scan_plot(psnr_results, "PSNR", gt.shape[2], os.path.join(odir, "PSNR_per_slice.png"))

                # --- 2. Store data for the master plots ---
                # Ensure models are in a specific order for the master plot
                model_order = ["FDK", "PL", "FBPCONVNet", "IResNet", "DDCNN"]
                ssim_results = {name: ssim_results[name] for name in model_order if name in ssim_results}
                psnr_results = {name: psnr_results[name] for name in model_order if name in psnr_results}
                scan_plot_data = {
                    'ssim': ssim_results,
                    'psnr': psnr_results,
                    'scan_type': scan_type,
                    'pid': pid,
                    'sid': sid,
                    'num_slices': gt.shape[2]
                }
                all_scans_data.append(scan_plot_data)

                # --- 3. Collect summary metrics for tables (your existing logic) ---
                ps_all = {name: np.nanmean(vals) for name, vals in psnr_results.items()}
                ms_all = {name: np.nanmean(vals) for name, vals in ssim_results.items()}

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
                    "scan_type": scan_type,
                    "patient_id": pid,
                    "scan_id": sid,
                    "view": view,
                    **{f"psnr_{k}": v for k, v in ps_all.items()},
                    **{f"mssim_{k}": v for k, v in ms_all.items()},
                }
                records.append(rec)

    # --- After the main loops finish, generate the master plots ---
    if all_scans_data:
        print("--- Generating Master Comparison Plots ---")
        master_dir = os.path.join(OUTPUT_DIR, "master_plots")
        os.makedirs(master_dir, exist_ok=True)

        # Generate plots
        generate_master_plot(all_scans_data, 'global', os.path.join(master_dir, "hf.png"), os.path.join(master_dir, "ff.png"))
        print("✅ Master plots generated.")

    # write summary
    df = pd.DataFrame(records)
    df.to_csv(os.path.join(OUTPUT_DIR, "summary.csv"), index=False)

    print("Done! Outputs in", OUTPUT_DIR)


if __name__ == "__main__":
    main()