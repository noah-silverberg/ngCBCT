#!/usr/bin/env python3
"""
Refactored script for generating 3D reconstruction comparison figures.

This script is designed to be highly configurable. The user should only need
to modify the 'USER CONFIGURATION' section to define which scans to plot,
which reconstruction methods to compare, and how to load the corresponding
data.
"""
import os
import re
import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from typing import List, Dict, Tuple, Callable, Union
from pipeline.paths import Directories, Files
from pipeline.utils import read_scans_agg_file

# =============================================================================
# -------------------------- USER CONFIGURATION -------------------------------
# =============================================================================

# --- General Paths ---
# Directory where the output figures (PNG, PDF) will be saved.
OUTPUT_DIR = "UQ_reconstruction_comparisons_PAPER"

# --- Scan Selection ---
# How to select scans to plot. Two modes are available:
# 1. 'direct': Manually list the scans in the `DIRECT_SCANS_TO_PLOT` list.
# 2. 'agg_file': Provide a path to a scan aggregation file. The script will
#    plot all scans from the TEST section.
SCAN_SELECTION_MODE = 'direct'  # Options: 'direct', 'agg_file'

# Used if SCAN_SELECTION_MODE is 'direct'.
# List of tuples, where each tuple is (scan_type, patient_id, scan_id).
DIRECT_SCANS_TO_PLOT = [
    ("FF", "26", "01"),
    ("FF", "28", "03"),
    ("FF", "29", "01"),
    ("HF", "25", "03"),
    ("HF", "27", "01"),
    ("HF", "29", "01"),
]

# Used if SCAN_SELECTION_MODE is 'agg_file'.
# Path to a text file listing scans (see `_read_scans_agg_file` for format).
AGG_FILE_PATH = "scans_to_agg.txt"

# --- Tumor Location Data ---
# Paths to the .pt files containing tumor locations for HF and FF scans.
# These files should contain a tensor where `tensor[pid, sid]` gives the
# tumor location as a (z, y, x) coordinate tuple.
TUMOR_LOC_PATHS = {
    "HF": "H:/Public/Noah/tumor_location_NEW.pt",
    "FF": "H:/Public/Noah/tumor_location_FF_NEW.pt",
}

# --- Scan Time Data ---
# Scan times for Gated and Nonstop acquisitions.
# TODO: Please replace these placeholder values with the actual data.
SCAN_TIMES = {
    # (scan_type, pid, sid): {"gated": "X min", "nonstop": "Y min"}
    ("FF", "26", "01"): {"gated": "???? min", "nonstop": "0.56 min"},
    ("FF", "28", "03"): {"gated": "???? min", "nonstop": "0.56 min"},
    ("FF", "29", "01"): {"gated": "???? min", "nonstop": "0.56 min"},
    ("HF", "25", "03"): {"gated": "???? min", "nonstop": "1 min"},
    ("HF", "27", "01"): {"gated": "???? min", "nonstop": "1 min"},
    ("HF", "29", "01"): {"gated": "???? min", "nonstop": "1 min"},
}

# --- Method & Data Loader Definitions ---
# Define the reconstruction methods to plot. Each method is a dictionary with:
#  - 'name': A string for the plot title (e.g., 'FDK', 'IResNet').
#  - 'loader': A function that takes (scan_type, pid, sid) and returns the
#              reconstruction volume as a NumPy array.
#
# IMPORTANT:
#  - The first method in the list is always treated as the ground truth
#    (Gated CBCT) and will have a dashed box drawn around it.
#  - Loader functions MUST return NumPy arrays with dimensions:
#      - (160, 512, 512) for 'HF' scans (z, y, x)
#      - (160, 256, 256) for 'FF' scans (z, y, x)
#  - The script assumes the loaded data is already normalized for display.
#  - You MUST implement the loader functions in the section below.

# Placeholder function type hint
LoaderFunc = Callable[[str, str, str], np.ndarray]

HF_DIRECTORIES = Directories(
    reconstructions_dir=os.path.join('H:\Public/Noah/phase7/DS13', "reconstructions"),
    reconstructions_gated_dir=os.path.join("H:\Public/Noah", "gated", "fdk_recon"),
    images_results_dir=os.path.join('H:\Public/Noah/phase7/DS13', "results", "images"),
)
HF_FILES = Files(HF_DIRECTORIES)
FF_DIRECTORIES = Directories(
    reconstructions_dir=os.path.join('H:\Public/Noah/phase7/DS14', "reconstructions"),
    reconstructions_gated_dir=os.path.join("H:\Public/Noah", "gated", "fdk_recon"),
    images_results_dir=os.path.join('H:\Public/Noah/phase7/DS14', "results", "images"),
)
FF_FILES = Files(FF_DIRECTORIES)

def load_gated_cbct(scan_type: str, pid: str, sid: str) -> np.ndarray:
    """USER-DEFINED: Loads the ground truth Gated CBCT volume."""
    # --- Example Implementation ---
    # print(f"Loading Gated CBCT for {scan_type} p{pid}_{sid}...")
    # if scan_type == 'HF':
    #     return np.random.rand(160, 512, 512)
    # else: # FF
    #     return np.random.rand(160, 256, 256)
    # --------------------------
    if scan_type == 'HF':
        path = HF_FILES.get_recon_filepath('fdk', pid, sid, 'HF', gated=True)
        recon = torch.load(path, map_location='cpu')
        recon = recon[20:-20].clone()
    else:
        path = FF_FILES.get_recon_filepath('fdk', pid, sid, 'FF', gated=True)
        recon = torch.load(path, map_location='cpu')
        recon = recon[20:-20].clone()
        if recon.shape[-2:] == (512, 512):
            recon = recon[:, 128:-128, 128:-128].clone()

    recon = torch.clip(recon, 0.0, 0.04) * 25.0  # Normalize to [0, 1] range
    return recon.numpy()

def load_fdk(scan_type: str, pid: str, sid: str) -> np.ndarray:
    if scan_type == 'HF':
        path = HF_FILES.get_recon_filepath('raw', pid, sid, 'HF', gated=False)
        recon = torch.load(path, map_location='cpu')
        recon = recon[20:-20].clone()
    else:
        path = FF_FILES.get_recon_filepath('raw', pid, sid, 'FF', gated=False)
        recon = torch.load(path, map_location='cpu')
        recon = recon[20:-20].clone()
        if recon.shape[-2:] == (512, 512):
            recon = recon[..., 128:-128, 128:-128].clone()

    recon = torch.clip(recon, 0.0, 0.04) * 25.0  # Normalize to [0, 1] range

    return recon.numpy()

def load_ddcnn(scan_type: str, pid: str, sid: str) -> np.ndarray:
    if scan_type == 'HF':
        path = HF_FILES.get_images_results_filepath('MK7_01', pid, sid)
        recon = torch.load(path, map_location='cpu')
    else:
        path = FF_FILES.get_images_results_filepath('MK7_01', pid, sid)
        recon = torch.load(path, map_location='cpu')
        if recon.shape[-2:] == (512, 512):
            recon = recon[..., 128:-128, 128:-128].clone()

    if recon.ndim == 4:
        recon = torch.squeeze(recon, dim=1)

    recon = torch.permute(recon, (0, 2, 1))
    return recon.numpy()

def load_mcdropout(scan_type: str, pid: str, sid: str) -> np.ndarray:
    if scan_type == 'HF':
        path = HF_FILES.get_images_results_filepath('MK7_MCDROPOUT_30_pct_NEW', pid, sid, passthrough_num=0)
        recon = torch.load(path, map_location='cpu')
    else:
        path = FF_FILES.get_images_results_filepath('MK7_MCDROPOUT_15_pct', pid, sid, passthrough_num=0)
        recon = torch.load(path, map_location='cpu')
        if recon.shape[-2:] == (512, 512):
            recon = recon[..., 128:-128, 128:-128].clone()

    if recon.ndim == 4:
        recon = torch.squeeze(recon, dim=1)
    recon = torch.permute(recon, (0, 2, 1))
    return recon.numpy()

def load_bbb(scan_type: str, pid: str, sid: str) -> np.ndarray:
    if scan_type == 'HF':
        path = HF_FILES.get_images_results_filepath('MK7_BBB_pi0.75_mu_0.0_sigma1_1e-1_sigma2_1e-3_beta_1e-2', pid, sid, passthrough_num=0)
        recon = torch.load(path, map_location='cpu')
    else:
        path = FF_FILES.get_images_results_filepath('MK7_BBB_pi0.5_mu_0.0_sigma1_1e-2_sigma2_3e-3_beta_1e-2', pid, sid, passthrough_num=0)
        recon = torch.load(path, map_location='cpu')
        if recon.shape[-2:] == (512, 512):
            recon = recon[..., 128:-128, 128:-128].clone()

    if recon.ndim == 4:
        recon = torch.squeeze(recon, dim=1)
    recon = torch.permute(recon, (0, 2, 1))
    return recon.numpy()

def load_evidential(scan_type: str, pid: str, sid: str) -> np.ndarray:
    if scan_type == 'HF':
        path = HF_FILES.get_images_results_filepath('MK7_EVIDENTIAL_nll1e0_reg1e-2_WUreg0_YEreg1e-2_NUreg1e-3_BETAreg1e-4_smooth1e0_ANNEAL', pid, sid)
        recon = torch.load(path, map_location='cpu')['gamma']
    else:
        path = FF_FILES.get_images_results_filepath('MK7_EVIDENTIAL_nll1e0_reg1e-3_WUreg0_YEreg1e-2_NUreg3e-4_BETAreg1e-4_smooth1e0_ANNEAL', pid, sid)
        recon = torch.load(path, map_location='cpu')['gamma']
        if recon.shape[-2:] == (512, 512):
            recon = recon[..., 128:-128, 128:-128].clone()

    if recon.ndim == 4:
        recon = torch.squeeze(recon, dim=1)
    recon = torch.permute(recon, (0, 2, 1))
    return recon.numpy()

METHODS_TO_PLOT: List[Dict[str, Union[LoaderFunc, str]]] = [
    {'name': 'Gated CBCT', 'loader': load_gated_cbct},
    {'name': 'FDK', 'loader': load_fdk},
    {'name': 'DDCNN', 'loader': load_ddcnn},
    {'name': 'MC Dropout', 'loader': load_mcdropout},
    {'name': 'BBB', 'loader': load_bbb},
    {'name': 'Evidential', 'loader': load_evidential},
]


# --- Arrow Customization ---
# Arrow parameters are defined by offsets from the tumor's center pixel.
# 'tail_offset': (dx, dy) for the arrow's tail.
# 'head_offset': (dx, dy) for the arrow's head (tip).
# 'lw': line width.
DEFAULT_ARROW_PARAMS = {
    'HF': {'tail_offset': (-55, -55), 'head_offset': (-15, -15), 'lw': 3},
    'FF': {'tail_offset': (-27, -27), 'head_offset': (-7, -7), 'lw': 3},
}

# Add custom arrow parameters for specific scans if defaults are not ideal.
# Key: (scan_type, pid, sid), Value: dict mapping view to params.
CUSTOM_ARROW_PARAMS = {
    ("FF", "26", "01"): {
        "index":  {'tail_offset': (-27, -27), 'head_offset': (-4, -4), 'lw': 3},
        "width":  {'tail_offset': (-22, -30), 'head_offset': (-4, -7), 'lw': 3},
        "height": {'tail_offset': (-30, -30), 'head_offset': (-7, -7), 'lw': 3},
    },
    ("FF", "28", "03"): {
        "index":  {'tail_offset': (-35, -35), 'head_offset': (-7, -7), 'lw': 3},
        "width":  {'tail_offset': (-32, -32), 'head_offset': (-4, -4), 'lw': 3},
        "height": {'tail_offset': (-35, -35), 'head_offset': (-7, -7), 'lw': 3},
    },
    ("FF", "29", "01"): {
        "index":  {'tail_offset': (-35, 35), 'head_offset': (-7, 7), 'lw': 3},
        "width":  {'tail_offset': (10, 40), 'head_offset': (3, 7), 'lw': 3},
        "height": {'tail_offset': (-44, 25), 'head_offset': (-5, 6), 'lw': 3},
    },
    ("HF", "25", "03"): {
        "index":  {'tail_offset': (-65, 45), 'head_offset': (-15, 10), 'lw': 3},
        "width":  {'tail_offset': (-55, 55), 'head_offset': (-10, 10), 'lw': 3},
        "height": {'tail_offset': (-10, -65), 'head_offset': (-2, -15), 'lw': 3},
    },
    ("HF", "27", "01"): {
        "index":  {'tail_offset': (-55, 47), 'head_offset': (-14, 14), 'lw': 3},
        "width":  {'tail_offset': (13, 48), 'head_offset': (11, 9), 'lw': 3},
        "height": {'tail_offset': (20, 58), 'head_offset': (3, 14), 'lw': 3},
    },
    ("HF", "29", "01"): {
        "index":  {'tail_offset': (-30, 70), 'head_offset': (-1, 9), 'lw': 3},
        "width":  {'tail_offset': (5, 55), 'head_offset': (3, 4), 'lw': 3},
        "height": {'tail_offset': (15, -60), 'head_offset': (7, -7), 'lw': 3},
    },
}

# --- Plot Layout & Views ---
VIEWS = ["index", "width", "height"]

# Per-scan crop regions for each view: (y0, y1, x0, x1)
CROPS = {
    ("FF", "26", "01"): {
        "index":  (8, 256 - 75, 8, 256 - 8),
        "width":  (0, 160 - 0, 8, 256 - 72),
        "height": (0, 160 - 0, 8, 256 - 8),
    },
    ("FF", "28", "03"): {
        "index":  (8, 256 - 8, 8, 256 - 8),
        "width":  (0, 160 - 0, 8, 256 - 8),
        "height": (0, 160 - 0, 8, 256 - 8),
    },
    ("FF", "29", "01"): {
        "index":  (8, 256 - 40, 8, 256 - 8),
        "width":  (0, 160 - 0, 8, 256 - 35),
        "height": (0, 160 - 0, 8, 256 - 8),
    },
    ("HF", "25", "03"): {
        "index":  (45, 512 - 205, 120, 512 - 15),
        "width":  (0, 160 - 0, 40, 512 - 200),
        "height": (0, 160 - 0, 135, 512 - 20),
    },
    ("HF", "27", "01"): {
        "index":  (100, 512 - 195, 95, 512 - 80),
        "width":  (0, 160 - 0, 100, 512 - 185),
        "height": (0, 160 - 0, 105, 512 - 80),
    },
    ("HF", "29", "01"): {
        "index":  (30, 512 - 165, 30, 512 - 30),
        "width":  (0, 160 - 0, 15, 512 - 160),
        "height": (0, 160 - 0, 20, 512 - 25),
    },
}

# =============================================================================
# ------------------------- SCRIPT CORE LOGIC ---------------------------------
# (You should not need to modify below this line)
# =============================================================================

def get_scans_to_plot() -> List[Tuple[str, str, str]]:
    """Determines which scans to plot based on the configuration."""
    if SCAN_SELECTION_MODE == 'direct':
        print(f"Using directly defined scans: {DIRECT_SCANS_TO_PLOT}")
        return DIRECT_SCANS_TO_PLOT
    elif SCAN_SELECTION_MODE == 'agg_file':
        print(f"Reading scans from aggregation file: {AGG_FILE_PATH}")
        scans_agg, _ = read_scans_agg_file(AGG_FILE_PATH)
        # Only plot test scans
        scans = scans_agg["TEST"]
        print(f"Found {len(scans)} scans to plot in validation/test sets.")
        return scans
    else:
        raise ValueError(f"Invalid SCAN_SELECTION_MODE: {SCAN_SELECTION_MODE}")

def load_tumor_location(scan_type: str, pid: str, sid: str) -> Tuple[int, int, int]:
    """Loads and adjusts the tumor (z, y, x) coordinates for a given scan."""
    path = TUMOR_LOC_PATHS.get(scan_type)
    if not path or not os.path.exists(path):
        raise FileNotFoundError(f"Tumor location file for type '{scan_type}' not found at '{path}'")

    tlocs = torch.load(path, weights_only=False)
    # Convert patient/scan IDs to integers for indexing
    pid_idx, sid_idx = int(pid), int(sid)

    # Assuming tlocs tensor is indexed by [pid, sid]
    t = list(tlocs[pid_idx, sid_idx])
    t = [t[2], t[0], t[1]]

    # The data is sliced from z=20 to z=180 (160 slices total)
    # Adjust the z-coordinate to match the sliced volume.
    t[0] -= 20

    if scan_type == 'FF':
        t[1] -= 128
        t[2] -= 128

    return tuple(t) # (z, y, x)

def get_arrow_params(scan: Tuple, view: str) -> Dict:
    """Gets arrow parameters for a scan, preferring custom over default."""
    # Check for a specific override for this scan and view
    if scan in CUSTOM_ARROW_PARAMS and view in CUSTOM_ARROW_PARAMS[scan]:
        return CUSTOM_ARROW_PARAMS[scan][view]
    # Fall back to the default for the scan type
    scan_type = scan[0]
    return DEFAULT_ARROW_PARAMS[scan_type]

def extract_view(vol: np.ndarray, tloc: Tuple, view: str) -> np.ndarray:
    """
    Extracts a 2D slice from a 3D volume (z, y, x) based on the view.

    Args:
        vol (np.ndarray): The 3D volume with shape (z, y, x).
        tloc (tuple): The tumor location (z, y, x).
        view (str): One of "index" (axial), "height" (coronal), "width" (sagittal).

    Returns:
        np.ndarray: The corresponding 2D slice.
    """
    z, y, x = tloc
    if view == "index":   # Axial view (y-x plane) at tumor's z
        return vol[z, :, :]
    elif view == "height":  # Coronal view (z-x plane) at tumor's y
        return vol[:, y, :]
    elif view == "width":   # Sagittal view (z-y plane) at tumor's x
        return vol[:, :, x]
    else:
        raise ValueError(f"Unknown view: {view}")

def plot_scan(scan: Tuple[str, str, str]):
    """
    Generates and saves a comparison figure for a single scan.
    """
    scan_type, pid, sid = scan
    print(f"\nProcessing scan: {scan_type} p{pid}_{sid}...")

    # Load all required data
    tloc = load_tumor_location(scan_type, pid, sid)
    vols = [m['loader'](scan_type, pid, sid) for m in METHODS_TO_PLOT]
    names = [m['name'] for m in METHODS_TO_PLOT]

    # --- Automatic Layout Calculation ---
    ncols = len(vols)
    nrows = len(VIEWS)
    heights = []
    for view in VIEWS:
        crop = CROPS.get(scan, {}).get(view)
        if crop:
            y0, y1, x0, x1 = crop
            height = (y1 - y0) / (x1 - x0) if (x1 - x0) > 0 else 1.0
            heights.append(height)
        else:
            heights.append(1.0)

    # Define base parameters for sizing
    subplot_base_width_inches = 3.0
    top_padding_inches = 1.2 # Increased padding for two rows of titles

    # Calculate final figure dimensions
    fig_width = subplot_base_width_inches * ncols
    image_area_height = subplot_base_width_inches * sum(heights)
    fig_height = image_area_height + top_padding_inches

    fig, axes = plt.subplots(
        nrows, ncols,
        figsize=(fig_width, fig_height),
        facecolor="black",
        gridspec_kw={"height_ratios": heights},
        squeeze=False,
    )
    fig.patch.set_facecolor("black")

    # --- Plotting Loop ---
    for i, view in enumerate(VIEWS):
        for j, (vol, name) in enumerate(zip(vols, names)):
            ax = axes[i, j]
            sl = extract_view(vol, tloc, view)

            if CROPS.get(scan, {}).get(view):
                y0, y1, x0, x1 = CROPS[scan][view]
                sl = sl[y0:y1, x0:x1]

            ax.imshow(sl, cmap="gray", vmin=0, vmax=1)

            # Draw arrow on the first (ground truth) column
            if j == 0:
                # Determine tumor pixel coords (y, x) for this view
                if view == "index":   # Axial view is y-x, tumor at (tloc_y, tloc_x)
                    ty, tx = tloc[1], tloc[2]
                elif view == "height":  # Coronal view is z-x, tumor at (tloc_z, tloc_x)
                    ty, tx = tloc[0], tloc[2]
                else:  # Sagittal view is z-y, tumor at (tloc_z, tloc_y)
                    ty, tx = tloc[0], tloc[1]

                if CROPS.get(scan, {}).get(view):
                    y0, y1, x0, x1 = CROPS[scan][view]
                    ty -= y0
                    tx -= x0

                params = get_arrow_params(scan, view)
                tail_dx, tail_dy = params['tail_offset']
                head_dx, head_dy = params['head_offset']

                ax.annotate(
                    "",
                    xy=(tx + head_dx, ty + head_dy), # Arrow head
                    xytext=(tx + tail_dx, ty + tail_dy), # Arrow tail
                    arrowprops=dict(color="white", arrowstyle="->", lw=params['lw']),
                )

            if i == 0 and j > 0: # Set titles for nonstop gated columns
                ax.set_title(name, fontsize=20, color="white", weight="bold")
            ax.axis("off")

    # --- Final Layout Adjustments and Titling ---
    top_of_plots = image_area_height / fig_height
    fig.subplots_adjust(
        left=0.01, right=0.99,
        top=top_of_plots,
        bottom=0.02, wspace=0.01,
        hspace=0.03,
    )

    # Add titles AFTER layout is finalized to ensure correct positioning
    top_margin_height = 1.0 - top_of_plots
    main_title_y = top_of_plots + top_margin_height * 0.63
    scan_time_y = top_of_plots + top_margin_height * 0.40

    # Gated Title and Time
    gated_ax_pos = axes[0, 0].get_position()
    gated_center_x = gated_ax_pos.x0 + gated_ax_pos.width / 2.0
    fig.text(x=gated_center_x, y=main_title_y, s=names[0], color="white", weight="bold", ha="center", fontsize=24)

    # Nonstop Title and Time
    nonstop_start_pos = axes[0, 1].get_position()
    nonstop_end_pos = axes[0, -1].get_position()
    nonstop_center_x = (nonstop_start_pos.x0 + nonstop_end_pos.x1) / 2.0
    fig.text(x=nonstop_center_x, y=main_title_y, s="Nonstop-Gated CBCT", color="white", weight="bold", ha="center", fontsize=24)

    times = SCAN_TIMES.get(scan)
    if times:
        fig.text(x=gated_center_x, y=scan_time_y, s=f"Scan Time: {times['gated']}", color="white", ha="center", fontsize=16)
        fig.text(x=nonstop_center_x, y=scan_time_y, s=f"Scan Time: {times['nonstop']}", color="white", ha="center", fontsize=16)


    # Draw dashed box around the first (gated) column to include titles
    pos_tl = axes[0, 0].get_position()
    pos_bl = axes[-1, 0].get_position()

    # Define the box boundaries in figure coordinates
    box_x = pos_tl.x0 - 0.005
    box_width = pos_tl.width + 0.01
    
    # Start the box halfway between the bottom of the figure and the bottom image
    box_y = pos_bl.y0 / 2.0 
    
    # Extend the box from its bottom to the very top of the figure
    box_height = 0.99 - box_y 

    fig.patches.append(patches.Rectangle(
        (box_x, box_y), box_width, box_height,
        transform=fig.transFigure, fill=False, edgecolor="white",
        linestyle="--", linewidth=2, clip_on=False
    ))

    # Save the figure
    outname_png = f"{scan_type}_p{pid}_{sid}.png"
    outname_pdf = f"{scan_type}_p{pid}_{sid}.pdf"
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    fig.savefig(os.path.join(OUTPUT_DIR, outname_png), dpi=600, facecolor='black')
    fig.savefig(os.path.join(OUTPUT_DIR, outname_pdf), dpi=600, facecolor='black')
    plt.close(fig)
    print(f"Saved: {outname_png} and {outname_pdf}")

def main():
    """Main execution function."""
    scans_to_run = get_scans_to_plot()
    if not scans_to_run:
        print("No scans selected for plotting. Please check the configuration.")
        return

    for scan_details in scans_to_run:
        plot_scan(scan_details)

if __name__ == "__main__":
    main()