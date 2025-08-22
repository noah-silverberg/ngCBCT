#!/usr/bin/env python3
"""
Refactored script for generating 3D reconstruction comparison figures.

This script is designed to be highly configurable. The user should only need
to modify the 'USER CONFIGURATION' section to define which scans to plot,
which reconstruction methods to compare, and how to load the corresponding
data from different datasets (e.g., different duty cycles).
"""
import os
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
OUTPUT_DIR = "SABOTAGE_HF_comparison"

# --- Scan Selection ---
# How to select scans to plot. Two modes are available:
# 1. 'direct': Manually list the scans in the `DIRECT_SCANS_TO_PLOT` list.
# 2. 'agg_file': Provide a path to a scan aggregation file. The script will
#    plot all scans from the TEST section.
SCAN_SELECTION_MODE = 'direct'  # Options: 'direct', 'agg_file'

# Used if SCAN_SELECTION_MODE is 'direct'.
# List of tuples, where each tuple is (scan_type, patient_id, scan_id).
DIRECT_SCANS_TO_PLOT = [
    # ("FF", "26", "01"),
    # ("FF", "28", "03"),
    # ("FF", "29", "01"),
    ("HF", "25", "03"),
    ("HF", "27", "01"),
    ("HF", "29", "01"),
]

# Used if SCAN_SELECTION_MODE is 'agg_file'.
# Path to a text file listing scans (see `read_scans_agg_file` for format).
AGG_FILE_PATH = "scans_to_agg.txt"

# --- Tumor Location Data ---
# Paths to the .pt files containing tumor locations for HF and FF scans.
# These files should contain a tensor where `tensor[pid, sid]` gives the
# tumor location as a (z, y, x) coordinate tuple.
TUMOR_LOC_PATHS = {
    "HF": "H:/Public/Noah/tumor_location_NEW.pt",
    "FF": "H:/Public/Noah/tumor_location_FF_NEW.pt",
}

# --- Dataset Definitions ---
# Define the datasets to plot as rows in the figure. Each requires a name
# and paths for both HF and FF scans. You must provide three datasets.
# NOTE: The paths below are examples based on the original script.
#       You MUST replace them with the actual paths to your datasets.
DATASETS = [
    {
        "name": "50%\nDuty Cycle",
        "HF_FILES": Files(Directories(
            reconstructions_dir=os.path.join('H:/Public/Noah/phase7/DS13', "reconstructions"),
            images_results_dir=os.path.join('H:/Public/Noah/phase7/DS13', "results", "images"),
        )),
        "FF_FILES": Files(Directories(
            reconstructions_dir=os.path.join('H:/Public/Noah/phase7/DS14', "reconstructions"),
            images_results_dir=os.path.join('H:/Public/Noah/phase7/DS14', "results", "images"),
        )),
    },
    {
        "name": "33%\nDuty Cycle",
        "HF_FILES": Files(Directories(
            reconstructions_dir=os.path.join('H:/Public/Noah/phase7/DS13_SABOTAGE_third', "reconstructions"),
            images_results_dir=os.path.join('H:/Public/Noah/phase7/DS13_SABOTAGE_third', "results", "images"),
        )),
        "FF_FILES": Files(Directories(
            reconstructions_dir=os.path.join('H:/Public/Noah/phase7/DS14_third', "reconstructions"),
            images_results_dir=os.path.join('H:/Public/Noah/phase7/DS14_third', "results", "images"),
        )),
    },
    {
        "name": "25%\nDuty cycle",
        "HF_FILES": Files(Directories(
            reconstructions_dir=os.path.join('H:/Public/Noah/phase7/DS13_SABOTAGE_fourth', "reconstructions"),
            images_results_dir=os.path.join('H:/Public/Noah/phase7/DS13_SABOTAGE_fourth', "results", "images"),
        )),
        "FF_FILES": Files(Directories(
            reconstructions_dir=os.path.join('H:/Public/Noah/phase7/DS14_SABOTAGE_fourth', "reconstructions"),
            images_results_dir=os.path.join('H:/Public/Noah/phase7/DS14_SABOTAGE_fourth', "results", "images"),
        )),
    },
]

# --- Method & Data Loader Definitions ---
# Define the reconstruction methods to plot. Each method is a dictionary with:
#  - 'name': A string for the plot title (e.g., 'FDK', 'IResNet').
#  - 'loader': A function that takes (scan_type, pid, sid, files_obj) and
#              returns the reconstruction volume as a NumPy array.
#
# IMPORTANT:
#  - Loader functions MUST return NumPy arrays with dimensions:
#      - (160, 512, 512) for 'HF' scans (z, y, x)
#      - (160, 256, 256) for 'FF' scans (z, y, x)
#  - The script assumes the loaded data is already normalized for display.

# Placeholder function type hint
LoaderFunc = Callable[[str, str, str, Files], np.ndarray]

def load_fdk(scan_type: str, pid: str, sid: str, files_obj: Files) -> np.ndarray:
    path = files_obj.get_recon_filepath('raw', pid, sid, scan_type, gated=False, odd=True)
    recon = torch.load(path, map_location='cpu')
    recon = recon[20:-20].clone()
    if scan_type == 'FF' and recon.shape[-2:] == (512, 512):
        recon = recon[..., 128:-128, 128:-128].clone()

    recon = torch.clip(recon, 0.0, 0.04) * 25.0  # Normalize to [0, 1] range
    return recon.numpy()

def load_ddcnn(scan_type: str, pid: str, sid: str, files_obj: Files) -> np.ndarray:
    if scan_type == 'HF':
        path = files_obj.get_images_results_filepath('MK7_01', pid, sid, odd=True)
    else: # FF
        path = files_obj.get_images_results_filepath('MK7_01', pid, sid, odd=True)

    recon = torch.load(path, map_location='cpu')
    if scan_type == 'FF' and recon.shape[-2:] == (512, 512):
        recon = recon[..., 128:-128, 128:-128].clone()

    if recon.ndim == 4:
        recon = torch.squeeze(recon, dim=1)

    recon = torch.permute(recon, (0, 2, 1))
    return recon.numpy()

def load_mcdropout(scan_type: str, pid: str, sid: str, files_obj: Files) -> np.ndarray:
    if scan_type == 'HF':
        path = files_obj.get_images_results_filepath('MK7_MCDROPOUT_30_pct_NEW', pid, sid, passthrough_num=0, odd=True)
    else: # FF
        path = files_obj.get_images_results_filepath('MK7_MCDROPOUT_15_pct', pid, sid, passthrough_num=0, odd=True)

    recon = torch.load(path, map_location='cpu')
    if scan_type == 'FF' and recon.shape[-2:] == (512, 512):
        recon = recon[..., 128:-128, 128:-128].clone()

    if recon.ndim == 4:
        recon = torch.squeeze(recon, dim=1)
    recon = torch.permute(recon, (0, 2, 1))
    return recon.numpy()

def load_bbb(scan_type: str, pid: str, sid: str, files_obj: Files) -> np.ndarray:
    if scan_type == 'HF':
        path = files_obj.get_images_results_filepath('MK7_BBB_pi0.75_mu_0.0_sigma1_1e-1_sigma2_1e-3_beta_1e-2', pid, sid, passthrough_num=0, odd=True)
    else: # FF
        path = files_obj.get_images_results_filepath('MK7_BBB_pi0.5_mu_0.0_sigma1_1e-2_sigma2_3e-3_beta_1e-2', pid, sid, passthrough_num=0, odd=True)

    recon = torch.load(path, map_location='cpu')
    if scan_type == 'FF' and recon.shape[-2:] == (512, 512):
        recon = recon[..., 128:-128, 128:-128].clone()

    if recon.ndim == 4:
        recon = torch.squeeze(recon, dim=1)
    recon = torch.permute(recon, (0, 2, 1))
    return recon.numpy()

def load_evidential(scan_type: str, pid: str, sid: str, files_obj: Files) -> np.ndarray:
    if scan_type == 'HF':
        path = files_obj.get_images_results_filepath('MK7_EVIDENTIAL_nll1e0_reg1e-2_WUreg0_YEreg1e-2_NUreg1e-3_BETAreg1e-4_smooth1e0_ANNEAL', pid, sid, odd=True)
    else: # FF
        path = files_obj.get_images_results_filepath('MK7_EVIDENTIAL_nll1e0_reg1e-3_WUreg0_YEreg1e-2_NUreg3e-4_BETAreg1e-4_smooth1e0_ANNEAL', pid, sid, odd=True)

    recon = torch.load(path, map_location='cpu')['gamma']
    if scan_type == 'FF' and recon.shape[-2:] == (512, 512):
        recon = recon[..., 128:-128, 128:-128].clone()

    if recon.ndim == 4:
        recon = torch.squeeze(recon, dim=1)
    recon = torch.permute(recon, (0, 2, 1))
    return recon.numpy()


METHODS_TO_PLOT: List[Dict[str, Union[LoaderFunc, str]]] = [
    {'name': 'FDK', 'loader': load_fdk},
    {'name': 'DDCNN', 'loader': load_ddcnn},
    {'name': 'MC Dropout', 'loader': load_mcdropout},
    {'name': 'BBB', 'loader': load_bbb},
    {'name': 'Evidential', 'loader': load_evidential},
]


# --- Arrow Customization ---
# Arrow points to the tumor. Parameters are defined by offsets from the tumor's center pixel.
# 'tail_offset': (dx, dy) for the arrow's tail.
# 'head_offset': (dx, dy) for the arrow's head (tip).
# 'lw': line width.
DEFAULT_ARROW_PARAMS = {
    'HF': {'tail_offset': (-55, -55), 'head_offset': (-15, -15), 'lw': 3},
    'FF': {'tail_offset': (-27, -27), 'head_offset': (-7, -7), 'lw': 3},
}

# Add custom arrow parameters for specific scans if defaults are not ideal.
# Key: (scan_type, pid, sid), Value: dict mapping view to params.
# NOTE: Only the "index" view is used, but other views are kept for reference.
CUSTOM_ARROW_PARAMS = {
    ("FF", "26", "01"): {
        "index":  {'tail_offset': (-27, -27), 'head_offset': (-4, -4), 'lw': 3},
    },
    ("FF", "28", "03"): {
        "index":  {'tail_offset': (-35, -35), 'head_offset': (-7, -7), 'lw': 3},
    },
    ("FF", "29", "01"): {
        "index":  {'tail_offset': (-35, 35), 'head_offset': (-7, 7), 'lw': 3},
    },
    ("HF", "25", "03"): {
        "index":  {'tail_offset': (-65, 45), 'head_offset': (-15, 10), 'lw': 3},
    },
    ("HF", "27", "01"): {
        "index":  {'tail_offset': (-55, 47), 'head_offset': (-14, 14), 'lw': 3},
    },
    ("HF", "29", "01"): {
        "index":  {'tail_offset': (-30, 70), 'head_offset': (-1, 9), 'lw': 3},
    },
}

# --- Plot Layout & Views ---
VIEW = "index"  # Only the axial view is plotted

# Per-scan crop regions for the axial view: (y0, y1, x0, x1)
CROPS = {
    ("FF", "26", "01"): {"index": (8, 256 - 75, 8, 256 - 8)},
    ("FF", "28", "03"): {"index": (8, 256 - 8, 8, 256 - 8)},
    ("FF", "29", "01"): {"index": (8, 256 - 40, 8, 256 - 8)},
    ("HF", "25", "03"): {"index": (45, 512 - 205, 120, 512 - 15)},
    ("HF", "27", "01"): {"index": (100, 512 - 195, 95, 512 - 80)},
    ("HF", "29", "01"): {"index": (30, 512 - 165, 30, 512 - 30)},
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
        scans = scans_agg["TEST"]
        print(f"Found {len(scans)} scans to plot in test set.")
        return scans
    else:
        raise ValueError(f"Invalid SCAN_SELECTION_MODE: {SCAN_SELECTION_MODE}")

def load_tumor_location(scan_type: str, pid: str, sid: str) -> Tuple[int, int, int]:
    """Loads and adjusts the tumor (z, y, x) coordinates for a given scan."""
    path = TUMOR_LOC_PATHS.get(scan_type)
    if not path or not os.path.exists(path):
        raise FileNotFoundError(f"Tumor location file for type '{scan_type}' not found at '{path}'")

    tlocs = torch.load(path, weights_only=False)
    pid_idx, sid_idx = int(pid), int(sid)

    t = list(tlocs[pid_idx, sid_idx])
    t = [t[2], t[0], t[1]]
    t[0] -= 20  # Adjust for z-slice cropping

    if scan_type == 'FF':
        t[1] -= 128
        t[2] -= 128

    return tuple(t)  # (z, y, x)

def get_arrow_params(scan: Tuple, view: str) -> Dict:
    """Gets arrow parameters for a scan, preferring custom over default."""
    if scan in CUSTOM_ARROW_PARAMS and view in CUSTOM_ARROW_PARAMS[scan]:
        return CUSTOM_ARROW_PARAMS[scan][view]
    scan_type = scan[0]
    return DEFAULT_ARROW_PARAMS[scan_type]

def extract_view(vol: np.ndarray, tloc: Tuple, view: str) -> np.ndarray:
    """Extracts a 2D slice from a 3D volume (z, y, x) based on the view."""
    z, y, x = tloc
    if view == "index":   # Axial view (y-x plane) at tumor's z
        return vol[z, :, :]
    # Other views removed as they are no longer used
    raise ValueError(f"Unknown or unused view: {view}")

def plot_scan(scan: Tuple[str, str, str]):
    """Generates and saves a comparison figure for a single scan."""
    scan_type, pid, sid = scan
    print(f"\nProcessing scan: {scan_type} p{pid}_{sid}...")

    tloc = load_tumor_location(scan_type, pid, sid)
    
    nrows = len(DATASETS)
    ncols = len(METHODS_TO_PLOT)
    
    # Determine aspect ratio from crop settings
    aspect_ratio = 1.0
    crop_coords = CROPS.get(scan, {}).get(VIEW)
    if crop_coords:
        y0, y1, x0, x1 = crop_coords
        if (x1 - x0) > 0: # Avoid division by zero
            aspect_ratio = (y1 - y0) / (x1 - x0)

    # --- New, robust figsize calculation ---
    # You can tweak these two values to customize the final look
    subplot_base_width_inches = 3.5  # The width of each individual image
    top_padding_inches = 0.6         # Extra space at the top for titles

    # Calculate final figure dimensions
    fig_width = subplot_base_width_inches * ncols
    image_area_height = nrows * subplot_base_width_inches * aspect_ratio
    fig_height = image_area_height + top_padding_inches

    fig, axes = plt.subplots(
        nrows, ncols,
        figsize=(fig_width, fig_height),
        facecolor="black",
        squeeze=False,
    )
    fig.patch.set_facecolor("black")

    for i, dataset in enumerate(DATASETS):
        # Determine which file collection to use (HF or FF)
        files_obj = dataset['HF_FILES'] if scan_type == 'HF' else dataset['FF_FILES']

        for j, method in enumerate(METHODS_TO_PLOT):
            ax = axes[i, j]
            
            # Load the volume for the current dataset and method
            vol = method['loader'](scan_type, pid, sid, files_obj)
            sl = extract_view(vol, tloc, VIEW)

            # Apply cropping if specified
            if crop_coords:
                y0, y1, x0, x1 = crop_coords
                sl = sl[y0:y1, x0:x1]

            ax.imshow(sl, cmap="gray", vmin=0, vmax=1)
            ax.axis("off")  # Hide axes

            # Add method titles to the top row
            if i == 0:
                ax.set_title(method['name'], fontsize=22, color="white", weight="bold", pad=20)

            # Draw arrow on the first column (FDK) to indicate tumor location
            ty, tx = tloc[1], tloc[2]  # Axial view is y-x
            if crop_coords:
                y0, _, x0, _ = crop_coords
                ty -= y0
                tx -= x0

            params = get_arrow_params(scan, VIEW)
            tail_dx, tail_dy = params['tail_offset']
            head_dx, head_dy = params['head_offset']

            ax.annotate(
                "",
                xy=(tx + head_dx, ty + head_dy),
                xytext=(tx + tail_dx, ty + tail_dy),
                arrowprops=dict(color="red", arrowstyle="->", lw=params['lw']),
            )
    
    # Adjust layout to be perfectly snug
    fig.subplots_adjust(
        left=0.055, right=0.995, bottom=0.01, 
        top=image_area_height / fig_height, 
        wspace=0.01, hspace=0.0
    )

    # Add labels AFTER the layout has been finalized to ensure they are centered
    for i, ax in enumerate(axes[:, 0]):
        pos = ax.get_position()
        fig.text(
            0.025,  # X-position from left edge
            pos.y0 + pos.height / 2, # Y-position is the vertical center of the FINAL axis position
            DATASETS[i]['name'],
            color='white',
            weight='bold',
            fontsize=22,
            rotation=90,
            ha='center',
            va='center'
        )

    # Save the figure
    outname_base = f"{scan_type}_p{pid}_{sid}_comparison"
    outname_png = f"{outname_base}.png"
    outname_pdf = f"{outname_base}.pdf"
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