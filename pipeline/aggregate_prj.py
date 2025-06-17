import torch
import os
from .config import DATA_DIR
from .utils import ensure_dir
from dsets import PrjSet


def aggregate_saved_projections(scan_type: str, sample: str):
    """
    Load per-scan projection tensors, concatenate across scans, and save aggregates.

    Args:
        scan_type (str): Type of scan, e.g. 'HF', 'FF'.
        sample (str): Data split name, e.g. 'train', 'validation', 'test'.

    Returns:
        None: Side effect—saves two files under DATA_DIR/agg:
            - {scan_type}_{sample}_gated.pt: torch.Tensor of shape (N_gated * H, 1, v_dim, 512)
            - {scan_type}_{sample}_ng.pt:    torch.Tensor of shape (N_ng * H, 1, v_dim, 512)
    """
    # Gated and nonstop-gated subdirectories
    g_dir = os.path.join(DATA_DIR, "gated")
    ng_dir = os.path.join(DATA_DIR, "ng")

    # Directory for aggregated data saving
    agg_dir = os.path.join(DATA_DIR, "agg")
    ensure_dir(agg_dir)

    # Create ground truth dataset, and concatenate all scans into one tensor
    # along the H dimension (i.e., the dimension where we already stacked them before saving -- see "divide_sinogram" in proj.py)
    truth_set = PrjSet(g_dir)
    prj_gcbct = truth_set[0]
    for idx in range(1, len(truth_set)):
        prj_gcbct = torch.cat((prj_gcbct, truth_set[idx]), dim=0)

    # Save the aggregated dataset
    torch.save(prj_gcbct, os.path.join(agg_dir, f"{scan_type}_{sample}_gated.pt"))

    # Repeat for nonstop-gated projections
    ns_set = PrjSet(ng_dir)
    prj_ngcbct = ns_set[0]
    for idx in range(1, len(ns_set)):
        prj_ngcbct = torch.cat((prj_ngcbct, ns_set[idx]), dim=0)
    torch.save(prj_ngcbct, os.path.join(agg_dir, f"{scan_type}_{sample}_ng.pt"))
