import torch
import os
from .config import DATA_DIR
from .utils import ensure_dir
from .dsets import PrjSet


def aggregate_saved_projections(scan_type: str, sample: str):
    """
    Load per-scan projection tensors, concatenate across scans, and save aggregates.

    Args:
        scan_type (str): Type of scan, e.g. 'HF', 'FF'.
        sample (str): Data split name, e.g. 'train', 'validation', 'test'.

    Returns:
        prj_gcbct (torch.Tensor): Concatenated gated projections tensor.
        prj_ngcbct (torch.Tensor): Concatenated nonstop-gated projections tensor.
    """
    # Gated and nonstop-gated subdirectories
    g_dir = os.path.join(DATA_DIR, "gated")
    ng_dir = os.path.join(DATA_DIR, "ng")

    # TODO we need some way to choose which datasets we actually want...

    # Create ground truth dataset, and concatenate all scans into one tensor
    # along the H dimension (i.e., the dimension where we already stacked them before saving -- see "divide_sinogram" in proj.py)
    truth_set = PrjSet(g_dir)
    prj_gcbct = truth_set[0]
    for idx in range(1, len(truth_set)):
        prj_gcbct = torch.cat((prj_gcbct, truth_set[idx]), dim=0)

    # Repeat for nonstop-gated projections
    ns_set = PrjSet(ng_dir)
    prj_ngcbct = ns_set[0]
    for idx in range(1, len(ns_set)):
        prj_ngcbct = torch.cat((prj_ngcbct, ns_set[idx]), dim=0)

    return prj_gcbct, prj_ngcbct
