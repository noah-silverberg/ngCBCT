import torch
import os
from .dsets import PrjSet
import logging

logger = logging.getLogger("pipeline")


def aggregate_saved_projections(scan_type: str, sample: str, PROJ_DIR: str):
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
    g_dir = os.path.join(PROJ_DIR, "gated")
    ng_dir = os.path.join(PROJ_DIR, "ng")

    # Create ground truth dataset, and concatenate all scans into one tensor
    # along the H dimension (i.e., the dimension where we already stacked them before saving -- see "divide_sinogram" in proj.py)
    truth_set = PrjSet(g_dir, scan_type, sample)

    # Ensure scans were found
    if len(truth_set) == 0:
        raise ValueError(
            f"No gated scans found for scan type {scan_type} and sample {sample}"
        )

    logger.debug(
        f"Found {len(truth_set)} gated scans for scan type {scan_type} and sample {sample}"
    )

    prj_gcbct = truth_set[0]
    for idx in range(1, len(truth_set)):
        prj_gcbct = torch.cat((prj_gcbct, truth_set[idx]), dim=0)
    logger.debug(f"Aggregated gated projections shape: {prj_gcbct.shape}")

    # Repeat for nonstop-gated projections
    ns_set = PrjSet(ng_dir, scan_type, sample)

    # Ensure scans were found
    if len(ns_set) == 0:
        raise ValueError(
            f"No nonstop-gated scans found for scan type {scan_type} and sample {sample}"
        )

    logger.debug(
        f"Found {len(ns_set)} nonstop-gated scans for scan type {scan_type} and sample {sample}"
    )

    prj_ngcbct = ns_set[0]
    for idx in range(1, len(ns_set)):
        prj_ngcbct = torch.cat((prj_ngcbct, ns_set[idx]), dim=0)
    logger.debug(f"Aggregated nonstop-gated projections shape: {prj_ngcbct.shape}")

    return prj_gcbct, prj_ngcbct
