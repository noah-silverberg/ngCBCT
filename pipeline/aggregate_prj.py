import torch
import os
import logging
from tqdm import tqdm

logger = logging.getLogger("pipeline")


def aggregate_saved_projections(scan_type: str, sample: str, prj_pt_dir: str, AGG_SCANS: dict, truth: bool):
    """
    Load per-scan projection tensors, concatenate across scans, and save aggregates.

    Args:
        scan_type (str): Type of scan, e.g. 'HF', 'FF'.
        sample (str): Data split name, e.g. 'train', 'validation', 'test'.
        prj_pt_dir (str): Directory where processed projections are saved.
        AGG_SCANS (dict): Dictionary with samples as keys and scans as values, as (patient, scan, scan_type).

    Returns:
        prj_gcbct (torch.Tensor): Concatenated gated projections tensor.
        prj_ngcbct (torch.Tensor): Concatenated nonstop-gated projections tensor.
    """
    # Gated or nonstop-gated subdirectory
    dir = os.path.join(prj_pt_dir, "gated" if truth else "ng")

    # Extract the scans for the given sample
    scans = AGG_SCANS[sample]

    # Concatenate all scans into one tensor
    # along the H dimension (i.e., the dimension where we already stacked them before saving -- see "divide_sinogram" in proj.py)
    for i, (patient, scan, scan_type) in tqdm(enumerate(scans), desc="Aggregating projections"):
        prj = torch.load(os.path.join(dir, f'{scan_type}_p{patient}_{scan}.pt')).detach()

        if i == 0:
            # Allocate an empty tensor for the aggregated projections
            # We assume all projections have the same shape
            prj_agg = torch.empty(
                (len(scans) * prj.shape[0], prj.shape[1], prj.shape[2], prj.shape[3]),
                dtype=prj.dtype,
            ).detach()

        # Fill the allocated tensor with the loaded projections
        prj_agg[i * prj.shape[0] : (i + 1) * prj.shape[0], ...] = prj

        del prj
        

    logger.debug(f"Aggregated {'gated' if truth else 'nonstop-gated'} projections shape: {prj_agg.shape}")

    return prj_agg
