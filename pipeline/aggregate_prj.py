import torch
import os
import logging
from tqdm import tqdm

logger = logging.getLogger("pipeline")


def aggregate_saved_projections(paths: list[str]):
    """
    Load per-scan projection tensors and concatenate across scans.

    Args:
        paths (list[str]): List of file paths to the projection tensors.

    Returns:
        torch.Tensor: Concatenated projection tensor.
    """
    # Concatenate all scans into one tensor
    # along the H dimension (i.e., the dimension where we already stacked them before saving -- see "divide_sinogram" in proj.py)
    for i, path in tqdm(enumerate(paths), desc="Aggregating projections"):
        prj = torch.load(path).detach()

        if i == 0:
            # Allocate an empty tensor for the aggregated projections
            # We assume all projections have the same shape
            prj_agg = torch.empty(
                (len(paths) * prj.shape[0], prj.shape[1], prj.shape[2], prj.shape[3]),
                dtype=prj.dtype,
            ).detach()

        # Fill the allocated tensor with the loaded projections
        prj_agg[i * prj.shape[0] : (i + 1) * prj.shape[0], ...] = prj

        del prj
        

    logger.debug(f"Aggregated {len(paths)} projections into shape: {prj_agg.shape}")

    return prj_agg
