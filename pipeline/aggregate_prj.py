import torch
import os
import logging
from tqdm import tqdm
from numpy.lib.format import open_memmap
import numpy as np

logger = logging.getLogger("pipeline")


def aggregate_saved_projections(paths: list[str], out_path: str):
    """
    Load per-scan projection tensors and concatenate across scans, and save aggregates
    directly to a memory-mapped file to avoid high RAM usage.

    Args:
        paths (list[str]): List of file paths to the projection tensors.
    """
    # Concatenate all scans into one tensor
    # along the H dimension (i.e., the dimension where we already stacked them before saving -- see "divide_sinogram" in proj.py)
    for i, path in tqdm(enumerate(paths), desc="Aggregating projections"):
        prj = torch.load(path).detach()

        if i == 0:
            # Allocate an empty tensor for the aggregated projections
            # We assume all projections have the same shape
            final_shape = (len(paths) * prj.shape[0], prj.shape[1], prj.shape[2], prj.shape[3])
            prj_agg = open_memmap(out_path, dtype=np.float32, mode="w+", shape=final_shape)
            logger.debug(f"Created memory-mapped file at {out_path} with shape {final_shape}")

        # Fill the allocated tensor with the loaded projections
        prj_agg[i * prj.shape[0] : (i + 1) * prj.shape[0], ...] = prj.cpu().numpy()

        del prj
        

    prj_agg.flush()  # Ensure all data is written to disk
    logger.debug(f"Aggregated {len(paths)} projections into shape: {prj_agg.shape}")

    return prj_agg
