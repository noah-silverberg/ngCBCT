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
    # First pass: Get all shapes to determine the final size.
    # This is necessary because the first dimension of each tensor can be different.
    shapes = [torch.load(p).shape for p in tqdm(paths, desc="Pass 1/2: Reading shapes")]
    
    # Calculate the total size for the first dimension.
    total_dim0 = sum(s[0] for s in shapes)
    # We assume all other dimensions are consistent, so we take them from the first tensor.
    final_shape = (total_dim0, *shapes[0][1:])

    # Allocate the memory-mapped file with the final, correct shape.
    prj_agg = open_memmap(out_path, dtype=np.float32, mode="w+", shape=final_shape)
    logger.debug(f"Created memory-mapped file at {out_path} with shape {final_shape}")

    # Second pass: Load data again and fill the array.
    current_pos = 0
    for i, path in tqdm(enumerate(paths), desc="Pass 2/2: Aggregating projections"):
        prj = torch.load(path).detach()
        # Fill the allocated tensor using a running index and the pre-calculated shape.
        rows_to_add = shapes[i][0]
        prj_agg[current_pos : current_pos + rows_to_add, ...] = prj.cpu().numpy()
        current_pos += rows_to_add

        del prj
        prj_agg.flush()  # Ensure all data is written to disk

    prj_agg.flush()  # Ensure all data is written to disk
    logger.debug(f"Aggregated {len(paths)} projections into shape: {prj_agg.shape}")

    return prj_agg