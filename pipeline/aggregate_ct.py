import torch
import os
import logging
from tqdm import tqdm
from pipeline.dsets import normalizeInputsClip
import numpy as np
from numpy.lib.format import open_memmap

logger = logging.getLogger("pipeline")


def aggregate_saved_recons(paths: list[str] | list[list[str]], out_path: str, scan_type: str):
    """
    Load per-scan reconstruction tensors, concatenate across scans, and save aggregates
    directly to a memory-mapped file to avoid high RAM usage.

    Args:
        paths (list[str], list[list[str]]): A list of file paths for the reconstruction tensors to aggregate.
            If a list of lists is provided, the sublists will be used as separate channels.
        out_path (str): The path to the output .npy file where the aggregated data will be saved.
        scan_type (str): The type of scan, either "FF" or "HF".
    """
    if not paths:
        logger.warning("No paths provided to aggregate_saved_recons. Nothing to do.")
        return
    
    if isinstance(paths[0], list):
        channels = len(paths)
        logger.debug(f"Received a list of lists for paths, aggregating across {channels} channels.")
    else:
        channels = 1
        paths = [paths]

    # 1. Determine the shape of the final aggregated array without loading all data
    first_recon = torch.load(paths[0]).detach().float()
    first_recon = normalizeInputsClip(first_recon, scan_type)
    first_recon = torch.unsqueeze(first_recon, 1)
    
    recon_shape = first_recon.shape
    num_slices_per_recon = recon_shape[0]
    num_recons = len(paths)

    if recon_shape[1] != 1:
        raise ValueError("Reconstruction tensors must have a single channel (shape should be [num_slices, 1, height, width]).")
    
    final_shape = (
        num_recons * num_slices_per_recon,
        channels,       # channel
        recon_shape[2], # height
        recon_shape[3]  # width
    )

    del first_recon

    # 2. Create a memory-mapped numpy array on disk
    # This creates the output file on disk without using significant RAM.
    recon_agg_memmap = open_memmap(out_path, dtype=np.float32, mode='w+', shape=final_shape)
    logger.debug(f"Created memory-mapped file at {out_path} with shape {final_shape}")

    # 3. Fill the memory-mapped array slice by slice
    # This keeps RAM usage low, as we only load one reconstruction at a time.
    for i, paths_ in tqdm(enumerate(zip(*paths)), total=len(paths[0]), desc="Aggregating reconstructions to disk"):
        for j, path in enumerate(paths_):
            recon = torch.load(path).detach().float()
            recon = normalizeInputsClip(recon, scan_type)
            recon = torch.unsqueeze(recon, 1)

            start_idx = i * num_slices_per_recon
            end_idx = (i + 1) * num_slices_per_recon

            recon_agg_memmap[start_idx:end_idx, j, ...] = recon.numpy()
        
        del recon

    # Ensure all data is written to disk
    recon_agg_memmap.flush()
    logger.debug(f"Finished aggregating reconstructions to {out_path} with shape {final_shape}")
    del recon_agg_memmap
