import torch
import os
import logging
from tqdm import tqdm
from pipeline.dsets import normalizeInputsClip
import numpy as np
from numpy.lib.format import open_memmap

logger = logging.getLogger("pipeline")


def aggregate_saved_recons(paths: list[str], augment: bool, out_path: str):
    """
    Load per-scan reconstruction tensors, concatenate across scans, and save aggregates
    directly to a memory-mapped file to avoid high RAM usage.

    Args:
        paths (list[str]): A list of file paths for the reconstruction tensors to aggregate.
        augment (bool): If True, applies horizontal and vertical flips, tripling the data size.
        out_path (str): The path to the output .npy file where the aggregated data will be saved.
    """
    if not paths:
        logger.warning("No paths provided to aggregate_saved_recons. Nothing to do.")
        return

    # 1. Determine the shape of the final aggregated array without loading all data
    first_recon = torch.load(paths[0]).detach().float()
    first_recon = normalizeInputsClip(first_recon)
    first_recon = torch.unsqueeze(first_recon, 1)
    
    recon_shape = first_recon.shape
    num_slices_per_recon = recon_shape[0]
    num_recons = len(paths)
    
    # The augmentation factor triples the number of slices
    augmentation_factor = 3 if augment else 1
    
    final_shape = (
        num_recons * num_slices_per_recon * augmentation_factor,
        recon_shape[1], # channel
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
    for i, path in tqdm(enumerate(paths), desc="Aggregating reconstructions to disk"):
        recon = torch.load(path).detach().float()
        recon = normalizeInputsClip(recon)
        recon = torch.unsqueeze(recon, 1)

        start_idx = i * num_slices_per_recon * augmentation_factor
        end_idx = (i + 1) * num_slices_per_recon * augmentation_factor

        if augment:
            # We need to handle the slices for original, flipped_h, and flipped_v
            orig_end = start_idx + num_slices_per_recon
            flip_h_end = orig_end + num_slices_per_recon
            
            recon_agg_memmap[start_idx:orig_end, ...] = recon.numpy()
            recon_agg_memmap[orig_end:flip_h_end, ...] = recon.flip(2).numpy()
            recon_agg_memmap[flip_h_end:end_idx, ...] = recon.flip(3).numpy()
        else:
            recon_agg_memmap[start_idx:end_idx, ...] = recon.numpy()
        
        del recon

    # Ensure all data is written to disk
    recon_agg_memmap.flush()
    logger.debug(f"Finished aggregating reconstructions to {out_path} with shape {final_shape}")
    del recon_agg_memmap
