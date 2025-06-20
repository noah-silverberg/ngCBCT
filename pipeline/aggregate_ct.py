import torch
import os
import logging
from tqdm import tqdm
from pipeline.dsets import normalizeInputsClip

logger = logging.getLogger("pipeline")


def aggregate_saved_recons(scan_type: str, sample: str, recon_pt_dir: str, AGG_SCANS: dict, truth: bool, augment: bool):
    """
    Load per-scan reconstruction tensors, concatenate across scans, and save aggregates.

    Args:
        scan_type (str): Type of scan, e.g. 'HF', 'FF'.
        sample (str): Data split name, e.g. 'train', 'validation', 'test'.
        recon_pt_dir (str): Directory where processed reconstructions are saved.
        AGG_SCANS (dict): Dictionary with samples as keys and scans as values, as (patient, scan, scan_type).

    Returns:
        recon_agg (torch.Tensor): Concatenated reconstructions tensor.
    """

    # Extract the scans for the given sample
    scans = AGG_SCANS[sample]

    # Concatenate all scans into one tensor
    # along the 1st dim (scan dimension)
    for i, (patient, scan, scan_type) in tqdm(enumerate(scans), desc="Aggregating reconstructions"):
        recon = torch.load(os.path.join(recon_pt_dir, f'FDK_{'gated' if truth else 'ng'}_{scan_type}_p{patient}_{scan}.pt')).detach().float()
        recon = normalizeInputsClip(recon)
        recon = recon.permute(2, 0, 1)
        recon = torch.unsqueeze(recon, 1)

        if i == 0:
            # Allocate an empty tensor for the aggregated reconstructions
            # We assume all reconstructions have the same shape
            if augment:
                recon_agg = torch.empty(
                    (3 * len(scans) * recon.shape[0], recon.shape[1], recon.shape[2], recon.shape[3]),
                    dtype=recon.dtype,
                ).detach()
            else:
                recon_agg = torch.empty(
                    (len(scans) * recon.shape[0], recon.shape[1], recon.shape[2], recon.shape[3]),
                    dtype=recon.dtype,
                ).detach()

        # Fill the allocated tensor with the loaded reconstructions
        if augment:
            recon_agg[3 * i * recon.shape[0] : (3 * i + 1) * recon.shape[0], ...] = recon
            recon_agg[(3 * i + 1) * recon.shape[0] : (3 * i + 2) * recon.shape[0], ...] = recon.flip(2)
            recon_agg[(3 * i + 2) * recon.shape[0] : (3 * i + 3) * recon.shape[0], ...] = recon.flip(3)
        else:
            recon_agg[i * recon.shape[0] : (i + 1) * recon.shape[0], ...] = recon

        del recon
        

    logger.debug(f"Aggregated {'gated' if truth else 'nonstop-gated'} reconstructions shape: {recon_agg.shape}")

    return recon_agg
