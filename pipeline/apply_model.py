import numpy as np
import torch
from pipeline import network_instance
import logging
from pipeline.dsets import normalizeInputsClip
from pipeline.proj import divide_sinogram

logger = logging.getLogger("pipeline")


def load_model(
    network_name: str, network_kwargs: dict, model_path: str, device: torch.device
):
    # Load and instantiate the network class dynamically
    model = getattr(network_instance, network_name)(**network_kwargs)

    # Load the model state and send it to the GPU
    state = torch.load(model_path)
    model.load_state_dict(state)
    model = model.to(device)

    # Set the model to eval mode for inference
    # NOTE: If you want to use it for training or MC dropout, you can just turn it back to train mode
    #       it's just safer to have this on eval mode by default
    model.eval()
    return model


def apply_model_to_projections(
    model: torch.nn.Module,
    scan_type: str,
    odd_index: np.ndarray,
    angles: np.ndarray,
    prj_gcbct: torch.Tensor,
    prj_ngcbct_li: torch.Tensor,
    device: torch.device,
    train_at_inference: bool = False,  # for MC dropout
    _batch_size: int = 4, # Number of slices (on each half of the scan) to process at once
):
    """Apply CNN model slice-wise to nonstop-gated projections to predict missing projections and combine."""
    if train_at_inference:
        # Set the model to train mode for MC dropout
        model.train()
        logger.info("Running model in train mode for MC dropout.")
    else:
        model.eval()

    # Flip angles if necessary
    angles = torch.from_numpy(np.array(sorted(angles))).float()
    angles1 = -(angles + np.pi / 2)
    if (angles1[-1:] - angles1[0]) < 0:
        angles1 = torch.flip(angles1, (0,))
    angles1 = angles1.detach().cpu().numpy()

    # Get the total number of angles from the gated scan
    num_angles = len(angles1)

    # Initialize output tensor for full sinogram output
    prj_ngcbct_cnn = torch.zeros((num_angles, 382, 510)).detach().to(device)

    # Handle the 3-patch case separately due to different input/output structure
    if scan_type == 'FF' and num_angles >= 520:
        v_dim = 256

        # Create 3 patches
        prj_ngcbct_li = divide_sinogram(prj_ngcbct_li, v_dim=v_dim, patches=3)

        # Loop over the outputted slices in batches of _batch_size indices (3 * _batch_size slices total), put the results in the output tensor
        for i in range(0, 382, _batch_size):
            batch_size = min(_batch_size, 382 - i)
            indices = [i + j for j in range(batch_size)] + [i + j + 382 for j in range(batch_size)] + [i + j + 2 * 382 for j in range(batch_size)]
            in_ = prj_ngcbct_li[indices].detach().to(device)
            with torch.no_grad():
                out = model(in_).cpu()

            for j in range(batch_size):
                idx = i + j

                # Define patch boundaries
                patch1_end = v_dim
                patch2_start = num_angles // 2 - v_dim // 2
                patch2_end = patch2_start + v_dim
                patch3_start = num_angles - v_dim
                overlap1 = patch1_end - patch2_start
                overlap2 = patch2_end- patch3_start

                # NOTE: For the 3-patch case, there won't be a gap, so we can just assume that there is a non-zero overlap
                #       and we can average the overlapping parts

                # Put the non-overlapping parts of the first part
                prj_ngcbct_cnn[0:(patch1_end - overlap1), idx, :] = out[j, 0, 0:-overlap1, 1:511]

                # Put the non-overlapping parts of the second part
                prj_ngcbct_cnn[patch1_end:patch2_end - overlap2, idx, :] = out[j + batch_size, 0, overlap1:-overlap2, 1:511]

                # Average the overlapping parts
                prj_ngcbct_cnn[(patch1_end - overlap1) : patch1_end, idx, :] = (
                    out[j, 0, -overlap1:, 1:511]
                    + out[j + batch_size, 0, :overlap1, 1:511]
                ) / 2.0

                # Put the non-overlapping parts of the third part
                prj_ngcbct_cnn[patch2_end:, idx, :] = out[j + 2 * batch_size, 0, overlap2:, 1:511]

                # Average the overlapping parts
                prj_ngcbct_cnn[(patch2_end - overlap2) : patch2_end, idx, :] = (
                    out[j + batch_size, 0, -overlap2:, 1:511]
                    + out[j + 2 * batch_size, 0, :overlap2, 1:511]
                ) / 2.0

    else:
        # This block handles HF scans and FF scans with < 520 angles
        v_dim = 512 if scan_type == "HF" else 256
        overlap = v_dim * 2 - num_angles
        
        # Prepare 2-patch input from the full projection
        prj_ngcbct_li = divide_sinogram(prj_ngcbct_li, v_dim=v_dim, patches=2)
        
        # Loop over the outputted slices in batches of _batch_size indices (2 * _batch_size slices total), put the results in the output tensor
        for i in range(0, 382, _batch_size):
            # Adjust the batch size for the last batch to avoid going out of bounds
            batch_size = min(_batch_size, 382 - i)
            indices = [i + j for j in range(batch_size)] + [i + j + 382 for j in range(batch_size)]
            in_ = prj_ngcbct_li[indices].detach().to(device)
            with torch.no_grad():
                out = model(in_).cpu()

            for j in range(batch_size):
                idx = i + j

                # Custom gap-filling for FF scans between 512 and 520 angles
                if scan_type == 'FF' and num_angles > 512:
                    odd_index_set = set(odd_index.astype(np.int64) - 1)

                    # Place the two predicted patches
                    prj_ngcbct_cnn[0:v_dim, idx, :] = out[j, 0, :, 1:511]
                    prj_ngcbct_cnn[(num_angles - v_dim):, idx, :] = out[j + batch_size, 0, :, 1:511]

                    # Fill the middle gap
                    gap_start, gap_end = v_dim, num_angles - v_dim
                    
                    # Fill with ground truth where available
                    known_in_gap = {k for k in range(gap_start, gap_end) if k in odd_index_set}
                    for k in known_in_gap:
                        prj_ngcbct_cnn[k, idx, :] = prj_gcbct[idx, k, 1:511]
                    
                    # Interpolate remaining empty parts of the gap
                    pos = gap_start
                    while pos < gap_end:
                        # If we find a missing position, we need to interpolate
                        if pos not in known_in_gap:
                            start_miss = pos
                            end_miss = start_miss

                            # Walk through the gap until we find the end of the missing section
                            while (end_miss + 1 < gap_end) and ((end_miss + 1) not in known_in_gap):
                                end_miss += 1
                            
                            # Now we take the surrounding values to interpolate
                            before_proj = prj_ngcbct_cnn[start_miss - 1, idx, :]
                            after_proj = prj_ngcbct_cnn[end_miss + 1, idx, :]
                            width = (end_miss + 1) - (start_miss - 1)

                            for k_off, miss_idx in enumerate(range(start_miss, end_miss + 1)):
                                frac = (k_off + 1) / width
                                prj_ngcbct_cnn[miss_idx, idx, :] = before_proj + (after_proj - before_proj) * frac

                            # Set the current position to the end of the missing section
                            pos = end_miss

                        # Step forward
                        pos += 1
                
                # Original logic for HF scans or FF scans <= 512 angles
                else:
                    # If there is overlap between the two halves, average the overlapping region
                    if overlap >= 0:
                        # Put the non-overlapping parts of the first half
                        prj_ngcbct_cnn[0 : (v_dim - overlap), idx, :] = out[
                            j, 0, 0 : (v_dim - overlap), 1:511
                        ]

                        # Put the non-overlapping parts of the second half
                        prj_ngcbct_cnn[v_dim:, idx, :] = out[j + batch_size, 0, overlap:, 1:511]

                        # Average the overlapping parts
                        prj_ngcbct_cnn[(v_dim - overlap) : v_dim, idx, :] = (
                            out[j, 0, (v_dim - overlap) : v_dim, 1:511]
                            + out[j + batch_size, 0, 0:overlap, 1:511]
                        ) / 2.0

                    # If there is no overlap, put the first half at the top
                    # and the second at the bottom
                    # and linearly interpolate the missing gap
                    else:
                        # Fill the top and bottom parts of the projection
                        prj_ngcbct_cnn[0:v_dim, idx, :] = out[j, 0, :, 1:511]
                        prj_ngcbct_cnn[(v_dim - overlap) :, idx, :] = out[j + batch_size, 0, :, 1:511]

                        # Calculate the linear interpolation for the gap
                        diff = (out[j + batch_size, 0, 0, 1:511] - out[j, 0, -1, 1:511]) / (
                            -overlap
                        ) # note: overlap < 0 here
                        for k in range(-overlap):
                            prj_ngcbct_cnn[v_dim + k, idx, :] = out[j, 0, -1, 1:511] + (k + 1) * diff

    # Now we need to:
    # 1. Create a ground truth tensor with the gated projections
    # 2. Create a tensor with the predicted nonstop-gated projections
    #    *but* we manually overwrite the angles that were actually acquired

    # 1.
    prj_gcbct_reshaped = prj_gcbct.permute(1, 0, 2)[:, :, 1:511]

    # 2.
    # Convert odd_index to int64 and adjust for 0-based indexing
    odd_index = (odd_index - 1).astype(np.int64)
    for i in odd_index:
        # Replace the acquired angles with the ground truth
        prj_ngcbct_cnn[i] = prj_gcbct_reshaped[i]

    g_mat = {
        "angles": angles1,
        "odd_index": odd_index,
        "prj": prj_gcbct_reshaped.cpu().numpy(),
    }
    cnn_mat = {
        "angles": angles1,
        "odd_index": odd_index,
        "prj": prj_ngcbct_cnn.cpu().numpy(),
    }

    return g_mat, cnn_mat

def apply_model_to_recons(
    model: torch.nn.Module,
    pt_path: str,
    device: torch.device,
    train_at_inference: bool = False, # for MC dropout
    _batch_size: int = 8, # Number of slices to process at once
):
    """
    Apply CNN model slice-wise to nonstop-gated reconstructions.
    Handles both standard (single-tensor output) and evidential (four-tensor output) models.
    """
    # Load the nonstop-gated reconstruction
    # NOTE: These each have shape (160, 512, 512) for HF and shape (160, 256, 256) for FF
    recon = torch.load(pt_path).detach().to(device)
    recon = normalizeInputsClip(recon)
    recon = torch.unsqueeze(recon, 1)  # Add channel dimension

    if train_at_inference:
        # Set the model to train mode for MC dropout
        logger.info("Running model in train mode for MC dropout.")
        model.train()
    else:
        # Set the model to eval mode for inference
        model.eval()

    # Determine output type by running one sample
    with torch.no_grad():
        sample_output = model(recon[0:1])
    
    is_evidential = isinstance(sample_output, (list, tuple)) and len(sample_output) == 4

    if is_evidential:
        logger.debug("Evidential model detected. Preparing for 4-channel output.")
        # Initialize output tensors for evidential regression
        gamma_out = torch.zeros_like(recon)
        nu_out = torch.zeros_like(recon)
        alpha_out = torch.zeros_like(recon)
        beta_out = torch.zeros_like(recon)
    else:
        # For standard models, we can modify the input tensor in place
        pass

    # Loop over the outputted slices in batches of _batch_size
    for i in range(0, recon.shape[0], _batch_size):
        # Adjust the batch size for the last batch to avoid going out of bounds
        batch_size = min(_batch_size, recon.shape[0] - i)
        indices = [i + j for j in range(batch_size)]
        with torch.no_grad():
            output = model(recon[indices])
            if is_evidential:
                gamma, nu, alpha, beta = output
                gamma_out[indices] = gamma
                nu_out[indices] = nu
                alpha_out[indices] = alpha
                beta_out[indices] = beta
            else:
                recon[indices] = output # replace the slices in place

    if is_evidential:
        return gamma_out, nu_out, alpha_out, beta_out
    else:
        return recon