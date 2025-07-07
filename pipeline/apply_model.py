import numpy as np
import torch
from pipeline import network_instance
import logging
from pipeline.dsets import normalizeInputsClip

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

    # Calculate the overlap between the two halves of the scan
    # (recall that there are two halves each with v_dim angles, combined into one tensor)
    v_dim = 512 if scan_type == "HF" else 256
    overlap = v_dim * 2 - num_angles

    # Loop over the outputted slices in batches of 4 indices (8 slices total), put the results in the output tensor
    for i in range(0, 382, _batch_size):
        # Adjust the batch size for the last batch to avoid going out of bounds
        batch_size = min(_batch_size, 382 - i)
        indices = [i + j for j in range(batch_size)] + [i + j + 382 for j in range(batch_size)]
        in_ = prj_ngcbct_li[indices].detach().to(device)
        with torch.no_grad():
            out = model(in_).cpu()

        for j in range(batch_size):
            idx = i + j

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
                )  # note: overlap < 0 here
                for k in range(-overlap):
                    prj_ngcbct_cnn[v_dim + k, idx, :] = out[j, 0, -1, 1:511] + (k + 1) * diff

    # Now we need to:
    # 1. Create a ground truth tensor with the gated projections
    # 2. Create a tensor with the predicted nonstop-gated projections
    #    *but* we manually overwrite the angles that were actually acquired

    # 1.
    prj_gcbct_reshaped = torch.zeros((num_angles, 382, 510)).detach()
    for i in range(382):
        # Fill in the top and bottom parts of the ground truth projection
        prj_gcbct_reshaped[0:v_dim, i, :] = prj_gcbct[i, 0, :, 1:511]
        prj_gcbct_reshaped[(v_dim - overlap) :, i, :] = prj_gcbct[i + 382, 0, :, 1:511]

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
    """Apply CNN model slice-wise to nonstop-gated reconstructions to clean up image domain artifacts."""
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

    # Loop over the outputted slices in batches of _batch_size
    for i in range(0, recon.shape[0], _batch_size):
        # Adjust the batch size for the last batch to avoid going out of bounds
        batch_size = min(_batch_size, recon.shape[0] - i)
        indices = [i + j for j in range(batch_size)]
        with torch.no_grad():
            recon[indices] = model(recon[indices]) # replace the slices in place

    return recon