import os
import numpy as np
import torch
import scipy.io
import mat73
from .config import (
    MODEL_DIR,
    PROJ_DIR,
    CUDA_DEVICE,
)
from .proj import load_projection_mat
from .utils import plot_loss, ensure_dir, display_slices_grid, plot_single_slices
from . import network_instance


def load_model(network_name: str, model_name: str, device=None):
    # Load and instantiate the network class dynamically
    model = getattr(network_instance, network_name)()

    # Load the model state and send it to the GPU
    state = torch.load(os.path.join(MODEL_DIR, f"{model_name}.pth"))
    model.load_state_dict(state)
    if device:
        model = model.to(device)

    # Set the model to eval mode for inference
    # NOTE: If you want to use it for training or MC dropout, you can just turn it back to train mode
    #       it's just safer to have this on eval mode by default
    model.eval()
    return model


def apply_model_to_projections(
    patient: str, scan: str, scan_type: str, sample: str, model: torch.nn.Module
):
    """Apply CNN model slice-wise to nonstop-gated projections to predict missing projections and combine."""
    # Set up CUDA
    device = torch.device(CUDA_DEVICE)

    # Get the acquired nonstop-gated indices and angles from the .mat file
    odd_index, angles, _ = load_projection_mat(patient, scan, scan_type)

    # Gated and nonstop-gated subdirectories
    g_dir = os.path.join(PROJ_DIR, "gated")
    ng_dir = os.path.join(PROJ_DIR, "ng")

    # Load the gated and (interpolated) nonstop-gated projections
    # NOTE: These each have shape (2*H, 1, v_dim, 512)
    prj_gcbct = torch.load(
        os.path.join(g_dir, f"{scan_type}_p{patient}_{scan}_{sample}.pt")
    ).detach()
    prj_ngcbct_li = torch.load(
        os.path.join(ng_dir, f"{scan_type}_p{patient}_{scan}_{sample}.pt")
    ).detach()

    # Flip angles if necessary
    angles = torch.from_numpy(np.array(sorted(angles))).float()
    angles1 = -(angles + np.pi / 2)
    if (angles1[-1:] - angles1[0]) < 0:
        angles1 = torch.flip(angles1, (0,))
    angles1 = angles1.detach().cpu().numpy()

    # Get the total number of angles from the gated scan
    num_angles = len(angles1)

    # Remove channel dim for processing
    prj_ngcbct_li = torch.squeeze(prj_ngcbct_li, 1)

    # Initialize output tensor for full sinogram output
    prj_ngcbct_cnn = torch.zeros((num_angles, 382, 510)).detach().to(device)

    # Calculate the overlap between the two halves of the scan
    # (recall that there are two halves each with v_dim angles, combined into one tensor)
    v_dim = 512 if scan_type == "HF" else 256
    overlap = v_dim * 2 - num_angles

    # Loop over the slices, pass them through the model, and put the results in the output tensor
    for i in range(382):
        # Run the first half of the projection through the model
        img1 = prj_ngcbct_li[i].detach().to(device)
        out1 = model(torch.unsqueeze(img1, 0)).detach().cpu()

        # Run the second half of the projection through the model
        img2 = prj_ngcbct_li[i + 382].detach().to(device)
        out2 = model(torch.unsqueeze(img2, 0)).detach().cpu()

        # If there is overlap between the two halves, average the overlapping region
        if overlap >= 0:
            # Put the non-overlapping parts of the first half
            prj_ngcbct_cnn[0 : (v_dim - overlap), i, :] = out1[
                0, 0, 0 : (v_dim - overlap), 1:511
            ]

            # Put the non-overlapping parts of the second half
            prj_ngcbct_cnn[v_dim:, i, :] = out2[0, 0, overlap:, 1:511]

            # Average the overlapping parts
            prj_ngcbct_cnn[(v_dim - overlap) : v_dim, i, :] = (
                out1[0, 0, (v_dim - overlap) : v_dim, 1:511]
                + out2[0, 0, 0:overlap, 1:511]
            ) / 2.0

        # If there is no overlap, put the first half at the top
        # and the second at the bottom
        # and linearly interpolate the missing gap
        else:
            # Fill the top and bottom parts of the projection
            prj_ngcbct_cnn[0:v_dim, i, :] = out1[0, 0, :, 1:511]
            prj_ngcbct_cnn[(v_dim - overlap) :, i, :] = out2[0, 0, :, 1:511]

            # Calculate the linear interpolation for the gap
            diff = (out2[0, 0, 0, 1:511] - out1[0, 0, -1, 1:511]) / (
                -overlap
            )  # note: overlap < 0 here
            for j in range(-overlap):
                prj_ngcbct_cnn[v_dim + j, i, :] = out1[0, 0, -1, 1:511] + (j + 1) * diff

    # Now we need to:
    # 1. Create a ground truth tensor with the gated projections
    # 2. Create a tensor with the predicted nonstop-gated projections
    #    *but* we manually overwrite the angles that were actually acquired

    # 1.
    prj_gcbct_reshaped = torch.zeros((num_angles, 382, 510)).detach().to(device)
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
        "prj": prj_gcbct_reshaped.numpy(),
    }
    cnn_mat = {
        "angles": angles1,
        "odd_index": odd_index,
        "prj": prj_ngcbct_cnn.numpy(),
    }

    return g_mat, cnn_mat

    # # TODO
    # out_dir = os.path.join(WORK_ROOT, "output", "prj_mat", mode)
    # ensure_dir(out_dir)
    # scipy.io.savemat(
    #     os.path.join(out_dir, f"panc{patient_id}.{mode}_ns.mat"), out_mat
    # )
    # return prj_ngcbct_mix
