# Common utility functions: plotting slices, creating directories, logging
import os
import matplotlib.pyplot as plt
import torch
import numpy as np
import CTorch.utils.geometry as geometry
from CTorch.reconstructor.fbpreconstructor import FBPReconstructor as FBP

def read_scans_agg_file(path, list_=False):
    # Read the aggregation scans file and split into TRAIN/VALIDATION/TEST
    with open(path, "r") as f:
        lines = [l.strip() for l in f.readlines()]

    # The first line is the scan type (HF or FF)
    scan_type = lines[0]

    if list_:
        all_scans = []
        for line in lines[1:]:
            if line:
                patient, scan = line.split()
                all_scans.append((patient, scan, scan_type))
                
        return all_scans, scan_type
    else:
        # Assemble the blocks of scans
        blocks = []
        current = []
        for line in lines[1:]:
            if not line:
                if current:
                    blocks.append(current)
                    current = []
            else:
                current.append(line)
        if current:
            blocks.append(current)

        # Now we have blocks of scans, each block corresponds to a sample (TRAIN, VALIDATION, TEST)
        samples = ["TRAIN", "VALIDATION", "TEST"]
        AGG_SCANS = {sample: [] for sample in samples}
        for sample, block in zip(samples, blocks):
            for entry in block:
                patient, scan = entry.split()
                AGG_SCANS[sample].append((patient, scan, scan_type))

        return AGG_SCANS, scan_type


def ensure_dir(path):
    """Create directory if not exists."""
    os.makedirs(path, exist_ok=True)
    
def get_geometry(prj, angles, scan_type):
    if scan_type == "HF":
        nView, nv, nu = prj.shape

        du, dv = 0.776, 0.776  # detecor pixel size
        detType = 'flat'
        SAD, SDD = [1000.0], [1500.0]  # source-axis-distance, source-detector-distance

        xOfst, yOfst, zOfst = [0.0], [0.0], [0.0]  # image center offset
        xSrc, zSrc = [0.0], [0.0]

        uOfst, vOfst = [-160.0], [0.0]  # detecor center offset for HF

        nx, ny, nz = 512, 512, 200 # image dimension
        dx, dy, dz = 1., 1., 1. # image voxel size

        # # The following are exactly the same for all projection data once the geometry is fixed
        padwidth = np.floor(2 * uOfst[0] / du)  # positive or negative value, but it's ok
        z_uOfst = [0.0]  # initialize
        z_uOfst[0] = uOfst[0] - padwidth/2 * du

        z_nu = int(nu + abs(padwidth))
        viewAngles = angles

        # CircGeom3D for HF after zeropadding and preweighting
        geom = geometry.CircGeom3D(
            nx, ny, nz, dx, dy, dz, z_nu, nv, nView, viewAngles, du, dv, detType, SAD, SDD,
            xOfst=xOfst, yOfst=yOfst, zOfst=zOfst, uOfst=z_uOfst, vOfst=vOfst,
            xSrc=xSrc, zSrc=zSrc, fixed=True
        )  # the only differences are z_nu and z_uOfst
    else:
        raise NotImplementedError("Only HF geometry has been implemented...")

    return geom, (nView, nu, nv, du, uOfst, SDD)

def pad_and_preweight(prj, nView, nu, nv, du, uOfst, SDD):
    # # The following are exactly the same for all projection data once the geometry is fixed
    padwidth = np.floor(2 * uOfst[0] / du)  # positive or negative value, but it's ok
    z_uOfst = [0.0]  # initialize
    z_uOfst[0] = uOfst[0] - padwidth/2 * du

    z_nu = int(nu + abs(padwidth))
    theta = (nu * du/2 - abs(uOfst[0])) * np.sign(uOfst[0])

    us = (np.arange(-z_nu/2 + 0.5, z_nu/2, 1) * du + abs(z_uOfst[0]))  # [-xx, xx] 1D vector
    abstheta = abs(theta)

    # weight1D
    weight1D = np.ones(z_nu, dtype=np.float32)

    mask1 = np.abs(us) <= abstheta
    weight1D[mask1] = 0.5 * (np.sin((np.pi/2) * np.arctan(us[mask1] / SDD[0]) / (np.arctan(abstheta/SDD[0]))) + 1)

    mask2 = us < -abstheta
    weight1D[mask2] = 0


    # weight2D
    weight2D = np.tile(weight1D, (nv, 1))  # repeat the weight1D

    if theta < 0:  # equivalent to uOfst[0] < 0
        weight2D = np.fliplr(weight2D)

    z_prj = np.zeros((nView, nv, z_nu), dtype=prj.dtype)  # initialize with zeros
    if uOfst[0] > 0:
        z_prj[:, :, padwidth:] = prj  # pad zeros in front
    else:
        z_prj[:, :, :nu] = prj  # pad zeros after


    # no need for looping around view
    z_prj[:, :, :] = z_prj[:, :, :] * weight2D * 2

    return z_prj

def CTorchReconstruct(prj, angles, scan_type, device):
    prj = torch.permute(prj, (2, 1, 0))
    prj = torch.flip(prj, [1])
    prj = prj.cpu().numpy()

    angles = angles - np.pi/2

    # Get geometry
    geom, pad_preweight_args = get_geometry(prj, angles, scan_type)

    # Pad and preweight projections
    prj_pad_preweight = pad_and_preweight(prj, *pad_preweight_args)

    # Convert to tensor
    prj_pad_preweight = torch.from_numpy(prj_pad_preweight).reshape(1, 1, *prj_pad_preweight.shape).float().to(device)

    # Initialize FBP reconstructor
    reconstructor = FBP(geom, "DD", window="hamming", cutoff=0.6) # reconstructed image is more blurred when cutoff is smaller

    # Reconstruct and remove extra dimensions
    recon = reconstructor(prj_pad_preweight).cpu()
    recon = torch.squeeze(recon, dim=1)
    recon = torch.squeeze(recon, dim=0)

    recon = torch.flip(recon, [1])

    return recon