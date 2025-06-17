# Implements Notebook 2: aggregate projection tensors
import torch
import os
from .proj import (
    load_projection_mat,
    reformat_for_tigre,
    interpolate_projections,
    pad_and_reshape,
)
from .config import DATA_DIR
from .utils import ensure_dir
from dsets import PrjSet  # assuming the project has this class available


def aggregate_saved_projections(data_ver: str, mode: str, sample: str, save_root=None):
    """Load saved per-scan .pt projection tensors and concatenate into one large tensor for sample split."""
    # base_dir: e.g., D:/MitchellYu/NSG_CBCT/phase6/data/DS12/
    base_dir = os.path.join(DATA_DIR, f"DS{data_ver}")
    if save_root is None:
        save_root = base_dir
    # Full projections
    full_dir = os.path.join(base_dir, sample, "full")
    ns_dir = os.path.join(base_dir, sample, "ns")
    # Using PrjSet to load each item, then concatenate
    truth_set = PrjSet(full_dir)
    prj_gcbct = truth_set[0]
    for idx in range(1, len(truth_set)):
        prj_gcbct = torch.cat((prj_gcbct, truth_set[idx]), dim=0)
    ensure_dir(os.path.join(full_dir))
    torch.save(prj_gcbct, os.path.join(full_dir, f"{sample}_full.pt"))
    # NS projections
    ns_set = PrjSet(ns_dir)
    prj_ngcbct = ns_set[0]
    for idx in range(1, len(ns_set)):
        prj_ngcbct = torch.cat((prj_ngcbct, ns_set[idx]), dim=0)
    ensure_dir(os.path.join(ns_dir))
    torch.save(prj_ngcbct, os.path.join(ns_dir, f"{sample}_ns.pt"))
