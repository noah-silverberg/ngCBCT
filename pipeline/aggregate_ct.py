# Implements Notebook 5: aggregate CT volumes into tensors
import torch
import os
from .utils import ensure_dir
from dsets import CTSet, AugCTSet  # assuming available
from .config import DATA_DIR, FLAGS


def aggregate_ct_volumes(
    data_ver: str, split: str, scan_type: int, augment: bool = False, save: bool = True
):
    """Aggregate CT volumes (full and ns) for given split (train/validation/test)."""
    base = os.path.join(DATA_DIR, f"DS{data_ver}")
    full = FLAGS.get("full", True)
    ns = FLAGS.get("ns", True)
    # Training or validation or test
    if full:
        if augment:
            dataset = AugCTSet(os.path.join(base, f"{split}/full/"))
        else:
            dataset = CTSet(os.path.join(base, f"{split}/full/"))
        imgs = dataset[0]
        for idx in range(1, len(dataset)):
            imgs = torch.cat((imgs, dataset[idx]), dim=0)
        # Crop for FF if needed
        if scan_type == 1:
            imgs = imgs[:, :, 128:384, 128:384]
        if save:
            ensure_dir(os.path.join(base, f"{split}/full/"))
            fname = f"{split}_full" + ("_aug" if augment else "") + ".pt"
            torch.save(imgs, os.path.join(base, f"{split}/full", fname))
    if ns:
        if augment:
            dataset = AugCTSet(os.path.join(base, f"{split}/ns/"))
        else:
            dataset = CTSet(os.path.join(base, f"{split}/ns/"))
        imgs = dataset[0]
        for idx in range(1, len(dataset)):
            imgs = torch.cat((imgs, dataset[idx]), dim=0)
        if scan_type == 1:
            imgs = imgs[:, :, 128:384, 128:384]
        if save:
            ensure_dir(os.path.join(base, f"{split}/ns/"))
            fname = f"{split}_ns" + ("_aug" if augment else "") + ".pt"
            torch.save(imgs, os.path.join(base, f"{split}/ns", fname))
