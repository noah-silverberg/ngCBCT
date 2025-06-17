# Implements Notebook 3 & 6: run helper for training apps
import logging
from .utils import ensure_dir
from .config import CUDA_DEVICE
from .config import MODEL_DIR
import torch


def run_app(app: str, args: list):
    """Run a training or other application via dynamic import and its main()."""
    import importlib

    # Prepend default args
    argv = []
    argv.append(f"--num_workers=0")
    argv.append(f"--batch_size=8")
    argv.extend(args)
    log = logging.getLogger("pipeline")
    log.info(f"Running: {app} {argv}")
    module_name, class_name = app.rsplit(".", 1)
    module = importlib.import_module(module_name)
    cls = getattr(module, class_name)
    instance = cls(argv)
    instance.main()
    log.info(f"Finished: {app}")
