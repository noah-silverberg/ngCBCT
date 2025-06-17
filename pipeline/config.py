# Configuration of paths and settings. Modify these to match your environment.
import os

# Debugging mode
DEBUG = False

# Base directories (modify as needed)
WORK_ROOT = r"D:/MitchellYu/NSG_CBCT/"

# Data versions
phase = "phase7"
data_version = "12.2"  # e.g., '12' or '12.2'
model_version = "MK6"

# PD training settings
PD_training_app = "train_app_MK6.TrainingApp"
PD_epochs = 20
PD_network_name = "IResNet"  # Network name for PD CNN (see network_instance.py)
PD_model_name = (
    f"{PD_network_name}_{model_version}_DS{data_version}_PD"  # Model name for saving
)
PD_batch_size = 8
PD_optimizer = "NAdam"

# ID training settings
ID_training_app = "train_app_MK6.TrainingApp"
ID_epochs = 50
ID_network_name = "IResNet"
ID_model_name = f"{ID_network_name}_{model_version}_DS{data_version}_ID"
ID_batch_size = 8
ID_optimizer = "NAdam"

# Directories derived from bases
PHASE_DIR = os.path.join(WORK_ROOT, f"phase{phase}/")
MODEL_DIR = os.path.join(PHASE_DIR, "model/")
# LOSS_DIR = os.path.join(MODEL_DIR, "loss/") TODO
RESULT_DIR = os.path.join(PHASE_DIR, "result/")
FIGURE_DIR = os.path.join(PHASE_DIR, "figure/")
DATA_DIR = os.path.join(
    PHASE_DIR, "data", f"DS{data_version}/"
)  # for input data (gated and non-stop gated projections)

# GPU settings
CUDA_DEVICE = "cuda:0"

# Default plotting clip ranges
CLIP_LOW = 0
CLIP_HIGH = 0.04

# Notebook flags defaults
FLAGS = {
    "full": True,
    "nsFDK": False,
    "nsPL": False,
    "PD": False,
    "save": True,
    "recon": False,
    "DEBUG": False,
    "augment": False,
}

SCANS = [
    # (patient_id, scan_id, scan_type)
    # e.g., ("13", "08", "HF")
]
