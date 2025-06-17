# Configuration of paths and settings. Modify these to match your environment.
import os

# Base directories (modify as needed)
CLOUD_ROOT = r"C:/Users/yum2/OneDrive - Memorial Sloan Kettering Cancer Center/Postdoc/Python/DeepLearning/NS_Gated_CBCT/"
WORK_ROOT = r"D:/MitchellYu/NSG_CBCT/"

# Data versions
data_version = "12.2"  # e.g., '12' or '12.2'
model_version = "MK6"

# Directories derived from bases
SAVE_DIR = os.path.join(CLOUD_ROOT, "phase6/")
MODEL_DIR = os.path.join(SAVE_DIR, "model/")
LOSS_DIR = os.path.join(SAVE_DIR, "model", "loss/")
RESULT_DIR = os.path.join(SAVE_DIR, "result/")
FIGURE_DIR = os.path.join(SAVE_DIR, "figure/")
DATA_DIR = os.path.join(WORK_ROOT, "data/")

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
