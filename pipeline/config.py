# Configuration of paths and settings. Modify these to match your environment.
import os

# Debugging mode
DEBUG = True

# GPU settings
# Should be near the top to ensure they are set before anything else that might use CUDA
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
CUDA_DEVICE = "cuda:0"

# Base directory
WORK_ROOT = os.path.abspath("./TESTING")

# Data versions
phase = "phase7"
data_version = "12.2"  # e.g., '12'
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
PD_num_workers = 0

# ID training settings
ID_training_app = "train_app_MK6.TrainingApp"
ID_epochs = 50
ID_network_name = "IResNet"
ID_model_name = f"{ID_network_name}_{model_version}_DS{data_version}_ID"
ID_batch_size = 8
ID_optimizer = "NAdam"
ID_num_workers = 0

# Directories derived from bases
PHASE_DATAVER_DIR = os.path.join(
    WORK_ROOT, f"DS{data_version}", f"phase{phase}/"
)  # everything should go inside this directory
MODEL_DIR = os.path.join(PHASE_DATAVER_DIR, "model/")  # for trained models
# LOSS_DIR = os.path.join(MODEL_DIR, "loss/") TODO
RESULT_DIR = os.path.join(PHASE_DATAVER_DIR, "result/")  # for outputs of CNN
# FIGURE_DIR = os.path.join(PHASE_DATAVER_DIR, "figure/") TODO
DATA_DIR = os.path.join(
    PHASE_DATAVER_DIR, "data"
)  # for input data (gated and non-stop gated projections)

# # Default plotting clip ranges
# CLIP_LOW = 0
# CLIP_HIGH = 0.04

# # Notebook flags defaults
# FLAGS = {
#     "full": True,
#     "nsFDK": False,
#     "nsPL": False,
#     "PD": False,
#     "save": True,
#     "recon": False,
#     "DEBUG": False,
#     "augment": False,
# }

SCANS = [
    # (patient_id, scan_id, scan_type, sample)
    # e.g., ("13", "08", "HF", "TRAIN")
    ("01", "01", "HF", "TRAIN"),
    ("02", "01", "FF", "TRAIN"),
]
