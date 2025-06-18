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
phase = "7"
data_version = "12.2"  # e.g., '12'
model_version = "MK6"

# PD training settings
PD_training_app = "train_app_MK6numpy.TrainingApp"
PD_epochs = 20
PD_learning_rate = 1e-3  # Either float or list (if the list is shorter than the number of epochs, the last value is used for the rest of the epochs)
PD_network_name = "IResNet"  # Network name for PD CNN (see network_instance.py)
PD_model_name = (
    f"{PD_network_name}_{model_version}_DS{data_version}_PD"  # Model name for saving
)
PD_batch_size = 8
PD_optimizer = "NAdam"
PD_num_workers = 0
PD_shuffle = True
PD_grad_clip = True
PD_grad_max = 0.01  # Only used if PD_grad_clip is True
PD_betas_NAdam = (0.9, 0.999)  # Only for NAdam, otherwise ignored
PD_momentum_decay_NAdam = 4e-4  # Only for NAdam, otherwise ignored
PD_momentum_SGD = 0.99  # Only for SGD, otherwise ignored
PD_weight_decay_SGD = 1e-8  # Only for SGD, otherwise ignored
PD_checkpoint_save_freq = 10  # Save checkpoint every N epochs
PD_tensor_board = False  # Whether to use TensorBoard for PD training
PD_tensor_board_comment = ""  # If using TensorBoard, a comment suffix
PD_train_during_inference = False  # Whether to put the model in training mode during inference (e.g., for MC dropout)

# ID training settings
ID_training_app = "train_app_MK6_numpy.TrainingApp"
ID_epochs = 50
ID_learning_rate = 1e-3
ID_network_name = "IResNet"
ID_model_name = f"{ID_network_name}_{model_version}_DS{data_version}_ID"
ID_batch_size = 8
ID_optimizer = "NAdam"
ID_num_workers = 0
ID_shuffle = True
ID_grad_clip = True
ID_grad_max = 0.01  # Only used if ID_grad_clip is True
ID_betas_NAdam = (0.9, 0.999)  # Only for NAdam
ID_momentum_decay_NAdam = 4e-4  # Only for NAdam
ID_momentum_SGD = 0.99  # Only for SGD
ID_weight_decay_SGD = 1e-8  # Only for SGD
ID_augment = True  # Whether to use augmented data for ID training
ID_checkpoint_save_freq = 10
ID_tensor_board = False
ID_tensor_board_comment = ""  # Only if using TensorBoard
ID_train_during_inference = False
ID_input_type = "FDK"

# Directories derived from bases
PHASE_DATAVER_DIR = os.path.join(
    WORK_ROOT, f"phase{phase}", f"DS{data_version}"
)  # everything should go inside this directory
MODEL_DIR = os.path.join(PHASE_DATAVER_DIR, "model")  # for trained models
# LOSS_DIR = os.path.join(MODEL_DIR, "loss/") TODO
RESULT_DIR = os.path.join(PHASE_DATAVER_DIR, "result")  # for outputs of CNN
# FIGURE_DIR = os.path.join(PHASE_DATAVER_DIR, "figure/") TODO
PROJ_DIR = os.path.join(
    PHASE_DATAVER_DIR, "proj_data"
)  # for input data (gated and non-stop gated projections)
AGG_DIR = os.path.join(
    PHASE_DATAVER_DIR, "agg"
)  # for aggregated data (for PD and ID training)

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
