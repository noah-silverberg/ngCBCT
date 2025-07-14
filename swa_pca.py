import torch
import numpy as np
import os
import matplotlib.pyplot as plt
from tqdm import tqdm

# Assuming these are in your project structure
from pipeline.paths import Files, Directories 
from pipeline.network_instance import IResNet, SWAG, flatten, unflatten_like, IResNetDropout

# --- 1. Configuration ---

# The original model that was trained (to get the snapshot paths)
SWAG_START_MODEL_VERSION = "MK7_MCDROPOUT_30_pct_NEW_SWAG_lr1e-2"
SWAG_LR = None #0.01 # The LR used to generate the snapshots
SWAG_MOM = None
SWAG_WEIGHT_DECAY = None
START_EPOCH = 50 
BURN_IN_EPOCHS = 0
SWA_EPOCHS = 10

# The final, saved SWAG model (to get the mean)
DOMAIN = "IMAG" # or "IMAG"

# The base model architecture
BASE_MODEL_CLS = IResNetDropout
BASE_MODEL_KWARGS = {'p' : 0.3}

# --- 2. Setup File Paths ---
WORK_ROOT = "D:/NoahSilverberg/ngCBCT"
PHASE = "7"
DATA_VERSION = "13"
# Base directory
WORK_ROOT = "E:/NoahSilverberg/ngCBCT"

# NSG_CBCT Path where the raw matlab data is stored
NSG_CBCT_PATH = "D:/MitchellYu/NSG_CBCT"

# Directory with all files specific to this phase/data version
PHASE_DATAVER_DIR = os.path.join(
    WORK_ROOT, f"phase{PHASE}", f"DS{DATA_VERSION}"
)

DIRECTORIES = Directories(
    # mat_projections_dir=os.path.join(NSG_CBCT_PATH, "data", "prj", "HF", "mat"),
    # pt_projections_dir=os.path.join(WORK_ROOT, "prj_pt"),
    # projections_aggregate_dir=os.path.join(PHASE_DATAVER_DIR, "aggregates", "projections"),
    # projections_model_dir=os.path.join(PHASE_DATAVER_DIR, "models", "projections"),
    # projections_results_dir=os.path.join(PHASE_DATAVER_DIR, "results", "projections"),
    # projections_gated_dir=os.path.join(WORK_ROOT, "gated", "prj_mat"),
    reconstructions_dir=os.path.join(PHASE_DATAVER_DIR, "reconstructions"),
    reconstructions_gated_dir=os.path.join(WORK_ROOT, "gated", "fdk_recon"),
    # images_aggregate_dir=os.path.join(PHASE_DATAVER_DIR, "aggregates", "images"),
    images_model_dir=os.path.join('H:\Public/Noah/phase7/DS13', "models", "images"),
    # images_results_dir=os.path.join(PHASE_DATAVER_DIR, "results", "images"),
)

FILES = Files(DIRECTORIES)


# --- 3. Load the SWA Snapshots and Final SWAG model ---

# Collect the paths to all the snapshots that were created
snapshot_paths = []
for i in range(1, SWA_EPOCHS + 1):
    epoch = START_EPOCH + BURN_IN_EPOCHS + i
    path = FILES.get_model_filepath(
        SWAG_START_MODEL_VERSION,
        DOMAIN,
        checkpoint=epoch,
        swag_lr=SWAG_LR,
        swag_momentum=SWAG_MOM,
        swag_weight_decay=SWAG_WEIGHT_DECAY,
    )
    if os.path.exists(path):
        snapshot_paths.append(path)
    print(path)

print(f"Found {len(snapshot_paths)} SWA snapshots.")

# Load the snapshots into a list of PyTorch vectors
swa_weight_vectors = []
for path in tqdm(snapshot_paths, desc="Loading snapshots"):
    state_dict = torch.load(path)['state_dict']
    temp_model = BASE_MODEL_CLS(**BASE_MODEL_KWARGS)
    temp_model.load_state_dict(state_dict)
    swa_weight_vectors.append(torch.nn.utils.parameters_to_vector(temp_model.parameters()))

# Create a matrix of all snapshot weights (num_snapshots x num_weights)
W = torch.stack(swa_weight_vectors)

# Directly calculate the mean of the SWA snapshots
swa_mean_vec = torch.mean(W, dim=0)


# --- 4. Perform PCA via SVD (No Scikit-learn) ---

# Center the weights around the SWA mean
W_centered = W - swa_mean_vec

print("Performing PCA via SVD on the weight trajectory...")
# Use PyTorch's SVD. full_matrices=False is more efficient for p >> n cases.
# U has shape (n_snapshots, k), S has shape (k,), Vh has shape (k, n_weights)
# where k = min(n_snapshots, n_weights)
U, S, Vh = torch.linalg.svd(W_centered.detach(), full_matrices=False)

# The projected coordinates of the centered data onto the principal components
# are given by U * S. We only need the first two columns for a 2D plot.
W_2d = torch.matmul(U.detach(), torch.diag(S.detach()))[:, :2].detach().cpu().numpy()

# The SWA mean is the origin (0,0) in this centered space
swa_mean_2d = np.zeros(2)


# --- 5. Create the Visualization ---

print("Generating plot...")
plt.style.use('seaborn-v0_8-whitegrid')
fig = plt.figure(figsize=(8, 8))
ax = fig.add_subplot(111)

# Plot the trajectory of the SWA snapshots
ax.plot(W_2d[:, 0], W_2d[:, 1], marker='o', markersize=4, alpha=0.7, linestyle='-', label='SWA Iterates')

# Plot the final SWA mean solution at the center
ax.plot(swa_mean_2d[0], swa_mean_2d[1], marker='*', color='r', markersize=15, label='SWA Mean', markeredgecolor='k')

ax.set_title('2D PCA Projection of SWA Trajectory', fontsize=16)
ax.set_xlabel('Principal Component 1', fontsize=12)
ax.set_ylabel('Principal Component 2', fontsize=12)
ax.legend(fontsize=12)
ax.axhline(0, color='grey', lw=1, linestyle='--')
ax.axvline(0, color='grey', lw=1, linestyle='--')

plt.show()