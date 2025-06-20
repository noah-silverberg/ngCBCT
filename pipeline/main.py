import os, yaml, copy, logging
from tqdm import tqdm
from pipeline.utils import ensure_dir, read_scans_agg_file, get_geometry
from pipeline.proj import load_projection_mat, reformat_sinogram, interpolate_projections, pad_and_reshape, divide_sinogram
from pipeline.aggregate_prj import aggregate_saved_projections
from pipeline.aggregate_ct import aggregate_saved_recons
from pipeline.apply_model import apply_model_to_projections, load_model
from pipeline.FDK_half.FDK_half import FDKHalf
from pipeline.train_app_MK6_numpy import TrainingApp as TrainApp
import torch
import scipy.io
import numpy as np
import gc

logger = logging.getLogger("pipeline.runner")

def convert_scans(scans_convert: list[tuple], mat_path: str, g_dir: str, ng_dir: str):
    """
    Convert raw scan data from .mat files into processed .pt files for gated and non-gated projections.
    
    The output files follow the naming convention:
    `{scan_type}_p{patient}_{scan}.pt`.

    Args:
        scans_convert (list[tuple]): A list of
            - patient (str): Identifier for the patient.
            - scan (str): Identifier for the scan.
            - scan_type (str): Type of scan ("HF" or "FF").
        mat_path (str): Path to the directory containing raw .mat files for the scans.
        g_dir (str): Directory to save gated projections.
        ng_dir (str): Directory to save non-gated projections.
    Notes:
        - Existing .pt files in the output directories will be overwritten.
    Example:
        scans = [("01", "01", "HF"), ("02", "02", "FF")]
        convert_scans(scans, mat_path="/path/to/mat_files", g_dir="/path/to/gated", ng_dir="/path/to/non_gated")
    """
    # Ensure the output directories exist
    ensure_dir(g_dir)
    ensure_dir(ng_dir)

    logger.info("Starting to convert raw scans to pt files...outputting to %s and %s", g_dir, ng_dir)

    # Loop through the scans we need to convert
    for patient, scan, scan_type in scans_convert:
        logger.debug(f"Converting scan {scan_type} for patient {patient}, scan {scan}...")

        # Load the raw scan data from mat_path
        odd_index, angles, g_projection = load_projection_mat(patient, scan, scan_type, mat_path)

        logger.debug(f"Loaded odd_index shape: {odd_index.shape}, angles shape: {angles.shape}, g_projection shape: {g_projection.shape}")

        # Reformat the projection and angle data (just flipping, permuting, etc.)
        g_projection, angles = reformat_sinogram(g_projection, angles)

        logger.debug(f"Reformatted g_projection shape: {g_projection.shape}, angles shape: {angles.shape}")

        # Simulate the non-stop gated projection data, and linearly interpolate the simulated gaps
        ng_projection = interpolate_projections(g_projection, odd_index)

        logger.debug(f"Interpolated ng_projection shape: {ng_projection.shape}")

        # Pad, reshape, and split each projection into two halves
        # these halves are stacked along dimension 0
        g_projection = divide_sinogram(pad_and_reshape(g_projection), v_dim=512 if scan_type == "HF" else 256)
        ng_projection = divide_sinogram(pad_and_reshape(ng_projection), v_dim=512 if scan_type == "HF" else 256)

        logger.debug(f"Divided gated projection shape: {g_projection.shape}, non-gated projection shape: {ng_projection.shape}")

        # Save gated projections
        g_path = os.path.join(g_dir, f"{scan_type}_p{patient}_{scan}.pt") # e.g., "HF_p01_01.pt"
        if os.path.exists(g_path):
            logger.warning(f"Gated projection file {g_path} already exists. Overwriting...")
        torch.save(g_projection, g_path)

        # Save non-gated projections
        ng_path = os.path.join(ng_dir, f"{scan_type}_p{patient}_{scan}.pt") # e.g., "HF_p01_01.pt"
        if os.path.exists(ng_path):
            logger.warning(f"Non-gated projection file {ng_path} already exists. Overwriting...")
        torch.save(ng_projection, ng_path)

        # Free up memory
        del odd_index, angles, g_projection, ng_projection
        gc.collect()

        logger.info(f"Converted scan {scan_type} for patient {patient}, scan {scan}")

    logger.info("Done converting scans.\n\n")

def aggregate_projections(scans_agg, scan_type, AGG_DIR, pt_prj_dir):
    ensure_dir(AGG_DIR)
    for sample in ("TRAIN","VALIDATION","TEST"):
        if not scans_agg[sample]: continue
        for truth in (False,True):
            arr = aggregate_saved_projections(scan_type, sample, pt_prj_dir, scans_agg, truth=truth)
            suffix = "gated" if truth else "ng"
            np.save(os.path.join(AGG_DIR,f"PROJ_{suffix}_{scan_type}_{sample}.npy"), arr.numpy())
            del arr
    logger.info("aggregate_projections done.")

def train_stage(config_files, domain, paths, id_or_pd):
    MODEL_DIR, AGG_DIR, DEBUG = paths
    for cfgf in config_files:
        cfg = yaml.safe_load(open(cfgf))
        S = cfg[f"{id_or_pd}_settings"]
        if not S["training"]: continue
        module_name, cls_name = S["training_app"].rsplit(".",1)
        for i in range(S["ensemble_size"]):
            c = copy.deepcopy(cfg)
            if S["ensemble_size"]>1:
                c[f"{id_or_pd}_settings"]["model_version"] = f"{S['model_version']}_{i+1:02}"
            c[f"{id_or_pd}_settings"]["data_version"] = c[f"{id_or_pd}_settings"]["data_version"]
            app = TrainApp(c, domain, DEBUG, MODEL_DIR, AGG_DIR)
            app.main(); del app; gc.collect()
    logger.info(f"{id_or_pd} training done.")

def apply_pd_and_fdk(config_files, paths, RESULT_DIR, RECON_DIR, work_root, NSG_CBCT_PATH, CUDA_DEVICE):
    MODEL_DIR, DATA_VERSION = paths
    for cfgf in config_files:
        cfg = yaml.safe_load(open(cfgf))
        P = cfg["PD_settings"]
        for i in range(P["ensemble_size"]):
            mv = P["model_version"]
            if P["ensemble_size"]>1: mv = f"{mv}_{i+1:02}"
            # load PD model
            m = load_model(P["network_name"], mv, torch.device(CUDA_DEVICE), MODEL_DIR, DATA_VERSION, "PROJ", P["scan_type"])
            result_dir = os.path.join(RESULT_DIR, mv)
            for f in tqdm(os.listdir(result_dir), desc=f"PD→FDK {mv}"):
                dom,truth,stype,pat,scan_ext = f.split("_")
                if dom!="PROJ" or stype!=P["scan_type"]: continue
                scan = scan_ext.split(".")[0]
                mat = scipy.io.loadmat(os.path.join(result_dir,f))
                rec = FDKHalf()(mat['prj'], get_geometry(), mat['angles'].flatten(), parker=True)
                rec = torch.from_numpy(rec)
                out_dir = os.path.join(RECON_DIR, mv); ensure_dir(out_dir)
                torch.save(rec, os.path.join(out_dir, f"FDK_{truth}_{stype}_{pat}_{scan}.pt"))
            del m; gc.collect()
    logger.info("apply_pd_and_fdk done.")

def aggregate_ct(scans_agg, scan_type, AGG_DIR, pt_recon_dir, AUGMENT_ID):
    ensure_dir(AGG_DIR)
    for sample in ("TRAIN","VALIDATION","TEST"):
        if not scans_agg[sample]: continue
        for truth in (False,True):
            arr = aggregate_saved_recons(scan_type, sample, pt_recon_dir, scans_agg, truth=truth, augment=AUGMENT_ID)
            suff = "gated" if truth else "ng"
            np.save(os.path.join(AGG_DIR,f"FDK_IMAG_{suff}_{scan_type}_{sample}.npy"), arr.numpy())
            del arr
    logger.info("aggregate_ct done.")

STAGES = {
    "convert" : convert_scans,
    "agg_prj" : aggregate_projections,
    "train_pd" : lambda *a: train_stage(*a, "PD"),
    "apply_pd_fdk" : lambda *a: apply_pd_and_fdk(*a, "PD"),
    "train_id" : lambda *a: train_stage(*a, "ID"),
    "apply_id" : lambda *a: apply_id(*a, "ID"),
}

def run_pipeline(steps, context):
    for s in steps:
        STAGES[s](*context[s])