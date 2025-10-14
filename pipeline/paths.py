import os
from dataclasses import dataclass
from pipeline.utils import ensure_dir
import logging

logger = logging.getLogger("ngCBCT")


@dataclass
class Directories:
    """
    Data class to hold the directories for the data/model paths.

    Attributes:
        mat_projections_dir (str): Absolute path to the directory containing projection `.mat` files.
        pt_projections_dir (str): Absolute path to the directory containing projection `.pt` files.
        projections_aggregate_dir (str): Absolute path to the directory containing aggregated PD data files.
        projections_model_dir (str): Absolute path to the directory containing PD model files.
        projections_results_dir (str): Absolute path to the directory containing PD results files.
        projections_gated_dir (str): Absolute path to the directory containing gated PD data files.
        reconstructions_dir (str): Absolute path to the directory containing FDK reconstruction files.
        reconstructions_gated_dir (str): Absolute path to the directory containing gated FDK reconstruction files.
        pl_reconstructions_dir (str): Absolute path to the directory containing PL reconstruction files.
        images_aggregate_dir (str): Absolute path to the directory containing aggregated ID data files.
        images_model_dir (str): Absolute path to the directory containing ID model files.
        images_results_dir (str): Absolute path to the directory containing ID results files.
        error_maps_dir (str): Absolute path to the directory containing ID absolute error map files.
        error_results_dir (str): Absolute path to the directory containing results files for auxiliary error-predicting models.

    Methods:
        get_model_dir(model_version, domain, ensure_exists=True): Get the directory path for the model of a specific version.
        get_projections_results_dir(model_version, passthrough_num, ensure_exists=True): Get the directory path for the projection results of a specific PD model version.
        get_reconstructions_dir(model_version, passthrough_num, ensure_exists=True): Get the directory path for the reconstructions of a specific PD model version.
        get_images_aggregate_dir(model_version, ensure_exists=True): Get the directory path for the aggregated images of a specific PD model version (after FDK).
        get_images_results_dir(model_version, passthrough_num, ensure_exists=True): Get the directory path for the images results of a specific ID model version.
        get_error_map_dir(model_version, passthrough_num, ensure_exists=True): Get the directory path for the absolute error results of a specific ID model version.
        get_error_results_dir(model_version, passthrough_num, ensure_exists=True): Get the directory path for the results of an auxiliary error-predicting model.

    Note:
        All paths must be absolute paths.
        You only need to specify the paths you want to use.
        If a path is not specified, it will default to `None`.
    """

    mat_projections_dir: str = None
    pt_projections_dir: str = None
    projections_aggregate_dir: str = None
    projections_model_dir: str = None
    projections_results_dir: str = None
    projections_gated_dir: str = None
    reconstructions_dir: str = None
    reconstructions_gated_dir: str = None
    pl_reconstructions_dir: str = None
    images_aggregate_dir: str = None
    images_model_dir: str = None
    images_results_dir: str = None
    error_maps_dir: str = None
    error_results_dir: str = None

    def __post_init__(self):
        for field in self.__dataclass_fields__:
            dir_path = getattr(self, field)
            if dir_path is not None:
                # Throw an error if the path is not an absolute path
                if not os.path.isabs(dir_path):
                    raise ValueError(f"{field} must be an absolute path.")
                
                # Create any directories that do not exist
                ensure_dir(dir_path)

        logger.debug(f"Directories initialized and verified:\n{self}")

    def __str__(self):
        # Print the dataclass fields and their values in a readable format
        return "\n".join(
            f"{field}: {getattr(self, field)}" for field in self.__dataclass_fields__
        )
    
    def get_model_dir(self, model_version, domain, ensure_exists=True):
        """
        Get the directory path for the model of a specific version.
        """
        if domain == "IMAG":
            dir_path = os.path.join(self.images_model_dir, model_version)
        else:
            dir_path = os.path.join(self.projections_model_dir, model_version)

        if ensure_exists:
            ensure_dir(dir_path)

        return dir_path

    def get_projections_results_dir(self, model_version, passthrough_num=None, ensure_exists=True):
        """
        Get the directory path for the projection results of a specific PD model version.
        """
        dir_path = os.path.join(self.projections_results_dir, model_version)

        if passthrough_num is not None:
            dir_path = os.path.join(dir_path, f"passthrough_{passthrough_num:02d}")

        if ensure_exists:
            ensure_dir(dir_path)

        return dir_path

    def get_reconstructions_dir(self, model_version, passthrough_num=None, ensure_exists=True):
        """
        Get the directory path for the reconstructions of a specific PD model version.

        Note:
            If using a reconstructions not from a PD model (e.g., FDK or PL),
            you can just pass that identifier instead (e.g., 'fdk' or 'pl').
        """
        dir_path = os.path.join(self.reconstructions_dir, model_version)

        if passthrough_num is not None:
            dir_path = os.path.join(dir_path, f"passthrough_{passthrough_num:02d}")

        if ensure_exists:
            ensure_dir(dir_path)

        return dir_path

    def get_images_aggregate_dir(self, model_version, ensure_exists=True):
        """
        Get the directory path for the aggregated images of a specific PD model version (after FDK).
        """
        dir_path = os.path.join(self.images_aggregate_dir, model_version)

        if ensure_exists:
            ensure_dir(dir_path)

        return dir_path

    def get_images_results_dir(self, model_version, passthrough_num, ensure_exists=True):
        """
        Get the directory path for the image results of a specific ID model version.
        """
        dir_path = os.path.join(self.images_results_dir, model_version)

        if passthrough_num is not None:
            dir_path = os.path.join(dir_path, f"passthrough_{passthrough_num:02d}")

        if ensure_exists:
            ensure_dir(dir_path)

        return dir_path
    
    def get_error_map_dir(self, model_version, passthrough_num, ensure_exists=True):
        """
        Get the directory path for the absolute error results of a specific ID model version.
        """
        dir_path = os.path.join(self.error_maps_dir, model_version)

        if passthrough_num is not None:
            dir_path = os.path.join(dir_path, f"passthrough_{passthrough_num:02d}")

        if ensure_exists:
            ensure_dir(dir_path)

        return dir_path
    
    def get_error_results_dir(self, model_version, passthrough_num, ensure_exists=True):
        """
        Get the directory path for the results of an auxiliary error-predicting model.
        """
        dir_path = os.path.join(self.error_results_dir, model_version)

        if passthrough_num is not None:
            dir_path = os.path.join(dir_path, f"passthrough_{passthrough_num:02d}")

        if ensure_exists:
            ensure_dir(dir_path)

        return dir_path


class Files:
    """
    Class to hold methods for generating filenames for various data files.

    Attributes:
        directories (Directories): An instance of the Directories class containing the directory paths.

    Methods:
        get_projection_mat_filepath(patient, scan, scan_type, pancreas, liver): Get the absolute file path for the projection `.mat` file.
        get_projection_pt_filepath(patient, scan, scan_type, gated, odd): Get the absolute file path for the projection `.pt` file.
        get_projections_aggregate_filepath(split, gated): Get the absolute file path for the aggregated projections `.npy` file.
        get_model_filepath(model_version, domain, checkpoint, ensure_exists): Get the absolute file path for the trained model file.
        get_projections_results_filepath(model_version, patient, scan, scan_type, gated, odd, ensure_exists): Get the absolute file path for the projection results `.mat` file.
        get_recon_filepath(self, model_version, patient, scan, scan_type, gated, odd, passthrough_num, ensure_exists): Get the absolute file path for the FDK reconstruction `.pt` file.
        get_images_aggregate_filepath(model_version, split, gated, ensure_exists): Get the absolute file path for the aggregated images `.npy` file.
        get_images_results_filepath(model_version, patient, scan, odd, ensure_exists): Get the absolute file path for the image results `.pt` file.
        get_error_results_filepath(model_version, patient, scan, odd, ensure_exists): Get the absolute file path for the error prediction results `.pt` file.
    """
    def __init__(self, directories: Directories):
        self.directories = directories

    @staticmethod
    def _get_projection_mat_filename(patient, scan, scan_type, pancreas, liver):
        """
        Get the filename for the projection `.mat` file based on patient, scan, scan type, and whether it's a pancreas/liver scan.

        Args:
            patient (str): Patient identifier, e.g. '01'.
            scan (str): Scan identifier, e.g. '01'.
            scan_type (str): Type of scan, e.g. 'HF', 'FF'.
            pancreas (bool): Whether the scan is for the pancreas.
            liver (bool): Whether the scan is for the liver.
        Returns:
            str: Filename for the projection `.mat` file.
        """
        if pancreas and liver:
            raise ValueError("pancreas and liver cannot both be True.")

        if pancreas:
            if scan_type == "HF":
                return f"panc{patient}.{scan_type}{scan}.mat" # e.g., panc01.HF01.mat
            else:
                return f"panc{int(patient)}.{scan_type}{scan}.{scan_type}.mat" # e.g., panc01.FF01.FF.mat
        elif liver:
            return f"liver{patient}.{scan_type}{scan}.{scan_type}.mat" # e.g., liver00.HF01.HF.mat
        else:
            return f"p{patient}.{scan_type}{scan}.{scan_type}.mat" # e.g., p01.HF01.HF.mat

    def get_projection_mat_filepath(self, patient, scan, scan_type, pancreas, liver):
        """
        Get the absolute file path for the projection `.mat` file.

        Args:
            patient (str): Patient identifier, e.g. '01'.
            scan (str): Scan identifier, e.g. '01'.
            scan_type (str): Type of scan, e.g. 'HF', 'FF'.
            pancreas (bool): Whether the scan is for the pancreas.
            liver (bool): Whether the scan is for the liver.
        Returns:
            str: Absolute file path for the projection `.mat` file.
        """
        filename = self._get_projection_mat_filename(patient, scan, scan_type, pancreas, liver)
        if liver:
            return os.path.join(self.directories.mat_projections_dir, scan_type, filename)
        else:
            return os.path.join(self.directories.mat_projections_dir, filename)

    @staticmethod
    def _get_projection_pt_filename(patient, scan, scan_type, gated, odd=None):
        """
        Get the filename for the projection `.pt` file based on patient, scan, and scan type.

        Args:
            patient (str): Patient identifier, e.g. '01'.
            scan (str): Scan identifier, e.g. '01'.
            scan_type (str): Type of scan, e.g. 'HF', 'FF'.
            gated (bool): Whether the data is gated or nonstop-gated.
            odd (bool, optional): Whether the data is even or odd indexed.
        Returns:
            str: Filename for the projection `.pt` file.
        """
        if gated:
            return f"gated_{scan_type}_p{patient}_{scan}.pt" # e.g., gated_HF_p01_01.pt
        else:
            if odd is None:
                raise ValueError("odd must be specified for nonstop-gated data.")
            
            if odd:
                return f"ng_{scan_type}_p{patient}_{scan}.pt" # e.g., ng_HF_p01_01.pt
            else:
                return f"ng_{scan_type}_p{patient}_{scan}_even.pt" # e.g., ng_HF_p01_01_even.pt

    def get_projection_pt_filepath(self, patient, scan, scan_type, gated, odd=None):
        """
        Get the absolute file path for the projection `.pt` file.

        Args:
            patient (str): Patient identifier, e.g. '01'.
            scan (str): Scan identifier, e.g. '01'.
            scan_type (str): Type of scan, e.g. 'HF', 'FF'.
            gated (bool): Whether the data is gated or nonstop-gated.
            odd (bool, optional): Whether the data is even or odd indexed.
        Returns:
            str: Absolute file path for the projection `.pt` file.
        """
        filename = self._get_projection_pt_filename(patient, scan, scan_type, gated, odd)
        return os.path.join(self.directories.pt_projections_dir, filename)
    
    @staticmethod
    def _get_projections_aggregate_filename(split, gated):
        """
        Get the filename for the aggregated projections `.npy` file based on split and gating.

        Args:
            split (str): Data split, e.g. 'train', 'val', 'test'.
            gated (bool): Whether the data is gated or nonstop-gated.

        Returns:
            str: Filename for the aggregated projections `.npy` file.
        """
        gated_str = "gated" if gated else "ng"
        split = split.lower()
        return f"{gated_str}_{split}.npy" # e.g., gated_train.npy
    
    def get_projections_aggregate_filepath(self, split, gated):
        """
        Get the absolute file path for the aggregated projections `.npy` file.

        Args:
            split (str): Data split, e.g. 'train', 'val', 'test'.
            gated (bool): Whether the data is gated or nonstop-gated.

        Returns:
            str: Absolute file path for the aggregated projections `.npy` file.
        """
        filename = self._get_projections_aggregate_filename(split, gated)
        return os.path.join(self.directories.projections_aggregate_dir, filename)
    
    @staticmethod
    def _get_checkpoint_swag_params(filename):
        """
        Extract the checkpoint epoch, SWAG learning rate, momentum, and weight decay from the filename.
        Args:
            filename (str): Filename of the model checkpoint.
        Returns:
            tuple: A tuple containing the checkpoint epoch (int), SWAG learning rate (float),
                SWAG momentum (float), and SWAG weight decay (float).
                Note if any SWAG parameters are not present, they will be None.
        """
        # Example filename: epoch-05_swag_lr-1.0e-3_swag_mom-0.9_swag_wd-1.0e-4.pth
        # Split off the .pth extension
        base = filename.replace('.pth', '')
        # Initialize default values
        checkpoint = None
        swag_lr = None
        swag_momentum = None
        swag_weight_decay = None

        # Extract the checkpoint epoch
        if 'epoch-' in base:
            epoch_part = base.split('_')[0]
            checkpoint = int(epoch_part.split('-')[1])

        # Extract SWAG learning rate, momentum, and weight decay if present
        if '_swag_lr-' in base:
            swag_lr_part = base.split('_swag_lr-', 1)[1]
            swag_lr = float(swag_lr_part.split('_')[0])

        if '_swag_mom-' in base:
            swag_momentum_part = base.split('_swag_mom-', 1)[1]
            swag_momentum = float(swag_momentum_part.split('_')[0])

        if '_swag_wd-' in base:
            swag_weight_decay_part = base.split('_swag_wd-', 1)[1]
            swag_weight_decay = float(swag_weight_decay_part.split('_')[0])

        return checkpoint, swag_lr, swag_momentum, swag_weight_decay
    
    @staticmethod
    def _get_model_filename(model_version, checkpoint=None, swag_lr=None, swag_momentum=None, swag_weight_decay=None):
        """
        Get the filename for the trained model file based on model version.

        Args:
            model_version (str): Model version identifier, e.g. 'v1', 'v2'.
            checkpoint (int, optional): If specified, indicates a checkpoint epoch number.
            swag_lr (float, optional): Only used if 'checkpoint' is specified.
                If specified, indicates the learning rate for SWAG.
                If not specified, the normal checkpoint is used.
            swag_momentum (float, optional): Only used if 'checkpoint' is specified.
            swag_weight_decay (float, optional): Only used if 'checkpoint' is specified.

        Returns:
            str: Filename for the trained model file.
        """
        if checkpoint:
            base = f"epoch-{checkpoint:02d}"
            if swag_lr is not None:
                base += f"_swag_lr-{swag_lr:.1e}"
            if swag_momentum is not None:
                base += f"_swag_mom-{swag_momentum:.1f}"
            if swag_weight_decay is not None:
                base += f"_swag_wd-{swag_weight_decay:.1e}"
        else:
            base = model_version

        return base + ".pth"

    def get_model_filepath(self, model_version, domain, checkpoint=None, swag_lr=None, swag_momentum=None, swag_weight_decay=None, ensure_exists=True):
        """
        Get the absolute file path for the trained model file.

        Args:
            model_version (str): Model version identifier, e.g. 'v1', 'v2'.
            domain (str): Domain of the model, either 'PROJ' for projections or 'IMAG' for images.
            checkpoint (int, optional): If specified, indicates a checkpoint epoch number.
            swag_lr (float, optional): Only used if 'checkpoint' is specified.
                If specified, indicates the learning rate for SWAG.
                If not specified, the normal checkpoint is used.
            swag_momentum (float, optional): Only used if 'checkpoint' is specified.
                If specified, indicates the momentum for SWAG.
            swag_weight_decay (float, optional): Only used if 'checkpoint' is specified.
                If specified, indicates the weight decay for SWAG.
            ensure_exists (bool, optional): Whether to ensure the directory exists.

        Returns:
            str: Absolute file path for the trained model file.
        """
        filename = self._get_model_filename(model_version, checkpoint, swag_lr, swag_momentum, swag_weight_decay)
        dir_ = self.directories.get_model_dir(model_version, domain, ensure_exists)
        return os.path.join(dir_, filename)
    
    @staticmethod
    def _get_train_loss_filename():
        """
        Get the filename for the training loss `.pth` file.

        Returns:
            str: Filename for the training loss `.pth` file.
        """
        return "train_loss.pth"
    
    def get_train_loss_filepath(self, model_version, domain, ensure_exists=True):
        """
        Get the absolute file path for the training loss `.pth` file.

        Args:
            model_version (str): Model version identifier, e.g. 'v1', 'v2'.
            domain (str): Domain of the model, either 'PROJ' for projections or 'IMAG' for images.
            ensure_exists (bool, optional): Whether to ensure the directory exists.

        Returns:
            str: Absolute file path for the training loss `.pth` file.
        """
        filename = self._get_train_loss_filename()
        dir_ = self.directories.get_model_dir(model_version, domain, ensure_exists)
        return os.path.join(dir_, filename)
    
    @staticmethod
    def _get_validation_loss_filename():
        """
        Get the filename for the validation loss `.pth` file.

        Returns:
            str: Filename for the validation loss `.pth` file.
        """
        return "validation_loss.pth"

    def get_validation_loss_filepath(self, model_version, domain, ensure_exists=True):
        """
        Get the absolute file path for the validation loss `.pth` file.

        Args:
            model_version (str): Model version identifier, e.g. 'v1', 'v2'.
            domain (str): Domain of the model, either 'PROJ' for projections or 'IMAG' for images.
            ensure_exists (bool, optional): Whether to ensure the directory exists.

        Returns:
            str: Absolute file path for the validation loss `.pth` file.
        """
        filename = self._get_validation_loss_filename()
        dir_ = self.directories.get_model_dir(model_version, domain, ensure_exists)
        return os.path.join(dir_, filename)

    @staticmethod
    def _get_projections_results_filename(patient, scan, gated, scan_type, odd=None):
        """
        Get the filename for the projection results `.mat` file based on patient and scan.

        Args:
            patient (str): Patient identifier, e.g. '01'.
            scan (str): Scan identifier, e.g. '01'.
            gated (bool): Whether the projections are gated or not.
            scan_type (str): Type of scan, either 'HF', 'FF'.
            odd (bool, optional): Whether the data is even or odd indexed.

        Returns:
            str: Filename for the projection results `.mat` file.
        """
        if gated:
            return f"{scan_type}_p{patient}_{scan}.mat" # e.g., p01_01.mat
        
        if odd is None:
            raise ValueError("odd must be specified for nonstop-gated data.")
        
        if odd:
            return f"p{patient}_{scan}.mat" # e.g., p01_01.mat
        else:
            return f"p{patient}_{scan}_even.mat"

    def get_projections_results_filepath(self, model_version, patient, scan, scan_type, gated, odd=None, passthrough_num=None, ensure_exists=True):
        """
        Get the absolute file path for the projection results `.mat` file.

        Args:
            model_version (str): Model version identifier, e.g. 'v1', 'v2'.
            patient (str): Patient identifier, e.g. '01'.
            scan (str): Scan identifier, e.g. '01'.
            scan_type (str): Type of scan, either 'HF', 'FF'.
            gated (bool): Whether the projections are gated or not.
            passthrough_num (int, optional): The passthrough number (leave as None for deterministic).
            ensure_exists (bool, optional): Whether to ensure the directory exists.

        Returns:
            str: Absolute file path for the projection results `.mat` file.
        """
        filename = self._get_projections_results_filename(patient, scan, gated, scan_type, odd)
        
        if gated:
            dir_ = self.directories.projections_gated_dir
        else:
            dir_ = self.directories.get_projections_results_dir(model_version, passthrough_num, ensure_exists)
        
        return os.path.join(dir_, filename)
    
    @staticmethod
    def _get_recon_filename(patient, scan, gated, scan_type, odd=None, pl=False):
        """
        Get the filename for the FDK reconstruction `.pt` file based on patient and scan.

        Args:
            patient (str): Patient identifier, e.g. '01'.
            scan (str): Scan identifier, e.g. '01'.
            gated (bool): Whether the reconstruction is gated or not.
            scan_type (str): Type of scan, either 'HF', 'FF'.
            odd (bool, optional): Whether the data is even or odd indexed.
            pl (bool, optional): Whether the reconstruction is a PL reconstruction.
        Returns:
            str: Filename for the FDK reconstruction `.pt` file.
        """
        if gated:
            return f"{scan_type}_p{patient}_{scan}_gated.pt"
        
        if odd is None:
            raise ValueError("odd must be specified for nonstop-gated data.")
        
        if pl:
            fname = f"recon_panc{patient}.{scan_type}{scan}.u_PL"
            if scan_type == "FF":
                fname += "_ROI"
            fname += ".b2.5"
            if not odd:
                fname += "_even"
            return fname + ".mat"
        
        if odd:
            return f"p{patient}_{scan}.pt" # e.g., p01_01.pt
        else:
            return f"p{patient}_{scan}_even.pt"

    def get_recon_filepath(self, model_version, patient, scan, scan_type, gated, odd=None, passthrough_num=None, ensure_exists=True):
        """
        Get the absolute file path for the FDK reconstruction `.pt` file.

        Args:
            model_version (str): Model version identifier, e.g. 'v1', 'v2'.
            patient (str): Patient identifier, e.g. '01'.
            scan (str): Scan identifier, e.g. '01'.
            scan_type (str): Type of scan, either 'HF', 'FF'.
            gated (bool): Whether the reconstruction is gated or not.
            odd (bool, optional): Whether the data is even or odd indexed.
            passthrough_num (int, optional): The passthrough number (leave as None for deterministic).
            ensure_exists (bool, optional): Whether to ensure the directory exists.

        Returns:
            str: Absolute file path for the FDK reconstruction `.pt` file.
        """
        filename = self._get_recon_filename(patient, scan, gated, scan_type, odd, pl=(model_version=="pl"))

        if gated:
            dir_ = self.directories.reconstructions_gated_dir
        else:
            if model_version == "pl":
                dir_ = self.directories.pl_reconstructions_dir
            else:
                dir_ = self.directories.get_reconstructions_dir(model_version, passthrough_num, ensure_exists)

        return os.path.join(dir_, filename)
    
    @staticmethod
    def _get_images_aggregate_filename(split, truth, error):
        """
        Get the filename for the aggregated images `.npy` file based on data split, truthiness,
        and whether it is for an error-based model.

        Args:
            split (str): Data split, e.g. 'train', 'val', 'test'.
            truth (bool): Whether the data is the ground truth or not.
            error (bool): Whether the data is for an error-based model.

        Returns:
            str: Filename for the aggregated images `.npy` file.
        """
        if truth:
            if not error:
                prefix = "gated" # For normal, ground truth = gated
            else:
                prefix = "truth_error" # For error-based, GT = absolute errors
        else:
            if not error:
                prefix = "ng" # For normal, !GT = !gated
            else:
                prefix = "three_channel" # For error-based, !GT = three channels (pre-PD, post-PD, and post-ID)

        split = split.lower()
        return f"{prefix}_{split}.npy" # e.g., gated_train.npy

    def get_images_aggregate_filepath(self, model_version, split, truth, error=False, ensure_exists=True):
        """
        Get the absolute file path for the aggregated images `.npy` file.

        Args:
            model_version (str): PD model version identifier, e.g. 'v1', 'v2'.
            split (str): Data split, e.g. 'train', 'val', 'test'.
            truth (bool): Whether the data is the ground truth or not.
            error (bool, optional): Whether the data is for an error-based model.
            ensure_exists (bool, optional): Whether to ensure the directory exists.

        Returns:
            str: Absolute file path for the aggregated images `.npy` file.

        Note:
            If using a reconstructions not from a PD model (e.g., FDK or PL),
            you can just pass that identifier instead (e.g., 'fdk' or 'pl').
        """
        if truth and not error and model_version != "fdk":
            raise ValueError("Gated images should be called with model_version='fdk'.")
        filename = self._get_images_aggregate_filename(split, truth, error)
        dir_ = self.directories.get_images_aggregate_dir(model_version, ensure_exists)
        return os.path.join(dir_, filename)

    @staticmethod
    def _get_images_results_filename(patient, scan, odd=None):
        """
        Get the filename for the image results `.pt` file based on patient and scan.

        Args:
            patient (str): Patient identifier, e.g. '01'.
            scan (str): Scan identifier, e.g. '01'.
            odd (bool, optional): Whether the data is even or odd indexed.

        Returns:
            str: Filename for the image results `.pt` file.
        """
        if odd is None:
            raise ValueError("odd must be specified for nonstop-gated data.")
        
        if odd:
            return f"p{patient}_{scan}.pt" # e.g., p01_01.pt
        else:
            return f"p{patient}_{scan}_even.pt"
    
    def get_images_results_filepath(self, model_version, patient, scan, odd=None, passthrough_num=None, ensure_exists=True):
        """
        Get the absolute file path for the image results `.pt` file.

        Args:
            model_version (str): Model version identifier, e.g. 'v1', 'v2'.
            patient (str): Patient identifier, e.g. '01'.
            scan (str): Scan identifier, e.g. '01'.
            odd (bool, optional): Whether the data is even or odd indexed.
            passthrough_num (int, optional): The passthrough number (leave as None for deterministic).
            ensure_exists (bool, optional): Whether to ensure the directory exists.

        Returns:
            str: Absolute file path for the image results `.pt` file.
        """
        filename = self._get_images_results_filename(patient, scan, odd)
        dir_ = self.directories.get_images_results_dir(model_version, passthrough_num, ensure_exists)
        return os.path.join(dir_, filename)

    @staticmethod
    def _get_error_results_filename(patient, scan, odd=None):
        """
        Get the filename for the error prediction results `.pt` file based on patient and scan.

        Args:
            patient (str): Patient identifier, e.g. '01'.
            scan (str): Scan identifier, e.g. '01'.
            odd (bool, optional): Whether the data is even or odd indexed.

        Returns:
            str: Filename for the error prediction results `.pt` file.
        """
        if odd is None:
            raise ValueError("odd must be specified for nonstop-gated data.")
        
        if odd:
            return f"p{patient}_{scan}_error.pt"
        else:
            return f"p{patient}_{scan}_error_even.pt"

    def get_error_results_filepath(self, model_version, patient, scan, odd=None, passthrough_num=None, ensure_exists=True):
        """
        Get the absolute file path for the error prediction results `.pt` file.

        Args:
            model_version (str): Model version identifier, e.g. 'v1', 'v2'.
            patient (str): Patient identifier, e.g. '01'.
            scan (str): Scan identifier, e.g. '01'.
            odd (bool, optional): Whether the data is even or odd indexed.
            passthrough_num (int, optional): The passthrough number (leave as None for deterministic).
            ensure_exists (bool, optional): Whether to ensure the directory exists.

        Returns:
            str: Absolute file path for the error prediction results `.pt` file.
        """
        filename = self._get_error_results_filename(patient, scan, odd)
        dir_ = self.directories.get_error_results_dir(model_version, passthrough_num, ensure_exists)
        return os.path.join(dir_, filename)