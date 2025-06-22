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
        reconstructions_dir (str): Absolute path to the directory containing FDK reconstruction files.
        images_aggregate_dir (str): Absolute path to the directory containing aggregated ID data files.
        images_model_dir (str): Absolute path to the directory containing ID model files.
        images_results_dir (str): Absolute path to the directory containing ID results files.

    Methods:
        get_projections_model_dir(model_version, ensure_exists=True): Get the directory path for the PD model of a specific version.
        get_projections_results_dir(model_version, ensure_exists=True): Get the directory path for the projection results of a specific PD model version.
        get_reconstructions_dir(model_version, ensure_exists=True): Get the directory path for the reconstructions of a specific PD model version.
        get_images_aggregate_dir(model_version, ensure_exists=True): Get the directory path for the aggregated images of a specific PD model version (after FDK).
        get_images_model_dir(model_version, ensure_exists=True): Get the directory path for the ID model of a specific version.
        get_images_results_dir(model_version, ensure_exists=True): Get the directory path for the image

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
    reconstructions_dir: str = None
    images_aggregate_dir: str = None
    images_model_dir: str = None
    images_results_dir: str = None

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
    
    def get_projections_model_dir(self, model_version, ensure_exists=True):
        """
        Get the directory path for the PD model of a specific version.
        """
        dir_path = os.path.join(self.projections_model_dir, model_version)
        
        if ensure_exists:
            ensure_dir(dir_path)

        return dir_path
    
    def get_projections_results_dir(self, model_version, ensure_exists=True):
        """
        Get the directory path for the projection results of a specific PD model version.
        """
        dir_path = os.path.join(self.projections_results_dir, model_version)
        
        if ensure_exists:
            ensure_dir(dir_path)

        return dir_path

    def get_reconstructions_dir(self, model_version, ensure_exists=True):
        """
        Get the directory path for the reconstructions of a specific PD model version.

        Note:
            If using a reconstructions not from a PD model (e.g., FDK or PL),
            you can just pass that identifier instead (e.g., 'fdk' or 'pl').
        """
        dir_path = os.path.join(self.reconstructions_dir, model_version)

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
    
    def get_images_model_dir(self, model_version, ensure_exists=True):
        """
        Get the directory path for the ID model of a specific version.
        """
        dir_path = os.path.join(self.images_model_dir, model_version)

        if ensure_exists:
            ensure_dir(dir_path)

        return dir_path

    def get_images_results_dir(self, model_version, ensure_exists=True):
        """
        Get the directory path for the image results of a specific ID model version.
        """
        dir_path = os.path.join(self.images_results_dir, model_version)

        if ensure_exists:
            ensure_dir(dir_path)

        return dir_path


class Files:
    """
    Class to hold methods for generating filenames for various data files.

    Attributes:
        directories (Directories): An instance of the Directories class containing the directory paths.

    Methods:
        get_projection_mat_filepath(patient, scan, scan_type): Get the absolute file path for the projection `.mat` file.
        get_projection_pt_filepath(patient, scan, scan_type): Get the absolute file path for the projection `.pt` file.
        get_projections_aggregate_filepath(split, gated): Get the absolute file path for the aggregated projections `.npy` file.
        get_projections_model_filepath(model_version, checkpoint, ensure_exists): Get the absolute file path for the trained PD model file.
        get_projections_results_filepath(model_version, patient, scan, ensure_exists): Get the absolute file path for the projection results `.mat` file.
        get_recon_filepath(model_version, patient, scan, ensure_exists): Get the absolute file path for the FDK reconstruction `.pt` file.
        get_images_aggregate_filepath(model_version, split, gated, ensure_exists): Get the absolute file path for the aggregated images `.npy` file.
        get_images_model_filepath(model_version, checkpoint, ensure_exists): Get the absolute file path for the trained ID model file.
        get_images_results_filepath(model_version, patient, scan, ensure_exists): Get the absolute file path for the image results `.pt` file.
    """
    def __init__(self, directories: Directories):
        self.directories = directories

    @staticmethod
    def _get_projection_mat_filename(patient, scan, scan_type):
        """
        Get the filename for the projection `.mat` file based on patient, scan, and scan type.

        Args:
            patient (str): Patient identifier, e.g. '01'.
            scan (str): Scan identifier, e.g. '01'.
            scan_type (str): Type of scan, e.g. 'HF', 'FF'.
        Returns:
            str: Filename for the projection `.mat` file.
        """
        return f"p{patient}.{scan_type}{scan}.{scan_type}.mat" # e.g., p01.HF01.HF.mat
    
    def get_projection_mat_filepath(self, patient, scan, scan_type):
        """
        Get the absolute file path for the projection `.mat` file.

        Args:
            patient (str): Patient identifier, e.g. '01'.
            scan (str): Scan identifier, e.g. '01'.
            scan_type (str): Type of scan, e.g. 'HF', 'FF'.
            ensure_exists (bool, optional): Whether to ensure the directory exists.
        Returns:
            str: Absolute file path for the projection `.mat` file.
        """
        filename = self._get_projection_mat_filename(patient, scan, scan_type)
        return os.path.join(self.directories.mat_projections_dir, filename)

    @staticmethod
    def _get_projection_pt_filename(patient, scan, scan_type, gated):
        """
        Get the filename for the projection `.pt` file based on patient, scan, and scan type.

        Args:
            patient (str): Patient identifier, e.g. '01'.
            scan (str): Scan identifier, e.g. '01'.
            scan_type (str): Type of scan, e.g. 'HF', 'FF'.
            gated (bool): Whether the data is gated or nonstop-gated.
        Returns:
            str: Filename for the projection `.pt` file.
        """
        gated_str = "gated" if gated else "ng"
        return f"{gated_str}_{scan_type}_p{patient}_{scan}.pt" # e.g., gated_HF_p01_01.pt

    def get_projection_pt_filepath(self, patient, scan, scan_type, gated):
        """
        Get the absolute file path for the projection `.pt` file.

        Args:
            patient (str): Patient identifier, e.g. '01'.
            scan (str): Scan identifier, e.g. '01'.
            scan_type (str): Type of scan, e.g. 'HF', 'FF'.
            gated (bool): Whether the data is gated or nonstop-gated.
        Returns:
            str: Absolute file path for the projection `.pt` file.
        """
        filename = self._get_projection_pt_filename(patient, scan, scan_type, gated)
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
    def _get_projections_model_filename(model_version, checkpoint=None):
        """
        Get the filename for the trained PD model file based on model version.

        Args:
            model_version (str): Model version identifier, e.g. 'v1', 'v2'.
            checkpoint (int, optional): If specified, indicates a checkpoint epoch number.

        Returns:
            str: Filename for the trained PD model file.
        """
        if checkpoint:
            return f"epoch-{checkpoint:02d}.pth" # e.g., epoch-05.pth

        return f"{model_version}.pth" # e.g., v1.pth

    def get_projections_model_filepath(self, model_version, checkpoint=None, ensure_exists=True):
        """
        Get the absolute file path for the trained PD model file.

        Args:
            model_version (str): Model version identifier, e.g. 'v1', 'v2'.
            checkpoint (int, optional): If specified, indicates a checkpoint epoch number.
            ensure_exists (bool, optional): Whether to ensure the directory exists.

        Returns:
            str: Absolute file path for the trained PD model file.
        """
        filename = self._get_projections_model_filename(model_version, checkpoint)
        dir_ = self.directories.get_projections_model_dir(model_version, ensure_exists)
        return os.path.join(dir_, filename)

    @staticmethod
    def _get_projections_results_filename(patient, scan):
        """
        Get the filename for the projection results `.mat` file based on patient and scan.

        Args:
            patient (str): Patient identifier, e.g. '01'.
            scan (str): Scan identifier, e.g. '01'.

        Returns:
            str: Filename for the projection results `.mat` file.
        """
        return f"p{patient}_{scan}.mat" # e.g., p01_01.mat
    
    def get_projections_results_filepath(self, model_version, patient, scan, ensure_exists=True):
        """
        Get the absolute file path for the projection results `.mat` file.

        Args:
            model_version (str): Model version identifier, e.g. 'v1', 'v2'.
            patient (str): Patient identifier, e.g. '01'.
            scan (str): Scan identifier, e.g. '01'.
            ensure_exists (bool, optional): Whether to ensure the directory exists.

        Returns:
            str: Absolute file path for the projection results `.mat` file.
        """
        filename = self._get_projections_results_filename(patient, scan)
        dir_ = self.directories.get_projections_results_dir(model_version, ensure_exists)
        return os.path.join(dir_, filename)
    
    @staticmethod
    def _get_recon_filename(patient, scan):
        """
        Get the filename for the FDK reconstruction `.pt` file based on patient and scan.

        Args:
            patient (str): Patient identifier, e.g. '01'.
            scan (str): Scan identifier, e.g. '01'.

        Returns:
            str: Filename for the FDK reconstruction `.pt` file.
        """
        return f"p{patient}_{scan}.pt" # e.g., p01_01.pt

    def get_recon_filepath(self, model_version, patient, scan, ensure_exists=True):
        """
        Get the absolute file path for the FDK reconstruction `.pt` file.

        Args:
            model_version (str): Model version identifier, e.g. 'v1', 'v2'.
            patient (str): Patient identifier, e.g. '01'.
            scan (str): Scan identifier, e.g. '01'.
            ensure_exists (bool, optional): Whether to ensure the directory exists.

        Returns:
            str: Absolute file path for the FDK reconstruction `.pt` file.
        """
        filename = self._get_recon_filename(patient, scan)
        dir_ = self.directories.get_reconstructions_dir(model_version, ensure_exists)
        return os.path.join(dir_, filename)
    
    @staticmethod
    def _get_images_aggregate_filename(split, gated):
        """
        Get the filename for the aggregated images `.npy` file based on data split and gating.

        Args:
            split (str): Data split, e.g. 'train', 'val', 'test'.
            gated (bool): Whether the data is gated or nonstop-gated.

        Returns:
            str: Filename for the aggregated images `.npy` file.
        """
        gated_str = "gated" if gated else "ng"
        split = split.lower()
        return f"{gated_str}_{split}.npy" # e.g., gated_train.npy

    def get_images_aggregate_filepath(self, model_version, split, gated, ensure_exists=True):
        """
        Get the absolute file path for the aggregated images `.npy` file.

        Args:
            model_version (str): PD model version identifier, e.g. 'v1', 'v2'.
            split (str): Data split, e.g. 'train', 'val', 'test'.
            gated (bool): Whether the data is gated or nonstop-gated.
            ensure_exists (bool, optional): Whether to ensure the directory exists.

        Returns:
            str: Absolute file path for the aggregated images `.npy` file.

        Note:
            If using a reconstructions not from a PD model (e.g., FDK or PL),
            you can just pass that identifier instead (e.g., 'fdk' or 'pl').
        """
        filename = self._get_images_aggregate_filename(split, gated)
        dir_ = self.directories.get_images_aggregate_dir(model_version, ensure_exists)
        return os.path.join(dir_, filename)
    
    @staticmethod
    def _get_images_model_filename(model_version, checkpoint=None):
        """
        Get the filename for the trained ID model file based on model version.

        Args:
            model_version (str): Model version identifier, e.g. 'v1', 'v2'.
            checkpoint (int, optional): If specified, indicates a checkpoint epoch number.

        Returns:
            str: Filename for the trained ID model file.
        """
        if checkpoint:
            return f"epoch-{checkpoint:02d}.pth" # e.g., epoch-05.pth
        
        return f"{model_version}.pth"

    def get_images_model_filepath(self, model_version, checkpoint=None, ensure_exists=True):
        """
        Get the absolute file path for the trained ID model file.

        Args:
            model_version (str): Model version identifier, e.g. 'v1', 'v2'.
            checkpoint (int, optional): If specified, indicates a checkpoint epoch number.
            ensure_exists (bool, optional): Whether to ensure the directory exists.

        Returns:
            str: Absolute file path for the trained ID model file.
        """
        filename = self._get_images_model_filename(model_version, checkpoint)
        dir_ = self.directories.get_images_model_dir(model_version, ensure_exists)
        return os.path.join(dir_, filename)

    @staticmethod
    def _get_images_results_filename(patient, scan):
        """
        Get the filename for the image results `.pt` file based on patient and scan.

        Args:
            patient (str): Patient identifier, e.g. '01'.
            scan (str): Scan identifier, e.g. '01'.

        Returns:
            str: Filename for the image results `.pt` file.
        """
        return f"p{patient}_{scan}.pt" # e.g., p01_01.pt
    
    def get_images_results_filepath(self, model_version, patient, scan, ensure_exists=True):
        """
        Get the absolute file path for the image results `.pt` file.

        Args:
            model_version (str): Model version identifier, e.g. 'v1', 'v2'.
            patient (str): Patient identifier, e.g. '01'.
            scan (str): Scan identifier, e.g. '01'.
            ensure_exists (bool, optional): Whether to ensure the directory exists.

        Returns:
            str: Absolute file path for the image results `.pt` file.
        """
        filename = self._get_images_results_filename(patient, scan)
        dir_ = self.directories.get_images_results_dir(model_version, ensure_exists)
        return os.path.join(dir_, filename)
