import os
from dataclasses import dataclass


@dataclass
class Directories:
    """
    Data class to hold the directories for the data/model paths.

    Attributes:
        mat_projections_dir (str): Path to the directory containing projection `.mat` files.
        pt_projections_dir (str): Path to the directory containing projection `.pt` files.
        projections_aggregate_dir (str): Path to the directory containing aggregated PD data files.
        projections_model_dir (str): Path to the directory containing PD model files.
        projections_results_dir (str): Path to the directory containing PD results files.
        recons_dir (str): Path to the directory containing FDK reconstruction files.
        images_aggregate_dir (str): Path to the directory containing aggregated ID data files.
        images_model_dir (str): Path to the directory containing ID model files.
        images_results_dir (str): Path to the directory containing ID results files.

    Note:
        You only need to specify the paths you want to use.
        If a path is not specified, it will default to `None`.
    """

    mat_projections_dir: str = None
    pt_projections_dir: str = None
    projections_aggregate_dir: str = None
    projections_model_dir: str = None
    projections_results_dir: str = None
    recons_dir: str = None
    images_aggregate_dir: str = None
    images_model_dir: str = None
    images_results_dir: str = None

    def __str__(self):
        # Print the dataclass fields and their values in a readable format
        return "\n".join(
            f"{field}: {getattr(self, field)}" for field in self.__dataclass_fields__
        )
    
    def get_model_projections_results_dir(self, model_version):
        """
        Get the directory path for the projection results of a specific PD model version.
        """
        return os.path.join(self.projections_results_dir, model_version)

    def get_model_recons_dir(self, model_version):
        """
        Get the directory path for the reconstructions of a specific PD model version.

        Note:
            If using a reconstructions not from a PD model (e.g., FDK or PL),
            you can just pass that identifier instead (e.g., 'fdk' or 'pl').
        """
        return os.path.join(self.recons_dir, model_version)

    def get_model_images_aggregate_dir(self, model_version):
        """
        Get the directory path for the aggregated images of a specific PD model version (after FDK).
        """
        return os.path.join(self.images_aggregate_dir, model_version)

    def get_model_images_results_dir(self, model_version):
        """
        Get the directory path for the image results of a specific ID model version.
        """
        return os.path.join(self.images_results_dir, model_version)


class Files:
    """
    Class to hold methods for generating filenames for various data files.
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
        return f"p{patient}.{scan_type}{scan}.{scan_type}.mat"
    
    def get_projection_mat_filepath(self, patient, scan, scan_type):
        """
        Get the full file path for the projection `.mat` file.

        Args:
            patient (str): Patient identifier, e.g. '01'.
            scan (str): Scan identifier, e.g. '01'.
            scan_type (str): Type of scan, e.g. 'HF', 'FF'.
        Returns:
            str: Full file path for the projection `.mat` file.
        """
        filename = self._get_projection_mat_filename(patient, scan, scan_type)
        return os.path.join(self.directories.mat_projections_dir, filename)

    @staticmethod
    def _get_projection_pt_filename(patient, scan, scan_type):
        """
        Get the filename for the projection `.pt` file based on patient, scan, and scan type.

        Args:
            patient (str): Patient identifier, e.g. '01'.
            scan (str): Scan identifier, e.g. '01'.
            scan_type (str): Type of scan, e.g. 'HF', 'FF'.
        Returns:
            str: Filename for the projection `.pt` file.
        """
        return f"{scan_type}_p{patient}_{scan}.pt"
    
    def get_projection_pt_filepath(self, patient, scan, scan_type):
        """
        Get the full file path for the projection `.pt` file.

        Args:
            patient (str): Patient identifier, e.g. '01'.
            scan (str): Scan identifier, e.g. '01'.
            scan_type (str): Type of scan, e.g. 'HF', 'FF'.
        Returns:
            str: Full file path for the projection `.pt` file.
        """
        filename = self._get_projection_pt_filename(patient, scan, scan_type)
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
        return f"{gated_str}_{split}.npy"
    
    def get_projections_aggregate_filepath(self, split, gated):
        """
        Get the full file path for the aggregated projections `.npy` file.

        Args:
            split (str): Data split, e.g. 'train', 'val', 'test'.
            gated (bool): Whether the data is gated or nonstop-gated.

        Returns:
            str: Full file path for the aggregated projections `.npy` file.
        """
        filename = self._get_projections_aggregate_filename(split, gated)
        return os.path.join(self.directories.projections_aggregate_dir, filename)
    
    @staticmethod
    def _get_projections_model_filename(model_version):
        """
        Get the filename for the trained PD model file based on model version.

        Args:
            model_version (str): Model version identifier, e.g. 'v1', 'v2'.

        Returns:
            str: Filename for the trained PD model file.
        """
        return f"{model_version}.pth"
    
    def get_projections_model_filepath(self, model_version):
        """
        Get the full file path for the trained PD model file.

        Args:
            model_version (str): Model version identifier, e.g. 'v1', 'v2'.

        Returns:
            str: Full file path for the trained PD model file.
        """
        filename = self._get_projections_model_filename(model_version)
        return os.path.join(self.directories.projections_model_dir, filename)
    
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
        return f"p{patient}_{scan}.mat"
    
    def get_projections_results_filepath(self, model_version, patient, scan):
        """
        Get the full file path for the projection results `.mat` file.

        Args:
            model_version (str): Model version identifier, e.g. 'v1', 'v2'.
            patient (str): Patient identifier, e.g. '01'.
            scan (str): Scan identifier, e.g. '01'.

        Returns:
            str: Full file path for the projection results `.mat` file.
        """
        filename = self._get_projections_results_filename(patient, scan)
        dir_ = self.directories.get_model_projections_results_dir(model_version)
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
        return f"p{patient}_{scan}.pt"

    def get_recon_filepath(self, model_version, patient, scan):
        """
        Get the full file path for the FDK reconstruction `.pt` file.

        Args:
            model_version (str): Model version identifier, e.g. 'v1', 'v2'.
            patient (str): Patient identifier, e.g. '01'.
            scan (str): Scan identifier, e.g. '01'.

        Returns:
            str: Full file path for the FDK reconstruction `.pt` file.
        """
        filename = self._get_recon_filename(patient, scan)
        dir_ = self.directories.get_model_recons_dir(model_version)
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
        return f"{gated_str}_{split}.npy"
    
    def get_images_aggregate_filepath(self, model_version, split, gated):
        """
        Get the full file path for the aggregated images `.npy` file.

        Args:
            model_version (str): PD model version identifier, e.g. 'v1', 'v2'.
            split (str): Data split, e.g. 'train', 'val', 'test'.
            gated (bool): Whether the data is gated or nonstop-gated.

        Returns:
            str: Full file path for the aggregated images `.npy` file.

        Note:
            If using a reconstructions not from a PD model (e.g., FDK or PL),
            you can just pass that identifier instead (e.g., 'fdk' or 'pl').
        """
        filename = self._get_images_aggregate_filename(split, gated)
        dir_ = self.directories.get_model_images_aggregate_dir(model_version)
        return os.path.join(dir_, filename)
    
    @staticmethod
    def _get_images_model_filename(model_version):
        """
        Get the filename for the trained ID model file based on model version.

        Args:
            model_version (str): Model version identifier, e.g. 'v1', 'v2'.

        Returns:
            str: Filename for the trained ID model file.
        """
        return f"{model_version}.pth"
    
    def get_images_model_filepath(self, model_version):
        """
        Get the full file path for the trained ID model file.

        Args:
            model_version (str): Model version identifier, e.g. 'v1', 'v2'.

        Returns:
            str: Full file path for the trained ID model file.
        """
        filename = self._get_images_model_filename(model_version)
        return os.path.join(self.directories.images_model_dir, filename)

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
        return f"p{patient}_{scan}.pt"
    
    def get_images_results_filepath(self, model_version, patient, scan):
        """
        Get the full file path for the image results `.pt` file.

        Args:
            model_version (str): Model version identifier, e.g. 'v1', 'v2'.
            patient (str): Patient identifier, e.g. '01'.
            scan (str): Scan identifier, e.g. '01'.

        Returns:
            str: Full file path for the image results `.pt` file.
        """
        filename = self._get_images_results_filename(patient, scan)
        dir_ = self.directories.get_model_images_results_dir(model_version)
        return os.path.join(dir_, filename)
