import h5py
import numpy as np
from scipy.interpolate import interp1d
import os

class KFactorInterpolator:
    """
    Efficiently loads and interpolates K-factor values from an HDF5 dataset.
    
    Usage:
        k_bend = KFactorInterpolator("K_bend.hdf5", dataset_name="K_Bend_Data")
        k_value = k_bend.get_k_factor(5e5, 1.75)
    """

    def __init__(self, hdf5_filename, dataset_name="K_Factor_Data"):
        """
        Initializes the interpolator by loading HDF5 data and precomputing interpolators.
        
        Args:
            hdf5_path (str): Path to the HDF5 file containing K-factor data.
            dataset_name (str): Name of the dataset group in HDF5 (default is "K_Factor_Data").
        """
        self.cleaned_k_data = {}  # Dictionary to store loaded data
        self.interpolators = {}  # Dictionary to store interpolation functions


        # Locate the HDF5 file in the same directory as this script
        script_dir = os.path.dirname(os.path.abspath(__file__))
        hdf5_path = os.path.join(script_dir, hdf5_filename)

        # Load data and build interpolators
        with h5py.File(hdf5_path, 'r') as hdf5_file:
            if dataset_name not in hdf5_file:
                raise KeyError(f"Dataset '{dataset_name}' not found in '{hdf5_path}'.")

            rd_group = hdf5_file[dataset_name]
            
            for rd_str in rd_group.keys():
                rd_value = float(rd_str)  # Convert R/D to float
                self.cleaned_k_data[rd_value] = {
                    "Reynolds": rd_group[rd_str]["Reynolds"][:],
                    "K": rd_group[rd_str]["K"][:],
                }
                self.interpolators[rd_value] = interp1d(
                    self.cleaned_k_data[rd_value]["Reynolds"],
                    self.cleaned_k_data[rd_value]["K"],
                    kind='linear',
                    fill_value="extrapolate"
                )

    def get_k_factor(self, re, rd):
        """
        Interpolates the K-factor for a given Reynolds number and R/D ratio.
        
        Args:
            re (float): Reynolds number.
            rd (float): Bend radius-to-diameter ratio (R/D).
        
        Returns:
            float: Interpolated K-factor.
        """
        rd_numeric = np.array(list(self.cleaned_k_data.keys()))

        # Handle cases where R/D is out of dataset range
        if rd <= rd_numeric[0]:  # Below minimum R/D
            return self.interpolators[rd_numeric[0]](re)
        elif rd >= rd_numeric[-1]:  # Above maximum R/D
            return self.interpolators[rd_numeric[-1]](re)
        else:
            # Interpolate between nearest R/D values
            lower_idx = np.searchsorted(rd_numeric, rd) - 1
            upper_idx = lower_idx + 1

            rd_lower, rd_upper = rd_numeric[lower_idx], rd_numeric[upper_idx]
            k_lower, k_upper = self.interpolators[rd_lower](re), self.interpolators[rd_upper](re)

            # Linear interpolation in R/D space
            return k_lower + (k_upper - k_lower) * (rd - rd_lower) / (rd_upper - rd_lower)


script_dir = os.path.dirname(os.path.abspath(__file__))
hdf5_path = os.path.join(script_dir, "K_bend.hdf5")


# Step 1: Load the Interpolator for Bend Data
K_bend = KFactorInterpolator("K_bend.hdf5", dataset_name="K_Factor_Data")
