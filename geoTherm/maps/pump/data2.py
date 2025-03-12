import h5py
import numpy as np
# Load the existing HDF5 file and convert pressures from PSI to bar
hdf5_filename_old = "D35e.h5"  # Original file in PSI
hdf5_filename_new = "D35e_bar.h5"  # New file in bar

# Conversion factor: 1 PSI = 0.0689476 bar
PSI_TO_BAR = 0.0689476

# Load and convert data
data_dict_bar = {}
with h5py.File(hdf5_filename_old, "r") as hdf5_file:
    for key in hdf5_file.keys():
        pressure_bar = round(float(key))  # Convert PSI to bar
        rpm = np.array(hdf5_file[key]["RPM"])
        q = np.array(hdf5_file[key]["Q"])

        data_dict_bar[str(round(pressure_bar, 4))] = {
            "RPM": rpm.tolist(),
            "Q": q.tolist()
        }

# Save the updated data in bar
with h5py.File(hdf5_filename_new, "w") as hdf5_file:
    for key, subdict in data_dict_bar.items():
        group = hdf5_file.create_group(key)
        group.create_dataset("RPM", data=subdict["RPM"])
        group.create_dataset("Q", data=subdict["Q"])

print(f"Data converted to bar and saved in {hdf5_filename_new}.")
