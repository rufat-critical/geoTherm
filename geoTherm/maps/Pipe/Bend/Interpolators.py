import pandas as pd
from scipy.interpolate import interp1d
from importlib.resources import files
import numpy as np


class KInterpolator:
    def __init__(self, data_file, x_name, y_name, interp_type='log'):
        # Try loading the file as a package resource
        if not data_file.endswith('.csv'):
            raise ValueError("Expected a CSV file")

        if not '/' in data_file and not '\\' in data_file:
            # Only filename provided, use default package location
            data_file = files("geoTherm.maps.Pipe.Bend").joinpath(data_file)
            with data_file.open("r") as f:
                self.data = pd.read_csv(f)
        else:
            self.data = pd.read_csv(data_file)

        self.dnames = []
        self.x_name = x_name
        self.y_name = y_name
        self.initialize(interp_type)

    def initialize(self, interp_type):

        self.dnames = list(self.data.columns[::2])  # Step by 2 starting from index 0

        # Rename unnamed columns based on previous column name
        for i in range(1, len(self.data.columns), 2):  # Step by 2 to check odd columns
            if self.data.columns[i].startswith('Unnamed'):
                prev_col_name = self.data.columns[i-1]
                self.data = self.data.rename(columns={self.data.columns[i]: f'{prev_col_name}'})

        # Get the first row which contains x and y names
        header_row = self.data.iloc[0]
        # Get actual data (excluding the header row)
        self.data = self.data.iloc[1:]

        self.interps = {name: {'x': None, 'y': None} for name in self.dnames}

        # First find all x columns and their indices
        for i, (header_name, header_value) in enumerate(header_row.items()):

            if header_value == self.x_name:
                self.interps[header_name]['x'] = self.data.iloc[:, i].astype(float).values
            elif header_value == self.y_name:
                self.interps[header_name]['y'] = self.data.iloc[:, i].astype(float).values

        # Loop and generate interpolations
        for name in self.dnames:
            self.interps[name] = self.generate_interpolation(name,
                                                             self.interps[name]['x'],
                                                             self.interps[name]['y'],
                                                             interp_type)

    def generate_interpolation(self, name, x, y, interp_type='cubic'):
        # Sort x and y together (required for interpolation)
        sort_idx = np.argsort(x)
        x = x[sort_idx]
        y = y[sort_idx]

        if interp_type == 'cubic':
            # Create interpolation function in log space
            f = interp1d(x, y, kind='cubic', bounds_error=False, fill_value='extrapolate')

            # Return a wrapper function that handles the log transformation
            return f
        else:
            from pdb import set_trace
            set_trace()
    

class BendCorrectionInterpolator(KInterpolator):
    def __init__(self,
                 data_file='BendCorrection.csv',
                 x_name = 'theta',
                 y_name = 'C',
                 interp_type = 'cubic'):
        
        # Convert relative path to absolute path
        super().__init__(data_file, x_name, y_name, interp_type)

    def evaluate(self, theta):
        return self.interps[self.dnames[0]](theta)

class Bend90Interpolator(KInterpolator):
    def __init__(self, data_file='90Bend.csv',
                 x_name='Re',
                 y_name='K', interp_type='cubic'):
        # Convert relative path to absolute path
        super().__init__(data_file, x_name, y_name, interp_type)


    def initialize(self, interp_type):
        super().initialize(interp_type)

        # Convert dictionary keys from 'RD=X' to float X
        self.interps = {float(key.split('=')[1]): value 
                       for key, value in self.interps.items()}

    def evaluate(self, Re, RD):
        # Get sorted list of available RD values
        rd_values = sorted(list(self.interps.keys()))
        
        # Find the bracketing RD values
        if RD <= rd_values[0]:
            return self.interps[rd_values[0]](Re)
        elif RD >= rd_values[-1]:
            return self.interps[rd_values[-1]](Re)
        
        # Find the two RD values that bracket our target RD
        for i in range(len(rd_values)-1):
            if rd_values[i] <= RD <= rd_values[i+1]:
                rd_low = rd_values[i]
                rd_high = rd_values[i+1]
                break
        
        # Get K values at the given Re for both RD values
        k_low = self.interps[rd_low](Re)
        k_high = self.interps[rd_high](Re)
        
        # Linear interpolation between the two K values
        weight = (RD - rd_low) / (rd_high - rd_low)
        k_interpolated = k_low + weight * (k_high - k_low)
        
        return k_interpolated

class KBendInterpolator(KInterpolator):
    def __init__(self, BendData='90Bend.csv',
                 BendCorrectionData='BendCorrection.csv', interp_type='cubic'):
        self.BendData = pd.read_csv(BendData)
        self.BendCorrectionData = pd.read_csv(BendCorrectionData)

    def get_K(self, Re, bend_angle):
        from pdb import set_trace
        set_trace()

class KBend(KInterpolator):
    def __init__(self, Bend90_Interpolator=Bend90Interpolator(),
                 BendCorrection_Interpolator=BendCorrectionInterpolator()):

        self.Bend90_Interpolator = Bend90_Interpolator
        self.BendCorrection_Interpolator = BendCorrection_Interpolator

    def evaluate(self, Re, RD, bend_angle):

        # Interpolate Bend90
        K_90 = self.Bend90_Interpolator.evaluate(Re, RD)

        # Interpolate BendCorrection
        C = self.BendCorrection_Interpolator.evaluate(bend_angle)

        # Calculate K
        K = K_90 * C

        return K


# Initialize Bend
KBend = KBend()
