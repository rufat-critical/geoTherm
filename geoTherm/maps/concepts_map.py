from geoTherm.utilities.loaders import concepts_excel_reader
import numpy as np
from scipy.interpolate import griddata, LinearNDInterpolator, NearestNDInterpolator
from geoTherm.common import logger
import pandas as pd
from geoTherm.units import toSI


def find_data_locations(df, sheet_name, row_names):
    """
    Find the row and column indices for specified row names in a DataFrame.
    Handles variable column structure where each variable has its own column group.
    Extracts data from ALL columns containing each variable name.
    
    Args:
        df (pd.DataFrame): DataFrame to search in
        sheet_name (str): Name of the sheet being processed
        row_names (list): List of row names to find
        
    Returns:
        dict: Dictionary with row names as keys and location info as values
               Each value is a list of dicts with 'units' and 'data' tuples (row_idx, col_idx)
    """
    data_locations = {}
    
    # Initialize data_locations with empty lists for each row_name
    for row_name in row_names:
        data_locations[row_name] = []
    
    # Search through all cells to find variable names
    for row_idx in range(len(df)):
        for col_idx in range(len(df.columns)):
            cell_value = str(df.iloc[row_idx, col_idx]).strip()
            
            # Check if this cell contains any of our target row names
            for row_name in row_names:
                if row_name == cell_value:
                    # Found a variable name - look for its associated data
                    unit_col_idx = col_idx + 1
                    data_start_col_idx = unit_col_idx + 1
                    
                    # Verify that the next column contains a unit (not another variable name)
                    if unit_col_idx < len(df.columns):
                        unit_cell = str(df.iloc[row_idx, unit_col_idx]).strip()
                        # Check if this looks like a unit (not another variable name)
                        if unit_cell and unit_cell not in row_names and not unit_cell.isdigit():
                            # This looks like a valid unit column
                            location_info = {
                                'units': (row_idx, unit_col_idx),
                                'data': (row_idx, data_start_col_idx)
                            }
                            data_locations[row_name].append(location_info)
                        else:
                            # No unit column found, but still record the data location
                            location_info = {
                                'units': (row_idx, col_idx),  # Use same column for units
                                'data': (row_idx, col_idx + 1)  # Data starts next column
                            }
                            data_locations[row_name].append(location_info)
                    else:
                        # No unit column found, but still record the data location
                        location_info = {
                            'units': (row_idx, col_idx),  # Use same column for units
                            'data': (row_idx, col_idx + 1)  # Data starts next column
                        }
                        data_locations[row_name].append(location_info)

    # Check for any missing row names
    missing_names = [name for name in row_names if not data_locations[name]]
    if missing_names:
        print(f"Warning: Could not find the following row names in sheet {sheet_name}: {missing_names}")

    return data_locations


def extract_row_data(df, data_locations):
    """
    Extract data for specified row locations from a DataFrame.
    Handles horizontal data structure: variable, units, data... variable, units, data...

    Args:
        df (pd.DataFrame): DataFrame to extract data from
        data_locations (dict): Dictionary with row names as keys and location info as values
                              Each value is a list of location dicts

    Returns:
        dict: Dictionary with row names as keys and extracted data as values
    """
    extracted_data = {}

    for row_name, location_list in data_locations.items():
        if location_list:  # Check if the list is not empty
            all_values = []
            unit = None

            # Process all locations for this variable
            for location in location_list:
                # Get the row that contains the variable name
                var_row_idx = location['data'][0]
                data_start_col = location['data'][1]

                # Extract unit from the unit column (use first non-None unit found)
                if unit is None:
                    unit = df.iloc[var_row_idx, location['units'][1]] if len(df.columns) > location['units'][1] else None

                # Extract data values from the same row, starting from data_start_col
                # Continue until we hit another variable name or end of row
                row_data = df.iloc[var_row_idx, data_start_col:].tolist()

                # Find where the data ends (when we hit another variable name or empty cell)
                data_end_col = data_start_col
                for col_idx in range(data_start_col, len(df.columns)):
                    cell_value = str(df.iloc[var_row_idx, col_idx]).strip()

                    # Stop if we hit another variable name
                    if cell_value in [name for name in data_locations.keys()]:
                        break

                    # Stop if we hit an empty cell
                    if not cell_value or cell_value == '' or cell_value == 'nan':
                        break

                    data_end_col = col_idx + 1

                # Extract the data values from the identified range
                data_values = df.iloc[var_row_idx, data_start_col:data_end_col].dropna().tolist()

                # Filter out non-numeric values
                valid_values = []
                for val in data_values:
                    try:
                        float_val = float(val)
                        valid_values.append(float_val)
                    except (ValueError, TypeError):
                        continue

                all_values.extend(valid_values)

            extracted_data[row_name] = {
                'unit': unit,
                'values': all_values,
            }
        else:
            extracted_data[row_name] = {
                'unit': None,
                'values': [],
            }

    return extracted_data


def concepts_data_reader(xlsx_path, row_names, sheet_names=None):
    """ 
    Load Excel data from specified sheets and return a combined DataFrame.

    Args:
        xlsx_path (str): Path to Excel file
        row_names (list): List of row names to read
        sheet_names (list, optional): List of sheet names to read. If None, all sheets are read.

    Returns:
        tuple: (combined DataFrame, dict of units)
                - Combined DataFrame contains all data from all sheets with a 'sheet' column
                - Units dict has variable names as keys and their units as values
    """
    # Load excel file
    xls = pd.read_excel(xlsx_path, sheet_name=None, header=None)

    # Determine which sheets to process
    if sheet_names is None:
        sheets_to_process = list(xls.keys())
    else:
        sheets_to_process = [sheet for sheet in sheet_names if sheet in xls.keys()]

    # Initialize containers
    all_dataframes = []
    units_dict = {}

    for sheet in sheets_to_process:
        data_locations = find_data_locations(xls[sheet], sheet, row_names)

        if not data_locations: # Check if data_locations is empty
            continue

        sheet_data = extract_row_data(xls[sheet], data_locations)

        # Create DataFrame for this sheet with just the values
        sheet_df_data = {}
        for row_name in row_names:
            if row_name in sheet_data:
                sheet_df_data[row_name] = sheet_data[row_name]['values']
                # Store units (use first non-None unit found)
                if row_name not in units_dict and sheet_data[row_name]['unit'] is not None:
                    units_dict[row_name] = sheet_data[row_name]['unit']
            else:
                sheet_df_data[row_name] = []

        # Create DataFrame and pad with NaN to make all columns same length
        max_length = max(len(values) for values in sheet_df_data.values()) if sheet_df_data else 0
        for row_name in sheet_df_data:
            if len(sheet_df_data[row_name]) < max_length:
                sheet_df_data[row_name].extend([None] * (max_length - len(sheet_df_data[row_name])))

        # Create DataFrame and drop rows with any NaN values
        df = pd.DataFrame(sheet_df_data)
        df_clean = df.dropna().copy()  # Add .copy() to create an explicit copy

        # Only add sheet name column if DataFrame is not empty
        if not df_clean.empty:
            df_clean['sheet'] = sheet
            all_dataframes.append(df_clean)

    combined_df = pd.concat(all_dataframes, ignore_index=True)

    # Remove duplicate rows
    print(f"Before removing duplicates: {len(combined_df)} rows")
    combined_df = combined_df.drop_duplicates()
    print(f"After removing duplicates: {len(combined_df)} rows")

    return combined_df, units_dict


# Variable mapping configuration
RITAL_MAPPING = {
    'STAGE InletTotalPressure': 'P0',
    'STAGE InletTotalTemperature': 'T0', 
    'STAGE Pexit': 'P_out',
    "STAGE PR_TS": 'PR_ts',
    "STAGE Power": 'Power',
    "STAGE RPM": 'RPM',
    'STAGE MFLOW CORRECTED': 'm_c',
    'STAGE EFF_TS': 'ETA_ts'
}

AXIAL_MAPPING = {'p00.in': 'P0',
                 'T00.in': 'T0',
                 'p.out': 'P_out',
                 'T.out': 'T_out',
                 'PR_ts': 'PR_ts',
                 'm.out': 'massflow',
                 'N': 'RPM',
                 'Power(Shaft)': 'Power',
                 'ETA_ts_ad': 'ETA_ts'}

Quantities = {
    'P0': 'PRESSURE',
    'T0': 'TEMPERATURE',
    'P_out': 'PRESSURE',
    'T_out': 'TEMPERATURE',
    'Power': 'POWER',
    'massflow': 'MASSFLOW',
    'm_c': 'MASSFLOW',
    'RPM': 'RPM',
}


# Unit corrections
UNIT_CORRECTIONS = {
    'KPa': 'kPa',
    'Kg/s': 'kg/s',
    'Bar': 'bar',
    'K': 'degK',
    }


def concepts_reader(xlsx_path, type='axial', sheet_names=None):

    if type == 'axial':
        mapping = AXIAL_MAPPING
    elif type == 'rital':
        mapping = RITAL_MAPPING
    else:
        raise ValueError(f"Invalid type: {type}")

    df, units = concepts_data_reader(xlsx_path, list(mapping.keys()), sheet_names)

    # Apply mapping to rename dataframe columns
    df = df.rename(columns=mapping)

    # Apply mapping to rename units dictionary keys
    units = {mapping[key]: value for key, value in units.items() if key in mapping}

    # Apply unit corrections to the units dictionary
    for key, unit in units.items():
        if unit is not None:
            # Apply unit corrections if the unit exists in UNIT_CORRECTIONS
            corrected_unit = UNIT_CORRECTIONS.get(unit, unit)
            units[key] = corrected_unit

    # Loop through each variable and convert to SI if it's in Quantities
    for column in df.columns:
        if column in Quantities:
            unit = units[column]
            # Get the quantity type for this variable
            quantity_type = Quantities[column]
            # Convert the data to SI units
            data_array = df[column].to_numpy()
            converted_data = toSI((data_array, unit), quantity_type)

            # Update the dataframe with converted values
            df[column] = converted_data

    # Generate corrected mass flow for axial maps
    if type == 'axial' and 'massflow' in df.columns:
        T_ref = 288.15  # K
        p_ref = 101325  # Pa

        # Calculate corrected mass flow from actual mass flow
        df['m_c'] = df['massflow'] * np.sqrt(df['T0']/T_ref) * (p_ref/df['P0'])

        # Add units for corrected mass flow
        units['m_c'] = 'kg/s'

    df = data_processor(df)

    return df


def data_processor(df):
    df_clean = df.copy()
    
    # Remove duplicate points (exact duplicates)
    df_clean = df.drop_duplicates()
    
    # Remove points that are too close together
    # Calculate distances between points in normalized space
    T_norm = (df_clean['T0'] - df_clean['T0'].min()) / (df_clean['T0'].max() - df_clean['T0'].min())
    P_norm = (df_clean['P0'] - df_clean['P0'].min()) / (df_clean['P0'].max() - df_clean['P0'].min())
    PR_norm = (df_clean['PR_ts'] - df_clean['PR_ts'].min()) / (df_clean['PR_ts'].max() - df_clean['PR_ts'].min())
    RPM_norm = (df_clean['RPM'] - df_clean['RPM'].min()) / (df_clean['RPM'].max() - df_clean['RPM'].min())
    
    # Calculate pairwise distances
    points = np.column_stack((T_norm, P_norm, PR_norm, RPM_norm))
    
    # Remove points that are too close (distance < threshold)
    threshold = 1e-6  # Adjust this value as needed
    keep_indices = []
    
    for i in range(len(points)):
        keep_point = True
        for j in keep_indices:
            if np.linalg.norm(points[i] - points[j]) < threshold:
                keep_point = False
                break
        if keep_point:
            keep_indices.append(i)
    
    # Update data with cleaned version
    df_clean.iloc[keep_indices].reset_index(drop=True)

    logger.info(f"Data cleaned: {len(df)} -> {len(keep_indices)} points")

    return df_clean

class ConceptsMap:
    """Class for interpolating turbine performance parameters."""

    def __init__(self, data_csv=None, type='axial', sheet_names=None, fallback_method='extrapolate', map=None):
        """
        Initialize interpolator with data.

        Args:
            data_csv (str): Path to CSV file containing turbine data
            type (str): Type of turbine map ('axial' or 'rital')
            sheet_names (list, optional): List of sheet names to read
            fallback_method (str): Method for handling extrapolation ('extrapolate' or 'nearest')
        """


        if map is None:
            self.data_csv = data_csv
            self.map = concepts_reader(data_csv, type, sheet_names)
        else:
            self.map = map


        self.fallback_method = fallback_method
        self.T_ref = 288.15  # K
        self.p_ref = 101325  # Pa

        self.initialize()

    def initialize(self):
        # Initialize interpolators

        # Initialize flags
        self.fixed_RPM = False
        self.fixed_T0 = False
        self.fixed_P_out = False

        if len(self.map['RPM'].unique()) == 1:
            self.fixed_RPM = True
        if len(self.map['T0'].unique()) == 1:
            self.fixed_T0 = True
        if len(self.map['P_out'].unique()) == 1:
            self.fixed_P_out = True

        self.T_range = self.map['T0'].max() - self.map['T0'].min()
        self.P_range = self.map['P0'].max() - self.map['P0'].min()
        self.RPM_range = self.map['RPM'].max() - self.map['RPM'].min()

        # Store original ranges for denormalization
        self.T_min, self.T_max = self.map['T0'].min(), self.map['T0'].max()
        self.P_min, self.P_max = self.map['P0'].min(), self.map['P0'].max()
        self.RPM_min, self.RPM_max = self.map['RPM'].min(), self.map['RPM'].max()
        self.PR_min, self.PR_max = self.map['PR_ts'].min(), self.map['PR_ts'].max()

        # Normalize the data
        self.map['T0'] /= self.T_range
        self.map['P0'] /= self.P_range
        self.map['RPM'] /= self.RPM_range

        # Create Grid for Interpolators
        if self.fixed_RPM and self.fixed_T0:
            self.grid = np.column_stack((self.map['P0'], self.map['PR_ts']))
        elif self.fixed_RPM and self.fixed_P_out:
            self.grid = np.column_stack((self.map['T0'], self.map['P0'], self.map['PR_ts']))
        elif self.fixed_RPM:
            self.grid = np.column_stack((self.map['T0'], self.map['P0'], self.map['PR_ts']))
        else:
            self.grid = np.column_stack((self.map['RPM'], self.map['T0'], self.map['P0'], self.map['PR_ts']))

        # Create primary interpolators
        self.LinearNDInterpolator = {
            'w': LinearNDInterpolator(points=self.grid, values=self.map['m_c']),
            'ETA_ts': LinearNDInterpolator(points=self.grid, values=self.map['ETA_ts']),
            'Power': LinearNDInterpolator(points=self.grid, values=self.map['Power'])
        }

        # Create fallback interpolators
        self.create_fallback_interpolators()

    def create_fallback_interpolators(self):
        """Create fallback interpolators for extrapolation."""
        if self.fallback_method == 'nearest':
            # Create nearest neighbor interpolators
            self.fallback_interpolators = {
                'w': NearestNDInterpolator(self.grid, self.map['m_c']),
                'ETA_ts': NearestNDInterpolator(self.grid, self.map['ETA_ts']),
                'Power': NearestNDInterpolator(self.grid, self.map['Power'])
            }
        else:  # 'extrapolate'
            # Create linear extrapolation functions, but always use nearest neighbor for mass flow
            self.fallback_interpolators = {
                'w': NearestNDInterpolator(self.grid, self.map['m_c']),  # Always nearest for mass flow
                'ETA_ts': self._create_extrapolation_function('ETA_ts'),
                'Power': self._create_extrapolation_function('Power')
            }

    def _create_extrapolation_function(self, param):
        """Create a linear extrapolation function for a given parameter."""
        def extrapolate(*point):
            # Find the k nearest points to use for extrapolation
            k = min(5, len(self.grid))  # Use up to 5 nearest points
            
            # Calculate distances to all points
            distances = np.sqrt(np.sum((self.grid - np.array(point))**2, axis=1))
            nearest_indices = np.argsort(distances)[:k]
            
            # Get the coordinates and values for these points
            nearest_points = self.grid[nearest_indices]
            nearest_values = self.map[param].iloc[nearest_indices].values
            
            # Fit a linear hyperplane through these points
            # Add a column of ones for the intercept term
            A = np.column_stack((nearest_points, np.ones(k)))
            # Solve for coefficients
            try:
                coeffs = np.linalg.lstsq(A, nearest_values, rcond=None)[0]
                # Return the extrapolated value
                return np.dot(np.append(point, 1), coeffs)
            except np.linalg.LinAlgError:
                # If linear fit fails, return the nearest neighbor value
                return nearest_values[0]
        
        return extrapolate

    def _check_if_point_in_bounds(self, point):
        """Check if a point is within the data bounds."""
        # Denormalize the point for bounds checking
        if self.fixed_RPM and self.fixed_T0:
            P_norm, PR = point
            P = P_norm * self.P_range
            return (self.P_min <= P <= self.P_max and 
                   self.PR_min <= PR <= self.PR_max)
        elif self.fixed_RPM:
            T_norm, P_norm, PR = point
            T = T_norm * self.T_range
            P = P_norm * self.P_range
            return (self.T_min <= T <= self.T_max and
                   self.P_min <= P <= self.P_max and
                   self.PR_min <= PR <= self.PR_max)
        else:
            RPM_norm, T_norm, P_norm, PR = point
            RPM = RPM_norm * self.RPM_range
            T = T_norm * self.T_range
            P = P_norm * self.P_range
            return (self.RPM_min <= RPM <= self.RPM_max and
                   self.T_min <= T <= self.T_max and
                   self.P_min <= P <= self.P_max and
                   self.PR_min <= PR <= self.PR_max)

    def interpolate_at_pressure(self, P0, Pout, RPM=None, T0=None, param=None):
        """Interpolate any property using pre-built interpolator with fallback extrapolation."""
        # Calculate pressure ratio
        PR_ts = P0 / Pout

        if param is None:
            raise ValueError("param must be specified")
        
        # Create the normalized point
        if self.fixed_RPM and self.fixed_T0:
            point = np.array([P0/self.P_range, PR_ts])
        elif self.fixed_RPM:
            point = np.array([T0/self.T_range, P0/self.P_range, PR_ts])
        else:
            point = np.array([RPM/self.RPM_range, T0/self.T_range, P0/self.P_range, PR_ts])

        # Check if we're extrapolating
        is_extrapolating = not self._check_if_point_in_bounds(point)
        
        # Try primary interpolation
        value = self.LinearNDInterpolator[param](point)
        
        # If the result is NaN (outside the convex hull), use the fallback interpolator
        if np.isnan(value):

            if param == 'ETA_ts':
                return 0

            if not is_extrapolating:
                method_name = "nearest neighbor" if self.fallback_method == 'nearest' else "linear extrapolation"
                logger.warn(f"Warning: Using {method_name} for point outside convex hull")

            # Use fallback interpolator
            value = self.fallback_interpolators[param](*point)
            
            # Log the extrapolation with appropriate message for mass flow
            if param == 'w':
                logger.warn(
                    f"Mass flow interpolation outside bounds - using nearest neighbor: "
                    f"P0={P0/1e5:.1f} bar, T0={T0:.1f}K, "
                    f"PR_ts={PR_ts:.2f}, RPM={RPM:.0f}, {param}={value:.6f}. "
                    f"Data ranges: P0={self.P_min/1e5:.1f}-{self.P_max/1e5:.1f} bar, "
                    f"T0={self.T_min:.1f}-{self.T_max:.1f}K, "
                    f"PR={self.PR_min:.2f}-{self.PR_max:.2f}, "
                    f"RPM={self.RPM_min:.0f}-{self.RPM_max:.0f}"
                )
            else:
                logger.warn(
                    f"Interpolation failed - point outside convex hull, using {self.fallback_method}: "
                    f"P0={P0/1e5:.1f} bar, T0={T0:.1f}K, "
                    f"PR_ts={PR_ts:.2f}, RPM={RPM:.0f}, {param}={value:.6f}. "
                    f"Data ranges: P0={self.P_min/1e5:.1f}-{self.P_max/1e5:.1f} bar, "
                    f"T0={self.T_min:.1f}-{self.T_max:.1f}K, "
                    f"PR={self.PR_min:.2f}-{self.PR_max:.2f}, "
                    f"RPM={self.RPM_min:.0f}-{self.RPM_max:.0f}"
                )

        return float(value)

    def get_massflow(self, P0, T0, Pout, N=None):
        """
        Calculate mass flow using both direct interpolation and corrected mass flow approach.

        Args:
            p0_in (float): Inlet total pressure [Pa]
            t0_in (float): Inlet total temperature [K]
            p_out (float): Outlet pressure [Pa]

        Returns:
            tuple: (direct_mass_flow, calculated_mass_flow) in kg/s
        """

        # Get corrected mass flow and convert to actual mass flow
        corrected_mass_flow = self.interpolate_at_pressure(P0=P0, T0=T0, Pout=Pout, RPM=N, param='w')
        mass_flow = float(corrected_mass_flow) * (P0/self.p_ref) * np.sqrt(self.T_ref/T0)
        
        return mass_flow
    
    def get_eta_ts(self, P0, T0, Pout, RPM=None):
        """Interpolate total-to-static efficiency."""
        eta = self.interpolate_at_pressure(P0=P0, T0=T0, Pout=Pout, RPM=RPM, param='ETA_ts')
    
        return np.minimum(1, np.maximum(0, eta))

    def get_W_shaft(self, P0, T0, Pout, RPM=None):
        """Interpolate shaft power."""
        return self.interpolate_at_pressure(P0=P0, T0=T0, Pout=Pout, RPM=RPM, param='Power')

    def eta_func(self, US_thermo, Pe, N, model):
        return self.get_eta_ts(US_thermo._P, US_thermo.T, Pe, N)
    
    def optimal_power(self, T0, Pout, RPM=None):
        """
        Find the optimal power by maximizing get_W_shaft over pressure ratio.
        
        Args:
            T0 (float): Inlet total temperature [K]
            Pout (float): Outlet pressure [Pa]
            RPM (float, optional): Rotational speed [rpm]
            
        Returns:
            dict: Dictionary containing optimal conditions:
                - 'power': Optimal shaft power [W]
                - 'P0_opt': Optimal inlet pressure [Pa]
                - 'PR_opt': Optimal pressure ratio
                - 'massflow': Mass flow at optimal conditions [kg/s]
                - 'eta': Efficiency at optimal conditions
        """
        from scipy.optimize import minimize_scalar
        
        # Define the objective function to maximize (negative for minimization)
        def objective(P0):
            power = self.get_W_shaft(P0, T0, 4e5, RPM)

            eta = self.get_eta_ts(P0, T0, 4e5, RPM)

            if eta == 0:
                return 0
            
            return -power  # Negative because we want to maximize
        
        # Define bounds for pressure ratio optimization
        # Use the data range with some safety margin
        P_min = self.P_min  # Minimum PR slightly above 1
        P_max = self.P_max
        
        # Optimize using scipy
        result = minimize_scalar(
            objective,
            bounds=(P_min, P_max),
            method='bounded'
        )

        P0_opt = result.x
        PR_opt = P0_opt / Pout
        power_opt = -result.fun
        massflow_opt = self.get_massflow(P0_opt, T0, Pout, RPM)
        eta_opt = self.get_eta_ts(P0_opt, T0, Pout, RPM)

        
        if not result.success:
            logger.warn(f"Optimization failed: {result.message}")
            # Fallback: try a grid search
            from pdb import set_trace
            set_trace()
            PR_range = np.linspace(PR_min, PR_max, 100)
            powers = []
            for PR in PR_range:
                P0 = PR * Pout
                power = self.get_W_shaft(P0, T0, Pout, RPM)
                powers.append(power)
            
            opt_idx = np.argmax(powers)
            PR_opt = PR_range[opt_idx]
            power_opt = powers[opt_idx]
        else:
            PR_opt = result.x
            power_opt = -result.fun  # Convert back from negative
        
        return {
            'power': power_opt,
            'P0_opt': P0_opt,
            'PR_opt': PR_opt,
            'massflow': massflow_opt,
            'eta': eta_opt
        }
    
    def optimal_eta(self, T0, Pout, RPM=None):
        """
        Find the optimal efficiency by maximizing get_eta_ts over pressure ratio.
        
        Args:
            T0 (float): Inlet total temperature [K]
            Pout (float): Outlet pressure [Pa]
            RPM (float, optional): Rotational speed [rpm]
            
        Returns:
            dict: Dictionary containing optimal conditions:
                - 'eta': Optimal efficiency
                - 'P0_opt': Optimal inlet pressure [Pa]
                - 'PR_opt': Optimal pressure ratio
                - 'power': Power at optimal conditions [W]
                - 'massflow': Mass flow at optimal conditions [kg/s]
        """
        from scipy.optimize import minimize_scalar
        
        # Define the objective function to maximize (negative for minimization)
        def objective(PR):
            P0 = PR * Pout
            eta = self.get_eta_ts(P0, T0, Pout, RPM)
            return -eta  # Negative because we want to maximize
        
        # Define bounds for pressure ratio optimization
        PR_min = max(1.1, self.PR_min * 0.9)
        PR_max = min(10.0, self.PR_max * 1.1)
        
        # Optimize using scipy
        result = minimize_scalar(
            objective,
            bounds=(PR_min, PR_max),
            method='bounded'
        )
        
        if not result.success:
            logger.warn(f"Optimization failed: {result.message}")
            # Fallback: try a grid search
            PR_range = np.linspace(PR_min, PR_max, 100)
            etas = []
            for PR in PR_range:
                P0 = PR * Pout
                eta = self.get_eta_ts(P0, T0, Pout, RPM)
                etas.append(eta)
            
            opt_idx = np.argmax(etas)
            PR_opt = PR_range[opt_idx]
            eta_opt = etas[opt_idx]
        else:
            PR_opt = result.x
            eta_opt = -result.fun  # Convert back from negative
        
        # Calculate other parameters at optimal conditions
        P0_opt = PR_opt * Pout
        power_opt = self.get_W_shaft(P0_opt, T0, Pout, RPM)
        massflow_opt = self.get_massflow(P0_opt, T0, Pout, RPM)
        
        return {
            'eta': eta_opt,
            'P0_opt': P0_opt,
            'PR_opt': PR_opt,
            'power': power_opt,
            'massflow': massflow_opt
        }
        