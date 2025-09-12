import numpy as np
import re
from scipy.optimize import root_scalar
from geoTherm.logger import logger
import geoTherm as gt
import yaml
from scipy.interpolate import interp1d
from scipy.interpolate import CloughTocher2DInterpolator
import matplotlib.pyplot as plt
from scipy.optimize import minimize_scalar
import pandas as pd
import inspect

# Get Machine precision
eps = np.finfo(float).eps
# G constant (m/s^2)
g = 9.80665
# Gas Constant (J/kmol/K)
R_ideal = 8314.46261815324


def dP_pipe(thermo, Dh, L, w, roughness=2.5e-6):
    """ Estimate pressure drop for pipe flow using mass flow.

    Args:
        thermo (thermostate): geoTherm thermostat object
        Dh (float): Hydraulic Diameter in m
        L (float): Pipe length in m
        w (float): Mass flow rate in kg/s
        roughness (float, optional): Pipe roughness in m (Default = 2.5e-6 m)

    Returns:
        float: Pipe pressure drop in Pa
    """
    K_pipe = pipe_K(thermo, L, Dh, w, roughness)
    A = np.pi / 4 * Dh ** 2
    return -K_pipe * (w / A) ** 2 / (2 * thermo._density)


def pipe_K(thermo, L, Dh, w, roughness=2.5e-6):
    """ Calculate the loss coefficient for pipe flow.

    Args:
        thermo (thermostate): geoTherm thermostat object
        L (float): Pipe length in m
        Dh (float): Hydraulic Diameter in m
        w (float): Mass flow rate in kg/s
        roughness (float, optional): Pipe roughness in m (Default = 2.5e-6 m)

    Returns:
        float: Loss coefficient
    """
    Re = Re_(thermo, Dh, w)
    f = friction_factor(Re, roughness / Dh)
    return f * L / Dh


def Re_(thermo, D_h, w):
    """ Calculate Reynolds number based on mass flow.

    Args:
        thermo (thermostate): geoTherm thermostat object
        Dh (float): Hydraulic Diameter in m
        w (float): Mass flow rate in kg/s

    Returns:
        float: Reynolds Number
    """
    return 4 * np.abs(w) / (np.pi * D_h * thermo._viscosity)

def Re_square(thermo, D_h, w_flux):

    return w_flux*D_h/thermo._viscosity

def Boiling_(G, q, dH_vap):
    """ Calculate boiling number"""
    # q: heat flux
    # G: mass flux
    # Hvap: Heat Of vaporization J/kg

    return q/(G*dH_vap)

def Bond_(thermo, D_h):

    # Create temnporary themro object
    temp_thermo = thermo.from_state(thermo.state)

    temp_thermo._PQ = thermo._P, 0
    rho_l = temp_thermo._density
    temp_thermo._PQ = thermo._P, 1
    rho_g = temp_thermo._density

    return (rho_l - rho_g)*g*D_h**2/temp_thermo._surface_tension




def friction_factor(Re, rel_roughness):
    """ Calculate the friction factor based on Reynolds number and 
        relative roughness.

    Args:
        Re (float): Reynolds Number
        rel_roughness (float): Relative roughness (roughness/Dh)

    Returns:
        float: Friction factor
    """
    if Re < 2100:
        return 64 / Re
    else:
        return colebrook(rel_roughness, Re)

def colebrook(k, Re):
    """ Get the friction factor using Colebook Equation 

    Args:
        k (float): Relative Roughness
        Re (float): Reynolds Number 

    Returns:
        friction factor """

    def res(f):
        # Residual function
        return 1/np.sqrt(f) + 2*np.log10(k/3.7 + 2.51/(Re*np.sqrt(f)))

    # Use root scalar to find root
    f = root_scalar(res, method='brentq', bracket=[1e-10, 1e4]).root

    return f


def pump_eta(phi):
    # Pump efficiency from Claudio
    eta_max = 0.83
    phi_max = 1.75
    k1 = 27
    k2 = 5000
    k3 = 10
    # Why is this 0 and why
    delta_eta_p = 0

    if phi <= 0.08:
        eta = eta_max*(1-k1*(phi_max*phi -phi)**2 - k2*(phi_max*phi-phi)**4) + delta_eta_p
    elif phi > 0.08:
        eta = eta_max*(1-k3*(phi_max*phi-phi)**2) + delta_eta_p

    return eta

def turb_axial_eta(phi, psi, psi_opt):

    eta_opt = 0.913 + 0.103*psi-0.0854*psi**2 + 0.0154*psi**3

    phi_opt = 0.375 + 0.25*psi_opt

    K = 0.375 - 0.125*psi

    return eta_opt - K*(phi-phi_opt)**2

def turb_radial_eta(ns):
    return 0.87 - 1.07*(ns-0.55)**2-0.5*(ns-0.55)**3

def parse_knob_string(knob_string):
    # Parse the knob string into a component name and variable
    
    if isinstance(knob_string, str):

        # Regular expression to match the pattern: Node.Variable
        pattern = r"(\w+)\.(\w+)"
        # Search for the pattern in the input string
        match = re.search(pattern, knob_string)

        if match:
            # If a match is found, create a dictionary from the groups
            return [match.group(1), match.group(2)]
        else:
            from pdb import set_trace
            set_trace()

    else:
        from pdb import set_trace
        set_trace()


    from pdb import set_trace
    set_trace()


def parse_component_attribute(attribute_string):
    """
    Parses an attribute string to extract the component name and the attribute
    chain.

    Args:
        attribute_string (str): The attribute string in the format 
                                'Component.Attribute' or 
                                'Component.SubComponent.Attribute'.

    Returns:
        list: A list where the first element is the component name and the 
              second element is the attribute chain.
    """
    if isinstance(attribute_string, str):
        # Regular expression to match the pattern: Component.Attribute or 
        # Component.SubComponent.Attribute
        pattern = r"(\w+)\.(.+)"
        
        # Search for the pattern in the input string
        match = re.search(pattern, attribute_string)

        if match:
            # Return a list where the first element is the component name 
            # and the second is the attribute chain
            return [match.group(1), match.group(2)]
        else:
            raise ValueError("Invalid attribute string format.")
    else:
        raise TypeError("Input should be a string.")


class thermo_data:

    def __init__(self, H, P, fluid, model):
        self._H = H
        self._P = P
        self.fluid = fluid
        self.model = model

    def update(self, state):

        self._H = state['H']
        self._P = state['P']


    @property
    def state(self):

        return {
            'fluid': self.fluid,
            'state': {'H': (self._H, 'J/kg'),
                      'P': (self._P, 'Pa')},
            'model': self.model
        }


def find_bounds(f, bounds, upper_limit=np.inf, lower_limit=-np.inf,
                max_iter=10, factor=2):
    """
    Expand bounds to ensure they bracket a root, returning
    bounds as [lower, upper] where lower < upper and within specified limits.

    Args:
        f (callable): Function for which the root is sought.
        bounds (list): Initial bounds as a list [lower, upper].
        upper_limit (float, optional): Maximum allowable value for the upper
                                       bound.
        lower_limit (float, optional): Minimum allowable value for the lower
                                       bound.
        max_iter (int, optional): Maximum number of iterations to extend
                                  bounds. Default is 10.
        factor (float, optional): Factor by which to extend the bounds.
        Default is 2.

    Returns:
        list: Adjusted bounds that bracket the root, ordered as [lower, upper].
    """
    lower, upper = min(bounds), max(bounds)

    if factor < 1:
        factor = 1/factor

    # Ensure initial bounds are within the specified limits

    lower = max(lower, lower_limit)
    upper = min(upper, upper_limit)

    f_lower_sign = np.sign(f(lower))
    f_upper_sign = np.sign(f(upper))

    if f_lower_sign != f_upper_sign:
        return [lower, upper]

    iteration = 0

    # Expand bounds until a root is bracketed, limits are reached, or
    # max iterations
    while f_lower_sign == f_upper_sign and iteration < max_iter:
        # Check if both bounds have reached their respective limits
        if ((upper_limit is not None and upper >= upper_limit) and
                (lower_limit is not None and lower <= lower_limit)):

            # Both limits reached, cannot extend further
            logger.info("Upper and lower bound limits reached in bound search")
            break

        if lower_limit < upper < upper_limit:
            upper = min(upper_limit, upper*factor)
            if np.sign(f(upper)) != f_upper_sign:
                from pdb import set_trace
                set_trace()

        if lower_limit < lower < upper_limit:
            lower = max(lower_limit, lower/factor)
            if np.sign(f(lower)) != f_lower_sign:
                return lower, lower*factor

        iteration += 1

    logger.info(f"Unable to find bounds after {iteration} iterations")
    # Return adjusted bounds
    return [lower, upper]


def has_cycle(node_map, node_list):
    # Detect if node_map has recirculation or not

    # Helper function to perform DFS
    def dfs(node, visited, stack):
        visited.add(node)
        stack.add(node)

        # Traverse all downstream nodes
        for neighbor in node_map[node]['DS']:
            if neighbor not in visited:
                # Recur for unvisited nodes
                if dfs(neighbor, visited, stack):
                    return True
            elif neighbor in stack:
                # If neighbor is in stack, we found a cycle
                return True

        # Remove the node from the stack after visiting all its neighbors
        stack.remove(node)
        return False

    visited = set()
    stack = set()

    # Check each node in the graph
    for node in node_map:
        if node not in visited:
            if dfs(node, visited, stack):
                if isinstance(node_list[node], gt.Boundary):
                    continue
                else:
                    return True
    return False


def process_config_dimensions(config):
    """
    Recursively process configuration dictionary to convert dimensional values to tuples.
    
    Args:
        config (dict): Configuration dictionary
    Returns:
        dict: Processed configuration with dimensional values converted to tuples
    """
    if isinstance(config, dict):
        return {key: process_config_dimensions(value) for key, value in config.items()}
    elif isinstance(config, list):
        return [process_config_dimensions(item) for item in list(config)]
    else:
        return parse_dimension(config)

def yaml_loader(yaml_path):
    """
    Load and parse the YAML model configuration file.
    
    Args:
        yaml_path (str): Path to the YAML configuration file
    
    Returns:
        dict: Parsed YAML configuration with processed dimensions
    """
    try:
        with open(yaml_path, 'r') as file:
            config = yaml.safe_load(file)
            # Process the dimensions in the config
            processed_config = process_config_dimensions(config)
            return processed_config

    except yaml.YAMLError as e:
        logger.critical(f"Error parsing YAML file: {e}")
    except FileNotFoundError:
        logger.critical(f"Could not find file: {yaml_path}")

class CleanDumper(yaml.SafeDumper):
    """Custom YAML dumper that creates cleaner output"""
    
    def represent_tuple(self, data):
        # Convert tuples to lists for cleaner output
        return self.represent_list(list(data))
        
    def represent_numpy_array(self, data):
        # Convert numpy arrays to lists for cleaner output
        return self.represent_list(data.tolist())
        
    def represent_numpy_float(self, data):
        # Convert numpy float types to regular Python float
        return self.represent_float(float(data))
        
    def represent_numpy_int(self, data):
        # Convert numpy int types to regular Python int
        return self.represent_int(int(data))

# Register the custom representers
CleanDumper.add_representer(tuple, CleanDumper.represent_tuple)
CleanDumper.add_representer(np.ndarray, CleanDumper.represent_numpy_array)
CleanDumper.add_representer(np.float64, CleanDumper.represent_numpy_float)
CleanDumper.add_representer(np.int64, CleanDumper.represent_numpy_int)

def yaml_writer(yaml_path, config):
    """
    Write a configuration dictionary to a YAML file with clean formatting.

    Args:
        yaml_path (str): Path to the YAML file.
        config (dict): Configuration dictionary.
    """
    with open(yaml_path, 'w') as file:
        yaml.dump(config, file, Dumper=CleanDumper, default_flow_style=False)
        

def parse_dimension(value):
    """
    Parse a dimensional value into a (number, unit) tuple.

    Args:
        value: String like "2 in" or number
    Returns:
        tuple: (float, str) or float if no unit
    """
    if isinstance(value, (int, float)):
        return value

    try:
        # Split the string into number and unit
        parts = str(value).strip().split()
        if len(parts) == 2:
            return (float(parts[0]), parts[1])
        else:
            return float(value)
    except (ValueError, TypeError):
        return value
    


#https://www.mydatabook.org/fluid-mechanics/flow-coefficient-opening-and-closure-curves-of-full-bore-ball-valves/
cv_2in = {
    "angle_deg": np.array([0, 10, 20, 30, 40, 50, 60, 70, 80, 90]),
    "Cv":        np.array([0, 15, 27, 45, 76, 118, 180, 301, 404, 470])
}


def get_SCFM(mdot, US):
    Q = mdot/US._density*2118.88
    return Q*(US._P/101325)*(288.71/US._T)

def CFM_to_SCFM(CFM, US):
    return CFM*(US._P/101325)*(288.71/US._T)


def get_Cv(US, DS, mdot):
    # based on Swagelok Valve Sizing Technical Bulletin
    # https://www.swagelok.com/downloads/webcatalogs/EN/MS-06-84.pdf
    # --- 1. Upstream state and fluid properties ---
    phase = US.phase

    # --- 2. Pressure conversions ---
    Pup_psi = US._P/6894.76
    Pdown_psi = DS._P/6894.76
    Tup_R = US._T*9/5
    deltaP_psi = (Pup_psi - Pdown_psi)

    if deltaP_psi <= 0:
        raise ValueError("Pressure drop must be positive.")

    # --- 3. Cv calculation ---
    if phase in ['liquid', 'supercritical_liquid']:
        Q_GPM = mdot/US._density*15850.3
        # Liquid-like behavior → use incompressible formula
        G = US._density / 999.0 # specific gravity
        Cv = Q_GPM / np.sqrt(deltaP_psi / G)

    else:
        gamma = US.gamma
        # Gas-like behavior → check for choked flow
        P_crit_ratio = (2 / (gamma + 1)) ** (gamma / (gamma - 1))
        is_choked = (DS._P / US._P) < P_crit_ratio

        Q_CFM = mdot/US._density*2118.88


        Q_SCFM = CFM_to_SCFM(Q_CFM, US)

        ref_thermo = US.copy()
        ref_thermo._TP = 288.71, 101325
        G = ref_thermo._density/1.225

        if is_choked:
            # Choked flow: no dependence on P2
            Cv = Q_SCFM / (0.471 * 22.67 * Pup_psi * np.sqrt(1/(G * Tup_R)))
        else:
            # Subsonic gas flow
            Cv = (Q_SCFM /
                  ((22.67 * Pup_psi * (1 -2/3*deltaP_psi/Pup_psi))
                   * np.sqrt(deltaP_psi/(Pup_psi*G * Tup_R)))
                  )

    return Cv

def mdot_from_Cv(Cv, US, DS):
    # based on Swagelok Valve Sizing Technical Bulletin
    # https://www.swagelok.com/downloads/webcatalogs/EN/MS-06-84.pdf
    # --- 1. Upstream state and fluid properties ---
    phase = US.phase

    # --- 2. Pressure conversions ---
    Pup_psi = US._P/6894.76
    Pdown_psi = DS._P/6894.76
    Tup_R = US._T*9/5
    deltaP_psi = (Pup_psi - Pdown_psi)

    if deltaP_psi <= 0:
        raise ValueError("Pressure drop must be positive.")

    # --- 3. Cv calculation ---
    if phase in ['liquid', 'supercritical_liquid']:
        SG = US._density / 999.0 # specific gravity
        Q_GPM = Cv * np.sqrt(deltaP_psi / SG)
        mdot = Q_GPM * US._density/15850.3

    else:
        gamma = US.gamma
        P_crit_ratio = (2 / (gamma + 1)) ** (gamma / (gamma - 1))
        is_choked = (DS._P / US._P) < P_crit_ratio
        





def get_valve_position(Cv, cv_curve=cv_2in):
    """
    Returns the valve position (in degrees) for a 2" full-bore ball valve
    given flow conditions and a fluid using CoolProp.
    Includes choked flow handling for gases.

    Parameters:
        mdot (float): Mass flow rate [kg/s]
        P1 (float): Upstream pressure [Pa]
        P2 (float): Downstream pressure [Pa]
        T1 (float): Inlet temperature [K]
        fluid (str): CoolProp-compatible fluid name

    Returns:
        float: Valve opening angle [deg]
    """
    # --- 5. Interpolate Cv → valve angle ---
    interp = interp1d(cv_curve['Cv'], cv_curve['angle_deg'], bounds_error=False, fill_value='extrapolate')
    angle = float(interp(Cv))

    return angle


class TurbineInterpolator:
    """Interpolator for turbine performance maps with extrapolation."""
    
    def __init__(self, csv_file):
        """Initialize interpolator from CSV file."""
        # Load the data

        self.csv_file = csv_file
        
        self.df = pd.read_csv(csv_file)
        
        # Store bounds for reference
        self.rpm_bounds = (self.df['N_rpm'].min(), self.df['N_rpm'].max())
        self.pr_bounds = (self.df['PRt-s'].min(), self.df['PRt-s'].max())
        
        # Add boundary points for each RPM value
        self._add_boundary_points()
        
        # Prepare points for interpolation
        self.points = np.column_stack((self.df['N_rpm'], self.df['PRt-s']))
        
        # Create interpolators
        self.deltaht_interpolator = CloughTocher2DInterpolator(
            self.points, 
            self.df['deltaht-s_J/kg']
        )
        self.etat_interpolator = CloughTocher2DInterpolator(
            self.points, 
            self.df['etat-s']
        )
        
        # Store the data for extrapolation
        self.deltaht_data = self.df['deltaht-s_J/kg'].values
        self.etat_data = self.df['etat-s'].values
    
    def _add_boundary_points(self):
        """Add boundary points for each RPM value to ensure complete coverage."""
        # Get unique RPM values
        unique_rpms = self.df['N_rpm'].unique()
        
        # Create new rows for boundary points
        new_rows = []
        for rpm in unique_rpms:
            # Get data for this RPM
            rpm_data = self.df[self.df['N_rpm'] == rpm]
            
            # Find min and max PR for this RPM
            pr_min = rpm_data['PRt-s'].min()
            pr_max = rpm_data['PRt-s'].max()
            
            # Get corresponding values
            min_row = rpm_data[rpm_data['PRt-s'] == pr_min].iloc[0]
            max_row = rpm_data[rpm_data['PRt-s'] == pr_max].iloc[0]
            
            # Add points at PR boundaries
            new_rows.append({
                'N_rpm': rpm,
                'PRt-s': self.pr_bounds[0],
                'deltaht-s_J/kg': min_row['deltaht-s_J/kg'],
                'etat-s': min_row['etat-s']
            })
            new_rows.append({
                'N_rpm': rpm,
                'PRt-s': self.pr_bounds[1],
                'deltaht-s_J/kg': max_row['deltaht-s_J/kg'],
                'etat-s': max_row['etat-s']
            })
        
        # Add new rows to the dataframe
        if new_rows:
            self.df = pd.concat([self.df, pd.DataFrame(new_rows)], ignore_index=True)
    
    def _extrapolate(self, rpm, pr, interpolator, data):
        """Helper method to handle extrapolation using k-nearest neighbors linear fit."""
        # Find the k nearest points to use for extrapolation
        k = 3  # Use 3 nearest points to establish the linear trend
        distances = np.sqrt((self.df['N_rpm'] - rpm)**2 + (self.df['PRt-s'] - pr)**2)
        nearest_indices = np.argsort(distances)[:k]
        
        # Get the coordinates and values for these points
        nearest_points = np.column_stack((self.df['N_rpm'].values[nearest_indices], 
                                        self.df['PRt-s'].values[nearest_indices]))
        nearest_values = data[nearest_indices]
        
        # Fit a linear plane through these points
        # Add a column of ones for the intercept term
        A = np.column_stack((nearest_points, np.ones(k)))
        # Solve for coefficients [a, b, c] in z = ax + by + c
        coeffs = np.linalg.lstsq(A, nearest_values, rcond=None)[0]
        
        # Return the extrapolated value
        return coeffs[0] * rpm + coeffs[1] * pr + coeffs[2]
    
    def get_values(self, rpm, pr):
        """Get interpolated deltaht and eta values with extrapolation."""
        # Check if point is outside bounds
        is_outside = (
            rpm < self.rpm_bounds[0] or rpm > self.rpm_bounds[1] or
            pr < self.pr_bounds[0] or pr > self.pr_bounds[1]
        )
        
        try:
            # Try to get interpolated values
            deltaht = float(self.deltaht_interpolator(rpm, pr))
            eta = float(self.etat_interpolator(rpm, pr))
            
            # Check if we got NaN values
            if np.isnan(deltaht) or np.isnan(eta):
                raise ValueError("Interpolation returned NaN")
                
        except (ValueError, TypeError):
            # If interpolation fails or returns NaN, use extrapolation
            deltaht = self._extrapolate(rpm, pr, self.deltaht_interpolator, self.deltaht_data)
            eta = self._extrapolate(rpm, pr, self.etat_interpolator, self.etat_data)
            is_outside = True
        
        # Add warning if extrapolating
        if is_outside:
            print(f"Warning: Point (RPM={rpm}, PR={pr}) is outside the data bounds. "
                  f"Using linear extrapolation from nearest points.")
        
        return deltaht, eta
    
    def plot_maps(self, extrapolation_range=0.2):
        """Create visualization of the performance maps with extrapolation region.
        
        Args:
            extrapolation_range (float): Fraction of the original range to extend
                the plot for extrapolation visualization (default: 0.2 = 20%)
        """
        # Calculate extended ranges for plotting
        rpm_range = self.rpm_bounds[1] - self.rpm_bounds[0]
        pr_range = self.pr_bounds[1] - self.pr_bounds[0]
        
        extended_rpm_min = self.rpm_bounds[0] - rpm_range * extrapolation_range
        extended_rpm_max = self.rpm_bounds[1] + rpm_range * extrapolation_range
        extended_pr_min = self.pr_bounds[0] - pr_range * extrapolation_range
        extended_pr_max = self.pr_bounds[1] + pr_range * extrapolation_range
        
        # Create a regular grid for plotting that includes the exact boundary values
        N_rpm_points = np.linspace(extended_rpm_min, extended_rpm_max, 100)
        PRt_s_points = np.linspace(extended_pr_min, extended_pr_max, 100)
        
        # Ensure the exact boundary values are included in the grid
        N_rpm_points = np.sort(np.unique(np.concatenate([
            N_rpm_points,
            [self.rpm_bounds[0], self.rpm_bounds[1]]
        ])))
        PRt_s_points = np.sort(np.unique(np.concatenate([
            PRt_s_points,
            [self.pr_bounds[0], self.pr_bounds[1]]
        ])))
        
        N_rpm_grid, PRt_s_grid = np.meshgrid(N_rpm_points, PRt_s_points)
        
        # Create figure with two subplots
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))
        
        # Plot deltaht surface
        self._plot_surface(ax1, self.deltaht_interpolator, 
                          N_rpm_grid, PRt_s_grid,
                          'deltaht-s_J/kg', 'viridis', '{:.0f}')
        
        # Plot eta surface
        self._plot_surface(ax2, self.etat_interpolator,
                          N_rpm_grid, PRt_s_grid,
                          'etat-s', 'plasma', '{:.4f}')
        
        plt.tight_layout()
        plt.show()
    
    def _plot_surface(self, ax, interpolator, N_rpm_grid, PRt_s_grid, 
                     title, cmap='viridis', format_str='{:.0f}'):
        """Helper function to create surface plot."""
        # Calculate interpolated values
        interpolated_values = np.zeros_like(N_rpm_grid)
        for i in range(N_rpm_grid.shape[0]):
            for j in range(N_rpm_grid.shape[1]):
                try:
                    interpolated_values[i,j] = interpolator(N_rpm_grid[i,j], PRt_s_grid[i,j])
                except (ValueError, TypeError):
                    # If interpolation fails, use extrapolation
                    interpolated_values[i,j] = self._extrapolate(
                        N_rpm_grid[i,j], PRt_s_grid[i,j], 
                        interpolator, 
                        self.deltaht_data if 'deltaht' in title else self.etat_data
                    )
        
        # Create contour plot with more levels for smoother appearance
        contour = ax.contourf(N_rpm_grid, PRt_s_grid, interpolated_values, 
                            levels=30, cmap=cmap, extend='both')
        plt.colorbar(contour, ax=ax, label=title)
        
        # Plot data points
        scatter = ax.scatter(self.df['N_rpm'], self.df['PRt-s'], 
                           c='red', s=30)
        
        # Add text labels for each point
        for i, row in self.df.iterrows():
            ax.text(row['N_rpm'], row['PRt-s'], 
                    format_str.format(row[title]), 
                    fontsize=8, 
                    ha='center', 
                    va='center',
                    color='black')
        
        # Calculate padding for axis limits (5% of the range)
        x_padding = (self.rpm_bounds[1] - self.rpm_bounds[0]) * 0.05
        y_padding = (self.pr_bounds[1] - self.pr_bounds[0]) * 0.05
        
        # Set axis limits with padding
        ax.set_xlim(self.rpm_bounds[0] - x_padding, self.rpm_bounds[1] + x_padding)
        ax.set_ylim(self.pr_bounds[0] - y_padding, self.pr_bounds[1] + y_padding)
        
        ax.set_xlabel('N_rpm')
        ax.set_ylabel('PRt-s')
        ax.set_title(title)

    def get_optimal_rpm(self, pr, rpm_range=None):
        """Find the RPM that gives maximum efficiency at a specific pressure ratio.
        
        Args:
            pr (float): Pressure ratio to evaluate
            rpm_range (tuple, optional): (min_rpm, max_rpm) range to search within.
                If None, uses the min/max from the data.
        
        Returns:
            tuple: (optimal_rpm, max_efficiency)
        """
        if rpm_range is None:
            rpm_range = self.rpm_bounds
        
        # Define the objective function (negative efficiency since we want to maximize)
        def objective(rpm):
            return -self.etat_interpolator(rpm, pr)
        
        # Use scipy's minimize_scalar to find the optimal RPM
        result = minimize_scalar(
            objective,
            bounds=rpm_range,
            method='bounded',
            options={'xatol': 1.0}  # 1 RPM tolerance
        )
        
        optimal_rpm = result.x
        max_efficiency = -result.fun  # Convert back from negative
        
        return optimal_rpm, max_efficiency
    

class TurbineInterpolator:
    """Interpolator for turbine performance maps with extrapolation."""
    
    def __init__(self, csv_file):
        """Initialize interpolator from CSV file."""
        # Load the data
        self.csv_file = csv_file
        raw_data = pd.read_csv(csv_file, header=None)
        
        # Process the data into a structured format
        self.process_data(raw_data)
        
        # Store bounds for reference
        self.temp_bounds = (self.df['T_in'].min(), self.df['T_in'].max())
        self.pr_bounds = (self.df['PR_ts'].min(), self.df['PR_ts'].max())
        
        # Add boundary points for each temperature value
        self._add_boundary_points()
        
        # Prepare points for interpolation
        self.points = np.column_stack((self.df['T_in'], self.df['PR_ts']))
        
        # Create interpolators
        self.power_interpolator = CloughTocher2DInterpolator(
            self.points, 
            self.df['Power_Shaft']
        )
        self.eta_interpolator = CloughTocher2DInterpolator(
            self.points, 
            self.df['ETA_ts_poly']
        )
        
        # Store the data for extrapolation
        self.power_data = self.df['Power_Shaft'].values
        self.eta_data = self.df['ETA_ts_poly'].values
    
    def process_data(self, raw_data):
        """Process the raw CSV data into a structured DataFrame."""
        # Initialize lists to store processed data
        data = []
        
        # Process each temperature section
        current_temp = None
        current_section = []
        
        for i in range(len(raw_data)):
            row = raw_data.iloc[i]
            
            # Check if this is a temperature header row
            if pd.notna(row[2]) and 'T_In' in str(row[2]):
                # If we have a previous section, process it
                if current_section and current_temp is not None:
                    self._process_section(current_section, current_temp, data)
                
                # Start new section
                current_temp = float(str(row[2]).split('=')[1].strip().split()[0])
                current_section = []
                continue
            
            # Skip empty rows
            if pd.isna(row[0]):
                continue
            
            # Add row to current section
            current_section.append(row)
        
        # Process the last section
        if current_section and current_temp is not None:
            self._process_section(current_section, current_temp, data)
        
        # Create DataFrame
        self.df = pd.DataFrame(data)
        
        if len(data) == 0:
            raise ValueError("No valid data points were found in the CSV file. Please check the file format.")
        
        # Print some debug information
        print(f"Processed {len(data)} data points")
        print(f"Temperature range: {self.df['T_in'].min()} to {self.df['T_in'].max()} C")
        print(f"PR range: {self.df['PR_ts'].min()} to {self.df['PR_ts'].max()}")
        
        # Print first few rows for debugging
        print("\nFirst few rows of processed data:")
        print(self.df.head())
    
    def _process_section(self, section, temp, data):
        """Process a section of data for a given temperature."""
        # Find the rows we need
        pr_row = None
        power_row = None
        eta_row = None
        
        for row in section:
            if pd.notna(row[0]):
                if 'PR_ts' in str(row[0]):
                    pr_row = row
                elif 'Power(Shaft)' in str(row[0]):
                    power_row = row
                elif 'ETA_ts_poly' in str(row[0]):
                    eta_row = row
        
        if pr_row is not None and power_row is not None and eta_row is not None:
            # Process each column (each pressure ratio point)
            for col in range(2, len(pr_row)):
                try:
                    if pd.notna(pr_row[col]) and pd.notna(power_row[col]) and pd.notna(eta_row[col]):
                        pr = float(pr_row[col])
                        power = float(power_row[col])
                        eta = float(eta_row[col])
                        
                        data.append({
                            'T_in': temp,
                            'PR_ts': pr,
                            'Power_Shaft': power,
                            'ETA_ts_poly': eta
                        })
                except (ValueError, IndexError) as e:
                    print(f"Warning: Skipping column {col} due to error: {e}")
                    continue
    
    def _add_boundary_points(self):
        """Add boundary points for each temperature value to ensure complete coverage."""
        # Get unique temperature values
        unique_temps = self.df['T_in'].unique()
        
        # Create new rows for boundary points
        new_rows = []
        for temp in unique_temps:
            # Get data for this temperature
            temp_data = self.df[self.df['T_in'] == temp]
            
            # Find min and max PR for this temperature
            pr_min = temp_data['PR_ts'].min()
            pr_max = temp_data['PR_ts'].max()
            
            # Get corresponding values
            min_row = temp_data[temp_data['PR_ts'] == pr_min].iloc[0]
            max_row = temp_data[temp_data['PR_ts'] == pr_max].iloc[0]
            
            # Add points at PR boundaries
            new_rows.append({
                'T_in': temp,
                'PR_ts': self.pr_bounds[0],
                'Power_Shaft': min_row['Power_Shaft'],
                'ETA_ts_poly': min_row['ETA_ts_poly']
            })
            new_rows.append({
                'T_in': temp,
                'PR_ts': self.pr_bounds[1],
                'Power_Shaft': max_row['Power_Shaft'],
                'ETA_ts_poly': max_row['ETA_ts_poly']
            })
        
        # Add new rows to the dataframe
        if new_rows:
            self.df = pd.concat([self.df, pd.DataFrame(new_rows)], ignore_index=True)
    
    def _extrapolate(self, temp, pr, interpolator, data):
        """Helper method to handle extrapolation using k-nearest neighbors linear fit."""
        # Find the k nearest points to use for extrapolation
        k = 3  # Use 3 nearest points to establish the linear trend
        distances = np.sqrt((self.df['T_in'] - temp)**2 + (self.df['PR_ts'] - pr)**2)
        nearest_indices = np.argsort(distances)[:k]
        
        # Get the coordinates and values for these points
        nearest_points = np.column_stack((self.df['T_in'].values[nearest_indices], 
                                        self.df['PR_ts'].values[nearest_indices]))
        nearest_values = data[nearest_indices]
        
        # Fit a linear plane through these points
        # Add a column of ones for the intercept term
        A = np.column_stack((nearest_points, np.ones(k)))
        # Solve for coefficients [a, b, c] in z = ax + by + c
        coeffs = np.linalg.lstsq(A, nearest_values, rcond=None)[0]
        
        # Return the extrapolated value
        return coeffs[0] * temp + coeffs[1] * pr + coeffs[2]
    
    def get_values(self, temp, pr):
        """Get interpolated power and efficiency values with extrapolation."""
        # Check if point is outside bounds
        is_outside = (
            temp < self.temp_bounds[0] or temp > self.temp_bounds[1] or
            pr < self.pr_bounds[0] or pr > self.pr_bounds[1]
        )
        
        try:
            # Try to get interpolated values
            power = float(self.power_interpolator(temp, pr))
            eta = float(self.eta_interpolator(temp, pr))
            
            # Check if we got NaN values
            if np.isnan(power) or np.isnan(eta):
                raise ValueError("Interpolation returned NaN")
                
        except (ValueError, TypeError):
            # If interpolation fails or returns NaN, use extrapolation
            power = self._extrapolate(temp, pr, self.power_interpolator, self.power_data)
            eta = self._extrapolate(temp, pr, self.eta_interpolator, self.eta_data)
            is_outside = True
        
        # Add warning if extrapolating
        if is_outside:
            print(f"Warning: Point (T={temp}, PR={pr}) is outside the data bounds. "
                  f"Using linear extrapolation from nearest points.")
        
        return power, eta
    
    def plot_maps(self, extrapolation_range=0.2):
        """Create visualization of the performance maps with extrapolation region."""
        # Calculate extended ranges for plotting
        temp_range = self.temp_bounds[1] - self.temp_bounds[0]
        pr_range = self.pr_bounds[1] - self.pr_bounds[0]
        
        extended_temp_min = self.temp_bounds[0] - temp_range * extrapolation_range
        extended_temp_max = self.temp_bounds[1] + temp_range * extrapolation_range
        extended_pr_min = self.pr_bounds[0] - pr_range * extrapolation_range
        extended_pr_max = self.pr_bounds[1] + pr_range * extrapolation_range
        
        # Create a regular grid for plotting
        temp_points = np.linspace(extended_temp_min, extended_temp_max, 100)
        pr_points = np.linspace(extended_pr_min, extended_pr_max, 100)
        
        # Ensure the exact boundary values are included
        temp_points = np.sort(np.unique(np.concatenate([
            temp_points,
            [self.temp_bounds[0], self.temp_bounds[1]]
        ])))
        pr_points = np.sort(np.unique(np.concatenate([
            pr_points,
            [self.pr_bounds[0], self.pr_bounds[1]]
        ])))
        
        temp_grid, pr_grid = np.meshgrid(temp_points, pr_points)
        
        # Create figure with two subplots
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))
        
        # Plot power surface
        self._plot_surface(ax1, self.power_interpolator, 
                          temp_grid, pr_grid,
                          'Power_Shaft', 'viridis', '{:.0f}')
        
        # Plot efficiency surface
        self._plot_surface(ax2, self.eta_interpolator,
                          temp_grid, pr_grid,
                          'ETA_ts_poly', 'plasma', '{:.4f}')
        
        plt.tight_layout()
        plt.show()
    
    def _plot_surface(self, ax, interpolator, temp_grid, pr_grid, 
                     title, cmap='viridis', format_str='{:.0f}'):
        """Helper function to create surface plot."""
        # Calculate interpolated values
        interpolated_values = np.zeros_like(temp_grid)
        for i in range(temp_grid.shape[0]):
            for j in range(temp_grid.shape[1]):
                try:
                    interpolated_values[i,j] = interpolator(temp_grid[i,j], pr_grid[i,j])
                except (ValueError, TypeError):
                    # If interpolation fails, use extrapolation
                    interpolated_values[i,j] = self._extrapolate(
                        temp_grid[i,j], pr_grid[i,j], 
                        interpolator, 
                        self.power_data if 'Power' in title else self.eta_data
                    )
        
        # Create contour plot
        contour = ax.contourf(temp_grid, pr_grid, interpolated_values, 
                            levels=30, cmap=cmap, extend='both')
        plt.colorbar(contour, ax=ax, label=title)
        
        # Plot data points
        scatter = ax.scatter(self.df['T_in'], self.df['PR_ts'], 
                           c='red', s=30)
        
        # Add text labels for each point
        for i, row in self.df.iterrows():
            ax.text(row['T_in'], row['PR_ts'], 
                    format_str.format(row[title]), 
                    fontsize=8, 
                    ha='center', 
                    va='center',
                    color='black')
        
        # Calculate padding for axis limits (5% of the range)
        x_padding = (self.temp_bounds[1] - self.temp_bounds[0]) * 0.05
        y_padding = (self.pr_bounds[1] - self.pr_bounds[0]) * 0.05
        
        # Set axis limits with padding
        ax.set_xlim(self.temp_bounds[0] - x_padding, self.temp_bounds[1] + x_padding)
        ax.set_ylim(self.pr_bounds[0] - y_padding, self.pr_bounds[1] + y_padding)
        
        ax.set_xlabel('T_in (C)')
        ax.set_ylabel('PR_ts')
        ax.set_title(title)

    def get_optimal_temp(self, pr, temp_range=None):
        """Find the temperature that gives maximum efficiency at a specific pressure ratio."""
        if temp_range is None:
            temp_range = self.temp_bounds
        
        # Define the objective function (negative efficiency since we want to maximize)
        def objective(temp):
            return -self.eta_interpolator(temp, pr)

        # Use scipy's minimize_scalar to find the optimal temperature
        result = minimize_scalar(
            objective,
            bounds=temp_range,
            method='bounded',
            options={'xatol': 1.0}  # 1 degree tolerance
        )

        optimal_temp = result.x
        max_efficiency = -result.fun  # Convert back from negative

        return optimal_temp, max_efficiency

def validate_function_signature(func, expected_params, func_name="function"):
    """
    Validate that a callable function has the expected parameters.
    
    Args:
        func: The callable function to validate
        expected_params (set): Set of expected parameter names
        func_name (str): Name of the function for error messages
    
    Returns:
        bool: True if validation passes
        
    Raises:
        ValueError: If required parameters are missing
    """
    if not callable(func):
        raise ValueError(f"{func_name} must be callable")
    
    sig = inspect.signature(func)
    actual_params = set(sig.parameters.keys())
    
    # Check for missing parameters
    missing_params = expected_params - actual_params
    if missing_params:
        raise ValueError(f"{func_name} missing required parameters: {missing_params}")
    
    # Optional: Check for unexpected parameters
    unexpected_params = actual_params - expected_params
    if unexpected_params:
        logger.warn(f"{func_name} has unexpected parameters: {unexpected_params}")
    
    return True


# User defined function
class UserDefinedFunction:
    """
    A flexible wrapper that handles both constant values and callable functions.

    This class allows users to specify either a constant value or a custom function
    that will be evaluated with specific arguments. It's commonly used in geoTherm
    for defining parameters that can be either fixed values or dynamic calculations.

    Attributes:
        _func: The internal function that will be called during evaluation

    Example:
        # Constant value
        udf = UserDefinedFunction(100.0)
        result = udf.evaluate(upstream_node, mass_flow)  # Returns 100.0

        # Custom function
        def my_func(US, w, model=None):
            return US._P * w  # Some calculation
        udf = UserDefinedFunction(my_func)
        result = udf.evaluate(upstream_node, mass_flow)  # Returns calculated value
    """

    parameters = {'US', 'w', 'model'}
    quantity = 'PRESSURE'

    def __init__(self, value):
        """
        Initialize the UserDefinedFunction.

        Args:
            value: Either a numeric constant (int/float) or a callable function
        """
        self._func = None
        self.set_func(value)

    def _validate_func_signature(self, func):
        """Validate the function signature."""

        sig = inspect.signature(func)
        actual_params = set(sig.parameters.keys())
        # Check for missing parameters
        missing_params = self.parameters - actual_params
        if missing_params:
            raise ValueError(f"Missing required parameters for {func.__name__}: {missing_params}")

        # Check for unexpected parameters
        unexpected_params = actual_params - self.parameters
        if unexpected_params:
            logger.warn(f"Unexpected parameters for {func.__name__}: {unexpected_params}")
        return True

    def set_func(self, val):
        """
        Set the internal function based on the input value.

        If val is numeric, creates a lambda that returns the constant value.
        If val is callable, uses it directly.

        Args:
            val: Numeric constant or callable function

        Raises:
            Critical log message if val is neither numeric nor callable
        """
        if isinstance(val, (int, float)):
            # Create a lambda that accepts any arguments and returns the constant value
            def constant_func(*args, **kwargs):
                return val
            self._func = constant_func
        elif callable(val):
            # Use the callable directly
            self._validate_func_signature(val)
            self._func = val
            logger.info(f"{self.quantity} function set to SI units")
        else:
            try:
                self.set_func(float(val))
            except:
                logger.critical(
                    f"Value must be numeric or callable. Got {type(val)}")

    def evaluate(self, *args, **kwargs):
        """
        Evaluate the function with the given arguments.

        Args:
            *args: Positional arguments passed to the function
            **kwargs: Keyword arguments passed to the function

        Returns:
            The result of calling the internal function
        """
        return self._func(*args, **kwargs)

    @property
    def _state(self):
        return {'func': self._func}


class UserDefinedMassFlow(UserDefinedFunction):
    """
    A UserDefinedFunction that returns a mass flow rate.
    """

    parameters = {'US', 'DS', 'model'}
    quantity = 'MASSFLOW'




class ThermalUserDefinedFunction(UserDefinedFunction):
    """
    Specialized UserDefinedFunction for thermal systems with hot/cold node arguments.

    This class extends UserDefinedFunction to handle thermal calculations that
    typically involve both hot and cold nodes (e.g., heat exchangers, thermal
    efficiency calculations).

    The main difference from the parent class is that when a constant value is
    provided, the lambda function expects (hot_node, cold_node, model=None) as
    arguments instead of (US, w, model=None).

    Example:
        # Constant Heat
        thermal_udf = ThermalUserDefinedFunction(1e6)
        result = thermal_udf.evaluate(hot_node, cold_node)  # Returns 1e6

        # Custom thermal calculation
        def thermal_func(hot_node, cold_node, model=None):
            return (hot_node._T - cold_node._T) / hot_node._T
        thermal_udf = ThermalUserDefinedFunction(thermal_func)
        result = thermal_udf.evaluate(hot_node, cold_node)  # Returns heat
    """

    def set_func(self, val):
        """
        Set the internal function for thermal calculations.

        Overrides the parent method to use (hot_node, cold_node, model=None) 
        as the default argument signature for constant values.

        Args:
            val: Numeric constant or callable function with thermal signature

        Raises:
            Critical log message if val is neither numeric nor callable
        """
        if isinstance(val, (int, float)):
            # Create a lambda that ignores hot/cold nodes and returns the constant value
            self._func = lambda hot_node, cold_node, model=None, val=val: val
        elif callable(val):
            # Use the callable directly
            self._func = val
        else:
            logger.critical("Value must be numeric or callable.")

