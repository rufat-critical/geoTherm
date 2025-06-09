from geoTherm.utilities.loaders import concepts_excel_reader
import numpy as np
from scipy.interpolate import griddata, LinearNDInterpolator
from geoTherm.common import logger


class ConceptsMap:
    """Class for interpolating turbine performance parameters."""
    
    def __init__(self, data_csv):
        """
        Initialize interpolator with data.
        
        Args:
            data_df (pd.DataFrame): DataFrame containing turbine data
        """
        

        self.data_csv = data_csv
        self.data = concepts_excel_reader(data_csv)
        self.T_ref = 288.15  # K
        self.p_ref = 101325  # Pa
        
        # Build interpolators once during initialization
        x = self.data['T0.in']  # Temperature
        y = self.data['PR_ts']  # Pressure ratio
        
        # Create interpolators for each parameter we need
        self.mass_flow_interpolator = LinearNDInterpolator(
            points=np.column_stack((x, y)),
            values=self.data['m_c']
        )
        self.eta_interpolator = LinearNDInterpolator(
            points=np.column_stack((x, y)),
            values=self.data['ETA_ts_ad']
        )
        self.power_interpolator = LinearNDInterpolator(
            points=np.column_stack((x, y)),
            values=self.data['Power(Shaft)']
        )
    
    def interpolate_2d(self, x, y, z, xi, yi):
        """Perform 2D interpolation using griddata."""
        return griddata((x, y), z, (xi, yi), method='linear')
    
    def interpolate_at_pressure(self, p0_in, t0_in, p_out, param_col):
        """Interpolate any property using pre-built interpolator."""
        # Calculate pressure ratio
        pr_ts = p0_in / p_out
        
        # Create the point to interpolate at
        point = np.array([t0_in, pr_ts])
        
        # Use appropriate interpolator based on parameter
        if param_col == 'm_c':
            value = self.mass_flow_interpolator(point)
        elif param_col == 'ETA_ts_ad':
            value = self.eta_interpolator(point)
        elif param_col == 'Power(Shaft)':
            value = self.power_interpolator(point)
        else:
            raise ValueError(f"Unknown parameter column: {param_col}")
        
        # Handle NaN
        if np.isnan(value):
            logger.warn(
                f"Interpolation failed - point outside data range: "
                f"p0_in={p0_in/1e5:.1f} bar, T0_in={t0_in:.1f}K, PR_ts={pr_ts:.2f}"
            )
            return 0
        
        return float(value)

    def get_massflow(self, p0_in, t0_in, p_out):
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
        corrected_mass_flow = self.interpolate_at_pressure(p0_in, t0_in, p_out, 'm_c')
        mass_flow = float(corrected_mass_flow) * (p0_in/self.p_ref) * np.sqrt(self.T_ref/t0_in)
        
        return mass_flow
    
    def get_eta_ts(self, p0_in, t0_in, p_out):
        """Interpolate total-to-static efficiency."""
        return self.interpolate_at_pressure(p0_in, t0_in, p_out, 'ETA_ts_ad')
    
    def get_W_shaft(self, p0_in, t0_in, p_out):
        """Interpolate shaft power."""
        return self.interpolate_at_pressure(p0_in, t0_in, p_out, 'Power(Shaft)')