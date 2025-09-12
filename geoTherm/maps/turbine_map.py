import numpy as np
from scipy.interpolate import LinearNDInterpolator
from geoTherm.common import logger
import pandas as pd
from geoTherm.units import toSI
from geoTherm.maps.massflow import MassFlowMap


class TurbineMap:
    """Class for interpolating turbine performance parameters."""

    def __init__(self, data=None, type='axial', sheet_names=None):
        """
        Initialize interpolator with data.

        Args:
            df (pandas.DataFrame): DataFrame containing turbine data
            type (str): Type of turbine map ('axial' or 'rital')
            sheet_names (list, optional): List of sheet names to read
            fallback_method (str): Method for handling extrapolation ('extrapolate' or 'nearest')
        """


        if isinstance(data, pd.DataFrame):
            self.map = data
        else:
            logger.critical(
                "TurbineMap must be initialized with a pandas DataFrame"
            )

        self.massflow_map = MassFlowMap(self.map)

        self.initialize()

    def initialize(self):
        # Initialize interpolators


        # Get the min and max of each column
        self.T_min, self.T_max = self.map['T0'].min(), self.map['T0'].max()
        self.P_min, self.P_max = self.map['P0'].min(), self.map['P0'].max()
        self.RPM_min, self.RPM_max = self.map['RPM'].min(), self.map['RPM'].max()
        self.PR_min, self.PR_max = self.map['PR_ts'].min(), self.map['PR_ts'].max()
        self.P_out_min, self.P_out_max = self.map['P_out'].min(), self.map['P_out'].max()

        # Initialize flags
        self.fixed_RPM = False
        self.fixed_T0 = False
        self.fixed_P_out = False

        if len(self.map['RPM'].unique()) == 1:
            self.fixed_RPM = True
            logger.info("Turbine Map has constant RPM")
        if len(self.map['T0'].unique()) == 1:
            self.fixed_T0 = True
            logger.info("Turbine Map has constant T0")

        rel_P_Out = (self.P_out_max - self.P_out_min)/self.P_out_min

        if rel_P_Out < 0.05:
            self.fixed_P_out = True
            logger.info("Turbine Map has constant P_out")


        if not self.fixed_RPM:
            self.RPM_range = self.map['RPM'].max() - self.map['RPM'].min()
        if not self.fixed_T0:
            self.T_range = self.map['T0'].max() - self.map['T0'].min()
        if not self.fixed_P_out:
            self.P_out_range = self.map['P_out'].max() - self.map['P_out'].min()

        self.P_range = self.P_max - self.P_min

        # Create Grid for Interpolators
        if self.fixed_RPM and self.fixed_T0:
            self.grid = np.column_stack((self.map['P0']/self.P_range,
                                         self.map['PR_ts']))
        elif self.fixed_RPM and self.fixed_P_out:
            self.grid = np.column_stack((#self.map['P0']/self.P_range,
                                         self.map['T0']/self.T_range,
                                         self.map['PR_ts']))
        elif self.fixed_RPM:
            self.grid = np.column_stack((self.map['P0']/self.P_range,
                                         self.map['T0']/self.T_range,
                                         self.map['PR_ts']))
        else:
            self.grid = np.column_stack((self.map['P0']/self.P_range,
                                         self.map['T0']/self.T_range,
                                         self.map['PR_ts'],
                                         self.map['RPM']/self.RPM_range))

        # Create primary interpolators
        self.LinearNDInterpolator = {
            'ETA_ts': LinearNDInterpolator(points=self.grid, values=self.map['ETA_ts']),
            'Power': LinearNDInterpolator(points=self.grid, values=self.map['Power'])
        }

    def interpolate(self, P0, T0, P_out, N, param):
        """Interpolate any property using pre-built interpolator with fallback extrapolation."""
        # Calculate pressure ratio
        PR_ts = P0 / P_out
        
        # Create the normalized point
        if self.fixed_RPM and self.fixed_T0:
            point = np.array([P0/self.P_range, PR_ts])
        elif self.fixed_RPM and self.fixed_P_out:
            point = np.array([T0/self.T_range, PR_ts])
        elif self.fixed_RPM:
            point = np.array([P0/self.P_range, T0/self.T_range, PR_ts])
        else:
            point = np.array([P0/self.P_range,
                              T0/self.T_range,
                              PR_ts,
                              N/self.RPM_range])

        # Try primary interpolation
        value = self.LinearNDInterpolator[param](point)        

        if np.isnan(value):
            logger.warn(f"Interpolation is outside convex hull for {param}\n"
                        f"P0={P0/1e5:.1f} bar, T0={T0:.1f}K, PR={PR_ts:.1f}, N={N:.0f}\n"
                        f"Data ranges: \n"
                        f"P0={self.P_min/1e5:.1f}-{self.P_max/1e5:.1f} bar,\n"
                        f"T0={self.T_min:.1f}-{self.T_max:.1f}K,\n"
                        f"P_out={self.P_out_min/1e5:.1f}-{self.P_out_max/1e5:.1f} bar,\n"
                        f"RPM={self.RPM_min:.0f}-{self.RPM_max:.0f}")
            logger.warn(f"returning 0 for {param}")
            return 0

        return float(value)

    def get_w(self, P0, T0, P_out, N=None):
        """
        Calculate mass flow using both direct interpolation and corrected mass flow approach.

        Args:
            p0_in (float): Inlet total pressure [Pa]
            t0_in (float): Inlet total temperature [K]
            p_out (float): Outlet pressure [Pa]

        Returns:
            tuple: (direct_mass_flow, calculated_mass_flow) in kg/s
        """

        if self.fixed_P_out:
            return self.massflow_map.get_w(P0, T0, P_out)
        else:
            return self.massflow_map.get_w(P0, T0, P_out)

        # Get corrected mass flow and convert to actual mass flow
        return self.massflow_map.get_w(P0, T0, P_out)

    def _get_w(self, P0, T0, P_out, N=None):
        return self.massflow_map._get_w(P0, T0, P_out)
    
    def get_eta_ts(self, P0, T0, P_out, N=None):
        """Interpolate total-to-static efficiency."""
        eta = self.interpolate(P0=P0, T0=T0, P_out=P_out, N=N, param='ETA_ts')
        return eta

    def get_W_shaft(self, P0, T0, P_out, N=None):
        """Interpolate shaft power."""
        return self.interpolate(P0=P0, T0=T0, P_out=P_out, N=N, param='Power')

    def eta_func(self, US_thermo, Pe, N, model):
        return self.get_eta_ts(US_thermo._P, US_thermo.T, Pe, N)
    
    def optimal_power(self, T0, Pout, N=None):
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
            power = self.get_W_shaft(P0, T0, Pout, N)
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
        massflow_opt = self.get_w(P0_opt, T0, Pout, N)
        eta_opt = self.get_eta_ts(P0_opt, T0, Pout, N)

        if not result.success:
            logger.warn(f"Optimization failed: {result.message}")
            # Fallback: try a grid search
            from pdb import set_trace
            set_trace()
            PR_range = np.linspace(PR_min, PR_max, 100)
            powers = []
            for PR in PR_range:
                P0 = PR * Pout
                power = self.get_W_shaft(P0, T0, Pout, N)
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
    
    def optimal_eta(self, T0, Pout, N=None):
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
            eta = self.get_eta_ts(P0, T0, Pout, N)
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
        