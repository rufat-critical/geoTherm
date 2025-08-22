import numpy as np
from scipy.interpolate import LinearNDInterpolator, NearestNDInterpolator
from scipy.spatial.distance import cdist
from scipy.optimize import fsolve
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from geoTherm.logger import logger
from scipy.optimize import brentq


class MassFlowMap:

    def __init__(self, df):
        """
        Initialize the MassFlowMap with a dataframe containing mass flow data.

        Parameters:
        -----------
        df : pandas.DataFrame
            Dataframe with columns 'P0', 'T0', 'P_out', 'w', 'w_c
        """
        
        self.T_ref = 288.15  # K
        self.p_ref = 101325  # Pa

        
        self.df = df
        self.initialize()

    def initialize(self):
        """Initialize the interpolators for mass flow and choked mass flow."""
        self.T0_min, self.T0_max = self.df['T0'].min(), self.df['T0'].max()
        self.P0_min, self.P0_max = self.df['P0'].min(), self.df['P0'].max()
        self.P_out_min, self.P_out_max = self.df['P_out'].min(), self.df['P_out'].max()

        self.T0_range = self.T0_max - self.T0_min
        self.P0_range = self.P0_max - self.P0_min
        self.P_out_range = self.P_out_max - self.P_out_min

        # Create grid for mass flow interpolator (T0, P0, PR_ts)
        self.grid = np.column_stack((self.df['P0']/self.P0_range,
                                     self.df['T0']/self.T0_range,
                                     self.df['P_out']/self.P_out_range))

        # Group by P0 and T0, then find the maximum massflow for each group (choked condition)
        self.choked_df = self.df.groupby(['P0', 'T0'])['massflow'].max().reset_index()
        self.choked_grid = np.column_stack((self.choked_df['P0']/self.P0_range,
                                            self.choked_df['T0']/self.T0_range))

        self.LinearNDInterpolator = {
            'w': LinearNDInterpolator(
                points=self.grid,
                values=self.df['massflow']
            ),
            'w_c': LinearNDInterpolator(
                points=self.grid,
                values=self.df['m_c']
            ),
            'w_choked': LinearNDInterpolator(
                points=self.choked_grid,
                values=self.choked_df['massflow']
            )
        }

        self.fallback_interpolator = {
            'w': NearestNDInterpolator(self.grid, self.df['massflow']),
            'w_c': NearestNDInterpolator(self.grid, self.df['m_c']),
            'w_choked': NearestNDInterpolator(self.choked_grid, self.choked_df['massflow'])
        }

    def interpolate(self, P0, T0, P_out, param):

        if param == 'w_choked':
            point = np.column_stack((P0/self.P0_range,
                                     T0/self.T0_range))
        else:
            point = np.column_stack((P0/self.P0_range,
                                    T0/self.T0_range,
                                    P_out/self.P_out_range))

        value = self.LinearNDInterpolator[param](point)

        if np.isnan(value):
            logger.warn(f"Interpolation is outside convex hull for {param}\n"
                        f"P0={P0/1e5:.1f} bar, T0={T0:.1f}K, P_out={P_out/1e5:.1f} bar\n"
                        f"Data ranges: \n"
                        f"P0={self.P0_min/1e5:.1f}-{self.P0_max/1e5:.1f} bar,\n"
                        f"T0={self.T0_min:.1f}-{self.T0_max:.1f}K,\n"
                        f"P_out={self.P_out_min/1e5:.1f}-{self.P_out_max/1e5:.1f} bar")

            if param == 'w':
                logger.warn(f"Using nearest neighbor fallback.")
                w_choked = self.interpolate(P0, T0, None, 'w_choked')
                w = self.fallback_interpolator['w'](point)
                return float(np.minimum(w, w_choked))
            else:
                return float(self.fallback_interpolator[param](point))

        return float(value)

    def get_w(self, P0, T0, P_out):
        """
        Get mass flow rate for given conditions.

        Parameters:
        -----------
        T0 : float or array-like
            Total temperature (K)
        P0 : float or array-like
            Total pressure (Pa)
        PR_ts : float or array-like
            Pressure ratio (total-to-static)

        Returns:
        --------
        float or array-like
            Mass flow rate (kg/s)
        """
        corrected_w = self.interpolate(P0=P0, T0=T0, P_out=P_out, param='w_c')
        w = corrected_w * (P0/self.p_ref) * np.sqrt(self.T_ref/T0)

        return np.minimum(w, self._w_max(P0, T0))

    def _get_w(self, P0, T0, P_out):
        return self.interpolate(P0, T0, P_out, 'w')

    def _w_max(self, P0, T0):
        """
        Get choked mass flow rate for given conditions.

        Parameters:
        -----------
        T0 : float or array-like
            Total temperature (K)
        P0 : float or array-like
            Total pressure (Pa)

        Returns:
        --------
        float or array-like
            Choked mass flow rate (kg/s)
        """
        return self.interpolate(P0, T0, 0, 'w_choked')


class TurboMassFlowMap(MassFlowMap):
    pass