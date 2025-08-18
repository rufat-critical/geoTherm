import numpy as np
from scipy.interpolate import LinearNDInterpolator
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
            Dataframe with columns 'P0', 'T0', 'PR_ts', 'massflow'
        """
        self.df = df
        self.initialize()

    def initialize(self):
        """Initialize the interpolators for mass flow and choked mass flow."""


        self.T_min, self.T_max = self.df['T0'].min(), self.df['T0'].max()
        self.P_min, self.P_max = self.df['P0'].min(), self.df['P0'].max()

        self.T_range = self.T_max - self.T_min
        self.P_range = self.P_max - self.P_min

        # Group by P0 and T0, then find the maximum massflow for each group (choked condition)
        self.choked_df = self.df.groupby(['P0', 'T0'])['massflow'].max().reset_index()

        # Create grid for mass flow interpolator (T0, P0, PR_ts)
        self.grid = np.column_stack((self.df['T0']/self.T_range, self.df['P0']/self.P_range, self.df['P0']/self.df['P_out']))

        self.massflow_interpolator = LinearNDInterpolator(
            points=self.grid,
            values=self.df['massflow']
        )
        
        self.grid_dP = np.column_stack((self.df['T0']/self.T_range, self.df['P0']/self.P_range, self.df['massflow']))

        self.dP_interpolator = LinearNDInterpolator(
            points=self.grid_dP,
            values=self.df['P0'] - self.df['P_out']
        )

        self.choked_grid = np.column_stack((self.choked_df['T0']/self.T_range, self.choked_df['P0']/self.P_range))
        self.choked_interpolator = LinearNDInterpolator(
            points=self.choked_grid,
            values=self.choked_df['massflow']
        )

    def _get_nearest_neighbor(self, points, grid, values):
        """
        Get nearest neighbor values for points that are outside the interpolation domain.

        Parameters:
        -----------
        points : array-like
            Query points (n_points, n_dimensions)
        grid : array-like
            Training points (n_samples, n_dimensions)
        values : array-like
            Values at training points (n_samples,)

        Returns:
        --------
        array-like
            Nearest neighbor values
        """
        # Calculate distances from query points to all grid points
        distances = cdist(points, grid)

        # Find indices of nearest neighbors
        nearest_indices = np.argmin(distances, axis=1)

        # Return values at nearest neighbor indices
        return values[nearest_indices]

    def _extrapolate_linear(self, points, grid, values, k=5):
        """
        Perform linear extrapolation for points outside the interpolation domain.

        Parameters:
        -----------
        points : array-like
            Query points (n_points, n_dimensions)
        grid : array-like
            Training points (n_samples, n_dimensions)
        values : array-like
            Values at training points (n_samples,)
        k : int
            Number of nearest points to use for extrapolation

        Returns:
        --------
        array-like
            Extrapolated values
        """
        extrapolated_values = np.zeros(len(points))
        
        for i, point in enumerate(points):
            # Find the k nearest points to use for extrapolation
            k_actual = min(k, len(grid))
            
            # Calculate distances to all points
            distances = np.sqrt(np.sum((grid - point)**2, axis=1))
            nearest_indices = np.argsort(distances)[:k_actual]
            
            # Get the coordinates and values for these points
            nearest_points = grid[nearest_indices]
            nearest_values = values[nearest_indices]
            
            # Fit a linear hyperplane through these points
            # Add a column of ones for the intercept term
            A = np.column_stack((nearest_points, np.ones(k_actual)))
            
            # Solve for coefficients
            try:
                coeffs = np.linalg.lstsq(A, nearest_values, rcond=None)[0]
                # Return the extrapolated value
                extrapolated_values[i] = np.dot(np.append(point, 1), coeffs)
            except np.linalg.LinAlgError:
                # If linear fit fails, return the nearest neighbor value
                extrapolated_values[i] = nearest_values[0]
        
        return extrapolated_values

    def get_massflow(self, P0, T0, P_out, *args, **kwargs):
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
        # For RBF interpolator, we need to handle arrays differently
        interpolated_values = self.massflow_interpolator(
            T0/self.T_range, 
            P0/self.P_range, 
            P0/P_out
        )

        # Check for NaN values (extrapolation) - only needed for LinearNDInterpolator
        if hasattr(interpolated_values, 'dtype') and np.issubdtype(interpolated_values.dtype, np.floating):
            nan_mask = np.isnan(interpolated_values)
            if np.any(nan_mask):
                points = np.column_stack((T0/self.T_range, P0/self.P_range, P0/P_out))
                # Use linear extrapolation for NaN points
                extrapolated_values = self._extrapolate_linear(
                    points[nan_mask], 
                    self.grid, 
                    self.df['massflow'].values
                )
                # Replace NaN values with extrapolated values
                interpolated_values[nan_mask] = np.maximum(0, extrapolated_values)

        choked = self._w_max(P0, T0)

        return np.minimum(interpolated_values, choked)[0]
    
    def _dP(self, P0, T0, w, *args, **kwargs):
        """
        Get pressure drop for given conditions using Brent's method solver.
        
        This method finds the pressure drop by solving for the outlet pressure
        that gives the desired mass flow rate using get_massflow.
        
        Parameters:
        -----------
        P0 : float
            Total pressure (Pa)
        T0 : float
            Total temperature (K)
        w : float
            Desired mass flow rate (kg/s)
        
        Returns:
        --------
        tuple
            (pressure_drop, None) where pressure_drop is negative
        """
        def objective_function(P_out):
            """
            Objective function for the solver.
            Returns the difference between desired and calculated mass flow.
            """
            try:
                calculated_massflow = self.get_massflow(P0, T0, P_out)
                return calculated_massflow - w
            except:
                # If interpolation fails, return a large value
                return 1e6
        
        # Get choked mass flow to determine bounds
        choked_massflow = self._w_max(P0, T0)
        
        # Set bounds for the solver
        P_out_min = P0 * 0.01  # Minimum 1% of inlet pressure
        P_out_max = P0 * 0.99  # Maximum 99% of inlet pressure
        
        # Check if desired mass flow is achievable
        if w >= choked_massflow:
            # If desired mass flow is greater than or equal to choked, 
            # use choked condition (minimum possible P_out)
            P_out_solution = P_out_min
            pressure_drop = P_out_solution - P0
            from pdb import set_trace
            set_trace()
            logger.warn(f"Desired mass flow {w:.4f} >= choked mass flow {choked_massflow:.4f}. "
                       f"Using choked condition.")
            return pressure_drop, None
        
        try:
            # Evaluate objective function at bounds to ensure we have a bracket
            f_min = objective_function(P_out_min)
            f_max = objective_function(P_out_max)
            
            # Check if we have a valid bracket (sign change)
            if f_min * f_max > 0:
                # No sign change, try to find a better bracket
                # Test some intermediate points
                test_points = [P0 * 0.1, P0 * 0.3, P0 * 0.5, P0 * 0.7, P0 * 0.9]
                f_values = [objective_function(p) for p in test_points]
                
                # Find where sign changes
                for i in range(len(f_values) - 1):
                    if f_values[i] * f_values[i + 1] <= 0:
                        P_out_min = test_points[i]
                        P_out_max = test_points[i + 1]
                        break
                else:
                    # Still no sign change, use fallback
                    raise ValueError("Could not find valid bracket for Brent's method")
            
            # Use Brent's method to find the root
            P_out_solution = brentq(
                objective_function, 
                P_out_min, 
                P_out_max,
                xtol=1e-6,  # Tolerance for x
                rtol=1e-6,  # Relative tolerance
                maxiter=100  # Maximum iterations
            )
            
            # Calculate pressure drop
            pressure_drop = P_out_solution - P0
            
            # Verify the solution gives reasonable mass flow
            final_massflow = self.get_massflow(P0, T0, P_out_solution)
            massflow_error = abs(final_massflow - w) / w if w > 0 else abs(final_massflow - w)
            
            if massflow_error > 0.05:  # 5% tolerance (tighter than before)
                logger.warn(f"Large mass flow error in _dP solver: {massflow_error:.3f}. "
                           f"Desired: {w:.4f}, Got: {final_massflow:.4f}")
            
            return pressure_drop, None
            
        except Exception as e:
            logger.warn(f"Failed to solve for pressure drop using Brent's method: {e}. "
                       f"Using fallback interpolation method.")
            
            # Fallback to original interpolation method
            interpolated_values = self.dP_interpolator(T0/self.T_range, P0/self.P_range, w)
            
            # Check for NaN values (extrapolation) - only needed for LinearNDInterpolator
            if hasattr(interpolated_values, 'dtype') and np.issubdtype(interpolated_values.dtype, np.floating):
                nan_mask = np.isnan(interpolated_values)
                if np.any(nan_mask):
                    points = np.column_stack((T0/self.T_range, P0/self.P_range, w))
                    # Use linear extrapolation for NaN points
                    extrapolated_values = self._extrapolate_linear(
                        points[nan_mask], 
                        self.grid_dP, 
                        (self.df['P0'] - self.df['P_out']).values
                    )
                    # Replace NaN values with extrapolated values
                    interpolated_values[nan_mask] = np.maximum(0, extrapolated_values)
            
            return -interpolated_values, None
    

    def _w_max(self, P0, T0, *args, **kwargs):
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
        points = np.column_stack((T0/self.T_range, P0/self.P_range))

        # Get interpolated values
        interpolated_values = self.choked_interpolator(points)
        
        # Check for NaN values (extrapolation)
        nan_mask = np.isnan(interpolated_values)

        if np.any(nan_mask):
            logger.warn(f"Choked Mass flow interpolation outside bounds "
                        f"- using nearest neighbor for: "
                        f"P0={P0/1e5:.1f} bar, T0={T0:.1f}K\n"
                        f"Data ranges: P0={self.P_min/1e5:.1f}-"
                        f"{self.P_max/1e5:.1f} bar, "
                        f"T0={self.T_min:.1f}-{self.T_max:.1f}K, ")
            # Get nearest neighbor values for NaN points
            nearest_values = self._get_nearest_neighbor(
                points[nan_mask],
                self.choked_grid,
                self.choked_df['massflow'].values
            )

            # Replace NaN values with nearest neighbor values
            interpolated_values[nan_mask] = nearest_values

        return interpolated_values

    def plot_choked_massflow(self):
        """
        Plot choked mass flow vs pressure at different temperatures.
        """
        # Get unique temperature values
        temperatures = sorted(self.choked_df['T0'].unique())

        # Create the plot
        plt.figure(figsize=(10, 6))

        # Plot a line for each temperature
        for T0 in temperatures:
            # Filter data for this temperature
            temp_data = self.choked_df[self.choked_df['T0'] == T0].sort_values('P0')

            # Plot the line
            plt.plot(temp_data['P0']/1e5, temp_data['massflow'], 
                    marker='s', label=f'T0 = {T0:.1f} K')

        # Customize the plot
        plt.xlabel('Pressure (bar)')
        plt.ylabel('Choked Mass Flow (kg/s)')
        plt.title('Choked Mass Flow vs Pressure at Different Temperatures')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.show()

    def plot_massflow(self, figsize=(1200, 800)):
        """
        Generate an interactive plot showing mass flow vs pressure ratio for each temperature.
        
        Parameters:
        -----------
        figsize : tuple
            Figure size in pixels (width, height)
            
        Returns:
        --------
        plotly.graph_objects.Figure
            Interactive plotly figure
        """
        # Get unique temperature and pressure values
        temperatures = sorted(self.df['T0'].unique())
        pressures = sorted(self.df['P0'].unique())
        
        # Calculate number of subplots needed
        n_temps = len(temperatures)
        n_cols = min(3, n_temps)  # Max 3 columns
        n_rows = (n_temps + n_cols - 1) // n_cols
        
        # Create subplots
        fig = make_subplots(
            rows=n_rows, 
            cols=n_cols,
            subplot_titles=[f'T0 = {T0:.1f} K' for T0 in temperatures],
            specs=[[{"secondary_y": False} for _ in range(n_cols)] for _ in range(n_rows)]
        )
        
        # Color palette for different pressures
        colors = px.colors.qualitative.Set1 + px.colors.qualitative.Set2 + px.colors.qualitative.Set3
        
        for i, T0 in enumerate(temperatures):
            row = i // n_cols + 1
            col = i % n_cols + 1
            
            # Filter data for this temperature
            temp_data = self.df[self.df['T0'] == T0]
            
            # Plot a line for each pressure
            for j, P0 in enumerate(pressures):
                # Filter data for this pressure and temperature
                pressure_data = temp_data[temp_data['P0'] == P0].sort_values('P_out')
                
                if len(pressure_data) > 0:
                    # Calculate pressure ratio
                    pressure_ratio = P0 / pressure_data['P_out']
                    
                    # Add trace to subplot
                    fig.add_trace(
                        go.Scatter(
                            x=pressure_ratio,
                            y=pressure_data['massflow'],
                            mode='lines+markers',
                            name=f'P0 = {P0/1e5:.1f} bar',
                            line=dict(color=colors[j % len(colors)]),
                            marker=dict(size=6),
                            showlegend=(i == 0),  # Only show legend for first subplot
                            hovertemplate='<b>Pressure Ratio:</b> %{x:.3f}<br>' +
                                        '<b>Mass Flow:</b> %{y:.4f} kg/s<br>' +
                                        '<b>P0:</b> ' + f'{P0/1e5:.1f} bar<br>' +
                                        '<b>T0:</b> ' + f'{T0:.1f} K<br>' +
                                        '<extra></extra>'
                        ),
                        row=row, col=col
                    )
            
            # Update subplot layout
            fig.update_xaxes(title_text="Pressure Ratio (P0/P_out)", row=row, col=col)
            fig.update_yaxes(title_text="Mass Flow (kg/s)", row=row, col=col)
        
        # Update overall layout
        fig.update_layout(
            title="Mass Flow vs Pressure Ratio by Temperature",
            width=figsize[0],
            height=figsize[1],
            showlegend=True,
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            )
        )
        
        return fig
    
    def plot_massflow_single(self, figsize=(1000, 600)):
        """
        Generate a single interactive plot with all temperatures on the same graph,
        using different colors for temperatures and line styles for pressures.
        
        Parameters:
        -----------
        figsize : tuple
            Figure size in pixels (width, height)
            
        Returns:
        --------
        plotly.graph_objects.Figure
            Interactive plotly figure
        """
        # Get unique temperature and pressure values
        temperatures = sorted(self.df['T0'].unique())
        pressures = sorted(self.df['P0'].unique())
        
        # Create figure
        fig = go.Figure()
        
        # Color palette for different temperatures
        temp_colors = px.colors.qualitative.Set1 + px.colors.qualitative.Set2
        # Line styles for different pressures
        line_styles = ['solid', 'dash', 'dot', 'dashdot']
        
        for i, T0 in enumerate(temperatures):
            # Filter data for this temperature
            temp_data = self.df[self.df['T0'] == T0]
            
            # Plot a line for each pressure
            for j, P0 in enumerate(pressures):
                # Filter data for this pressure and temperature
                pressure_data = temp_data[temp_data['P0'] == P0].sort_values('P_out')
                
                if len(pressure_data) > 0:
                    # Calculate pressure ratio
                    pressure_ratio = P0 / pressure_data['P_out']
                    
                    # Add trace
                    fig.add_trace(
                        go.Scatter(
                            x=pressure_ratio,
                            y=pressure_data['massflow'],
                            mode='lines+markers',
                            name=f'T0={T0:.1f}K, P0={P0/1e5:.1f}bar',
                            line=dict(
                                color=temp_colors[i % len(temp_colors)],
                                dash=line_styles[j % len(line_styles)]
                            ),
                            marker=dict(size=4),
                            hovertemplate='<b>Pressure Ratio:</b> %{x:.3f}<br>' +
                                        '<b>Mass Flow:</b> %{y:.4f} kg/s<br>' +
                                        '<b>P0:</b> ' + f'{P0/1e5:.1f} bar<br>' +
                                        '<b>T0:</b> ' + f'{T0:.1f} K<br>' +
                                        '<extra></extra>'
                        )
                    )
        
        # Update layout
        fig.update_layout(
            title="Mass Flow vs Pressure Ratio",
            xaxis_title="Pressure Ratio (P0/P_out)",
            yaxis_title="Mass Flow (kg/s)",
            width=figsize[0],
            height=figsize[1],
            showlegend=True,
            legend=dict(
                orientation="v",
                yanchor="top",
                y=1,
                xanchor="left",
                x=1.02
            )
        )
        
        return fig


class TurboMassFlowMap(MassFlowMap):
    pass