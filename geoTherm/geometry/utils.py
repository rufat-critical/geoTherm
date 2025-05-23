import matplotlib.pyplot as plt
import numpy as np
from geoTherm.common import units

class GeometryPlotter:
    """
    A utility class for plotting different geometry objects.
    
    This class provides methods to visualize various geometry types including
    cylinders, tube banks, and finned tube banks.
    
    Attributes
    ----------
    geometry : Geometry
        The geometry object to be plotted
    """
    
    def __init__(self, geometry):
        """
        Initialize the GeometryPlotter with a geometry object.
        
        Parameters
        ----------
        geometry : Geometry
            The geometry object to be plotted
        """
        self.geometry = geometry
        
    def _setup_plot(self, figsize=(12, 6)):
        """Set up the basic plot structure with three subplots."""
        fig = plt.figure(figsize=figsize)
        gs = fig.add_gridspec(1, 3, width_ratios=[1, 1, 0.3])
        ax1 = fig.add_subplot(gs[0])  # Top view
        ax2 = fig.add_subplot(gs[1])  # Side view
        ax3 = fig.add_subplot(gs[2])  # Info panel
        return fig, ax1, ax2, ax3
        
    def _plot_tubes(self, ax1, ax2, x_positions, y_positions):
        """Plot the basic tube structure in both views."""
        # Plot tubes in top view
        for i in range(self.geometry.N_l):
            for j in range(self.geometry.N_t):
                tube = plt.Circle((x_positions[i], y_positions[j]), 
                                self.geometry.D/2,
                                facecolor='gray', alpha=0.5, edgecolor='black')
                ax1.add_patch(tube)

        # Plot tubes in side view
        for i in range(self.geometry.N_l):
            for j in range(self.geometry.N_t):
                tube = plt.Rectangle((0, y_positions[j] - self.geometry.D/2), 
                                   self.geometry.L, self.geometry.D,
                                   facecolor='gray', alpha=0.5, edgecolor='black')
                ax2.add_patch(tube)
                
    def _plot_spacing(self, ax1, x_positions, y_positions):
        """Plot spacing indicators and labels."""
        # Add S_t line in top view
        if self.geometry.N_t > 1:
            ax1.plot([x_positions[0], x_positions[0]], 
                    [y_positions[0], y_positions[1]],
                    'r--', linewidth=2)
            ax1.text(x_positions[0], (y_positions[0] + y_positions[1])/2,
                    f'S_t = {self.geometry.S_t:.3f}m',
                    ha='right', va='center', color='red')

        # Add S_l line in top view
        if self.geometry.N_l > 1:
            ax1.plot([x_positions[0], x_positions[1]], 
                    [y_positions[0], y_positions[0]],
                    'b--', linewidth=2)
            ax1.text((x_positions[0] + x_positions[1])/2, y_positions[0],
                    f'S_l = {self.geometry.S_l:.3f}m',
                    ha='center', va='bottom', color='blue')
                    
    def _plot_fins(self, ax1, ax2, x_positions, y_positions):
        """Plot the fins in both views."""
        if not hasattr(self.geometry, 'D_fin'):
            return
            
        # Plot fins in top view
        for i in range(self.geometry.N_l):
            for j in range(self.geometry.N_t):
                fin = plt.Circle((x_positions[i], y_positions[j]), 
                               self.geometry.D_fin/2,
                               facecolor='none', edgecolor='gray', alpha=0.3)
                ax1.add_patch(fin)

        # Plot fins in side view
        for i in range(self.geometry.N_l):
            for j in range(self.geometry.N_t):
                y_center = y_positions[j]
                if hasattr(self.geometry, 'N_fin') and self.geometry.N_fin > 0:
                    fin_spacing = self.geometry.L / self.geometry.N_fin
                    for k in range(int(self.geometry.N_fin)):
                        fin_x = k * fin_spacing
                        fin = plt.Rectangle(
                            (fin_x, y_center - self.geometry.D_fin/2),
                            self.geometry.th_fin, self.geometry.D_fin,
                            facecolor='none', edgecolor='gray', alpha=0.7, linewidth=1
                        )
                        ax2.add_patch(fin)
                        
    def _setup_axes(self, ax1, ax2):
        """Set up axes labels, titles, and limits."""
        # Set labels and titles
        ax1.set_xlabel(f'Longitudinal direction ({units._output_units_for_display["LENGTH"]})')
        ax1.set_ylabel(f'Transverse direction ({units._output_units_for_display["LENGTH"]})')
        ax1.set_title('Top View')
        
        ax2.set_xlabel(f'Length ({units._output_units_for_display["LENGTH"]})')
        ax2.set_ylabel(f'Transverse direction ({units._output_units_for_display["LENGTH"]})')
        ax2.set_title('Side View')
        
        # Calculate the maximum extent in each dimension
        max_x = (self.geometry.N_l - 1) * self.geometry.S_l + self.geometry.D
        max_y = (self.geometry.N_t - 1) * self.geometry.S_t + self.geometry.D
        max_z = self.geometry.L
        
        # Set the plot limits with padding
        padding = 0.1  # 10% padding
        ax1.set_xlim(-self.geometry.D/2 - self.geometry.D * padding, 
                     max_x + self.geometry.D * padding)
        ax1.set_ylim(-self.geometry.D * padding, max_y + self.geometry.D * padding)
        
        ax2.set_xlim(-self.geometry.L * padding, max_z + self.geometry.L * padding)
        ax2.set_ylim(-self.geometry.D * padding, max_y + self.geometry.D * padding)
        
        # Set aspect ratios
        ax1.set_aspect('equal')
        ax2.set_aspect('auto')
        
        # Add grid
        ax1.grid(True, linestyle='--', alpha=0.3)
        ax2.grid(True, linestyle='--', alpha=0.3)
        
    def _add_info_panel(self, ax3):
        """Add information panel with geometry details."""
        ax3.axis('off')
        
        # Base info for all geometries
        info_text = (
            f'N_t (Transverse) = {self.geometry.N_t}\n'
            f'N_l (Longitudinal) = {self.geometry.N_l}\n'
            f'Surface Area = {self.geometry.surface:.2f} {units._output_units_for_display["AREA"]}\n'
            f'Flow Area = {self.geometry.area_flow:.2f} {units._output_units_for_display["AREA"]}'
        )
        
        # Add fin-specific info if available
        if hasattr(self.geometry, 'surface_fin'):
            info_text = (
                f'N_t (Transverse) = {self.geometry.N_t}\n'
                f'N_l (Longitudinal) = {self.geometry.N_l}\n'
                f'N_fin (Fins per tube) = {self.geometry.N_fin:.1f}\n'
                f'Bare Tube Area = {self.geometry.surface_outer_exposed:.2f} {units._output_units_for_display["AREA"]}\n'
                f'Fin Area = {self.geometry.surface_fin:.2f} {units._output_units_for_display["AREA"]}\n'
                f'Total Surface Area = {self.geometry.surface:.2f} {units._output_units_for_display["AREA"]}\n'
                f'Flow Area = {self.geometry.area_flow_min:.2f} {units._output_units_for_display["AREA"]}'
            )
            
        ax3.text(0.1, 0.5, info_text,
                 transform=ax3.transAxes,
                 verticalalignment='center',
                 bbox=dict(facecolor='white', edgecolor='gray', alpha=0.8))
        
    def plot(self, figsize=(12, 6), show=True):
        """
        Plot the geometry in 2D views.
        
        Parameters
        ----------
        figsize : tuple, optional
            Figure size in inches (width, height), by default (12, 6)
        show : bool, optional
            Whether to display the plot immediately, by default True
            
        Returns
        -------
        matplotlib.figure.Figure
            The generated figure
        """
        # Create figure and axes
        fig, ax1, ax2, ax3 = self._setup_plot(figsize)
        
        # Calculate tube positions
        x_positions = np.arange(self.geometry.N_l) * self.geometry.S_l
        y_positions = np.arange(self.geometry.N_t) * self.geometry.S_t
        
        # Plot tubes and spacing
        self._plot_tubes(ax1, ax2, x_positions, y_positions)
        self._plot_spacing(ax1, x_positions, y_positions)
        
        # Add fins if available
        self._plot_fins(ax1, ax2, x_positions, y_positions)
        
        # Setup axes and add info panel
        self._setup_axes(ax1, ax2)
        self._add_info_panel(ax3)
        
        plt.tight_layout()
        
        if show:
            plt.show()
            
        return fig
        
