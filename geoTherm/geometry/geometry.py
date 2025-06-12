
from ..logger import logger
from ..units import inputParser, addQuantityProperty, units
from typing import final
import numpy as np

GEOMETRY_CLASSES = {}

def register_geometry(cls):
    """Register a geometry class with the GeometryGroup class."""
    GEOMETRY_CLASSES[cls.__name__] = cls
    return cls


@addQuantityProperty
class GeometryGroup:
    """A container class for managing a collection of geometric components.

    This class provides functionality to group multiple geometry objects together,
    allowing for collective operations and property calculations across all geometries
    in the group.

    Attributes:
        geometries (list): A list of Geometry objects contained in the group.

    Methods:
        from_dict(config_dict): Creates a GeometryGroup from a dictionary configuration.
        to_dict(): Converts the GeometryGroup to a dictionary representation.
        __iadd__(geometry): Implements += operator for adding geometries to the group.
        __add__(other): Implements + operator for combining geometry groups.

    Properties:
        _L: Total length of all geometries.
        _area_avg: Average cross-sectional area of all geometries.
        _surface_inner: Total inner surface area of all geometries.
        _surface_outer: Total outer surface area of all geometries.
        _Do: Average outer diameter of all geometries.
        _Di: Average inner diameter of all geometries.
        _Ain: Average inner surface area of all geometries.
        _Aout: Average outer surface area of all geometries.

    Example:
        >>> group = GeometryGroup()
        >>> group += Cylinder(D=0.1, L=1.0)  # Add a single geometry
        >>> group += another_geometry_group   # Combine with another group
    """

    _units = {'L': 'LENGTH', 'area_avg': 'AREA', 'surface_inner': 'AREA',
              'surface_outer': 'AREA', 'Do': 'LENGTH', 'Di': 'LENGTH',
              'Ain': 'AREA', 'Aout': 'AREA', 'dz': 'LENGTH'}

    def __init__(self, geometries=None):
        """Initialize a GeometryGroup.

        Args:
            geometries (list or dict, optional): A list of Geometry instances or
                a dictionary of named geometries.
        """
        
        # Initialize empty list of geometries
        self.geometries = []

        # If no geometries are provided, return
        if geometries is None:
            return

        # If geometries is a list, add each geometry to the group
        if isinstance(geometries, list):
            for geometry in geometries:
                if isinstance(geometry, Geometry):
                    self.geometries.append(geometry)
                elif isinstance(geometry, dict):
                    self.geometries.append(Geometry.from_dict(geometry))
                else:
                    logger.critical("All geometries in list must be instances "
                                    "of Geometry or a dictionary")
        # If geometries is a single Geometry instance, add it to the group
        elif isinstance(geometries, Geometry):
            self.geometries.append(geometries)
        else:
            logger.critical("GeometryGroup must be initialized with a list or "
                            "a Geometry instance")


    @classmethod
    def from_dict(cls, config_dict):
        """Create a GeometryGroup instance from a dictionary configuration."""
        #group = cls()


        components = config_dict['components']

        geometries = [Geometry.from_dict(component) for component in components]

        return cls(geometries=geometries)

    @property
    def _state(self):
        """Return a list of the state of the geometries in the group."""
        geometries = []
        for geometry in self.geometries:
            geometries.append(geometry._state)
        return geometries

    def __iadd__(self, geometry):
        if isinstance(geometry, Geometry):
            self.geometries.append(geometry)
        elif isinstance(geometry, GeometryGroup):
            self.geometries.extend(geometry.geometries)
        else:
            logger.critical("All geometries must be instances of Geometry")

        return self

    def __add__(self, other):

        if isinstance(other, GeometryGroup):
            new_group = GeometryGroup(self.geometries.copy())
            new_group += other
            return new_group
        elif isinstance(other, Geometry):
            new_group = GeometryGroup(self.geometries.copy())
            new_group += other
            return new_group
        else:
            logger.critical("Can only add GeometryGroup or Geometry instances")

    def __str__(self):
        if not self.geometries:
            return ""
        return " => ".join(str(geometry) for geometry in self.geometries)

    @property
    def _L(self):
        return np.sum([geometry._L for geometry in self.geometries])

    @property
    def _area_avg(self):
        return np.average([geometry._area for geometry in self.geometries])

    @property
    def _surface_inner(self):
        return np.sum([geometry._surface_inner for geometry in self.geometries])

    @property
    def _surface_outer(self):
        return np.sum([geometry._surface_outer for geometry in self.geometries])

    @property
    def _Do(self):
        return np.average([geometry._Do for geometry in self.geometries])

    @property
    def _Di(self):
        return np.average([geometry._Di for geometry in self.geometries])

    @property
    def _Ain(self):
        return np.average([geometry._Ain for geometry in self.geometries])

    @property
    def _Aout(self):
        return np.average([geometry._Aout for geometry in self.geometries])

    @property
    def _dz(self):
        return np.sum([geometry._dz for geometry in self.geometries])

class Geometry:
    """Base class for different geometries with a formatted string output."""

    def __str__(self):
        """
        Returns a formatted string of key properties.

        Output Format:
        - Dh: Hydraulic Diameter (m)
        - A: Cross-sectional Area (m²)
        - L: Length of the geometry (m)
        """

        class_type = type(self).__name__

        return (f"{class_type}:\n"
                f"Inner: {self.inner.__str__()}\n"
                f"Outer: {self.outer.__str__()}")


    @classmethod
    def from_dict(cls, state_dict):
        if state_dict['type'] not in GEOMETRY_CLASSES:
            logger.critical(f"Unknown geometry type: {state_dict['type']}")

        return GEOMETRY_CLASSES[state_dict['type']](**state_dict['parameters'])

    @property
    @final
    def state(self):
        """Convert the geometry to a dictionary representation."""
        return {'type': type(self).__name__,
                'parameters': self._state}



@addQuantityProperty
class TubeBank(Geometry):
    """
    A bundle of tubes in a rectangular array.
    """

    _units = {'D': 'LENGTH', 'L': 'LENGTH', 't': 'LENGTH',
              'Di': 'LENGTH', 'Do': 'LENGTH', 'th': 'LENGTH',
              'surface_outer': 'AREA', 'surface_inner': 'AREA',
              'area_t': 'AREA', 'Dh': 'LENGTH', 'S_t': 'LENGTH',
              'S_l': 'LENGTH'}

    def __init__(self, Di: 'LENGTH', L: 'LENGTH', S_t: 'LENGTH', S_l: 'LENGTH', N_L, N_T, th: 'LENGTH', N_L_passes=1):
        """
        Initialize a tube bundle geometry.

        Args:
            Di: Inner diameter of the tubes (LENGTH)
            L: Length of the tubes (LENGTH)
            S_t: Transverse spacing between tubes (LENGTH)
            S_l: Longitudinal spacing between tubes (LENGTH)
            N_L: Number of tubes in the longitudinal direction
            N_T: Number of tubes in the transverse direction
            th: Wall thickness of the tubes (LENGTH)
            N_passes: Number of passes
        """
        self._Di = Di
        self._L = L
        self._S_t = S_t
        self._S_l = S_l
        self.N_L = N_L
        self.N_T = N_T
        self._th = th
        self._Do = Di + 2 * th
        self.N_L_passes = N_L_passes
        # UPDATE TO BE COMPOSITE OBJECT, USE INTERNAL FLOW OBJECT
        self._roughness = 1e-5

    @property
    def N_tubes(self):  
        return self.N_L * self.N_T

    @property
    def _area(self):
        return self.N_tubes * self._area_tube/self.N_L_passes
    
    @property
    def _area_tube(self):
        return np.pi * self._Di**2/4
    
    @property
    def _surface_inner(self):
        return self.N_tubes * np.pi * self._Di * self._L/self.N_L_passes

    @property
    def _surface_outer(self):
        return self._surface_outer_bare

    @property
    def _surface_outer_bare(self):
        return self.N_tubes * np.pi * self._Do * self._L

    @property
    def _Dh(self):
        return np.sqrt(self._area*4/np.pi)

    @property
    def _area_flow(self):
        """ Flow area across the tube bank """
        return (self._S_t - self._Do) * self.N_T * self._L

    def plot(self, figsize=(12, 6), show=True):
        """
        Plot the tube bank geometry in 2D views.
        
        Parameters:
        -----------
        figsize : tuple, optional
            Figure size in inches (width, height). Default is (12, 6).
        show : bool, optional
            Whether to display the plot immediately. Default is True.
            
        Returns:
        --------
        matplotlib.figure.Figure
            The generated figure.
        """
        import matplotlib.pyplot as plt
        import numpy as np
        
        # Create figure with three subplots (two for views, one for text)
        fig = plt.figure(figsize=figsize)
        gs = fig.add_gridspec(1, 3, width_ratios=[1, 1, 0.3])
        ax1 = fig.add_subplot(gs[0])
        ax2 = fig.add_subplot(gs[1])
        ax3 = fig.add_subplot(gs[2])
        
        # Calculate tube positions
        x_positions = np.arange(self.N_L) * self.S_l  # Longitudinal direction
        y_positions = np.arange(self.N_T) * self.S_t  # Transverse direction
        
        # Plot 1: Top view (longitudinal vs transverse)
        for i in range(self.N_L):
            for j in range(self.N_T):
                # Draw tube as a circle
                tube = plt.Circle((x_positions[i], y_positions[j]), 
                                self.Do/2,
                                facecolor='gray', alpha=0.5, edgecolor='black')
                ax1.add_patch(tube)

        # Add S_t line in top view
        if self.N_T > 1:
            # Draw a line between centers of the first two tubes in the first row
            ax1.plot([x_positions[0], x_positions[0]], 
                    [y_positions[0], y_positions[1]],
                    'r--', linewidth=2)
            # Add label
            ax1.text(x_positions[0], (y_positions[0] + y_positions[1])/2,
                    f'S_t = {self.S_t:.3f}m',
                    ha='right', va='center', color='red')

        # Add S_l line in top view
        if self.N_L > 1:
            # Draw a line between centers of the first two tubes in the first column
            ax1.plot([x_positions[0], x_positions[1]], 
                    [y_positions[0], y_positions[0]],
                    'b--', linewidth=2)
            # Add label
            ax1.text((x_positions[0] + x_positions[1])/2, y_positions[0],
                    f'S_l = {self.S_l:.3f}m',
                    ha='center', va='bottom', color='blue')

        # Plot 2: Side view (Length vs Transverse)
        for i in range(self.N_L):
            for j in range(self.N_T):
                # Draw tube as a rectangle
                tube = plt.Rectangle((0, y_positions[j] - self.Do/2), 
                                   self.L, self.Do,
                                   facecolor='gray', alpha=0.5, edgecolor='black')
                ax2.add_patch(tube)
        
        # Set labels and titles
        ax1.set_xlabel(f'Longitudinal direction ({units._output_units_for_display["LENGTH"]})')
        ax1.set_ylabel(f'Transverse direction ({units._output_units_for_display["LENGTH"]})')
        ax1.set_title('Top View')
        
        ax2.set_xlabel(f'Length ({units._output_units_for_display["LENGTH"]})')
        ax2.set_ylabel(f'Transverse direction ({units._output_units_for_display["LENGTH"]})')
        ax2.set_title('Side View')
        
        # Calculate the maximum extent in each dimension
        max_x = (self.N_L - 1) * self.S_l + self.Do
        max_y = (self.N_T - 1) * self.S_t + self.Do
        max_z = self.L
        
        # Set the plot limits with padding
        padding = 0.1  # 10% padding
        # Adjust left bound to ensure circles are fully visible
        ax1.set_xlim(-self.Do/2 - self.Do * padding, max_x + self.Do * padding)
        ax1.set_ylim(-self.Do * padding, max_y + self.Do * padding)
        
        ax2.set_xlim(-self.L * padding, max_z + self.L * padding)
        ax2.set_ylim(-self.Do * padding, max_y + self.Do * padding)
        
        # Set equal aspect ratio only for top view
        ax1.set_aspect('equal')
        # Remove equal aspect ratio for side view to allow different scaling
        ax2.set_aspect('auto')
        
        # Add grid
        ax1.grid(True, linestyle='--', alpha=0.3)
        ax2.grid(True, linestyle='--', alpha=0.3)
        
        # Add text information in the third subplot
        ax3.axis('off')  # Turn off axis for text subplot
        info_text = (
            f'N_T (Transverse) = {self.N_T}\n'
            f'N_L (Longitudinal) = {self.N_L}\n'
            f'Surface Area = {self.surface_outer:.2f} {units._output_units_for_display["AREA"]}\n'
            f'Flow Area = {self.area_t:.2f} {units._output_units_for_display["AREA"]}'
        )
        ax3.text(0.1, 0.5, info_text,
                 transform=ax3.transAxes,
                 verticalalignment='center',
                 bbox=dict(facecolor='white', edgecolor='gray', alpha=0.8))
        
        plt.tight_layout()
        
        if show:
            plt.show()
        
        return fig


@addQuantityProperty
class FinnedTubeBank2(TubeBank):
    """
    A finned tube geometry.
    """

    _units = TubeBank._units | {'D_f': 'LENGTH', 'L_l': 'LENGTH', 'L_t': 'LENGTH', 'th_f': 'LENGTH', 'N_f': '1/LENGTH'}

    def __init__(self, D: 'LENGTH', L: 'LENGTH', S_t: 'LENGTH', S_l: 'LENGTH', N_L, N_T, th: 'LENGTH', N_f:'1/LENGTH', D_f: 'LENGTH', L_t: 'LENGTH', th_f: 'LENGTH', N_L_passes=1):
        """Initialize a finned tube bank geometry.

        A finned tube bank consists of multiple tubes arranged in a rectangular array,
        with fins attached to each tube to increase the heat transfer surface area.

        Parameters
        ----------
        D : LENGTH
            Inner diameter of the tubes.
        L : LENGTH
            Length of the tubes.
        S_t : LENGTH
            Transverse spacing between tubes (center-to-center distance).
        S_l : LENGTH
            Longitudinal spacing between tubes (center-to-center distance).
        N_L : int
            Number of tubes in the longitudinal direction.
        N_T : int
            Number of tubes in the transverse direction.
        th : LENGTH
            Wall thickness of the tubes.
        N_f : 1/LENGTH
            Number of fins per unit length on each tube.
        D_f : LENGTH
            Outer diameter of the fins.
        L_t : LENGTH
            Length of the finned section.
        th_f : LENGTH
            Thickness of the fins.
        N_L_passes : int, optional
            Number of longitudinal passes in the tube bank, by default 1.

        Notes
        -----
        - The total number of tubes is calculated as N_L * N_T
        - The fin diameter (D_f) must be greater than the tube outer diameter
        - The fin thickness (th_f) affects the flow area and heat transfer characteristics
        - The number of fins per unit length (N_f) determines the fin density
        """
        super().__init__(D, L, S_t, S_l, N_L, N_T, th, N_L_passes)
        self._N_f = N_f
        self._D_f = D_f
        self._D_fin = D_f
        self._L_t = L_t
        self._th_f = th_f
        self._th_fin = th_f

    @property
    def _h_fin(self):
        return (self._D_f - self._Do)/2

    @property
    def _surface_fin(self):
        return self._surface_fin_each*self._N_f*self._L*self.N_tubes

    @property
    def _surface_fin_each(self):
        return (2*np.pi*(self._D_f**2-self._Do**2)/4
                 + np.pi*self._D_f*self._th_f)

    @property
    def _surface_fin_per_tube(self):
        return self._surface_fin_each*self._N_f*self._L

    @property
    def _surface_fin_per_tube_longitudinal(self):
        return self._surface_fin_per_tube*self.N_L

    @property
    def _surface_fin_per_tube_transverse(self):
        return self._surface_fin_per_tube*self.N_T
    
    @property
    def _L_tube_exposed(self):
        return self._L*(1-self._N_f*self._th_f)

    @property
    def _dL_tube_exposed(self):
        return 1/self._N_f - self._th_f

    @property
    def _surface_outer_exposed(self):
        return self.N_tubes*np.pi*self._Do*self._L_tube_exposed

    @property
    def _surface_outer(self):
        return self._surface_fin + self._surface_outer_exposed

    @property
    def area_increase(self):
        return self._surface_outer/self._surface_outer_bare

    @property
    def _area_flow_min(self):
        return self._area_flow_transverse

    @property
    def _area_flow_transverse(self):
        return (self._S_t - self._Do - 2*self._th_f*self._h_fin*self._N_f)*self._L*self.N_T

    @property
    def _area_face(self):
        return self._S_t * self._L * (self.N_T + 1)
    
    @property
    def flow_area_contraction_ratio(self):
        return self._area_flow_min/self._area_face
    
    def plot(self, figsize=(12, 6), show=True):
        """
        Plot the finned tube bank geometry in 2D views.
        
        Parameters:
        -----------
        figsize : tuple, optional
            Figure size in inches (width, height). Default is (12, 6).
        show : bool, optional
            Whether to display the plot immediately. Default is True.
            
        Returns:
        --------
        matplotlib.figure.Figure
            The generated figure.
        """
        import matplotlib.pyplot as plt
        import numpy as np
        
        # Create figure with three subplots (two for views, one for text)
        fig = plt.figure(figsize=figsize)
        gs = fig.add_gridspec(1, 3, width_ratios=[1, 1, 0.3])
        ax1 = fig.add_subplot(gs[0])
        ax2 = fig.add_subplot(gs[1])
        ax3 = fig.add_subplot(gs[2])
        
        # Calculate tube positions
        x_positions = np.arange(self.N_L) * self.S_l  # Longitudinal direction
        y_positions = np.arange(self.N_T) * self.S_t  # Transverse direction
        
        # Plot 1: Top view (longitudinal vs transverse)
        for i in range(self.N_L):
            for j in range(self.N_T):
                # Draw fin as a circle
                fin = plt.Circle((x_positions[i], y_positions[j]), 
                               self.D_f/2,
                               facecolor='none', edgecolor='gray', alpha=0.3)
                ax1.add_patch(fin)
                
                # Draw tube as a circle
                tube = plt.Circle((x_positions[i], y_positions[j]), 
                                self.Do/2,
                                facecolor='gray', alpha=0.5, edgecolor='black')
                ax1.add_patch(tube)
        
        # Add S_t line in top view
        if self.N_T > 1:
            # Draw a line between centers of the first two tubes in the first row
            ax1.plot([x_positions[0], x_positions[0]], 
                    [y_positions[0], y_positions[1]],
                    'r--', linewidth=2)
            # Add label
            ax1.text(x_positions[0], (y_positions[0] + y_positions[1])/2,
                    f'S_t = {self.S_t:.3f}m',
                    ha='right', va='center', color='red')
        
        # Add S_l line in top view
        if self.N_L > 1:
            # Draw a line between centers of the first two tubes in the first column
            ax1.plot([x_positions[0], x_positions[1]], 
                    [y_positions[0], y_positions[0]],
                    'b--', linewidth=2)
            # Add label
            ax1.text((x_positions[0] + x_positions[1])/2, y_positions[0],
                    f'S_l = {self.S_l:.3f}m',
                    ha='center', va='bottom', color='blue')
        
        # Plot 2: Side view (Length vs Transverse)
        for i in range(self.N_L):
            for j in range(self.N_T):
                y_center = y_positions[j]
                # Draw tube as a rectangle
                tube = plt.Rectangle((0, y_center - self.Do/2), 
                                     self.L, self.Do,
                                     facecolor='gray', alpha=0.5, edgecolor='black')
                ax2.add_patch(tube)
                
                # Draw fins as thin rectangles along the tube length
                if self.N_f > 0:
                    fin_spacing = self.L / self.N_f
                    for k in range(self.N_f):
                        fin_x = k * fin_spacing
                        fin = plt.Rectangle(
                            (fin_x, y_center - self.D_f/2),
                            self.th_f, self.D_f,
                            facecolor='none', edgecolor='gray', alpha=0.7, linewidth=1
                        )
                        ax2.add_patch(fin)
        
        # Set labels and titles
        ax1.set_xlabel(f'Longitudinal direction ({units._output_units_for_display["LENGTH"]})')
        ax1.set_ylabel(f'Transverse direction ({units._output_units_for_display["LENGTH"]})')
        ax1.set_title('Top View')
        
        ax2.set_xlabel(f'Length ({units._output_units_for_display["LENGTH"]})')
        ax2.set_ylabel(f'Transverse direction ({units._output_units_for_display["LENGTH"]})')
        ax2.set_title('Side View')
        
        # Calculate the maximum extent in each dimension
        max_x = (self.N_L - 1) * self.S_l + self.D_f
        max_y = (self.N_T - 1) * self.S_t + self.D_f
        max_z = self.L
        
        # Set the plot limits with padding
        padding = 0.1  # 10% padding
        # Adjust left bound to ensure circles are fully visible
        ax1.set_xlim(-self.D_f/2 - self.D_f * padding, max_x + self.D_f * padding)
        ax1.set_ylim(-self.D_f * padding, max_y + self.D_f * padding)
        
        ax2.set_xlim(-self.L * padding, max_z + self.L * padding)
        ax2.set_ylim(-self.D_f * padding, max_y + self.D_f * padding)
        
        # Set equal aspect ratio only for top view
        ax1.set_aspect('equal')
        # Remove equal aspect ratio for side view to allow different scaling
        ax2.set_aspect('auto')
        
        # Add grid
        ax1.grid(True, linestyle='--', alpha=0.3)
        ax2.grid(True, linestyle='--', alpha=0.3)
        
        # Add text information in the third subplot
        ax3.axis('off')  # Turn off axis for text subplot
        info_text = (
            f'N_T (Transverse) = {self.N_T}\n'
            f'N_L (Longitudinal) = {self.N_L}\n'
            f'N_f (Fins per tube) = {self.N_f}\n'
            f'Bare Tube Area = {self._surface_outer_exposed:.2f} {units._output_units_for_display["AREA"]}\n'
            f'Fin Area = {self._surface_fin:.2f} {units._output_units_for_display["AREA"]}\n'
            f'Total Surface Area = {self.surface_outer:.2f} {units._output_units_for_display["AREA"]}\n'
            f'Flow Area = {self._area_flow_min:.2f} {units._output_units_for_display["AREA"]}'
        )
        ax3.text(0.1, 0.5, info_text,
                 transform=ax3.transAxes,
                 verticalalignment='center',
                 bbox=dict(facecolor='white', edgecolor='gray', alpha=0.8))
        
        plt.tight_layout()
        
        if show:
            plt.show()
        
        return fig


@addQuantityProperty
class Cylinder(Geometry):
    """
    Cylindrical pipe geometry.

    Attributes:
    ----------
    - D: Outer diameter of the pipe (LENGTH)
    - L: Length of the pipe (LENGTH)
    - t: Wall thickness (LENGTH)
    - Di: Inner diameter (LENGTH)
    - Do: Outer diameter (LENGTH)
    - area: Cross-sectional area of the inner pipe (AREA)
    - perimeter: Wetted perimeter (LENGTH)
    - roughness: Surface roughness (LENGTH)
    """

    _units = {'D': 'LENGTH', 'L': 'LENGTH', 't': 'LENGTH',
              'Di': 'LENGTH', 'Do': 'LENGTH',
              'area': 'AREA', 'perimeter': 'LENGTH',
              'surface': 'AREA',
              'roughness': 'LENGTH'}
    _type = 'Cylinder'
    @inputParser
    def __init__(self, D: 'LENGTH',             # noqa
                 L: 'LENGTH',                   # noqa
                 dz: 'LENGTH' = 0,              # noqa
                 t: 'LENGTH' = 0,               # noqa
                 roughness: 'LENGTH' = 1e-4):   # noqa
        """
        Initialize a cylindrical geometry.

        Parameters:
        ----------
        D : float
            Outer diameter of the pipe (m).
        L : float
            Length of the pipe (m).
        dz : float, optional
            Height change of the pipe (m). Default is 0.
        t : float, optional
            Wall thickness of the pipe (m). Default is 0 (thin-wall assumption).
        roughness : float, optional
            Surface roughness of the pipe (m). Default is 1e-4.

        Notes:
        ------
        - The inner diameter (Di) is computed as `Di = D - 2*t`.
        - The cross-sectional area is computed based on the inner diameter.
        """
        self._Do = D
        self._Di = D - 2 * t
        self._D = D
        self._L = L
        self._t = t
        self._roughness = roughness
        self._dz = dz

    @property
    def _area(self):
        """Cross-sectional area of the inner pipe (m²)."""
        return np.pi * (self._Di)**2 / 4

    @property
    def _perimeter(self):
        """Wetted perimeter (circumference) of the inner pipe (m)."""
        return np.pi * self._Di

    @property
    def _surface_inner(self):
        """Inner pipe surface area (m²)."""
        return 2 * np.pi * self._Di * self._L

    @property
    def _surface_outer(self):
        """Outer pipe surface area (m²)."""
        return 2 * np.pi * self._Do * self._L

    @property
    def _Ain(self):
        """Inner pipe surface area (m²)."""
        return 2 * np.pi * self._Di * self._L

    @property
    def _Aout(self):
        """Outer pipe surface area (m²)."""
        return 2 * np.pi * self._Do * self._L
        

@addQuantityProperty
class Rectangular(Geometry):
    """
    Rectangular duct geometry.

    Attributes:
    ----------
    - width: Width of the duct (LENGTH)
    - height: Height of the duct (LENGTH)
    - L: Length of the duct (LENGTH)
    - area: Cross-sectional area (AREA)
    - perimeter: Wetted perimeter (LENGTH)
    - D: Hydraulic diameter (LENGTH)
    - roughness: Surface roughness (LENGTH)
    """

    _units = {'length': 'LENGTH', 'width': 'LENGTH', 'L': 'LENGTH',
              'area': 'AREA', 'perimeter': 'LENGTH', 'D': 'LENGTH',
              'roughness': 'LENGTH'}

    @inputParser
    def __init__(self, width: 'LENGTH',         # noqa
                 height: 'LENGTH',              # noqa
                 L: 'LENGTH',                   # noqa
                 roughness: 'LENGTH' = 1e-4):   # noqa
        """
        Initialize a rectangular duct geometry.

        Parameters:
        ----------
        width : float
            Width of the duct (m).
        height : float
            Height of the duct (m).
        L : float
            Length of the duct (m).
        roughness : float, optional
            Surface roughness of the duct (m). Default is 1e-4.

        Notes:
        ------
        - The cross-sectional area is calculated as `area = width * height`.
        - The hydraulic diameter is calculated as `D = 4 * area / perimeter`.
        """
        self._width = width
        self._height = height
        self._L = L
        self._roughness = roughness

    @property
    def _area(self):
        """Cross-sectional area of the rectangular duct (m²)."""
        return self._width * self._height

    @property
    def _perimeter(self):
        """Wetted perimeter of the rectangular duct (m)."""
        return 2 * (self._width + self._height)

    @property
    def _D(self):
        """Hydraulic diameter for a rectangular duct (m)."""
        return (4 * self._area) / self._perimeter

@addQuantityProperty
class CylinderBend(Cylinder):
    """
    Cylinder Bend geometry.
    """

    _units = {'R': 'LENGTH', 'D': 'LENGTH', 't': 'LENGTH', 'roughness': 'LENGTH'}

    _type = 'CylinderBend'
    @inputParser
    def __init__(self, D: 'LENGTH', angle, R: 'LENGTH', dz: 'LENGTH'=0, t: 'LENGTH'=0, roughness: 'LENGTH'=1e-4):
        """
        Initialize a bend geometry.
        """
        self._Do = D
        self._Di = D - 2 * t
        self._D = D
        self.angle = angle
        # Bend radius
        self._R = R
        self._t = t
        self._roughness = roughness
        self._dz = dz

    @property
    def theta(self):
        return self.angle * np.pi / 180

    @property
    def RD(self):
        return self._R / self._D

    @property
    def _L(self):
        return self._R * self.theta

    @property
    def _perimeter(self):
        from pdb import set_trace
        set_trace()
    
    @property
    def _area(self):
        return self._R * self.theta
    
    def __str__(self):
        """
        Returns a formatted string of key properties.

        Output Format:
        - Dh: Hydraulic Diameter (m)
        - A: Cross-sectional Area (m²)
        - L: Length of the geometry (m)
        """
        if self._dz != 0:
            return f"{self._type}(Dh: {self.D:.4f}, L: {self.L:.4f}, dz: {self._dz:.4f}, angle: {self.angle:.2f})"
        else:
            return f"{self._type}(Dh: {self.D:.4f}, L: {self.L:.4f}, angle: {self.angle:.2f})"


class GeometryProperties:
    """Class to expose geometry properties in a parent class. This should
        be inhereted.

    Provides access to:
    - _area: Cross-sectional area of the geometry
    - _perimeter: Wetted perimeter of the geometry
    - _D: Hydraulic diameter
    - _L: Length of the geometry
    - _roughness: Surface roughness
    """

    _units = {'area': 'AREA', 'perimeter': 'LENGTH', 'D': 'LENGTH',
              'L': 'LENGTH', 'roughness': 'LENGTH'}

    @property
    def _area(self):
        return self.geometry._area

    @property
    def _perimeter(self):
        return self.geometry._perimeter

    @property
    def _D(self):
        """Hydraulic diameter (universal for all geometries)."""
        return self.geometry._D

    @property
    def _L(self):
        """Returns the length of the geometry (m)."""
        return self.geometry._L

    @property
    def _roughness(self):
        """Surface roughness of the geometry."""
        return self.geometry._roughness
