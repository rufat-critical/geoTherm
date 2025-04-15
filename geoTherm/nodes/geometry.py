import numpy as np
from ..logger import logger
from ..units import inputParser, addQuantityProperty

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

    _units = {'L': 'LENGTH', 'area_avg': 'AREA', 'surface_inner': 'AREA', 'surface_outer': 'AREA',
              'Do': 'LENGTH', 'Di': 'LENGTH', 'Ain': 'AREA', 'Aout': 'AREA'}

    def __init__(self, geometries=None):
        if geometries is None:
            self.geometries = []
        else:
            for geometry in geometries:
                if not isinstance(geometry, Geometry):
                    logger.critical("All geometries must be instances of Geometry")
            self.geometries = geometries

    @classmethod
    def from_dict(cls, config_dict):
        """Create a GeometryGroup instance from a dictionary configuration."""
        #group = cls()


        components = config_dict['components']

        geometries = [Geometry.from_dict(component) for component in components]

        return cls(geometries=geometries)


    def to_dict(self):
        """Convert the GeometryGroup to a dictionary representation."""
        config_dict = {}
        for i, geometry in enumerate(self.geometries):
            geom_dict = geometry.to_dict()
            key = list(geom_dict.keys())[0]  # Get the geometry type
            
            # If there are multiple geometries of the same type, append a number
            if key in config_dict:
                key = f"{key}_{i}"
            config_dict[key] = geom_dict[list(geom_dict.keys())[0]]
            
        return config_dict
    
    def _state_dict(self):
        from pdb import set_trace
        set_trace()

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
        if self._dz != 0:
            return f"{self._type}(Dh: {self.D:.4f}, L: {self.L:.4f}, dz: {self._dz:.4f})"
        else:
            return f"{self._type}(Dh: {self.D:.4f}, L: {self.L:.4f})"

    @classmethod
    def from_dict(cls, config_dict):
        """Create a geometry instance from a dictionary configuration."""
        geom_type = list(config_dict.keys())[0]  # Get the geometry type
        params = config_dict[geom_type]

        if geom_type == 'Cylinder':
            return Cylinder(**params)
        elif geom_type == 'CylinderBend':
            return CylinderBend(**params)
        elif geom_type == 'Rectangular':
            return Rectangular(**params)
        else:
            logger.critical(f"Unknown geometry type: {geom_type}")

    def to_dict(self):
        """Convert the geometry to a dictionary representation."""
        if isinstance(self, Cylinder):
            return {'Cylinder': {
                'D': self._Do,
                'L': self._L,
                't': self._t,
                'dz': self._dz,
                'roughness': self._roughness
            }}
        elif isinstance(self, CylinderBend):
            return {'CylinderBend': {
                'D': self._Do,
                'angle': self.angle,
                'R': self._R,
                't': self._t,
                'dz': self._dz,
                'roughness': self._roughness
            }}
        elif isinstance(self, Rectangular):
            return {'Rectangular': {
                'width': self._width,
                'height': self._height,
                'L': self._L,
                'roughness': self._roughness
            }}


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
