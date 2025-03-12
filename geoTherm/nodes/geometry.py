import numpy as np
from ..units import inputParser, addQuantityProperty


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
        return f"Dh: {self.D:.4f}, A: {self.area:.4f}, L: {self.L:.4f}"


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
              'roughness': 'LENGTH'}

    @inputParser
    def __init__(self, D: 'LENGTH',             # noqa
                 L: 'LENGTH',                   # noqa
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
        self._L = L
        self._t = t
        self._roughness = roughness

    @property
    def _area(self):
        """Cross-sectional area of the inner pipe (m²)."""
        return np.pi * (self._Di)**2 / 4

    @property
    def _perimeter(self):
        """Wetted perimeter (circumference) of the inner pipe (m)."""
        return np.pi * self._Di

    @property
    def _D(self):
        """Hydraulic diameter for a cylindrical pipe (same as inner diameter)."""
        return self._Di


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
