from abc import ABC, abstractmethod
from geoTherm.common import addQuantityProperty
from ..utils import GeometryPlotter

@addQuantityProperty
class ExternalGeometry(ABC):
    """
    Abstract base class for external flow geometries.

    Subclasses must implement methods to compute:
    - External heat transfer surface area
    - Characteristic wetted perimeter or hydraulic diameter
    - State dictionary for reporting or reconstruction
    """

    _units = {"area": "AREA", "surface": "AREA", "perimeter": "LENGTH",
              "volume": "VOLUME"}

    @property
    @abstractmethod
    def _area(self) -> float:
        """Return cross-sectional flow area [m²]."""
        pass

    @property
    @abstractmethod
    def _surface(self) -> float:
        """Return external wetted surface area [m²]."""
        pass

    @property
    def _volume(self) -> float:
        """Return volume of the geometry [m³]."""
        return self._area*self._L

    @property
    @abstractmethod
    def _state(self) -> dict:
        """Return a dictionary describing geometry state (for rebuilding)."""
        pass

    def plot(self, figsize=(12, 6), show=True):
        """
        Plot the finned tube bank geometry in 2D views.
        
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
        
        return GeometryPlotter(self).plot(figsize=figsize, show=show)