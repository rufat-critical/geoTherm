from abc import ABC, abstractmethod
from geoTherm.common import addQuantityProperty
from ..geometry import Geometry


@addQuantityProperty
class InternalGeometry(ABC, Geometry):
    """
    Abstract base class for internal flow geometries.

    Subclasses must implement methods to compute:
    - Cross-sectional flow area
    - Wetted surface area
    - Wetted perimeter
    - State dictionary for reporting/debugging
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
        """Return internal wetted surface area [m²]."""
        pass

    @property
    @abstractmethod
    def _perimeter(self) -> float:
        """Return wetted perimeter [m]."""
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
