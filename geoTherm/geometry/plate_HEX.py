import numpy as np
from .geometry import Geometry, register_geometry
from geoTherm.common import addQuantityProperty, inputParser


@register_geometry
@addQuantityProperty
class PlateHEX(Geometry):
    """Plate Heat Exchanger geometry class.

    This class represents the geometry of a plate heat exchanger, including
    dimensions and properties of the plates, channels, and ports.

    Attributes:
        V (float): Volume of the heat exchanger
        As (float): Surface area of the heat exchanger
        Dp (float): Port diameter
        Lv (float): Vertical length of the plate
        Lh (float): Horizontal length of the plate
        N_plates (int): Number of plates
    """

    _units = {
        'V': 'VOLUME',
        'surface_area': 'AREA',
        'D_port': 'LENGTH',
        'Lv': 'LENGTH',
        'Lh': 'LENGTH',
    }


    @inputParser
    def __init__(self, V: 'VOLUME', surface_area: 'AREA', D_port: 'LENGTH', Lv: 'LENGTH', Lh: 'LENGTH', chevron_angle, N_plates):
        """Initialize a PlateHEX instance.

        Args:
            V (float): Volume of the heat exchanger
            As (float): Surface area of the heat exchanger
            Dp (float): Port diameter
            Lv (float): Vertical length of the plate
            Lh (float): Horizontal length of the plate
            chevron_angle (float): Chevron angle of the plate
            N_plates (int): Number of plates
        """
        self._V = V
        self._surface = surface_area
        self._D_port = D_port
        self._Lv = Lv
        self._Lh = Lh
        self.N_plates = N_plates
        self.chevron_angle = chevron_angle

    @property
    def _th(self) -> float:
        """Channel thickness.

        Returns:
            float: Channel thickness calculated as volume divided by surface area
        """
        return self._V/self._surface

    @property
    def _Dh(self) -> float:
        """Hydraulic diameter.

        Returns:
            float: Hydraulic diameter of the channel
        """
        return 2*self._th/self.phi

    @property
    def _Lp(self) -> float:
        """Total plate length.

        Returns:
            float: Sum of vertical and horizontal lengths
        """
        return self._Lv - self._D_port

    @property
    def _Lw(self) -> float:
        """Plate width.

        Returns:
            float: Plate width including port diameter
        """
        return self._Lh + self._D_port

    @property
    def phi(self) -> float:
        """Surface area density.

        Returns:
            float: Surface area density of the heat exchanger
        """
        return self._surface/(self._Lp*self._Lw)

    @property
    def _area_channel(self) -> float:
        """Channel cross-sectional area.

        Returns:
            float: Cross-sectional area of the channel
        """
        return self._Lw*self._th

    @property
    def _area_flow_total(self):
        """Total flow area.

        Returns:
            float: Total flow area of the heat exchanger
        """
        return self._area_channel*self.N_plates

    @property
    def _area_port(self) -> float:
        """Port cross-sectional area.

        Returns:
            float: Cross-sectional area of the port
        """
        return np.pi*self._D_port**2/4
