
from .base_internal import InternalGeometry
from geoTherm.common import addQuantityProperty, inputParser
import numpy as np

@addQuantityProperty
class InternalCylinder(InternalGeometry):


    @inputParser
    def __init__(self, D:'LENGTH', L:'LENGTH', roughness:'LENGTH'=1e-4, dz:'LENGTH'=0, n_streams=1):

        self._D = D
        self._L = L
        self._roughness = roughness
        self._dz = dz
        # Used to calculate the number of streams
        self.n_streams = n_streams

    @property
    def _area(self):
        return np.pi * self._D**2 / 4*self.n_streams

    @property
    def _perimeter(self):
        return np.pi * self._D*self.n_streams

    @property
    def _surface(self):
        return np.pi * self._D * self._L*self.n_streams

    @property
    def _Dh(self):
        return self._D

    @property
    def _state(self):
        return {
            'D': (self._D, 'm'),
            'L': (self._L, 'm'),
            'roughness': (self._roughness, 'm')}

    def __str__(self):
        return f"Di: {self._D:.4f}"


@addQuantityProperty
class InternalCylinderBend(InternalCylinder):
    """
    Cylinder Bend geometry.
    """

    _units = {'R': 'LENGTH', 'D': 'LENGTH', 't': 'LENGTH', 'roughness': 'LENGTH', 'L': 'LENGTH'}

    @inputParser
    def __init__(self, D: 'LENGTH', angle, R: 'LENGTH', dz: 'LENGTH'=0, t: 'LENGTH'=0, roughness: 'LENGTH'=1e-4, n_streams =1):
        """
        Initialize a bend geometry.
        """
        self._D = D
        self.angle = angle
        # Bend radius
        self._R = R
        self._t = t
        self._roughness = roughness
        self._dz = dz
        self.n_streams = n_streams


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
        return f"CylinderBend(Dh: {self.D:.4f}, L: {self.L:.4f}, dz: {self._dz:.4f}, angle: {self.angle:.2f})"
