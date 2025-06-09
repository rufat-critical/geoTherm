from .base_external import ExternalGeometry
from geoTherm.common import addQuantityProperty, inputParser
import numpy as np

@addQuantityProperty
class ExternalCylinder(ExternalGeometry):

    @inputParser
    def __init__(self, D:'LENGTH', L:'LENGTH', roughness:'LENGTH'=1e-4):

        self._D = D
        self._L = L
        self._roughness = roughness

    @property
    def _area(self):
        return np.pi * self._D**2 / 4

    @property
    def _Dh(self):
        return self._D

    @property
    def _perimeter(self):
        return np.pi * self._D

    @property
    def _surface(self):
        return np.pi * self._D * self._L

    @property
    def _state(self):
        return {
            'D': (self._D, 'm'),
            'L': (self._L, 'm'),
            'roughness': (self._roughness, 'm')}

    def __str__(self):
        return f"Do: {self._D:.4f}"
