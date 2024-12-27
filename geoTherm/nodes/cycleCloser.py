from .baseNodes.baseFlow import baseFlow
from ..units import addQuantityProperty, inputParser
import numpy as np

@addQuantityProperty
class cycleCloser(baseFlow):

    _displayVars = ['w', 'dP', 'dH']
    _units = {'w': 'MASSFLOW', 'dP': 'PRESSURE', 'dH': 'ENERGY'}

    def __init__(self, name, US, DS,
                 w:'MASSFLOW'=0):

        super().__init__(name, US, DS)

        self._w = w

    @property
    def _dP(self):
        return (self.DS_node.thermo._P
                - self.US_node.thermo._P)

    def _set_flow(self, w):
        self._w = w

    @property
    def _dH(self):
        return (self.DS_node.thermo._H
                - self.US_node.thermo._H)

    def evaluate(self):
        self._w += self.US_node.flux[0]


    def get_outlet_state(self, US, w):

        return {'P': self.DS_node.thermo._P,
                'H': self.DS_node.thermo._H}
