from .baseNodes.baseFlow import baseFlowResistor, FixedFlow
from geoTherm.common import inputParser, addQuantityProperty
from geoTherm.utils import cv_2in
from scipy.interpolate import interp1d
from geoTherm.flow_funcs import CvFlow
import numpy as np
from geoTherm.units import toSI, fromSI
import geoTherm as gt
from geoTherm.maps.valve.Cv_Map import Cv_Map

@addQuantityProperty
class Valve(baseFlowResistor):

    _units = {'w': 'MASSFLOW', 'dP': 'PRESSURE', 'Cv': 'FLOWCOEFFICIENT'}
    _displayVars = ['w', 'dP', 'Cv']

    @inputParser
    def __init__(self, name, US, DS, Cv:'FLOWCOEFFICIENT'):
        super().__init__(name, US, DS)
        self.flow = CvFlow(Cv)

    @property
    def _Cv(self):
        return self.flow._Cv

    @_Cv.setter
    def _Cv(self, value):
        self.flow._Cv = value


class PositionValve(Valve):

    @inputParser
    def __init__(self, name, US, DS, position, Cv_map):
        super().__init__(name, US, DS, 1)
        self.Cv_map = Cv_Map(Cv_map)
        self.position = position

    @property
    def position(self):
        return self._position

    @position.setter
    def position(self, value):
        self._position = value
        self._Cv = self.Cv_map._Cv(self._position)


@addQuantityProperty
class FixedFlowValve(FixedFlow):

    _displayVars = ['w', 'dP', 'Cv']

    _units = FixedFlow._units | {'Cv': 'FLOWCOEFFICIENT'}

    @inputParser
    def __init__(self, name, US, DS, w:'MASSFLOW'):
        super().__init__(name, US, DS, w)

        self._w = w
        self.flow = CvFlow(1)

    @property
    def _Cv(self):
        US, DS, _ = self.thermostates()
        self.flow.update_Cv(US, DS, self._w)
        return self.flow._Cv
