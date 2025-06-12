from ..units import inputParser, addQuantityProperty
import numpy as np
from ..logger import logger
from .baseNodes.baseFlowResistor import baseFlowResistor
from .baseNodes.baseFlow import FixedFlow
from ..flow_funcs import FlowModel


@addQuantityProperty
class resistor(baseFlowResistor):
    """ Resistor where mass flow is calculated based on
    flow func"""

    _displayVars = ['w', 'area', 'dP', 'PR', 'flow_func']
    _units = {'w': 'MASSFLOW', 'area': 'AREA', 'dP': 'PRESSURE'}

    @inputParser
    def __init__(self, name, US, DS, area: 'AREA', flow_func='isen'):  # noqa

        super().__init__(name, US, DS)

        self.flow_func = flow_func
        self.flow = FlowModel(flow_func, cdA=area)

    @property
    def _area(self):
        return self.flow._cdA

    @_area.setter
    def _area(self, value):
        self.flow._cdA = value


    def get_inlet_state(self, DS, w):

        dP = self.flow2._dP_reverse(DS, w/self._area, US_thermo=None)

        return {'H': DS._H, 'P': DS._P + dP}

    def _get_outlet_state_PR(self, US, PR):


        return {'P': US._P*PR, 'H': US._H}

    def _get_dP(self, US, w):
        from pdb import set_trace
        set_trace()

class orifice(resistor):
    pass

class Resistor(resistor):
    pass

@addQuantityProperty
class FixedFlowResistor(FixedFlow):

    _units = FixedFlow._units | {'area': 'AREA'}

    _displayVars =  ['w', 'area', 'dP', 'PR', 'flow_func']

    def __init__(self, name, US, DS, w, flow_func='isen'):  # noqa

        super().__init__(name, US, DS, w)

        self.flow_func = flow_func
        self.flow = FlowModel(flow_func, 1)

    @property
    def _area(self):
        return self._w/self.flow._w(self.US_node.thermo,
                                    self.DS_node.thermo)
