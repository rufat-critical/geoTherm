from ..units import inputParser, addQuantityProperty
import numpy as np
from ..logger import logger
from .baseNodes.baseFlowResistor import baseFlowResistor
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
        self._area = area
        self.flow = FlowModel(flow_func, self._area)

    def initialize(self, model):
        super().initialize(model)


    def evaluate(self):

        # Get US, DS Thermo
        US, DS, flow_sign = self.thermostates()
        self._w = self.flow._w(US, DS)*flow_sign

    def get_outlet_state(self, US, w):

        dP, error = self.flow._dP(US, np.abs(w))


        if dP is None:
            # Set outlet to high dP to tell
            # solver to reduce mass flow
            dP = -1e9

        return {'H': US._H, 'P': US._P + dP}


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
