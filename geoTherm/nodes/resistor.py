from ..units import inputParser, addQuantityProperty
import numpy as np
from ..logger import logger
from .baseNodes.baseFlow import FixedFlow, baseFlowResistor, baseInertantFlow
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



class orifice(resistor):
    pass

class Resistor(resistor):
    pass

class InertantResistor(baseInertantFlow):

    @inputParser
    def __init__(self, name, US, DS, area: 'AREA', flow_func='isen', w=0, Z=1):  # noqa

        super().__init__(name, US, DS, w, Z)

        self.flow_func = flow_func
        self.flow = FlowModel(flow_func, cdA=area)

    def get_outlet_state(self, US, w):
        dP, error = self.flow._dP(US, np.abs(w))

        if dP is None:
            dP = -1e9
            
        return {'P': US._P + dP, 'H': US._H}

    
    def get_outlet_state_PR(self, US, PR):
        from pdb import set_trace
        set_trace()

    def _w_max(self, US, DS):
        from pdb import set_trace
        set_trace()

        return self.flow._w_max(US, DS)

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
