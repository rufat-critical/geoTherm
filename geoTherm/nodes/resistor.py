from ..units import inputParser, addQuantityProperty
from ..flow_funcs import (_w_incomp, _dP_incomp, _w_isen, _dP_isen,
                          _w_comp, _dP_comp, _w_isen_max)
from ..flow_funcs import flow_func
import numpy as np
from ..logger import logger
from .baseNodes.baseFlowResistor import baseFlowResistor


@addQuantityProperty
class resistor(baseFlowResistor):
    """ Resistor where mass flow is calculated based on
    flow func"""

    _displayVars = ['w', 'area', 'dP', 'flow_func']
    _units = {'w': 'MASSFLOW', 'area': 'AREA', 'dP': 'PRESSURE'}

    @inputParser
    def __init__(self, name, US, DS, area: 'AREA', flow_func='isentropic'):  # noqa

        super().__init__(name, US, DS)

        self.flow_func = flow_func
        self._area = area

    def initialize(self, model):
        super().initialize(model)

        if self.flow_func == 'isentropic':
            self._flow_func = _w_isen
            self._dP_func = _dP_isen
        elif self.flow_func == 'incomp':
            self._flow_func = _w_incomp
            self._dP_func = _dP_incomp
        elif self.flow_func == 'comp':
            self._flow_func = _w_comp
            self._dP_func = _dP_comp

        self.flow = flow_func(self.flow_func)

    def evaluate(self):

        # Get US, DS Thermo
        US, DS, flow_sign = self.thermostates()
        self._w = self.flow._w_flux(US, DS)*self._area*flow_sign
        self._dP = DS._P - US._P

    def get_outlet_state(self, US, w):

        dP = self._dP_func(US, w/self._area)
        
        if dP is None:
            dP = 1e9

        return {'H': US._H, 'P': US._P + dP}


    def get_inlet_state(self, DS, w):

        dP = self.flow._dP_reverse(DS, w/self._area, US_thermo=None)

        return {'H': DS._H, 'P': DS._P + dP}



class orifice(resistor):
    pass



@addQuantityProperty
class fixedFlow2(baseFlowResistor):
    """ Resistor Object where mass flow is fixed """

    _units = {'w': 'MASSFLOW', 'dP': 'PRESSURE'}
    _displayVars = ['w']

    @inputParser
    def __init__(self, name, US, DS,
                 w:'MASSFLOW',      # noqa
                 dP:'PRESSURE'=0):  # noqa
        """
        Initialize the fixedFlow node with given parameters.

        Args:
            name (str): Name of the node.
            US (str): Upstream node identifier.
            DS (str): Downstream node identifier.
            w (float): Mass flow rate.
            dP (float, optional): Pressure difference. Defaults to None.
        """

        super().__init__(name, US, DS)

        self._dP = dP
        self._w = w


    def get_outlet_state(self, US, w):
        """
        Calculate the thermodynamic state at the outlet (downstream).

        Returns:
            dict: A dictionary containing the enthalpy ('H') and pressure ('P')
                  at the downstream node.
        """

        # Get US, DS Thermo
        US = self.model.nodes[self.US].thermo

        return {'H': US._H, 'P': US._P + self._dP}  # Outlet state

    def get_inlet_state(self, DS, w):

        return {'H': DS._H,
                'P': DS._P - self._dP}

