from ..units import inputParser, addQuantityProperty
from .baseClasses import flowNode, fixedFlowNode
from ..flow_funcs import (_w_incomp, _dP_incomp, _w_isen, _dP_isen,
                          _w_comp, _dP_comp)


@addQuantityProperty
class resistor(flowNode):
    """ Resistor where mass flow is calculated based on
    flow func"""

    _displayVars = ['w', 'area', 'dP']
    _units = {'w': 'MASSFLOW', 'area': 'AREA', 'dP': 'PRESSURE'}

    @inputParser
    def __init__(self, name, US, DS, area: 'AREA', flow_func='isentropic'):  # noqa
        self.name = name
        self.US = US
        self.DS = DS
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

    def evaluate(self):

        # Evaluate flow func to get flow rate
        self._w = self._flow_func(self.US_node.thermo,
                                  self.DS_node.thermo,
                                  self._area)

    def get_outlet_state(self):
        # The enthalpy is equal but pressure drop is present

        # Get US, DS Thermo
        US, _ = self._get_thermo()

        self._dP = self._dP_func(US, self._w/self._area)

        return {'H': US._H,
                'P': US._P + self._dP}


@addQuantityProperty
class fixedFlow(flowNode, fixedFlowNode):
    """ Resistor Object where mass flow is fixed """

    _units = {'w': 'MASSFLOW', 'dP': 'PRESSURE'}
    _displayVars = ['w']

    @inputParser
    def __init__(self, name, US, DS,
                 w:'MASSFLOW',      # noqa
                 dP:'PRESSURE'=0):  # noqa

        self.name = name
        self.US = US
        self.DS = DS
        self._w = w
        self._dP = dP

    def get_outlet_state(self):

        # Get the Downstream thermo state
        US = self.model.nodes[self.US].thermo
        self._dH = self._Q/self._w

        return {'H': US._H + self._dH,
                'P': US._P + self._dP}
