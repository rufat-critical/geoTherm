from ..units import inputParser, addQuantityProperty
from .baseClasses import flowNode, fixedFlowNode
from ..flow_funcs import (_w_incomp, _dP_incomp, _w_isen, _dP_isen,
                          _w_comp, _dP_comp)
from .boundary import PBoundary
from ..logger import logger


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

        if self._dP:
            return {'H': US._H,
                    'P': US._P + self._dP}
        else:
            return self._dP

@addQuantityProperty
class fixedFlow(fixedFlowNode):
    """ Resistor Object where mass flow is fixed """

    _units = {'w': 'MASSFLOW', 'dP': 'PRESSURE'}
    _displayVars = ['w']

    @inputParser
    def __init__(self, name, US, DS,
                 w:'MASSFLOW',      # noqa
                 dP:'PRESSURE'=None):  # noqa
        """
        Initialize the fixedFlow node with given parameters.

        Args:
            name (str): Name of the node.
            US (str): Upstream node identifier.
            DS (str): Downstream node identifier.
            w (float): Mass flow rate.
            dP (float, optional): Pressure difference. Defaults to None.
        """

        self.name = name
        self.US = US
        self.DS = DS
        self.__dP = dP

        self.initialize_flow(w)

    def _get_thermo(self):
        """
        Get the inlet and outlet thermo states based on flow direction.
        """

        return (self.model.nodes[self.US].thermo,
                self.model.nodes[self.DS].thermo)

    def initialize(self, model):
        """
        Initialize the node within the model and check for consistency
        in terms of the pressure boundary conditions.

        Args:
            model (object): The model in which the node is to be initialized.
        """
        super().initialize(model)  # Call the parent class initializer

        # Check if pressure difference is specified and check if proper
        # US/DS node specified
        if self.__dP is not None:  #
            if not isinstance(self.DS_node, PBoundary) and self._w > 0:
                logger.critical(
                    f"fixedFlow Node {self.name} has dP specified and "
                    "requires the downstream node to be a PBoundary!"
                )
            elif not isinstance(self.US_node, PBoundary) and self._w < 0:
                logger.critical(
                    f"fixedFlow Node {self.name} has dP specified and "
                    "requires the upstream node to be a PBoundary!"
                )
            else:
                # Update thermodynamic state for PBoundary nodes
                self.DS_node.update_thermo(self.get_outlet_state())

    @property
    def _dP(self):

        if self.__dP is not None:
            return self.__dP
        else:
            US, DS = self._get_thermo()
            return DS._P - US._P

    def get_outlet_state(self):
        """
        Calculate the thermodynamic state at the outlet (downstream).

        Returns:
            dict: A dictionary containing the enthalpy ('H') and pressure ('P')
                  at the downstream node.
        """

        # Get US, DS Thermo
        US = self.model.nodes[self.US].thermo

        if self._w == 0:
            self._dH = 0  # No enthalpy change if no mass flow
        else:
            self._dH = self._Q/self._w  # Calculate enthalpy change

        return {'H': US._H + self._dH, 'P': US._P + self._dP}  # Outlet state
