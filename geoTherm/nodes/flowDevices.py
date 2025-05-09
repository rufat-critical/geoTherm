from .baseNodes.baseFlow import baseFlow, baseInertantFlow
from ..units import inputParser, addQuantityProperty
from .baseNodes.baseTurbo import baseTurbo
from ..flow_funcs import _dH_isentropic
from ..decorators import state_dict

@addQuantityProperty
class PressureController(baseInertantFlow):
    """Controls pressure by adjusting mass flow rate."""

    _displayVars = ['w', 'P', 'dP']
    _units = baseInertantFlow._units | {
        'P_setpoint': 'PRESSURE'
    }

    _bounds = [0, 1e5]
    @inputParser
    def __init__(self, name, US, DS, w:'MASSFLOW'=0, 
                 P_setpoint:'PRESSURE'=None, Z=(1, 'm**-3')):
        """Initialize pressure controller.

        Args:
            name: Component identifier
            US: Upstream node reference
            DS: Downstream node reference
            w: Initial mass flow rate
            P_setpoint: Target pressure to maintain
            Z: Flow inertance
        """
        super().__init__(name, US, DS, w, Z)
        self._P_setpoint = P_setpoint

    @state_dict
    def _state_dict(self):
        return {'P_setpoint': self._P_setpoint}

    def get_outlet_state(self, US, w):
        """Calculate outlet state based on current conditions."""
        # Pass through enthalpy, pressure will be controlled by evaluate()
        return {'H': US._H, 'P': self._P_setpoint}

    def evaluate(self):
        """Adjust flow to achieve target pressure."""
        DS = self.DS_node.thermo

        # Flow acceleration based on pressure error
        pressure_error = self._P_setpoint - DS._P
        self._wdot = pressure_error / self._Z

@addQuantityProperty
class fixedFlow(baseFlow):

    _units = baseFlow._units | {
        'area': 'AREA'
    }
    _bounds = [1e-5, 1]
    _displayVars = ['w', 'area', 'flow_func_name', 'PR']

    @inputParser
    def __init__(self, name, US, DS,
                 w:"MASSFLOW",
                 flow_func='isentropic'):

        super().__init__(name, US, DS)

        self._w = w

        self.flow_func_name = flow_func
        if flow_func == 'isentropic':
            from geoTherm.flow_funcs import IsentropicFlow
            self.flow_func = IsentropicFlow
        else:
            from pdb import set_trace
            set_trace()

    @state_dict
    def _state_dict(self):
        """
        Get the state dictionary containing the node's current state information.

        """

        return {'w': (self._w, 'kg/s'),
                'flow_func': self.flow_func_name}

    def thermostates(self):
        if self._w > 0:
            return self.US_node.thermo, self.DS_node.thermo, 1
        else:
            return self.DS_node.thermo, self.US_node.thermo, -1

    @property
    def _dP(self):

        US, DS, _ = self.thermostates()

        return DS._P - US._P

    def get_outlet_state(self, US, PR):
        """
        Calculate the thermodynamic state at the outlet (downstream).

        Returns:
            dict: A dictionary containing the enthalpy ('H') and pressure ('P')
                  at the downstream node.
        """

        # Get US, DS Thermo
        #US = self.model.nodes[self.US].thermo
        return {'H': US._H, 'P': US._P*PR}


    def _get_dP(self, US, PR):
        return US._P*PR

    def get_inlet_state(self, DS, PR):

        return {'H': DS._H,
                'P': DS._P}#/PR}

    def _set_flow(self, w):
        self._w = w

    @property
    def _area(self):
        return self.flow_func._cdA(self.US_node.thermo, self.DS_node.thermo, self._w)
