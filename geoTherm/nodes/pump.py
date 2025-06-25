from .baseNodes.baseTurbo import Turbo, pumpParameters, fixedFlowTurbo, baseTurbo, fixedPressureRatioTurbo, turboParameters
from .baseNodes.baseFlow import baseInertantFlow, FixedFlow
#from .flowDevices import fixedFlow
from ..units import addQuantityProperty
from ..utils import pump_eta
from ..flow_funcs import _dH_isentropic
from ..logger import logger
import numpy as np
from ..maps.pump.D35e import PumpMap
from ..decorators import state_dict

@addQuantityProperty
class basePump(baseTurbo, turboParameters):
    """Base class for pump components.

    Implements pump-specific behavior for pressure increase and
    isentropic enthalpy calculations.
    """

    _units = baseTurbo._units | turboParameters._units | {'NPSP': 'PRESSURE'}


    _displayVars = baseTurbo._displayVars + ['Q_in', 'Q_out']

    @property
    def _dH_is(self):
        """Calculate isentropic enthalpy change for compression.

        Returns:
            float: Isentropic enthalpy change [J/kg]
        """
        US, DS, _ = self.thermostates()
        return _dH_isentropic(US, DS._P)

    @property
    def _dH(self):
        return self._dH_is / self.eta

    @property
    def PR(self):
        """Calculate actual pressure ratio (outlet/inlet).

        For pumps, PR > 1 indicates normal operation (compression),
        while PR < 1 indicates reverse flow.

        Returns:
            float: Pressure ratio (P_downstream / P_upstream)
        """
        US, DS, _ = self.thermostates()
        return DS._P / US._P

    @property
    def _NPSP(self):
        try:
            return self.US_node.thermo._P - self._ref_thermo._Pvap
        except:
            return 0


class baseRotorPump(basePump):
    pass


@addQuantityProperty
class baseInertantPump(basePump, baseInertantFlow):
    pass


@addQuantityProperty
class Pump(baseInertantPump):

    def __init__(self, name, US, DS, rotor, w, eta, PumpMap,
                 Z:'INERTANCE'=(1, 'm**-3')):   # noqa
        super().__init__(name, US, DS, w, Z)
        self.rotor = rotor
        self.eta = eta
        self.PumpMap = PumpMap

    
    def evaluate(self):

        US, DS, _ = self.thermostates()
        N = self.rotor_node.N
        dP = self.PumpMap.get_dP(N, self._Q_in)

        self._wdot = ((US._P + dP)- DS._P)/self._Z


    def initialize(self, model):
        super().initialize(model)

        self.rotor_node = self.model.nodes[self.rotor]


    def get_outlet_state(self, US, w):

        Q = w/US._density
        N = self.rotor_node.N
        dP = self.PumpMap.get_dP(N, Q)

        
        dH_is = _dH_isentropic(US, US._P + dP)

        #if error:
        #    return {'P': 1e9, 'H': US._H}
        #else:
        return {'P': US._P + dP, 'H': US._H + dH_is/self.eta}

    def _get_dP(self, US, w):
        from pdb import set_trace
        set_trace()


@addQuantityProperty
class fixedFlowPump(basePump, FixedFlow):
    """Pump with a fixed mass flow rate.

    A pump that maintains a constant mass flow rate.
    Inherits initialization parameters from fixedFlow.

    Args:
        name (str): Component identifier
        US: Upstream node reference
        DS: Downstream node reference
        w (float): Fixed mass flow rate [kg/s]
        eta (float): Isentropic efficiency
        """
    _displayVars = ['w', 'eta', 'dH', 'W', 'PR']
    _bounds = [1, 100]

    def __init__(self, name, US, DS, w, eta):
        super().__init__(name, US, DS, w)
        self.eta = eta

    @state_dict
    def _state_dict(self):
        """
        Get the state dictionary containing the node's current state information.

        This property extends basePump's state dictionary by adding the
        current efficiency value to it.
        """
        return {'eta': self.eta}

    def get_outlet_state(self, US, PR):
        dH_is = _dH_isentropic(US, US._P*PR)/self.eta

        return {'P': US._P*PR, 'H': US._H + dH_is}


@addQuantityProperty
class fixedPressureRatioPump(basePump, fixedPressureRatioTurbo):
    """Pump with a fixed pressure ratio.

    A pump that maintains a constant ratio between outlet and inlet pressure.
    Inherits initialization parameters from fixedPressureRatioTurbo.

    Args:
        name (str): Component identifier
        US: Upstream node reference
        DS: Downstream node reference
        PR (float): Fixed pressure ratio (outlet/inlet pressure)
        w (float): Mass flow rate [kg/s]
        eta (float): Isentropic efficiency
        Z (tuple, optional): Tuple of (value, units) for compressibility factor,
                           defaults to (1, 'm**-3')

    Attributes:
        _bounds (list): Flow rate bounds [min, max] in kg/s, set to [0, 1e5]
    
    Note:
        Unlike the base turbo class, pump flow bounds are restricted to positive values only.
    """

    # Bounds on flow rate
    _bounds = [0, 1e5]

    def get_outlet_state(self, US, w):

        if self.PR_setpoint == 1:
            # No pressure change, so no enthalpy change
            dH = 0
        else:
            # Isentropic enthalpy change across the pump
            dH_is = _dH_isentropic(US, US._P*self.PR_setpoint)
            # Actual enthalpy change is isentropic enthalpy change divided by efficiency
            dH = dH_is/self.eta

        return {
            'P': US._P*self.PR_setpoint,
            'H': US._H + dH
        }
    
    def _get_dP(self, US, w):
        from pdb import set_trace
        set_trace()



@addQuantityProperty
class simplePump(basePump, baseInertantFlow):
    """Simple pump model with inertial dynamics.

    Implements a basic pump where flow acceleration is proportional
    to deviation from target pressure ratio.
    """

    _displayVars = ['w', 'wdot', 'eta', 'dH', 'W', 'dP', 'PR', 'Z']

    def __init__(self, name, US, DS, PR_setpoint, w:'MASSFLOW', eta,
                 Z:'INERTANCE'=(1, 'm**-3')):   # noqa
        """Initialize simple pump.

        Args:
            name (str): Component identifier
            US: Upstream node reference
            DS: Downstream node reference
            PR_setpoint (float): Target pressure ratio
            w (float): Initial mass flow rate [kg/s]
            eta (float): Isentropic efficiency
            Z (tuple, optional): Tuple of (value, units) for compressibility,
                               defaults to (1, 'm**-3')
        """
        super().__init__(name, US, DS, w, Z)  # Calls baseInertantFlow.__init__
        self.PR_setpoint = PR_setpoint
        self.eta = eta

    def evaluate(self):
        """Update pump state based on pressure ratio deviation."""
        self._wdot = (self.PR - self.PR_setpoint) / self._Z

    def get_outlet_state(self, US, w):
        """Calculate pump outlet thermodynamic state."""

        dH_is = _dH_isentropic(US, US._P*self.PR_setpoint)

        return {
            'P': US._P * self.PR_setpoint,
            'H': US._H + dH_is/self.eta
        }


@addQuantityProperty
class Pump2(Turbo, pumpParameters):
    """Pump class inheriting from Turbo."""

    _displayVars = ['w', 'dP:\u0394P', 'dH:\u0394H', 'W', 'PR', 'Q_in',
                    'Q_out', 'Ns', 'Ds', 'D', 'Mach_in', 'Mach_out',
                    'phi:\u03C6', 'psi:\u03C8', 'psi_is:\u03C8_is', 'U_tip',
                    'eta:\u03B7', 'NSS', 'NPSP']

    _units = {'w': 'MASSFLOW', 'W': 'POWER', 'dH': 'SPECIFICENERGY',
              'dP': 'PRESSURE', 'Q_in': 'VOLUMETRICFLOW',
              'Q_out': 'VOLUMETRICFLOW', 'Q': 'POWER',
              'Ns': 'SPECIFICSPEED', 'Ds': 'SPECIFICDIAMETER',
              'NSS': 'SPECIFICSPEED', 'NPSP': 'PRESSURE',
              'D': 'LENGTH', 'U_tip': 'VELOCITY'}

    # Bounds on flow variables
    _bounds = [1, 1000]

    def thermostates(self):
        """
        Get the inlet and outlet thermo states based on Pressure
        """

        # Handle Backflow
        if self.US_node.thermo._P <= self.DS_node.thermo._P:
            US = self.US_node.thermo
            DS = self.DS_node.thermo
            flow_sign = 1
        else:
            US = self.DS_node.thermo
            DS = self.US_node.thermo
            flow_sign = -1

        return US, DS, flow_sign

    def _get_dP(self):
        """Get delta P across Pump."""

        US, _, _ = self.thermostates

        return US._P*(self.PR-1)
    


    def _get_dH(self):
        """Get enthalpy change across Pump."""

        #if self.update_eta:
        #    self.eta = pump_eta(self.phi)

        return self._dH_is/self.eta

    @property
    def _dP(self):
        US, _, _ = self.thermostates()

        return US._P*(self.PR-1)

    @property
    def _dH(self):
        return self._dH_is/self.eta

    @property
    def _dH_is(self):
        # Isentropic Enthalpy across Turbo Component

        # Get Upstream Thermo
        #US,_,_ = self.thermostates()

        US = self.US_node.thermo


        return _dH_isentropic(US, US._P*self.PR)


    def get_outlet_state(self, US, w):

        dP = US._P*(self.PR-1)
        dH_is = _dH_isentropic(US, US._P*self.PR)


        return {'P': US._P + dP,
                'H': US._H + dH_is/self.eta}

    def evaluate(self):
        pass
        #US, _, flow_sign = self.thermostates()
        #US = self.US_node.thermo
        #self._dP = US._P*(self.PR-1)
        #self._dH = _dH_isentropic(US, US._P*self.PR)/self.eta


    def _set_flow(self, w):
        self._w = w


class fixedFlowPump2(basePump, FixedFlow):
    """Pump class with fixed mass flow.

    Args:
        name (str): Component identifier
        US: Upstream node reference
        DS: Downstream node reference
        w (float): Fixed mass flow rate [kg/s]
        eta (float): Isentropic efficiency
        PR (float): Fixed pressure ratio (outlet/inlet pressure)
    """

    # State Bounds, defining them here for now
    # In the future can make a control class to check/update bounds
    _bounds = [1, 500]


    def __init__(self, name, US, DS, w, eta, PR):

        super().__init__(name, US, DS, w)
        self.eta = eta
        self.PR_setpoint = PR

    def evaluate(self):
        pass
        #self.PR = self.DS_node.thermo._P/self.US_node.thermo._P

    def get_outlet_state(self, US, w):

        dH = _dH_isentropic(US, US._P*self.PR_setpoint)/self.eta

        return {'H': US._H + dH, 'P': US._P*self.PR_setpoint}
