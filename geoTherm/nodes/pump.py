from .baseNodes.baseTurbo import (Turbo, baseTurbo, baseInertantTurbo, FixedPressureRatioTurbo,
                                  FixedFlowTurbo, turboParameters)
from .baseNodes.baseFlow import baseInertantFlow
#from .flowDevices import fixedFlow
from ..units import addQuantityProperty
from ..utils import pump_eta
from ..flow_funcs import _dH_isentropic
from ..logger import logger
import numpy as np
from ..decorators import state_dict


class pumpParameters(turboParameters):

    _units = turboParameters._units | {'NPSP': 'PRESSURE'}

    @property
    def _NPSP(self):
        try:
            return self.US_node.thermo._P - self._ref_thermo._Pvap
        except:
            return 0

    @property
    def _Ds(self):
        """ Pump Specific Diameter """
        return self._D/np.sqrt(self._Q_in)*(self._dH_is)**0.25

    @property
    def phi(self):
        return self._Q_in/(self._D**2*self._u_tip)

    @property
    def psi(self):
        dH = self._get_dH()

        return dH/self._U_tip**2

    @property
    def _Ns(self):
        """ Pump Specific Speed Dimensional in SI """
        return self.rotor_node.N*np.sqrt(self._Q_in)/(self._dH_is)**(0.75)

    @property
    def _NSS(self):
        return (self.rotor_node.N*np.sqrt(self._Q_in)
                / (np.abs(self._NPSP)
                   / (9.81*self.US_node.thermo._density))**0.75)

    @property
    def _dH(self):
        return self._dH_is / self.eta


@addQuantityProperty
class basePump(pumpParameters, baseTurbo):
    """Base class for pump components.

    Implements pump-specific behavior for pressure increase and
    isentropic enthalpy calculations.
    """

    _units = baseTurbo._units | turboParameters._units | {'NPSP': 'PRESSURE'}


@addQuantityProperty
class baseInertantPump(pumpParameters, baseInertantTurbo):
    _units = baseInertantTurbo._units | pumpParameters._units



@addQuantityProperty
class Pump(baseInertantPump):

    def __init__(self, name, US, DS, rotor, w, eta, PumpMap,
                 Z:'INERTANCE'=(1, 'm**-3')):   # noqa
        
        super().__init__(name=name,
                         US=US,
                         DS=DS,
                         w=w,
                         rotor=rotor,
                         eta=eta,
                         Z=Z)

        self.PumpMap = PumpMap

    def evaluate(self):

        US, DS, _ = self.thermostates()
        N = self.rotor_node.N
        dP = self.PumpMap.get_dP(N, self._Q_in)

        self._wdot = ((US._P + dP)- DS._P)/self._Z

    def initialize(self, model):
        super().initialize(model)

        self.rotor_node = self.model.nodes[self.rotor]

    def get_outlet_state(self, US, *, w=None, PR=None):

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
class FixedFlowPump(pumpParameters, FixedFlowTurbo):
    _units = FixedFlowTurbo._units | pumpParameters._units

    @state_dict
    def _state_dict(self):
        """
        Get the state dictionary containing the node's current state information.

        This property extends basePump's state dictionary by adding the
        current efficiency value to it.
        """
        return {'eta': self.eta}

    def get_outlet_state(self, US, *, w=None, PR=None):
        dH_is = _dH_isentropic(US, US._P*PR)
        dH = dH_is/self.eta

        return {'P': US._P*PR, 'H': US._H + dH}


@addQuantityProperty
class FixedFlowPumpMap(FixedFlowPump):
    
    def __init__(self, name, US, DS, rotor, w, eta, PumpMap):
        super().__init__(name, US, DS, w, eta)
        self.rotor = rotor
        self.PumpMap = PumpMap


    def initialize(self, model):
        super().initialize(model)
        self.rotor_node = self.model.nodes[self.rotor]

    def evaluate(self):
        US, DS, _ = self.thermostates()

        N, _ = self.PumpMap.get_rpm((DS._P - US._P), self._Q_in)

        self.rotor_node.N = N


@addQuantityProperty
class FixedPressureRatioPump(basePump, FixedPressureRatioTurbo):
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

    _units = FixedPressureRatioTurbo._units | basePump._units

    # Bounds on flow rate
    _bounds = [0, 1e5]

    def get_outlet_state(self, US, *, w=None, PR=None):

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

