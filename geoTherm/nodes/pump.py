from .baseClasses import fixedFlowNode
from .baseTurbo import Turbo, pumpParameters, fixedFlowTurbo
from ..units import addQuantityProperty
from ..utils import pump_eta
from ..flow_funcs import _dH_isentropic
from ..logger import logger
import numpy as np


@addQuantityProperty
class Pump(Turbo, pumpParameters):
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

    def _get_dP(self):
        """Get delta P across Pump."""

        US, _ = self._get_thermo()

        return US._P*(self.PR-1)

    def _get_dH(self):
        """Get enthalpy change across Pump."""

        #if self.update_eta:
        #    self.eta = pump_eta(self.phi)

        return self._dH_is/self.eta

    @property
    def _dH_is(self):
        # Isentropic Enthalpy across Turbo Component

        # Get Upstream Thermo
        US,_ = self._get_thermo()

        return _dH_isentropic(US, US._P*self.PR)



class fixedFlowPump(fixedFlowTurbo, Pump):
    """Pump class with fixed mass flow."""

    # State Bounds, defining them here for now
    # In the future can make a control class to check/update bounds
    _bounds = [1, 500]

    @property
    def x(self):
        """
        fixedWPump PR pressure ratio state.

        Returns:
            np.array: Pressure ratio.
        """
        return np.array([self.PR])

    def update_state(self, x):
        """
        Update the state of the Pump.

        Args:
            x (float): New state value to set.
        """

        # Update X if it is within boudns or apply penalty
        if x < self._bounds[0]:
            self.penalty = (self._bounds[0] - x[0] + 10)*1e5
            self.PR = self._bounds[0]
        elif x > self._bounds[1]:
            self.penalty = (self._bounds[1] - x[0] - 10)*1e5
            self.PR = self._bounds[1]
        else:
            self.penalty = False
            self.PR = x[0]

    def __init__(self, *args, **kwargs):
        self._w_correction = 0
        # Run init method
        super().__init__(*args, **kwargs)

    @property
    def error(self):
        if self.penalty is not False:
            return self.penalty
        else:
            return (self.DS_node.thermo._P/self.US_node.thermo._P
                    - self.PR)

    @property
    def _w2(self):
        # Correction Term
        #if hasattr(self, 'DS_node'):
        #    corr = (self.DS_node.thermo._P/self.US_node.thermo._P
        #                - self.PR)
        #else:
        #    corr = 0
        #return self._w_setpoint + corr*.1

        from pdb import set_trace
        set_trace()

        return self._w_setpoint*(1+self._w_correction)

    @_w2.setter
    def _w2(self, w):
        self._w_setpoint = w
