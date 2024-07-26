from .baseClasses import Turbo
from .turbine import Turbine
from ..units import inputParser, addQuantityProperty
from ..logger import logger
import numpy as np

@addQuantityProperty
class Pump(Turbo):
    """Pump class inheriting from Turbo."""

    _displayVars = ['w', 'dP', 'dH', 'W', 'PR', 'vol_flow', 'NPSP']
    _units = {'w': 'MASSFLOW', 'W': 'POWER', 'dH': 'SPECIFICENERGY',
              'dP': 'PRESSURE', 'vol_flow':'VOLUMETRICFLOW', 'Q':'POWER',
              'NPSP': 'PRESSURE'}

    def _get_dP(self, US, DS):
        """Get delta P across Pump."""
        return US._P*(self.PR-1)
    
    def _get_dH(self, US, DS):
        """Get enthalpy change across Pump."""

        # Generate temporary thermo
        isentropic = US.from_state(US.state)
        isentropic._SP = US._S, US._P*self.PR

        return (isentropic._H - US._H)/self.eta

    @property
    def _NPSP(self):
        US, DS = self._getThermo()

        self._refThermo._TQ = US._T, 0
        return US._P - self._refThermo._P


@addQuantityProperty
class fixedWPump(Pump):
    """Pump class with fixed mass flow."""

    # State Bounds, defining them here for now
    # In the future can make a control class to check/update bounds
    _bounds = [1, 100]

    @property
    def x(self):
        """
        fixedWPump PR pressure ratio state.

        Returns:
            np.array: Pressure ratio.
        """
        return np.array([self.PR])

    def updateState(self, x):
        """
        Update the state of the Pump.

        Args:
            x (float): New state value to set.
        """

        # Update X if it is within boudns or apply penalty
        if x < self._bounds[0]:
            self.penalty = (self._bounds[0] - x[0] + 10)*1e8
            self.PR = self._bounds[0]
        elif x > self._bounds[1]:
            self.penalty = (x[0] - self._bounds[1] - 10)*1e8
            self.PR = self._bounds[1]
        else:
            self.penalty = False
            self.PR = x[0]
