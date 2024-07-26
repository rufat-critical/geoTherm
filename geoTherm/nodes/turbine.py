import numpy as np
from .baseClasses import Turbo
from ..units import addQuantityProperty, inputParser
from ..logger import logger


@addQuantityProperty
class Turbine(Turbo):

    def _get_dP(self, US, DS):
        # Get delta P across Turbine
        return US._P*(1/self.PR - 1)
    
    def _get_dH(self, US, DS):
        # Get enthalpy change across turbine

        # Generate temporary thermo
        isentropic = US.from_state(US.state)
        isentropic._SP = US._S, US._P/self.PR

        return (isentropic._H - US._H)*self.eta



class fixedWTurbine(Turbine):
    """ 
    Turbine class where mass flow is fixed to initialization value.
    """

    # State Bounds, defining them here for now
    # In the future can make a control class to check/update bounds
    _bounds = [1, 100]

    @property
    def x(self):
        """
        fixedWTurbine PR pressure ratio state.

        Returns:
            np.array: Pressure ratio.
        """
        return np.array([self.PR])

    def updateState(self, x):
        """
        Update the state of the turbine.

        Args:
            x (float): New state value to set.
        """

        # Update X if it is within boudns or apply penalty
        if x < self._bounds[0]:
            self.penalty = (self._bounds[0] - x[0] + 10)*1e5
            self.PR = self._bounds[0]
        elif x > self._bounds[1]:
            self.penalty = (x[0] - self._bounds[1] - 10)*1e5
            self.PR = self._bounds[1]
        else:
            self.penalty = False
            self.PR = x[0]