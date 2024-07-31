import numpy as np
from .baseClasses import Turbo, fixedFlowNode
from ..units import addQuantityProperty, inputParser
from ..utils import dH_isentropic
from ..logger import logger


@addQuantityProperty
class Turbine(Turbo):

    def __init__(self, *args, Ns=None, Ds=None, **kwargs):
        """
        Initialize the Turbine Node.

        Args:
            name (str): Name of the turbine.
            eta (float): Efficiency of the turbine.
            US (str): Upstream node name.
            DS (str): Downstream node name.
            rotor (str): Rotor Object.
            PR (float): Pressure ratio.
            phi (float): Head Coefficient
            Ns (float): Specific Speed
            Ds (float): Specific Diameter
            w (float): Mass flow rate.
        """

        # Store NS and DS for Turbo 
        self._Ns = Ns
        self._Ds = Ds

        # Do rest of initialization
        super().__init__(self, *args, **kwargs)

    def _get_dP(self, US, DS):
        # Get delta P across Turbine
        return US._P*(1/self.PR - 1)
    
    def _get_dH(self, US, DS):
        """Get enthalpy change across Turbine."""

        return dH_isentropic(US, US._P/self.PR)*self.eta

    @property
    def _c_is(self):
        # Isentropic Spouting Velocity
        return np.sqrt(2*self._dH_is)

    @property
    def _Mach_in(self):
        return self._c_is/self.US_node.thermo.sound

    @property
    def _Mach_out(self):
        return self._c_is/self.DS_node.thermo.sound

    @property
    def _vol_flow_out(self):
        return self._w/self.DS_node.thermo._density

    @property
    def _D(self):
        return self._Ds*np.sqrt(self._vol_flow_out)/self._dH_is**0.25
    
    @property
    def N(self):
        return self._Ns*self._dH_is**0.75/np.sqrt(self._vol_flow_out)

    @property
    def _u_tip(self):
        return self._D*self.rotor_node.omega/2
    
    @property
    def psi(self):
        return self._dH_is/self._u_tip**2
    
    @psi.setter
    def psi(self, input):
        pass

class fixedWTurbine(fixedFlowNode, Turbine):
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
    
    @property
    def error(self):
        from pdb import set_trace
        set_trace()