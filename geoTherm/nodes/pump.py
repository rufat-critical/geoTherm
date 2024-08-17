from .baseClasses import fixedFlowNode
from .baseTurbo import Turbo
from ..units import addQuantityProperty
from ..utils import dH_isentropic, pump_eta
from ..logger import logger
import numpy as np

@addQuantityProperty
class Pump(Turbo):
    """Pump class inheriting from Turbo."""

    _displayVars = ['w', 'dP', 'dH', 'W', 'PR']#'vol_flow',
    #                'specific_speed', 'NPSP', 'NSS']
    _units = {'w': 'MASSFLOW', 'W': 'POWER', 'dH': 'SPECIFICENERGY',
              'dP': 'PRESSURE', 'vol_flow':'VOLUMETRICFLOW', 'Q':'POWER',
              'NPSP': 'PRESSURE', 'specific_speed': 'SPECIFICSPEED',
              'NSS': 'SPECIFICSPEED'}
    bounds = [1, 1000]

    def _get_dP(self):
        """Get delta P across Pump."""
        
        US, _ = self.get_thermostates()
        
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
        US,_ = self.get_thermostates()

        return dH_isentropic(US, US._P*self.PR)


    def update_pump_parameters(self):
        self._NPSP = self.US_node.thermo._P - self._refThermo._Pvap
        self._u_tip = np.sqrt(self._dH_is/self.psi)
        self.phi = self._vol_flow/(self._D**2*self._u_tip)
        self._NSS = (self.rotor_node.N*np.sqrt(self._vol_flow)
                     / (self._NPSP
                        / (9.81*self.US_node.thermo._density))**0.75)

    @property
    def _NPSP(self):
        return self.US_node.thermo._P - self._refThermo._Pvap

    @property
    def _u_tip(self):
        return np.sqrt(self._dH_is/self.psi)

    @property
    def Ds(self):
        return None
    #@property
    #def _D(self):
    #    return 2*self._u_tip/self.rotor_node.omega

    @property
    def phi(self):
        return self._Q_in/(self._D**2*self._u_tip)

    @property
    def psi(self):
        dH = self._get_dH()

        return dH/self._U**2

    @property
    def _NSS(self):
        return (self.rotor_node.N*np.sqrt(self._vol_flow)
                / (np.abs(self._NPSP)
                   / (9.81*self.US_node.thermo._density))**0.75)


@addQuantityProperty
class fixedFlowPump(fixedFlowNode, Pump):
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
        return np.array(self._x)

    def updateState(self, x):
        """
        Update the state of the Pump.

        Args:
            x (float): New state value to set.
        """

        # Update X if it is within boudns or apply penalty

        self._x = x
        PR = x[0]*np.diff(self._bounds)

        if self._bounds[0] < PR < self._bounds[1]:
            self.penalty = False
            self.PR = PR[0]
        else:
            self.penalty = (self._bounds[0] - PR - 10*np.sign(PR))*1e8

    def initialize(self, model):
        self._x = self.PR/np.diff(self._bounds)
        return super().initialize(model)

    def __init__(self, *args, **kwargs):
        self._w_correction = 0
        # Run init method
        super().__init__(*args, **kwargs)

    @property
    def error(self):
        if self.penalty is not False:
            return self.penalty
        else:
        # Get Thermo States
            #self._w_correction = (self.US_node.thermo._P/self.DS_node.thermo._P
            #                      - self.PR)*.0001

            return (self.DS_node.thermo._P/self.US_node.thermo._P
                    - self.PR)
    @property
    def _w(self):
        # Correction Term
        #if hasattr(self, 'DS_node'):
        #    corr = (self.DS_node.thermo._P/self.US_node.thermo._P
        #                - self.PR)
        #else:
        #    corr = 0
        #return self._w_setpoint + corr*.1

        return self._w_setpoint*(1+self._w_correction)

    @_w.setter
    def _w(self, w):
        self._w_setpoint = w