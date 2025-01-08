from .baseNodes.baseTurbo import Turbo, pumpParameters, fixedFlowTurbo
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


class fixedFlowPump(fixedFlowTurbo, Pump):
    """Pump class with fixed mass flow."""

    # State Bounds, defining them here for now
    # In the future can make a control class to check/update bounds
    _bounds = [1, 500]

    def evaluate(self):

        self.PR = self.DS_node.thermo._P/self.US_node.thermo._P

    def get_outlet_state(self, US, PR):
        
        dH = _dH_isentropic(US, US._P*self.PR)/self.eta

        return {'H': US._H + dH, 'P': US._P*PR}
