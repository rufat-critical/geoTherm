import numpy as np
#from .baseTurbo import Turbo, fixedFlowTurbo, TurboSizer, turbineParameters
from .baseNodes.baseTurbo import Turbo, fixedFlowTurbo, TurboSizer, turbineParameters
from ..units import addQuantityProperty
from ..utils import turb_axial_eta, turb_radial_eta
from ..flow_funcs import _dH_isentropic
from scipy.optimize import fsolve


@addQuantityProperty
class Turbine(Turbo, turbineParameters):

    def _get_dP(self):
        # Get Upstream Thermo
        US, _ = self._get_thermo()

        # Get delta P across Turbine
        return US._P*(1/self.PR - 1)

    def _get_dH(self):
        """Get enthalpy change across Turbine."""

        return self._dH_is*self.eta

    @property
    def _dH(self):
        return self._dH_is*self.eta

    @property
    def _dH_is(self):

        # Get Upstream Thermo
        US, _,_ = self.thermostates()

        # Isentropic Enthalpy across Turbo Component
        return _dH_isentropic(US, US._P/self.PR)

    def _set_flow(self, w):
        self._w = w
        
    def _update_eta(self):
        # Update Turbine efficiency using efficiency curves

        def hunt_eta(x):
            # Function used to find eta using fsolve

            # Set eta
            self.eta = x[0]

            if self.axial:
                eta_calc = turb_axial_eta(self.phi, self.psi, self.psi_is)
            else:
                eta_calc = turb_radial_eta(self.ns)

            return x[0]-eta_calc

        # Fsolve to find eta
        eta = fsolve(hunt_eta, self.eta)

        # Update Turbine eta and check it's converged
        eta_root = hunt_eta(eta)

        if np.abs(eta_root) > 1e-5:
            # Eta was not updated correctly
            # so go debug and figure out what happened
            from pdb import set_trace
            set_trace()

    def evaluate(self):

        if self.update_eta:
            self._update_eta()

        super().evaluate()

    def get_outlet_state(self, US, w):

        dP = US._P*(1/self.PR - 1)
        dH_is  = _dH_isentropic(US, US._P/self.PR)

        return {'P': US._P + dP,
                'H': US._H + dH_is*self.eta}


class Turbine_sizer(Turbine, TurboSizer):
    """ Turbine Class that sets shaft speed based on input Ns Ds psi or phi"""

    def _update_eta(self):
        # Update Turbine efficiency using efficiency curves

        def hunt_eta(x):
            # Function used to find eta using fsolve

            # Set eta
            self.eta = x[0]
            # Update Rotor
            self._update_rotor()
            # Update Rotor Diameter
            self._D = (self._targets['Ds']*np.sqrt(self._Q_out)
                       / (-self._dH_is)**0.25)

            if self.axial:
                eta_calc = turb_axial_eta(self.phi, self.psi, self.psi_is)
            else:
                eta_calc = turb_radial_eta(self.ns)

            return x[0]-eta_calc

        # Fsolve to find eta
        eta = fsolve(hunt_eta, self.eta)

        # Update Turbine eta and check it's converged
        eta_root = hunt_eta(eta)

        if np.abs(eta_root) > 1e-5:
            # Eta was not updated correctly
            # so go debug and figure out what happened
            from pdb import set_trace
            set_trace()


class fixedFlowTurbine(fixedFlowTurbo, Turbine):
    """
    Turbine class where mass flow is fixed to initialization value.
    """

    # State Bounds, defining them here for now
    # In the future can make a control class to check/update bounds
    _bounds = [1, 100]

    def get_outlet_state(self, US, PR):
        
        dH = _dH_isentropic(US, US._P*self.PR)*self.eta

        return {'H': US._H + dH, 'P': US._P*PR}

    def evaluate(self):
       self.PR = self.US_node.thermo._P/self.DS_node.thermo._P