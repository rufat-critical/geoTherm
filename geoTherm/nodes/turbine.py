import numpy as np
from .baseClasses import fixedFlowNode
from .baseTurbo import Turbo, TurboSizer
from ..units import addQuantityProperty
from ..utils import dH_isentropic, turb_axial_eta, turb_radial_eta
from scipy.optimize import fsolve


@addQuantityProperty
class Turbine(Turbo):

    def _get_dP(self):
        # Get Upstream Thermo
        US, _ = self.get_thermostates()

        # Get delta P across Turbine
        return US._P*(1/self.PR - 1)

    def _get_dH(self):
        """Get enthalpy change across Turbine."""

        return self._dH_is*self.eta

    @property
    def _dH_is(self):

        # Get Upstream Thermo
        US, _ = self.get_thermostates()

        # Isentropic Enthalpy across Turbo Component
        return dH_isentropic(US, US._P/self.PR)

    @property
    def phi(self):
        return self._Q_out/(self._D**2*self._U)

    @property
    def psi(self):

        dH = self._get_dH()

        return -dH/self._U**2

    @property
    def _Ns(self):
        """ Turbine Specific Speed Dimensional in SI """
        return self.rotor_node.N*np.sqrt(self._Q_out)/(-self._dH_is)**(0.75)

    @property
    def ns(self):
        """ Turbine Specific Speed Dimensionless """
        return self.rotor_node.Nrad*np.sqrt(self._Q_out)/(-self._dH_is)**(0.75)

    @property
    def _Ds(self):
        """ Turbine Specific Diameter"""
        return self._D/np.sqrt(self._Q_out)*(-self._dH_is)**0.25

    @property
    def AN2(self):
        return self.rotor_node.N**2*self._Q_out

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


class fixedFlowTurbine(fixedFlowNode, Turbine):
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
    def xdot(self):
        if self.penalty is not False:
            return self.penalty
        else:
            return (self.US_node.thermo._P/self.DS_node.thermo._P - self.PR)
