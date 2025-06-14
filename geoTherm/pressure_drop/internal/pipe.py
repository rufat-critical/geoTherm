from ..flow import PipeLoss2
from scipy.optimize import root_scalar
import numpy as np
from geoTherm.utils import eps
from ..base_loss import baseLossModel
from geoTherm.geometry.internal.cylinder import InternalCylinder, InternalCylinderBend
from maps.Pipe.Bend.Interpolators import KBend

def Re_(Dh, area, w, viscosity):
    """ Calculate the Reynolds number.

    Args:
        thermo (thermostate): geoTherm thermostat object
        Dh (float): Hydraulic Diameter in m
        w (float): Mass flow rate in kg/s

    Returns:
        float: Reynolds number
    """
    return w*Dh / (viscosity * area)


def friction_factor(Re, rel_roughness):
    """ Calculate the friction factor based on Reynolds number and 
        relative roughness.

    Args:
        Re (float): Reynolds Number
        rel_roughness (float): Relative roughness (roughness/Dh)

    Returns:
        float: Friction factor
    """
    if Re < 2100:
        return 64 / (Re + eps)
    else:
        return colebrook(rel_roughness, Re)


def colebrook(k, Re):
    """ Get the friction factor using Colebook Equation 

    Args:
        k (float): Relative Roughness
        Re (float): Reynolds Number 

    Returns:
        friction factor """

    def res(f):
        # Residual function
        return 1/np.sqrt(f) + 2*np.log10(k/3.7 + 2.51/(Re*np.sqrt(f)))

    # Use root scalar to find root
    f = root_scalar(res, method='brentq', bracket=[1e-10, 1e4]).root

    return f


def pipe_K(thermo, L, Dh, area, w, roughness=2.5e-6):
    """ Calculate the loss coefficient for pipe flow.

    Args:
        thermo (thermostate): geoTherm thermostat object
        L (float): Pipe length in m
        Dh (float): Hydraulic Diameter in m
        w (float): Mass flow rate in kg/s
        roughness (float, optional): Pipe roughness in m (Default = 2.5e-6 m)

    Returns:
        float: Loss coefficient
    """
    Re = Re_(Dh, area, w, thermo._viscosity)
    f = friction_factor(Re, roughness / Dh)
    return f * L / Dh


class StraightLoss(baseLossModel):
    """Pressure loss calculations for a pipe."""

    def evaluate(self, thermo, w):
        """Compute loss coefficient and pressure drop dynamically."""

        # Get K Factor
        self.K = pipe_K(thermo, self.geometry._L,
                        self.geometry._Dh,
                        self.geometry._area,
                        w,
                        self.geometry._roughness)

        self.Re = Re_(self.geometry._Dh,
                     self.geometry._area,
                     w,
                     thermo._viscosity)
        
        self.f = friction_factor(self.Re,
                                 self.geometry._roughness / self.geometry._Dh)

        self._dP = -self.K*((w/self.geometry._area)**2
                            / (2 * thermo._density))

        return self._dP

class BendLoss(baseLossModel):
    """Pressure loss calculations for a pipe bend."""

    def evaluate(self, thermo, w):
        """Compute loss coefficient and pressure drop."""
        self.Re = Re_(self.geometry._Dh,
                     self.geometry._area,
                     w,
                     thermo._viscosity)

        self.Kbend = KBend.evaluate(self.Re, self.geometry.RD, self.geometry.angle)

        self.Kpipe = pipe_K(thermo, self.geometry._L,
                            self.geometry._D,
                            w,
                            self.geometry._roughness)

        self.K = self.Kpipe + self.Kbend

        self._dP = -self.K*((w/self.geometry._area)**2
                            / (2 * thermo._density))

        return self._dP


class PipeLoss(baseLossModel):
    """Pressure loss calculations for a pipe."""

    def __init__(self, geometry):
        super().__init__(geometry)
        self.initialize()

    def initialize(self):

        if isinstance(self.geometry, InternalCylinderBend):
            self.loss = BendLoss(self.geometry)       
        elif isinstance(self.geometry, InternalCylinder):
            self.loss = StraightLoss(self.geometry)

        else:
            from pdb import set_trace
            set_trace()

    def evaluate(self, thermo, w):
        """Compute loss coefficient and pressure drop."""
        return self.loss.evaluate(thermo, w) + self._dP_z(thermo)

    def _dP_z(self, thermo):
        return -self.geometry._dz*9.81*thermo._density