from ..utils import Re_, eps
from ..logger import logger
import numpy as np
from scipy.optimize import root_scalar


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


def pipe_K(thermo, L, Dh, w, roughness=2.5e-6):
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
    Re = Re_(thermo, Dh, w)
    f = friction_factor(Re, roughness / Dh)
    return f * L / Dh


class baseLoss:
    pass


class PipeLoss2(baseLoss):
    """Pressure loss calculations for a pipe."""
    def __init__(self, geometry):
        self.geometry = geometry

        # Initialize K and dP
        self.K = None
        self.Re = None
        self.f = None
        self._dP = None



    def evaluate(self, thermo, w):
        """Compute loss coefficient and pressure drop."""

        # Get K Factor
        self.K = pipe_K(thermo, self.geometry._L,
                        self.geometry._Dh,
                        w,
                        self.geometry._roughness)

        self.Re = Re_(thermo, self.geometry._Dh, w)
        self.f = friction_factor(self.Re,
                                    self.geometry._roughness / self.geometry._Dh)

        self._dP = -self.K*((w/self.geometry._area)**2
                            / (2 * thermo._density))

        return self._dP







class StraightLoss(baseLoss):
    """Pressure loss calculations for a pipe."""
    def __init__(self, node, geometry, dP=None):
        self.node = node  # Associate with pipe node
        self.geometry = geometry

        self.fixed_dP = dP

        # Initialize K and dP
        self.K = None
        self._dP = None

        if self.fixed_dP is not None:
            logger.info(f"Setting Loss for {self.node.name} to {dP}")
            self._dP = self.fixed_dP

    def evaluate(self, w, thermo):
        """Compute loss coefficient and pressure drop dynamically."""
        if self.fixed_dP is not None:
            return self.fixed_dP  # If dP is set, return it directly
        else:

            # Get K Factor
            self.K = pipe_K(thermo, self.geometry._L,
                            self.geometry._D,
                            w,
                            self.geometry._roughness)
            
            self.Re = Re_(thermo, self.geometry._D, w)
            self.f = friction_factor(self.Re,
                                     self.geometry._roughness / self.geometry._D)

            self._dP = -self.K*((w/self.geometry._area)**2
                                / (2 * thermo._density))

            return self._dP



class BendLoss(baseLoss):

    def __init__(self, node, geometry):
        self.node = node
        self.geometry = geometry

        self.K = None
        self._dP = None

    def evaluate(self, w, thermo):

        self.Re = Re_(thermo, self.geometry._D, w)

        self.Kbend = KBend.evaluate(self.Re, self.geometry.RD, self.geometry.angle)

        self.Kpipe = pipe_K(thermo, self.geometry._L,
                            self.geometry._D,
                            w,
                            self.geometry._roughness)

        self.K = self.Kpipe + self.Kbend

        self._dP = -self.K*((w/self.geometry._area)**2
                            / (2 * thermo._density))
        
        return self._dP

def load_physics(physics):



    if isinstance(physics, dict):

        if physics['type'] == 'ConstantPhysics':
            return ConstantPhysics(**physics['inputs'])
        elif physics['type'] == 'CustomPhysics':
            from pdb import set_trace
            set_trace()
        elif physics['type'] == 'PipePhysics':
            return PipePhysics(physics['inputs']['type'])


        from pdb import set_trace
        set_trace()

    elif isinstance(physics, BasePhysicsModel):
        return physics


class BasePhysicsModel:

    def evaluate(self, w, US, DS):
        pass


    def initialize(self, node):
        self.node = node
        self.geometry = node.geometry
        

class ConstantPhysics(BasePhysicsModel):

    def __init__(self, dP=0, dH=0, node=None):
        self.dP = dP
        self.dH     = dH

        if node is not None:
            self.initialize(node)

    def evaluate(self, w, US, DS):
        return {'dP': self.dP, 'dH': self.dH}
    
    @property
    def _state(self):
        return {'type':'ConstantPhysics',
                'inputs': {'dP': self.dP, 'dH': self.dH}}

class CustomPhysics(BasePhysicsModel):

    def __init__(self, node=None, dP_func=None, dH_func=None):
        self.node = node
        self.dP_func = dP_func
        self.dH_func = dH_func

        if node is not None:
            self.initialize(node)

    def evaluate(self, w, US, DS):
        return {'dP': self.dP_func(w, US, DS),
                'dH': 0}

class PipePhysics(BasePhysicsModel):

    def __init__(self, type='straight', node=None):

        self.type = type

        if node is not None:
            self.initialize(node)


    def initialize(self, node):
        super().initialize(node)

        if self.type == 'straight':
            self.dP_model = StraightLoss(self.node, self.geometry)
        elif self.type == 'bend':
            self.dP_model = BendLoss(self.node, self.geometry)
        else:
            from pdb import set_trace
            set_trace()

    def evaluate(self, w, US, DS):
         
        dP = self.dP_model.evaluate(w, US)

        dP += -self.geometry._dz*9.81*US._density
    
        return {'dP': dP, 'dH': 0}

class PipeLoss:

    _units = {'dP': 'PRESSURE'}

    def __init__(self, node, geometry, dP=None, loss_type='straight'):
        self.node = node
        self.geometry = geometry
        self.fixed_dP = dP
    
        if loss_type == 'straight':
            self.loss_model = StraightLoss(self.node, self.geometry)
        elif loss_type == 'bend':
            self.loss_model = BendLoss(self.node, self.geometry)
        else:
            from pdb import set_trace
            set_trace()

    def evaluate(self, w, thermo):

        if self.fixed_dP is not None:
            if isinstance(self.fixed_dP, float):
                return self.fixed_dP
            elif callable(self.fixed_dP):
                return self.fixed_dP(self.node)
            else:
                from pdb import set_trace
                set_trace()

        self._dP = self.loss_model.evaluate(w, thermo)

        # I should move this to somewhere else but putting here for now
        self._dP += -self.geometry._dz*9.81*thermo._density
        return self._dP


class PipeLossModel:
    def __init__(self, node, geometry):
        self.node = node
        self.geometry = geometry
        self.losses = []

        for geometry in self.geometry:
            if geometry._type == 'Cylinder':
                self.losses.append(PipeLoss(self.node, geometry))
            elif geometry._type == 'CylinderBend':
                self.losses.append(BendLoss(self.node, geometry))
            else:
                from pdb import set_trace
                set_trace()

    def evaluate(self, w, thermo):

        # Get Reynolds Number
        # G
        #self.Re = Re_(thermo, self.geometry._D, w)

        # Get Friction Factor
        #f = friction_factor(Re, self.geometry._roughness / self.geometry._D)

        # Get Loss Coefficient
        #K = f * self.geometry._L / self.geometry._D

        # Evaluate Losses
        for loss in self.losses:
            loss.evaluate(w, thermo)

        from pdb import set_trace
        set_trace()
        self._dP = sum([loss._dP for loss in self.losses])

        return self._dP

