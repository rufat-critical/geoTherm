from .node import Node
from .geometry import Cylinder, GeometryProperties
from .baseNodes.baseFlow import baseInertantFlow
from ..units import inputParser, addQuantityProperty
from ..utils import dP_pipe
from ..logger import logger
import numpy as np
from ..resistance_models.flow import PipeLoss


@addQuantityProperty
class Pipe(baseInertantFlow, GeometryProperties):
    _displayVars = ['w', 'dP', 'geometry']
    _units = {**GeometryProperties._units, **{
        'w': 'MASSFLOW', 'U': 'VELOCITY'
    }
    }
    #_units = {'D': 'LENGTH', 'L': 'LENGTH', 'w': 'MASSFLOW',
    #          'roughness': 'LENGTH', 'dP': 'PRESSURE',
    #          'Q': 'POWER', 'dH': 'SPECIFICENERGY',
    #          'U': 'VELOCITY'}

    _bounds = [-1e5, 1e5]

    @inputParser
    def __init__(self, name, US, DS,
                 L=None,
                 D=None,
                 roughness=1e-4,
                 t=0,
                 w:'MASSFLOW'=0,
                 dP:'PRESSURE'=None):


        #super().__init__(name, US, DS)
        self.name = name
        self.US = US
        self.DS = DS
        self._w = w
        self.penalty = False

        # Initialize Geometry
        if L is not None and D is not None:
            self.geometry = Cylinder(D, L, t, roughness)
            self._Z = self.geometry._L/self.geometry._area**2
        else:
            self.geometry = None
            self._Z = 1
        # Initialize Loss Model
        self.loss = PipeLoss(self, self.geometry, dP=dP)


    @property
    def _dP(self):
        US, _, _ = self.thermostates()
        return self.loss.evaluate(self._w, US)


    @property
    def _U(self):
        # Incompressible velocity
        US, _, _ = self.thermostates()
        return self._w/(US._density*self.geometry._area)

    @property
    def _cdA(self):
        # Incompressible cdA
        US, DS, _ = self.thermostates()
        dP = np.abs(US.thermo._P - DS.thermo._P)
        return self._w/np.sqrt(2*US._density*dP)

    def get_outlet_state(self, US, w):

        # Evaluate loss
        dP = self.loss.evaluate(np.abs(w), US)

        return {'H': US._H,
                'P': US._P + dP}

    @property
    def xdot2(self):
        if self.penalty is not False:
            return np.array([self.penalty])

        #if self._w >= 0:
        #    US, DS = self.US_node.thermo, self.DS_node.thermo
        #else:
        #    US, DS = self.DS_node.thermo, self.US_node.thermo
        US, DS, _ = self.thermostates()

        DS_target = self.get_outlet_state(US, self._w)

        return np.array([DS_target['P'] - DS._P])*np.sign(self._w)        


class LumpedPipe(Node):

    def __init__(self, name):
        pass

class discretePipe(Node):

    def __init__(self, name):
        pass

    # Solve for pressure drop with pressure drop

    # Check for choked flow condition
    # sqrt(gam*R*T)

# ESTIMATE Q LOSS FOR PIPE SECTION


#class PipeBend(Node):

