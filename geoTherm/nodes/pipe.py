from .node import Node
from .baseNodes.baseFlow import baseInertantFlow
from ..units import inputParser, addQuantityProperty
from ..utils import dP_pipe
from ..logger import logger
import numpy as np


@addQuantityProperty
class Pipe(baseInertantFlow):
    _displayVars = ['w', 'dP', 'dH']
    _units = {'D': 'LENGTH', 'L': 'LENGTH', 'w': 'MASSFLOW',
              'roughness': 'LENGTH', 'dP': 'PRESSURE',
              'Q': 'POWER', 'dH': 'SPECIFICENERGY',
              'U': 'VELOCITY'}

    _bounds = [-1e5, 1e5]

    @inputParser
    def __init__(self, name, US, DS,
                 w:'MASSFLOW'=0,
                 L:'LENGTH'=None,
                 D:'LENGTH'=1,
                 roughness:'LENGTH'=1e-5,
                 dP:'PRESSURE'=None,
                 dH:'ENERGY'=0):

        self.name = name
        self.US = US
        self.DS = DS
        self._D = D
        self._L = L
        self._w = w
        self._roughness = roughness
        self.penalty = False
        self.fixed_dP = dP
        self.fixed_dH = dH

        if self.fixed_dP is None:
            if (L is None) or (D is None):
                logger.critical(f"Pipe {self.name} needs to have D and L "
                                "specified if dP is not specified!")

    @property
    def _area(self):
        return np.pi*self._D**2/4

    @property
    def _dP(self):
        US, _, _ = self.thermostates()

        if self.fixed_dP is None:
            return dP_pipe(US,
                           self._D,
                           self._L,
                           np.abs(self._w),
                           self._roughness)*self.sign(self._w)
        else:
            return self.fixed_dP

    @_dP.setter
    def _dP(self, dP):
        self.fixed_dP = dP


    @property
    def _U(self):
        # Incompressible velocity
        US, _, _ = self.thermostates()

        return self._w/(US._density*self._area)

    @property
    def _cdA(self):
        # Incompressible cdA
        US, DS, _ = self.thermostates()
        dP = np.abs(US.thermo._P - DS.thermo._P)
        return self._w/np.sqrt(2*US._density*dP)

    def get_outlet_state(self, US, w):

        if self.fixed_dP is not None:
            dP = self.fixed_dP
        else:
            dP = dP_pipe(US,
                         self._D,
                         self._L,
                         np.abs(w),
                         self._roughness)

        return {'H': US._H,
                'P': US._P + dP}

    @property
    def xdot(self):
        if self.penalty is not False:
            return np.array([self.penalty])

        #if self._w >= 0:
        #    US, DS = self.US_node.thermo, self.DS_node.thermo
        #else:
        #    US, DS = self.DS_node.thermo, self.US_node.thermo
        US, DS, _ = self.thermostates()

        DS_target = self.get_outlet_state(US, self._w)

        return np.array([DS_target['P'] - DS._P])*np.sign(self._w)        

    @property
    def _dH(self):

        if self.fixed_dH is not None:
            return self.fixed_dH
        else:
            US, DS, _ = self.thermostates()

            return DS._H - US._H

    @_dH.setter
    def _dH(self, dH):
        self.fixed_dH = dH

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
