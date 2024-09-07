from .node import Node
from .flow import flow
from ..units import inputParser, addQuantityProperty
from ..utils import dP_pipe
from ..logger import logger
import numpy as np


@addQuantityProperty
class Pipe(flow):
    pass


@addQuantityProperty
class heatedPipe(Pipe):

    @inputParser
    def __init__(self, name, US, DS, Q:'POWER', D:'LENGTH',
                 L: 'LENGTH',
                 w: 'MASSFLOW',
                 roughness:'LENGTH'=1e-5,
                 dP:'PRESSURE'= None):
        self.name = name
        self.US = US
        self.DS = DS
        self.cool = DS
        self._D = D
        self._L = L
        self._w = w
        self._roughness = roughness
        self._Q = Q

        if dP is None:
            self.update_dP = True
        else:
            self.update_dP = False
            self._dP = dP

    def get_outlet_state(self):
        # Get US, DS Thermo
        US, DS = self._get_thermo()

        self._dP = self._get_dP(US)

        if self._w == 0:
            self._dH = 0
        else:
            self._dH = self._Q/abs(self._w)

        return {'H': US._H + self._dH, 'P': US._P - self._dP}


@addQuantityProperty
class heatedPipe2(Pipe):

    @inputParser
    def __init__(self, name, US, DS, Q:'POWER', D:'LENGTH',
                 L: 'LENGTH',
                 w: 'MASSFLOW',
                 roughness:'LENGTH'=1e-5):
        self.name = name
        self.US = US
        self.DS = DS
        self._D = D
        self._L = L
        self._w = w
        self._roughness = roughness
        self._Q = Q

    def get_outlet_state(self):
        # Get US, DS Thermo
        US, DS = self._get_thermo()

        self._dP = self._get_dP(US)

        if self._w == 0:
            self._dH = 0
        else:
            self._dH = self._Q/abs(self._w)

        return {'H': US._H + self._dH, 'P': US._P - self._dP}


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
