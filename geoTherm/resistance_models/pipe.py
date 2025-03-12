from ..utils import pipe_K
import numpy as np

class baseResistance:
    pass



class Pipe(baseResistance):

    def __init__(self, D, L, roughness, w, thermo):

        self._D = D
        self._L = L
        self._roughness = roughness
        self._w = w
        #self.K = 0

        self.thermo = thermo

    def initialize(self, node):

        # Check if flow node is attached
        self.flow_node = node


    def evaluate(self):

        self._w = self.flow_node._w

        self.thermo, _, _ = self.flow_node.thermostates()

    @property
    def _A(self):
        return np.pi / 4 * self._D**2

    @property
    def K(self):
        return pipe_K(self.thermo, self._L, self._D, self._w, self._roughness)

    def _dP(self):
        return self.K * (self._w/self._A)**2 / (2 * self.thermo._density)


class Bend(baseResistance):
    pass
