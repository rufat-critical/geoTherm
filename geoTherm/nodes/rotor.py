from .node import Node
from ..units import addQuantityProperty, inputParser
import numpy as np

@addQuantityProperty
class Rotor(Node):
    """ Rotor Node that connects spinny components"""
    """ Speed is Constant """

    _displayVars = ['N', 'loads']
    _units = {'N': 'ROTATIONSPEED'}

    @inputParser
    def __init__(self, name, N:'ROTATIONSPEED'):
        self.name = name
        self._N = N
        self.loads = []

    @property
    def Nrad(self):
        return self._N*2*np.pi/60

    def initialize(self, model):

        self.loads = []
        self.model = model
        # Find the components this rotor is attached to
        for name, node in model.nodes.items():
            if hasattr(node, 'rotor'):
                if self.name == node.rotor:
                    self.loads.append(name)

class Rotor_Ns(Node):

    def __init__(self, name, Ns, turbo_node):
        # N is calculated to satisfy specific speed

        pass