from .node import Node
import numpy as np

class Rotor(Node):
    """ Rotor Node that connects spinny components"""
    """ Speed is Constant """

    _displayVars = ['N', 'loads']

    def __init__(self, name, N):
        self.name = name
        self.N = N
        self.loads = []

    @ property
    def omega(self):
        return self.N*2*np.pi/60

    def initialize(self, model):

        self.loads = []
        self.model = model
        # Find the components this rotor is attached to
        for name, node in model.nodes.items():
            if hasattr(node, 'rotor'):
                if self.name == node.rotor:
                    self.loads.append(name)
