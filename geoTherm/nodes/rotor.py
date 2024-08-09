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

    @property
    def Nrad(self):
        return self.N*2*np.pi/60

    def evaluate(self):
        if 'Turb' in self.model.nodes:
            self.N = self.model.nodes['Turb'].N

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