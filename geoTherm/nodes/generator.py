import numpy as np
from .baseNodes.baseNode import Node
from geoTherm.common import logger, addQuantityProperty
from .rotor import Rotor


@addQuantityProperty
class Generator(Node):

    _units = {
        'Welectric': 'POWER',
    }

    _displayVars = ['Welectric', 'eta', 'drag_loss', 'gearbox_loss', 'generator_loss']

    def __init__(self, name, rotor,
                 drag_loss,
                 gearbox_loss,
                 generator_loss):
        self.name = name
        self.rotor = rotor
        self.drag_loss = drag_loss
        self.gearbox_loss = gearbox_loss
        self.generator_loss = generator_loss

    def initialize(self, model):
        super().initialize(model)

        self.rotorNode = self.model.nodes[self.rotor]

        if not isinstance(self.rotorNode, Rotor):
            logger.critical(f"Rotor {self.rotor} is not a baseRotor. "
                            "The Generator node must be connected to a "
                            "baseTurbine node.")

    @property
    def eta(self):
        return (1-self.drag_loss)*(1-self.gearbox_loss)*(1-self.generator_loss)

    @property
    def _Welectric(self):
        total_W = 0
        for load_name in self.rotorNode.loads:
            # Skip the generator itself to avoid self-reference
            if load_name == self.name:
                continue
            
            load_node = self.model.nodes[load_name]
            if hasattr(load_node, '_W'):
                total_W += load_node._W
        
        return total_W * self.eta
