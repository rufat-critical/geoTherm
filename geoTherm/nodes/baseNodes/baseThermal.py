from .baseNode import Node
from .baseThermo import baseThermo
from ...logger import logger


class baseThermal(Node):

    def __init__(self, name, hot=None, cool=None):
        self.name = name

        self.hot = hot
        self.cool = cool

    def _set_heat(self, Q):
        self._Q = Q
        return False
    
    def get_DS_state(self):

        if self._Q > 0:
            DS_node = self.model.node_map[self.name]['cool'][0]
        else:
            DS_node = self.model.node_map[self.name]['hot'][0]

        DS_state = self.get_outlet_state()

        if DS_state:
            return DS_node, DS_state
        else:
            from pdb import set_trace
            set_trace()

    def get_US_state(self):
        if self._Q > 0:
            US_node = self.model.node_map[self.name]['hot'][0]
        else:
            US_node = self.model.node_map[self.name]['cool'][0]

        US_state = self.get_inlet_state()

        if US_state:
            return US_node, US_state
        else:
            from pdb import set_trace
            set_trace()

        return US_state, US_node

    def initialize(self, model):

        super().initialize(model)

        node_map = self.model.node_map[self.name]

        if self.cool is not None:
            if not isinstance(self.model.nodes[self.cool],
                              baseThermo):
                logger.critical(f"Thermal Component {self.name} can only "
                                "be attached to a thermo node")
        if self.hot is not None:
            if not isinstance(self.model.nodes[self.hot],
                              baseThermo):
                logger.critical(f"Thermal Component {self.name} can only "
                                "be attached to a thermo node")         

        if node_map['hot']:
            self.hot_node = model.nodes[node_map['hot'][0]]
        else:
            self.hot_node = []

        if node_map['cool']:
            self.cool_node = model.nodes[node_map['cool'][0]]
        else:
            self.cool_node = []
