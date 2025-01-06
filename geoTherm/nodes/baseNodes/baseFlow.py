#from .baseFlowResistor import baseFlowResistor
from .baseNode import Node
from ...logger import logger
from ...units import addQuantityProperty, inputParser
import numpy as np

@addQuantityProperty
class baseFlow(Node):

    _displayVars = ['w', 'dP']
    _units = {'w': 'MASSFLOW', 'dP': 'PRESSURE'}    
    _bounds = [1e-5, 1e5]

    def __init__(self, name, US, DS):

        self.name = name
        self.US = US
        self.DS = DS

        # Initialize flow variable
        self._w = 0

    def initialize(self, model):
        """
        Initialize the node with the model, setting up connections to upstream
        and downstream nodes.

        Args:
            model: The model containing the node map and other nodes.
        """
        
        super().initialize(model)
        
        node_map = self.model.node_map[self.name]

        # Validate node connections
        if len(node_map['US']) != 1 or len(node_map['DS']) != 1:
            logger.critical(
                f"Flow Node {self.name} must have exactly one "
                f"upstream and one downstream node. Current connections: "
                f"US: {node_map['US']}, DS: {node_map['DS']}"
            )

        # Validate the node map strings are generated correctly
        if node_map['US'][0] != self.US or node_map['DS'][0] != self.DS:
            logger.critical(
                f"Node mapping mismatch for '{self.name}': Expected "
                f"US: {self.US}, DS: {self.DS}, but got "
                f"US: {node_map['US'][0]}, DS: {node_map['DS'][0]}"
            )

        if node_map['hot'] or node_map['cool']:
            logger.critical(
                f"Flow Node {self.name} cannot have any hot or cool "
                "nodes attached to.\nCurrent connections:\n"
                f"\bhot: {node_map['hot']}\ncool: {node_map['cool']}"
            )

        # Attach references to upstream and downstream nodes
        self.US_node = model.nodes[node_map['US'][0]]
        self.DS_node = model.nodes[node_map['DS'][0]]    

    def thermostates(self):
        """
        Get the inlet and outlet thermo states based on Pressure
        """

        # Handle Backflow
        if self.US_node.thermo._P >= self.DS_node.thermo._P:
            US = self.US_node.thermo
            DS = self.DS_node.thermo
            flow_sign = 1
        else:
            US = self.DS_node.thermo
            DS = self.US_node.thermo
            flow_sign = -1


        return US, DS, flow_sign

    def get_DS_state(self, US, w):

        # Get Downstream Node
        # w must be positive

        # Get the Outlet State
        DS_state = self.get_outlet_state(US, w)
        
        return DS_state

    def get_US_state(self, DS, w):

        # Get Upstream Node
        # w must be positive

        # Get the Inlet State
        US_state = self.get_inlet_state(DS, w)

        return US_state
    
    @property
    def _dH(self):
        return 0


@addQuantityProperty
class baseInertantFlow(baseFlow):
    """Base class for a flow node that calculates flow in between stations."""

    _displayVars = ['w', 'dP']
    _units = {'w': 'MASSFLOW', 'dP': 'PRESSURE'}    
    _bounds = [1e-5, 1e5]

    def __init__(self, name, US, DS, w:'MASSFLOW'=0):

        self.name = name
        self.US = US
        self.DS = DS
        self._w = w
        self.penalty = False

    def update_state(self, x):
        """
        Update the state of the node.

        Args:
            x (float): New state value to set.
        """

        if self._bounds[0] < x[0] < self._bounds[1]:
            self._w = x[0]
            self.penalty = False
        else:
            logger.warn(f"Can't update state for {self.name} to: "
                        f"{x[0]}, it's outside the bounds: {self._bounds}")
            if x < self._bounds[0]:
                self.penalty = (self._bounds[0] - x + 10)*1e8
                self._w = self._bounds[0]
            elif x > self._bounds[1]:
                self.penalty = (x - self._bounds[1] - 10)*1e8
                self._w = self._bounds[1]           

    @property
    def x(self):
        """
        Mass flow rate state.

        Returns:
            np.array: Mass flow rate (kg/s).
        """

        return np.array([self._w])

    @property
    def xdot(self):
        if self.penalty is not False:
            return np.array([self.penalty])

        #if self._w >= 0:
        #    US, DS = self.US_node.thermo, self.DS_node.thermo
        #else:
        #    US, DS = self.DS_node.thermo, self.US_node.thermo
        US, DS = self.thermostates()

        DS_target = self.get_outlet_state(US, self._w)

        return np.array([DS._P-DS_target['P']])*np.sign(self._w)

    @property
    def _W(self):
        return 0

    def thermostates(self):
        """
        Get the inlet and outlet thermo states based on flow
        """

        if self._w >= 0:
            return self.US_node.thermo, self.DS_node.thermo, 1
        else:
            return self.DS_node.thermo, self.US_node.thermo, -1


    def _set_flow(self, w):
        """
        Set the flow rate and get outlet state.

        Args:
            w (float): Flow rate.

        Returns:
            tuple: Downstream node name and downstream state.
        """

        self._w = w

        return False
        # Get Downstream Node
        #if self._w >= 0:
        #    dsNode = self.model.node_map[self.name]['DS'][0]
        #else:
        #    dsNode = self.model.node_map[self.name]['US'][0]

        # Get the Outlet State
        #dsState = self.get_outlet_state()

        #if dsState:
            # Return the downstream node and downstream state
        #    return dsNode, dsState
        #else:
        #    return dsNode, None


    def get_DS_state(self, US, w):

        DS_state = self.get_outlet_state(US, w)

        return DS_state


    def get_US_state(self, DS, w):

        # Get the Inlet State
        US_state = self.get_inlet_state(DS, w)

        return US_state