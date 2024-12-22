from .node import Node
import numpy as np
from ..logger import logger
from ..units import inputParser, addQuantityProperty
from ..thermostate import thermo, addThermoAttributes
from numbers import Number


class flowNode(Node):
    """Base class for a flow node that calculates flow in between stations."""

    def initialize(self, model):
        """
        Initialize the node with the model, setting up connections to upstream
        and downstream nodes.

        Args:
            model: The model containing the node map and other nodes.
        """
        node_map = model.node_map[self.name]

        # Validate node connections
        if len(node_map['US']) != 1 or len(node_map['DS']) != 1:
            logger.critical(
                f"Flow Node {self.name} must have exactly one upstream and "
                f"one downstream node. Current connections: "
                f"US: {node_map['US']}, DS: {node_map['DS']}"
            )

        # Validate the node map strings are generated correctly
        if node_map['US'][0] != self.US or node_map['DS'][0] != self.DS:
            logger.critical(
                f"Node mapping mismatch for {self.name}: Expected "
                f"US: {self.US}, DS: {self.DS}, but got "
                f"US: {node_map['US'][0]}, DS: {node_map['DS'][0]}"
            )

        # Attach references to upstream and downstream nodes
        self.US_node = model.nodes[node_map['US'][0]]
        self.DS_node = model.nodes[node_map['DS'][0]]
        self.hot_nodes = [model.nodes[name] for name in node_map['hot']]
        self.cool_nodes = [model.nodes[name] for name in node_map['cool']]

        # Initialize attributes if not already defined and no property
        # is defined
        if not hasattr(self, '_w'):
            self._w = 0
        if not hasattr(self, '_dP'):
            self._dP = 0
        if not hasattr(self, '_dH'):
            self._dH = 0

        # Continue with further initialization
        return super().initialize(model)

    @property
    def _W(self):
        return 0

    @property
    def _Q(self):
        # Get Q from hot and cool nodes
        Qnet = 0
        for hot in self.hot_nodes:
            Qnet += hot._Q
        for cool in self.cool_nodes:
            Qnet += -cool._Q

        return Qnet

    def _get_thermo(self):
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

    def get_outlet_state(self):
        """
        Placeholder method to be overwritten by specific flow nodes.
        """
        logger.critical(f"{self.name} of type {type(self)} is missing a "
                        "getOutletState method, geoTherm cannot run "
                        "without this!")

    def _get_dH(self, US, DS):
        """
        Placeholder method to be overwritten by specific flow nodes.
        """
        logger.critical(f"{self.name} of type {type(self)} is missing a "
                        "_get_dH(self, US, DS) method")

    def _get_dP(self, US, DS):
        """
        Placeholder method to be overwritten by specific flow nodes.
        """
        logger.critical(f"{self.name} of type {type(self)} is missing a "
                        "_get_dP(self, US, DS) method")


    def get_DS_state(self):

        # Get Downstream Node
        if self._w >= 0:
            DS_node = self.model.node_map[self.name]['DS'][0]
        else:
            DS_node = self.model.node_map[self.name]['US'][0]

        # Get the Outlet State
        DS_state = self.get_outlet_state()

        if DS_state:
            # Return the downstream node and downstream state
            return DS_node, DS_state
        else:
            return DS_node, None

    def get_US_state(self, w, DS_thermo=None):

        # Get Upstream Node
        if w >= 0:
            US_name = self.US_node.name
        else:
            US_name = self.DS_node.name

        # Get the Inlet State
        US_state = self.get_inlet_state(w, DS_thermo)

        if US_state:
            return US_name, US_state
        else:
            return US_name, None


class fixedFlowNode(flowNode):
    """
    Node for classes where flow is fixed, e.g., for fixedFlow resistors, pumps, 
    or turbines.
    Fixed flow nodes are associated with a FlowController.
    """

    def __init__(self, name, US, DS, w) -> None:

        # Component Name
        self.name = name
        # Upstream Station
        self.US = US
        # Downstream Station
        self.DS = DS
        # Flow Controller
        self.flow_controller = None

    def initialize_flow(self, w):

        if isinstance(w, Number):
            self._w = w
            self.flow_controller = None
        elif isinstance(w, (flowController, str)):
            self.flow_controller = w
        else:
            logger.critical(f"w to '{self.name}' must be a number, "
                            "flowController Object or string of the "
                            "flowController Object in the model")

    def initialize(self, model):
        # Add Flow Controller Node Reference
        if isinstance(self.flow_controller, str):
            if self.flow_controller not in model.nodes:
                logger.critical(
                    f"{self.name} references flow controller: "
                    f"{self.flow_controller} but it's not defined in the"
                    "model")
            else:
                self.flow_controller = model.nodes[self.flow_controller]

        if self.flow_controller is not None:
            self.flow_controller.add_node(self)

            if self.flow_controller.name not in model.nodes:
                logger.warn(
                    f"Flow Controller '{self.flow_controller.name}' is "
                    f"controlling '{self.name}' but is not part of the model."
                    " Adding it to model. Ignore the model initialization "
                    "failure message."
                )
                model.addNode(self.flow_controller)

        super().initialize(model)


@addQuantityProperty
class flowController(Node):
    """ Object that sets mass flow"""

    _units = {'w': 'MASSFLOW'}
    _displayVars = ['w', 'nodes']

    @inputParser
    def __init__(self, name,
                 w:'MASSFLOW',  # noqa
                 nodes=None):

        self.name = name
        self.__w = w
        self._nodes = {}

    @property
    def _w(self):
        return self.__w

    @_w.setter
    def _w(self, w):
        self.__w = w
        self.evaluate()

    def evaluate(self):

        if self._nodes:
            for _, node in self._nodes.items():
                node._w = self._w

    def add_node(self, node):
        # Add Node to internal nodes dictionary
        self._nodes.update({node.name: node})
        # Update Node Mass Flow
        node._w = self._w

    @property
    def nodes(self):
        return [node for node in self._nodes]



class statefulFlowNode(flowNode):
    """
    Node class with mass flow as state variable. This needs to be inherited
    and not standalone
    """

    # Variable Bounds
    _bounds = [-1e5, 1e5]

    def _get_thermo(self):
        """
        Get the inlet and outlet thermo states based on flow direction.
        """

        # Handle Backflow
        if self._w >= 0:
            US = self.US_node.thermo
            DS = self.DS_node.thermo
        else:
            US = self.DS_node.thermo
            DS = self.US_node.thermo

        return US, DS

    def initialize(self, model):
        """
        Initialize the node with the model.

        Args:
            model: The model to initialize with.
        """

        self.penalty = False

        return super().initialize(model)

    def evaluate(self):
        """
        Evaluate the flow node and update pressure and enthalpy differences.
        """

        # Get the target outlet state
        # This should be a dictionary in the form of:
        # {'H': Enthalpy, 'P':Pressure}
        outletState = self.get_outlet_state()

        US, _ = self._get_thermo()

        # Update dP and dH
        self._dP = (outletState['P']
                    - US._P)

        self._dH = (outletState['H']
                    - US._H)

    @property
    def x(self):
        """
        Mass flow rate state.

        Returns:
            np.array: Mass flow rate (kg/s).
        """

        return np.array([self._w])

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
            if x < self._bounds[0]:
                self.penalty = (self._bounds[0] - x + 10)*1e8
                self._w = self._bounds[0]
            elif x > self._bounds[1]:
                self.penalty = (x - self._bounds[1] - 10)*1e8
                self._w = self._bounds[1]

    @property
    def xdot(self):
        return self.error

    @property
    def error(self):
        """
        Get the error between the target outlet state and actual outlet state.

        Returns:
            np.array: Difference in downstream property and outlet state
            property.
        """

        if self.penalty is not False:
            return np.array([self.penalty])

        outletState = self.get_outlet_state()

        # Handle reverse flow case
        US, DS = self._get_thermo()

        # Get Difference in DS property and outletState property
        # via list comprehension
        return np.array([(outletState['P'] - DS._P)*np.sign(self._w)])

    def get_outlet_state(self):
        """
        Calculate the thermodynamic state at the outlet (downstream).

        Returns:
            dict: A dictionary containing the enthalpy ('H') and pressure ('P')
                  at the downstream node.
        """
        # Get US, DS Thermo
        US, DS = self._get_thermo()

        # get dh and dP
        self._dH = self._get_dH()
        self._dP = self._get_dP()

        # Return outlet state
        return {'H': US._H + self._dH,
                'P': US._P + self._dP}


    def get_inlet_state(self, w, DS=None):

        self._w = w

        if DS is None:
            if w> 0:
                DS = self.DS_node.thermo
            else:
                DS = self.US_node.thermo

        if self.update_dP is False:
            dP = self._dP

        else:
            from ..utils import dP_pipe
            dP = self._get_dP()
            #dP = dP_pipe(self.US_node.thermo,
            #        self._U,
            #        self._D,
            #        self._L,
            #        self._roughness)
            
            #dP = self._get_dP()
            #state = {'H': DS._H, 'P': DS._P-dP}

            #from pdb import set_trace
            #set_trace()      


        return {'H': DS._H,
                'P': DS._P-dP}


class Heat(Node):
    
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
        # Get the Outlet State


class statefulHeatNode(Node):

    @property
    def x(self):
        return np.array([self._Q])

    def update_state(self, x):
        self._Q = x[0]


@addThermoAttributes
@addQuantityProperty
class ThermoNode(Node):
    """
    Base thermodynamic node for handling thermodynamic states

    This class extends the base Node class to include thermodynamic properties
    and their initialization.
    """

    _displayVars = ['P', 'T', 'H', 'phase']
    _units = {'w': 'MASSFLOW'}

    @inputParser
    def __init__(self, name, fluid,
                 P: 'PRESSURE'=None,           # noqa
                 T: 'TEMPERATURE'=None,        # noqa
                 H: 'SPECIFICENTHALPY'=None,   # noqa
                 S: 'SPECIFICENTROPY'=None,    # noqa
                 Q=None,
                 state=None):
        """
        Initialize a thermodynamic node with a given fluid and state.

        Args:
            name (str): Node Name.
            fluid (str or Thermo): Fluid name or a Thermo object.
            P (float, optional): Pressure.
            T (float, optional): Temperature.
            H (float, optional): Enthalpy.
            S (float, optional): Entropy.
            Q (float, optional): Fluid Quality.
            state (dict, optional): Dictionary with a predefined
                                    thermodynamic state.

        Notes:
            If `state` is provided, it overrides individual parameters
            (P, T, H, S, Q).
        """
        self.name = name

        if state is None:
            # Generate and trim the state dictionary based on the provided
            # parameters
            state = {var: val for var, val in {'P': P, 'T': T, 'H': H,
                                               'S': S, 'Q': Q}.items()
                     if val is not None}
            state = state if state else None

        # Handle the fluid argument
        if isinstance(fluid, str):
            # If fluid is a string, create a new thermo object with it
            self.thermo = thermo(fluid, state=state)
        elif isinstance(fluid, thermo):
            # If fluid is a thermo object, use it for calculations
            self.thermo = fluid

            # Update the thermo object with the provided state, if any
            if state is not None:
                self.thermo._update_state(state)

    def initialize(self, model):
        """
        Initialize the node within the model.

        This method attaches the model to the node, initializes connections
        with neighboring nodes, and prepares the node for simulation.

        Args:
            model: The model instance to which this node belongs.
        """

        # Initialize the node using the base class method
        # (This adds a reference of the model to the thermoNode instance)
        super().initialize(model)

        # Retrieve the node map for this node from the model
        node_map = self.model.node_map[self.name]

        # Initialize neighbor connections
        self.US_neighbors = node_map['US']
        self.US_nodes = [self.model.nodes[name] for name in node_map['US']]
        self.DS_neighbors = node_map['DS']
        self.DS_nodes = [self.model.nodes[name] for name in node_map['DS']]
        self.hot_neighbors = node_map['hot']
        self.hot_nodes = [self.model.nodes[name] for name in node_map['hot']]
        self.cool_neighbors = node_map['cool']
        self.cool_nodes = [self.model.nodes[name] for name in node_map['cool']]

    def update_thermo(self, state):
        """
        Update the thermodynamic state of the node.

        Args:
            state (dict): Dictionary defining the thermodynamic state.

        Returns:
            bool: False if successful, True if an error occurs.
        """
        try:
            # Attempt to update the thermodynamic state
            self.thermo.update_state(state)
            return False
        except Exception as e:
            # If an error occurs, trigger debugging and return True
            logger.error(f"Failed to update thermo state for {self.name}: {e}")
            return True

    @property
    def _w_avg(self):
        """
        Calculate the average mass flow through the node.

        Returns:
            float: The average mass flow rate.
        """
        # Average mass flow from node objects

        # Get the node map
        node_map = self.model.node_map[self.name]

        # Get the average flow from inlet/outlet flowNodes
        w_avg = sum(self.model.nodes[name]._w for name in node_map['US']) + \
            sum(self.model.nodes[name]._w for name in node_map['DS'])

        return w_avg/2
