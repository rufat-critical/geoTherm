from .node import Node
import numpy as np
from ..logger import logger
from ..units import inputParser, addQuantityProperty
from ..thermostate import thermo, addThermoAttributes


class flowNode(Node):
    """Base class for a flow node that calculates flow in between stations."""

    def initialize(self, model):
        """
        Initialize the node with the model.

        Args:
            model: The model to initialize with.
        """

        # Attach reference to upstream and downstream nodes
        self.US_node = model.nodes[self.US]
        self.DS_node = model.nodes[self.DS]

        # Add w attribute if not defined
        if not hasattr(self, '_w'):
            self._w = 0

        # Initialize dP
        if not hasattr(self, '_dP'):
            self._dP = 0

        # Initialize dH
        if not hasattr(self, '_dH'):
            self._dH = 0

        # Store Upstream and Downstream condi

        # Do rest of initialization
        return super().initialize(model)

    def _getThermo(self):
        """
        Get the inlet and outlet thermo states based on flow direction.
        """

        # Handle Backflow
        if self.US_node.thermo._P >= self.DS_node.thermo._P:
            US = self.US_node.thermo
            DS = self.DS_node.thermo
        else:
            US = self.DS_node.thermo
            DS = self.US_node.thermo

        return US, DS

    def _setFlow(self, w):
        """
        Set the flow rate and get outlet state.

        Args:
            w (float): Flow rate.

        Returns:
            tuple: Downstream node name and downstream state.
        """

        self._w = w

        # Get Downstream Node
        if self._w >= 0:
            dsNode = self.model.nodeMap[self.name]['DS'][0]
        else:
            dsNode = self.model.nodeMap[self.name]['US'][0]

        # Get the Outlet State
        dsState = self.getOutletState()

        # Return the downstream node and downstream state
        return dsNode, dsState

    def getOutletState(self):
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


class fixedFlowNode:
    # Node for classes where flow is fixed, fixedFlow Resistor, Pump, Turb
    pass


class statefulFlowNode(flowNode):
    """
    Node class with mass flow as state variable. This needs to be inherited
    and not standalone
    """

    # Variable Bounds
    _bounds = [-1e5, 1e5]

    def get_thermostates(self):
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

        if not hasattr(self, '_W'):
            self._W = 0

        if not hasattr(self, '_Q'):
            self._Q = 0

        self.penalty = False

        return super().initialize(model)

    def evaluate(self):
        """
        Evaluate the flow node and update pressure and enthalpy differences.
        """

        # Get the target outlet state
        # This should be a dictionary in the form of:
        # {'H': Enthalpy, 'P':Pressure}
        outletState = self.getOutletState()

        US, _ = self.get_thermostates()

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

    def updateState(self, x):
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

        outletState = self.getOutletState()

        # Handle reverse flow case
        US, DS = self.get_thermostates()

        # Get Difference in DS property and outletState property
        # via list comprehension
        return np.array([(outletState['P'] - DS._P)*np.sign(self._w)])

    def getOutletState(self):

        # Get US, DS Thermo
        US, DS = self.get_thermostates()

        # get dh and dP
        self._dH = self._get_dH()
        self._dP = self._get_dP()

        # Return outlet state
        return {'H': US._H + self._dH,
                'P': US._P + self._dP}

    def get_outlet_state(self):

        # Get US, DS Thermo
        US, DS = self.get_thermostates()

        # get dh and dP
        self._dH = self._get_dH()
        self._dP = self._get_dP()

        # Return outlet state
        return {'H': US._H + self._dH,
                'P': US._P + self._dP}


class Heat(Node):
    pass


class statefulHeatNode(Node):

    @property
    def x(self):
        return np.array([self._Q])

    def updateState(self, x):
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
        nodeMap = self.model.nodeMap[self.name]

        # Initialize neighbor connections
        self.US_neighbors = nodeMap['US']
        self.US_nodes = [self.model.nodes[name] for name in nodeMap['US']]
        self.DS_neighbors = nodeMap['DS']
        self.DS_nodes = [self.model.nodes[name] for name in nodeMap['DS']]
        self.hot_neighbors = nodeMap['hot']
        self.hot_nodes = [self.model.nodes[name] for name in nodeMap['hot']]
        self.cool_neighbors = nodeMap['cool']
        self.cool_nodes = [self.model.nodes[name] for name in nodeMap['cool']]

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
            logger.error(f"Failed to update thermo state: {e}")
            return True

    @property
    def _w_avg(self):
        """
        Calculate the average mass flow through the node.

        Returns:
            float: The average mass flow rate.
        """
        # Average mass flow from node objects

        # Get the nodeMap
        nMap = self.model.nodeMap[self.name]

        # Get the average flow from inlet/outlet flowNodes
        w_avg = sum(self.model.nodes[name]._w for name in nMap['US']) + \
            sum(self.model.nodes[name]._w for name in nMap['DS'])

        return w_avg/2
