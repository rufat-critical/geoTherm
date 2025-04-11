from .baseNode import Node
from geoTherm.logger import logger
from geoTherm.units import inputParser, addQuantityProperty
from geoTherm.thermostate import thermo, addThermoAttributes

@addThermoAttributes
@addQuantityProperty
class baseThermo(Node):
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
        self.name = name
        self.model = None

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

    def _update_thermo(self, state):
        """
        Update the thermodynamic state of the node.

        Args:
            state (dict): Dictionary defining the thermodynamic state.

        Returns:
            bool: False if successful, True if an error occurs.
        """

        HP0 = self.thermo._HP
        try:
            # Attempt to update the thermodynamic state
            self.thermo._update_state(state)
            return False
        except Exception as e:
            # If an error occurs, trigger debugging and return True
            logger.error(f"Failed to update thermo state for {self.name} to: "
                         f"{state}\nThermo error:{e}")

            self.thermo._HP = HP0
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

    @property
    def flux(self):
        if self.model is None:
            logger.critical(f"thermoNode '{self.name} must be asssociated with"
                            " a geoTherm model before flux can be calculated")

        return self.model.get_flux(self)

    @property
    def influx(self):
        if self.model is None:
            logger.critical(f"thermoNode '{self.name} must be asssociated with"
                            " a geoTherm model before flux can be calculated")

        return self.model.get_in_flux(self)

    @property
    def outflux(self):
        if self.model is None:
            logger.critical(f"thermoNode '{self.name} must be asssociated with"
                            " a geoTherm model before flux can be calculated")

        return self.model.get_out_flux(self)
