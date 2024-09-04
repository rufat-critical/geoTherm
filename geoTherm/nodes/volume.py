from .surfaces import Wall
from .baseClasses import ThermoNode
from ..utils import Re_calc
import numpy as np
from ..thermostate import thermo, addThermoAttributes
from ..units import inputParser, addQuantityProperty
from ..logger import logger
from .node import Node


class Station(ThermoNode):
    """
    Station Node where the thermodynamic state is defined.
    """

    def initialize(self, model):
        """
        Initialize the Station node.

        Args:
            model: The model instance to which this node belongs.
        """

        super().initialize(model)

        # Penalty in case an out of bounds state is specified
        # i.e if step length is too large in Fsolve then it
        # can sometimes try to update thermo state to negative
        # density, the penalty helps tell fsolve to step back
        self.penalty = False

    @property
    def x(self) -> np.ndarray:
        return np.array([self.thermo._density, self.thermo._U])

    @property
    def xdot(self) -> np.ndarray:
        """
        Return the rate of change of the state vector.

        Returns:
            np.ndarray: Rate of change of state vector [mass flux, energy flux]
        """

        # Check if penalty is triggered
        if self.penalty is not False:
            return self.penalty

        # Calculate fluxes (net mass and energy flow)
        wNet, Hnet, Wnet, Qnet = self.model.getFlux(self)

        return np.array([wNet, Hnet + Wnet + Qnet])

    def updateState(self, x):
        """
        Update the state of the station node based on the provided state
        vector.

        Args:
            x (np.ndarray): State vector [density, internal energy].
        """
        x0 = self.x
        try:
            self.thermo._DU = x[0], x[1]
            self.penalty = False
        except Exception:
            logger.warn(f'Failed to update thermo state for {self.name} to:'
                        f'D, U: {x}, resetting to D0, U0: {x0}')
            self.thermo._DU = x0
            self.penalty = (x0 - x) * 1e5


@addQuantityProperty
class Volume(Station):
    """
    Volume Node where the thermodynamic state is defined via mass and energy 
    state properties.
    """

    _displayVars = ['P', 'T', 'H', 'volume', 'phase']

    _units = {'volume': 'VOLUME', 'mass': 'MASS', 'U': 'ENERGY', 'w': 'MASSFLOW'}

    @inputParser
    def __init__(self, name, fluid,
                 P: 'PRESSURE' = None,          # noqa
                 T: 'TEMPERATURE' = None,       # noqa
                 H: 'SPECIFICENTHALPY' = None,  # noqa
                 S: 'SPECIFICENTROPY' = None,   # noqa
                 Q=None,
                 state=None,
                 volume:'VOLUME'=1.0):               # noqa
        """
        Initialize a volume node with a given fluid and state.

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
            V (float, optional): Volume (default is 1m^3).
        """

        super().__init__(name, fluid, P, T, H, S, Q, state)
        self._volume = volume

    def initialize(self, model):
        """
        Initialize the Volume node.

        Args:
            model: The model instance to which this node belongs.
        """

        # Initialize using the Stations's initialization process
        super().initialize(model)

        self._reinit_state_vars()

    def _reinit_state_vars(self):
        """
        Reinitialize the state variables for mass and internal energy.
        """
        self._mass = self.thermo._density*self._volume
        self._U = self.thermo._U*self._mass

    def updateState(self, x):
        """
        Update the state of the Volume node based on mass and internal energy.

        Args:
            x (np.ndarray): State vector [mass, internal energy].
        """

        # Save the initial state for potential rollback
        x0 = self.x
        mass0, U0 = self._mass, self._U

        try:
            # Update the state with new values
            self._mass, self._U = x
            self.thermo._DU = self._mass/self._volume, self._U/self._mass
            self.penalty = False
        except Exception:
            # If thermo fails to update, log it and revert to the initial state
            logger.warn(f"Failed to update thermostate for {self.name} to:\n"
                        f"Mass, U:{x}, resetting to D0, U0: {x0}")

            # Reset Mass and U
            self._mass, self._U = mass0, U0
            self.thermo._DU = self._mass/self._volume, self._U/self._mass
            self.penalty = (x0 - x)*1e5

    def update_thermo(self, state):
        """
        Update the station thermodynamic state.

        Args:
            state (dict): Dictionary defining the thermodynamic state.

        Returns:
            bool: False if successful, True if an error occurs.
        """
        try:
            # Attempt to update the thermodynamic state
            self.thermo.update_state(state)
            self._reinit_state_vars()
            return False
        except Exception as e:
            # If an error occurs, trigger debugging and return True
            logger.error(f"Failed to update thermo state: {e}")
            return True

    @property
    def x(self) -> np.ndarray:
        """
        Return the state vector for the node (mass and internal energy).

        Returns:
            np.ndarray: State vector [mass, internal energy].
        """

        return np.array([self._mass, self._U])


@addThermoAttributes
class flowVol(Node):
    # Used in LMTD Calcs

    _displayVars = ['P', 'T']
    _units = {'flowU': 'VELOCITY', 'Per': 'LENGTH'}

    @inputParser
    def __init__(self, name, fluid,
                 P:'PRESSURE'=None,
                 T:'TEMPERATURE'=None,
                 H:'SPECIFICENTHALPY'=None,
                 S:'SPECIFICENTROPY'=None,
                 Q=None,
                 A:'AREA'=None,
                 Per:'LENGTH'=None,
                 w:'MASSFLOW'=None,
                 state=None):
        """ Initialize a Thermodynamic Station via fluid
        name and thermodynamic state

        Args:
            name (str): Node Name
            fluid (str): Fluid name
            P (float, optional): Pressure
            T (float, optional): Temperature
            H (float, optional): Enthalpy
            S (float, optional): Entropy
            Q (float, optional): Fluid Quality
            state (dict, Optional): Dictionary with thermodynamic state
        """

        # Store name
        self.name = name

        # Check if state or P,T,H... have been defined
        if state is not None:
            pass
        else:
            # Generate State Dictionary
            state = {'P':P, 'T': T, 'H': H, 'S': S, 'Q': Q}
            # Trim the state by removing entries with None Variables
            state = {var:val for var, val in state.items() if val is not None}
            
            if len(state) == 0:
                # If the state dict is 0 then set the state to None
                # thermostate will use default initializiation values
                state = None

        # If fluid is a string then this is the composition
        if isinstance(fluid, (str, dict)):
            # Generate the thermo object
            self.thermo = thermo(fluid, state=state)
        elif isinstance(fluid, thermo):
            # If thermo object is specified for fluid then use that
            # thermo object for calcs
            self.thermo = fluid

            if state is not None:
                self.thermo._update_state(state)

        self._w = w
        self._A = A
        self._Per = Per
        # Get Perimeter
        if self._Per is None:
            self._Per = np.sqrt(4*np.pi*self._A)

    def update_thermo(self, state):
        """ Update the station thermodynamic state
        
        Args:
            state (dict): Dictionary defining the thermodynamic state """
        
        try:
            self.thermo.updateState(state)
            return False
        except:
            return True

    @property
    def Re(self):
        # Calculate Hydraulic Diameter
        Dh = 4*self._A/self._Per
        return Re_calc(self.thermo, self._flowU, Dh)

    @property
    def _flowV(self):
        return self._w/(self.thermo._density*self._A)

    @property
    def _flowU(self):
         return self._w/(self.thermo._density*self._A)       


class hexVolume(Node):

    @inputParser
    def __init__(self, name, fluid,
                 P:'PRESSURE'=None,
                 T:'TEMPERATURE'=None,
                 H:'SPECIFICENTHALPY'=None,
                 S:'SPECIFICENTROPY'=None,
                 Q=None,
                 A:'AREA'=None,
                 Per:'LENGTH'=None,
                 w:'MASSFLOW'=None,
                 state=None):
        """ Initialize a Thermodynamic Station via fluid
        name and thermodynamic state

        Args:
            name (str): Node Name
            fluid (str): Fluid name
            P (float, optional): Pressure
            T (float, optional): Temperature
            H (float, optional): Enthalpy
            S (float, optional): Entropy
            Q (float, optional): Fluid Quality
            state (dict, Optional): Dictionary with thermodynamic state
        """

        # Store name
        self.name = name

        # Check if state or P,T,H... have been defined
        if state is not None:
            pass
        else:
            # Generate State Dictionary
            state = {'P':P, 'T': T, 'H': H, 'S': S, 'Q': Q}
            # Trim the state by removing entries with None Variables
            state = {var:val for var, val in state.items() if val is not None}
            
            if len(state) == 0:
                # If the state dict is 0 then set the state to None
                # thermostate will use default initializiation values
                state = None

        # If fluid is a string then this is the composition
        if isinstance(fluid, (str, dict)):
            # Generate the thermo object
            self.thermo = thermo(fluid, state=state)
        elif isinstance(fluid, thermo):
            # If thermo object is specified for fluid then use that
            # thermo object for calcs
            self.thermo = fluid

            if state is not None:
                self.thermo._update_state(state)

    def initialize(self, model):
        
        nodeMap = model.nodeMap[self.name]

        # Get Upstream and downstream nodes
        self.US_flow_node = model.nodes[nodeMap['US'][0]]
        self.DS_flow_node = model.nodes[nodeMap['DS'][0]]

        self._w = (self.US_flow_node._w
                   + self.DS_flow_node._w)/2

        # Get the Upstream and Downstream volume nodes
        self.US_vol = model[self.US_flow_node.US]
        self.DS_vol = model[self.DS_flow_node.DS]

