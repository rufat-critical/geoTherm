from ..utils import Re_
import numpy as np
from ..thermostate import thermo, addThermoAttributes
from ..units import inputParser, addQuantityProperty
from ..logger import logger
from .node import Node
from .baseNodes.baseThermo import baseThermo


class Station(baseThermo):
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

        # This is the current state
        self.state = np.array([self.thermo._H, self.thermo._P])
        
        #self._x = np.array([self.thermo._H, self.thermo._P])
        # This is the latest state that didn't make thermostate
        # complain
        self._state = np.array([self.thermo._H, self.thermo._P])
        #self.__x = np.array([self.thermo._H, self.thermo._P])

    def update_thermo(self, state):
        error = self._update_thermo(state)

        if not error:
            self._reinit_state_vars()

        return error

    def _reinit_state_vars(self):
        self.state = np.array([self.thermo._H, self.thermo._P])

    @property
    def x(self) -> np.ndarray:
        return self.state

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
        wNet, Hnet, Wnet, Qnet = self.flux

        return np.array([(Hnet + Wnet + Qnet), wNet])

    def update_state(self, x):
        """
        Update the state of the station node based on the provided state
        vector.

        Args:
            x (np.ndarray): State vector [density, internal energy].
        """

        # Store the state
        self.state = x
        try:
            #self.thermo._DU = x[0], x[1]
            self.thermo._HP = x[0], x[1]
            self.penalty = False
            # If thermo did not complain then update __x
            self._state = x
        except Exception as e:
            logger.warn(f'Failed to update thermo state for {self.name} to:'
                        f'H, P: {x}, resetting to H0, P0: {self._state}. '
                        f'Error: {e}')
            # If thermo complained then revert it back to __x state
            #self.thermo._DU = x0
            self.thermo._HP = self._state
            # Point penalty in direction of working state
            self.penalty = (self._state - x) * 1e5


class TStation(Station):

    def initialize(self, model):
        super().initialize(model)

        self.state = np.array([self.thermo._P])
        self._state = np.array([self.thermo._T, self.thermo._P])

    @property
    def xdot(self):

        wNet, _, _, _ = self.flux

        return np.array([wNet])

    def update_state(self, x):

        self.state = x
        try:
            self.thermo._TP = self.thermo._T, x[0]
            self.penalty = False
            self._state = np.array([self.thermo._T, x[0]])
        except:
            logger.warn(f'Failed to update thermo state for {self.name} to:'
                        f'T, P: {x}, resetting to T0, P0: {self._state}. ')
            self.thermo._TP = self._state
            self.penalty = (self._state - x)*1e5        

    def update_thermo(self, state):

        state = {'T': self.thermo._T, 'P': state['P']}
        error = self._update_thermo(state)


        return error

class fixedStation(Station):

    @inputParser
    def __init__(self, name, fluid,
                 P: 'PRESSURE' = None,          # noqa
                 T: 'TEMPERATURE' = None,       # noqa
                 H: 'SPECIFICENTHALPY' = None,  # noqa
                 S: 'SPECIFICENTROPY' = None,   # noqa
                 Q=None,
                 state=None,
                 fixed_state='T'):               # noqa
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
        self.fixed_state = fixed_state

        from pdb import set_trace
        set_trace()

        self._mass = self.thermo._density*self._volume
        self._U = self.thermo._U*self._mass

        #self._x = np.array([self._mass, self._U])   
        



class PStation(Station):

    
    def initialize(self, model):
        super().initialize(model)

        self._reinit_state_vars()
        self.__x = np.array([self.thermo._H, self.thermo._P])
        

    def _reinit_state_vars(self):
        self._x = np.array([self.thermo._P])
    


    @property
    def xdot(self):

        if self.penalty is not False:
            return self.penalty
        wNet, _, _, _ = self.flux

        return np.array([wNet*1e6])
    

    @property
    def x(self):
        return self._x
    
    def update_energy(self):

        win = 0
        H= 0
        for node in self.US_nodes:
            if node._w > 0:
                win += node._w
                H += node.US_node.thermo._H*node._w

        for node in self.DS_nodes:
            if node._w < 0:
                win += -node._w
                H+= node.DS_node.thermo._H*(-node._w)  


        if H == 0:
            Hmix = self.thermo._H
        else:
            Hmix = H/(win+1e-10)


        return Hmix
        try:
            self.thermo._HP = Hmix, self.thermo._P
        except:
            print('Failed')
            pass


    def update_state(self, x):

        self._x = x[0]
        
        Hmix = self.update_energy()

        try:
            self.thermo._HP = Hmix, x[0]
        except:
            from pdb import set_trace
            set_trace()



        try:
            self.thermo._HP = Hmix, x[0]
            #self.thermo._HP = self.thermo._H, x[0]
            self.penalty = False
            self.__x = np.array([self.thermo._H, x[0]])
        except:
            self.thermo._HP = self.__x
            print('penalty triggered: ')
            self.penalty = np.array([(self.__x[0] - x[0])*1e5])



        from pdb import set_trace
        #set_trace()


@addQuantityProperty
class Volume(Station):
    """
    Volume Node where the thermodynamic state is defined via mass and energy
    state properties.
    """

    _displayVars = ['P', 'T', 'H', 'volume', 'phase']

    _units = {'volume': 'VOLUME', 'mass': 'MASS', 'U': 'ENERGY',
              'w': 'MASSFLOW'}

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

        self._mass = self.thermo._density*self._volume
        self._U = self.thermo._U*self._mass

        #self._x = np.array([self._mass, self._U])

    def initialize(self, model):
        #"""
        #Initialize the Volume node.

        #Args:
        #    model: The model instance to which this node belongs.
        #"""

        # Initialize using the Stations's initialization process
        super().initialize(model)

        self._reinit_state_vars()

    def _reinit_state_vars(self):
        """
        Reinitialize the state variables for mass and internal energy.
        """
        self._mass = self.thermo._density*self._volume
        self._U = self.thermo._U*self._mass
        
        self._x = np.array([self._mass, self._U])


    def update_state(self, x):
        """
        Update the state of the Volume node based on mass and internal energy.

        Args:
            x (np.ndarray): State vector [mass, internal energy].
        """

        # Save the initial state for potential rollback
        #x0 = self.x

        self._x = x

        mass0, U0 = self._mass, self._U
        try:
            # Update the state with new values
            self._mass, self._U = x
            # Try to update
            self.thermo._DU = self._mass/self._volume, self._U/self._mass
            self.penalty = False
        except Exception:
            x0 = np.array([mass0, U0])
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

        x0 = self.x
        mass0, U0 = self._mass, self._U
        try:
            # Attempt to update the thermodynamic state
            self.thermo._update_state(state)
            self._reinit_state_vars()
            return False
        except Exception as e:
            self.update_state(x0)
            # If an error occurs, trigger debugging and return True
            logger.error(f"Failed to update thermo state to state: {state} "
                         f"for {self.name}:\n {e}")
            return True

    @property
    def x(self) -> np.ndarray:
        """
        Return the state vector for the node (mass and internal energy).

        Returns:
            np.ndarray: State vector [mass, internal energy].
        """
        return self._x
        return np.array([self._mass, self._U])

    @property
    def xdot(self):

        if self.penalty is not False:
            return self.penalty

        # Calculate fluxes (net mass and energy flow)
        wNet, Hnet, Wnet, Qnet = self.flux

        return np.array([wNet, (Hnet + Wnet + Qnet)])


@addQuantityProperty
class lumpedMass(Volume):
    "Lumped Mass with Constant Density"
    _bounds = [50, 3500]

    def initialize(self, model):
        super().initialize(model)

        self._reinit_state_vars()

    def _reinit_state_vars(self):
        self._x = np.array([self.thermo._T])

    def update_thermo(self, state):

        x0 = self.x

        error = self._update_thermo(state)

        if error:
            # Update state back to OG state
            self.update_state(x0)

        self._reinit_state_vars()

        return error

    @property
    def x(self):
        return self._x

    def update_state(self, x):

        T0 = self.thermo._T
        self._x = x
        state0 = {'D': self.thermo._density,
                  'T': self.thermo._T}

        try:
            self.thermo._TD = x[0], self.thermo._density
            self.penalty = False
        except Exception:
            # If thermo fails to update, log it and revert to the initial state
            logger.warn(f"Failed to update thermostate for {self.name} to:\n"
                        f"T:{x[0]}, resetting to T0: {T0}")

            self.thermo._TD = T0, self.thermo._density
            self.penalty = (T0 - x)*1e5            

    @property
    def xdot(self):

        if self.penalty is not False:
            return self.penalty

        _, _, _, Qnet = self.model.get_flux(self)

        return np.array([Qnet])

@addThermoAttributes
class flowVol(Node):
    # Used in LMTD Calcs

    _displayVars = ['P', 'T']
    _units = {'flowU': 'VELOCITY', 'Per': 'LENGTH'}

    @inputParser
    def __init__(self, name, fluid,
                 P: 'PRESSURE'=None,            # noqa
                 T: 'TEMPERATURE'=None,         # noqa
                 H: 'SPECIFICENTHALPY'=None,    # noqa
                 S: 'SPECIFICENTROPY'=None,     # noqa
                 Q=None,
                 A: 'AREA'=None,                # noqa
                 Per: 'LENGTH'=None,            # noqa
                 w: 'MASSFLOW'=None,            # noqa
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
            state = {'P': P, 'T': T, 'H': H, 'S': S, 'Q': Q}
            # Trim the state by removing entries with None Variables
            state = {var: val for var, val in state.items() if val is not None}

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
            self.thermo.update_state(state)
            return False
        except:
            return True

    @property
    def Re(self):
        # Calculate Hydraulic Diameter
        Dh = 4*self._A/self._Per
        return Re_(thermo, Dh, self._w)

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
        
        node_map = model.node_map[self.name]

        # Get Upstream and downstream nodes
        self.US_flow_node = model.nodes[node_map['US'][0]]
        self.DS_flow_node = model.nodes[node_map['DS'][0]]

        self._w = (self.US_flow_node._w
                   + self.DS_flow_node._w)/2

        # Get the Upstream and Downstream volume nodes
        self.US_vol = model[self.US_flow_node.US]
        self.DS_vol = model[self.DS_flow_node.DS]

