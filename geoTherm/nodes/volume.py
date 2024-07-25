from .station import Station
from ..thermostate import thermo, addThermoAttributes
from ..units import inputParser
from .surfaces import Wall
from .baseClasses import Node
from ..utils import Re_calc
import numpy as np

class Volume(Station):
    """ Volume Node where the thermodynamic state is defined """

    _displayVars = ['P', 'T', 'H', 'phase']

    stateVars = ['thermo._density', 'thermo._u']

    @inputParser
    def __init__(self, name, fluid,
                 P:'PRESSURE'=None,
                 T:'TEMPERATURE'=None,
                 H:'SPECIFICENTHALPY'=None,
                 S:'SPECIFICENTROPY'=None,
                 Q=None,
                 A=None,
                 L=None,
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
        if isinstance(fluid, str):
            # Generate the thermo object
            self.thermo = thermo(fluid, state=state)
        elif isinstance(fluid, thermo):
            # If thermo object is specified for fluid then use that
            # thermo object for calcs
            self.thermo = fluid

            if state is not None:
                self.thermo._updateState(state)

        # Store area and Length
        self._A = A
        self._L = L

        # Penalty in case an out of bounds state is specified
        # i.e if step length is too large in Fsolve then it 
        # can sometimes try to update thermo state to negative 
        # density, the penalty helps tell fsolve to step back
        self.penalty = None

    @property
    def _w(self):
        # Calculate mass flow thru the volume
        # Average mass flow from node objects
        
        # Get the nodeMap
        nMap = self.model.nodeMap[self.name]

        # Get the average flow from inlet/outlet flowNodes
        wAvg= 0
        for name in nMap['US']:
            wAvg+= self.model.nodes[name]._w
        for name in nMap['DS']:
            wAvg+= self.model.nodes[name]._w

        return wAvg/2


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
                self.thermo._updateState(state)

        self._w = w
        self._A = A
        self._Per = Per
        # Get Perimeter
        if self._Per is None:
            self._Per = np.sqrt(4*np.pi*self._A)

    def updateThermo(self, state):
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
                self.thermo._updateState(state)

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

