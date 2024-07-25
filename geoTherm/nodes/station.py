from ..thermostate import thermo, addThermoAttributes
from ..units import inputParser
from ..logger import logger
from .node import Node
import numpy as np


@addThermoAttributes
class Station(Node):
    """ Station Node where there the thermodynamic state is defined"""

    _displayVars = ['P', 'T', 'H', 'phase']

    stateVars = ['thermo._density', 'thermo._u']

    @inputParser
    def __init__(self, name, fluid,
                 P:'PRESSURE'=None,
                 T:'TEMPERATURE'=None,
                 H:'SPECIFICENTHALPY'=None,
                 S:'SPECIFICENTROPY'=None,
                 Q=None,
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

        # Penalty in case an out of bounds state is specified
        # i.e if step length is too large in Fsolve then it 
        # can sometimes try to update thermo state to negative 
        # density, the penalty helps tell fsolve to step back
        self.penalty = None

    @property
    def x(self):
        return np.array([self.thermo._density,
                         self.thermo._U])

    @property    
    def error(self):
        # Get Fluxes

        wNet, Hnet, Wnet, Qnet = self.model.getFlux(self)

        if self.penalty is not None:
            return self.penalty

        return np.array([wNet, Hnet + Wnet + Qnet])


    def updateState(self, x):

        # Get the initial state
        X0 = self.x
        try:
            self.thermo._DU = x[0], x[1]
            self.penalty = None
        except:
            msg = f'Failed to update thermostate for {self.name} to:\n'
            msg += f"D, U:{x}, resetting to D0, U0: {X0}"
            self.thermo._DU = X0
            self.penalty = (X0 - x)*1e5 

    def updateThermo(self, state):
        """ Update the station thermodynamic state
        
        Args:
            state (dict): Dictionary defining the thermodynamic state """
        
        try:
            self.thermo.updateState(state)
            return False
        except:
            return True
