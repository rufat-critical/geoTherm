from .node import Node
import numpy as np
from ..logger import logger
from ..units import inputParser

class flowNode(Node):
    """Base class for a flow node that calculates flow in between stations."""

    def initialize(self, model):
        """
        Initialize the node with the model.

        Args:
            model: The model to initialize with.
        """

        # Add w attribute if not defined
        if not hasattr(self, '_w'):
            self._w = 0

        # Initialize dP using inlet/outlet node pressures
        if not hasattr(self, '_dP'):
            self._dP = (model.nodes[self.US].thermo._P -
                        model.nodes[self.DS].thermo._P)

        # Initialize dH using inlet/outlet node enthalpies
        if not hasattr(self, '_dH'):
            self._dH = (model.nodes[self.US].thermo._H -
                        model.nodes[self.DS].thermo._H)           

        # Do rest of initialization 
        return super().initialize(model)

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
        if self._w > 0:
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


class statefulFlowNode(flowNode):
    """
    Node class with mass flow as state variable. This needs to be inherited
    and not standalone
    """

    # Variable Bounds
    _bounds = [-1e5, 1e5]

    def _getThermo(self):
        """
        Get the inlet and outlet thermo states based on flow direction.
        """

        # Handle Backflow
        if self._w >= 0:
            US = self.model.nodes[self.US].thermo
            DS = self.model.nodes[self.DS].thermo
        else:
            US = self.model.nodes[self.DS].thermo
            DS = self.model.nodes[self.US].thermo

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

        US, _ = self._getThermo()

        # Update dP and dH
        self._dP = (outletState['P']
                    - US._P)

        self._dH = (outletState['H']
                    - US._H)

        #if abs(self._dP - self._get_dP(US, DS))>1e-9:
        #    from pdb import set_trace
        #    set_trace()


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
        US, DS = self._getThermo()

        # Get Difference in DS property and outletState property
        # via list comprehension
        return np.array([(outletState['P'] - DS._P)*np.sign(self._w)])        

    def getOutletState(self):

        # Get US, DS Thermo
        US, DS = self._getThermo()

        # get dh and dP
        self._dH = self._get_dH(US, DS)
        self._dP = self._get_dP(US, DS)

        # Return outlet state
        return {'H': US._H + self._dH,
                'P': US._P + self._dP}


class Turbo(statefulFlowNode):
    """Base Turbo Class for turbines and pumps."""

    _displayVars = ['w', 'dP', 'dH', 'W', 'PR', 'vol_flow']
    _units = {'w': 'MASSFLOW', 'W': 'POWER', 'dH': 'SPECIFICENERGY',
              'dP': 'PRESSURE', 'vol_flow':'VOLUMETRICFLOW', 'Q':'POWER'}

    @inputParser
    def __init__(self, name, eta,
                 US,
                 DS,
                 PR=2,
                 w:'MASSFLOW'=1):
        """
        Initialize the base turbine.

        Args:
            name (str): Name of the turbine.
            eta (float): Efficiency of the turbine.
            US (str): Upstream node name.
            DS (str): Downstream node name.
            PR (float): Pressure ratio.
            w (float): Mass flow rate.
        """
        
        # Component Name
        self.name = name
        # Component Efficiency
        self.eta = eta
        # Upstream Station
        self.US = US
        # Downstream Station
        self.DS = DS
        # Pressure Ratio
        self.PR = PR
        # Mass Flow
        self._w = w

        if self.PR is None:
            msg = f'No PR input specified for {self.name}, setting it to 2'
            logger.warn(msg)
            self.PR = 1.5

        if self._w is None:
            msg = f'No w input specified for {self.name}, setting it to 1'
            logger.warn(msg)
            self._w = 1

        # Initialize Variables
        self._Q = 0

    def initialize(self, model):
        from geoTherm import thermo
        self._refThermo = thermo.from_state(model.nodes[self.US].thermo.state)

        return super().initialize(model)

    def getOutletState5(self):
        """
        Calculate the outlet state of the turbine.

        Args:
            model: The model containing all nodes.

        Returns:
            dict: Outlet state with pressure and enthalpy.
        """

        # Handle reverse flow
        US, DS = self._getThermo()

        self._dP = self._get_dP(US, DS)
        self._dH = self._get_dH(US, DS)

        self._W = -self._dH*np.abs(self._w)

        return {'H': US._H + self._dH, 'P': US._P + self._dP}

    @property
    def _W(self):
        US, DS = self._getThermo()
        dH = self._get_dH(US, DS)
        return -dH*np.abs(self._w)

    @property
    def _vol_flow(self):

        # Get Thermo States
        US, DS = self._getThermo()

        # Get Upstream
        return self._w/US._density

class Heat(Node):
    pass

class statefulHeatNode(Node):

    @property
    def x(self):
        return np.array([self._Q])

    def updateState(self, x):
        self._Q = x[0]
