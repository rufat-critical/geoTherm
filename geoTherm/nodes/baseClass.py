from .node import Node
import numpy as np
from geoTherm.logger import logger
from geoTherm.units import inputParser
from geoTherm.utils import dPpipe
from geoTherm import Boundary, TBoundary


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


    def getOutletState(self):
        """
        Placeholder method to be overwritten by specific flow nodes.
        """
        msg = f"{self.name} of type {type(self)} is missing a getOutletState "\
            "method, geoTherm cannot run without this!"
        logger.error(msg)
        raise RuntimeError(msg)
    
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

    def _get_dH(US, DS):
        """
        Placeholder method to get enthalpy difference.
        """

        msg = f"{self.name} of type {type(self)} is missing a " \
            "_get_dH(US, DS) method"
        logger.error(msg)

    def _get_dP(US, DS):
        """
        Placeholder method to get pressure difference.
        """

        msg = f"{self.name} of type {type(self)} is missing a " \
            "_get_dP(US, DS) method"
        logger.error(msg)


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

        US, DS = self._getThermo()

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
            np.array: Difference in downstream property and outlet state property.
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

    _displayVars = ['w', 'dP', 'dH', 'W', 'PR']
    _units = {'w': 'MASSFLOW', 'W': 'POWER', 'dH': 'SPECIFICENERGY',
              'dP': 'PRESSURE'}

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


    def _get_dH(US, DS):
        """
        Placeholder method to get enthalpy change.
        """

        msg = f"{self.name} of type {type(self)} is missing a " \
            "_get_dH(US, DS) method, geoTherm cannot run without this!"
        logger.error(msg)
        raise RuntimeError(msg)

    def _get_dP(US, DS):
        """
        Placeholder method to get pressure drop.
        """

        msg = f"{self.name} of type {type(self)} is missing a " \
            "_get_dP(US, DS) method, geoTherm cannot run without this!"
        logger.error(msg)
        raise RuntimeError(msg)
    

class flow(statefulFlowNode):
    """ Base Class for Flow Node """

    _displayVars = ['w', 'dP', 'dH', 'Q']
    _units = {'D': 'LENGTH', 'L': 'LENGTH', 'w': 'MASSFLOW',
              'roughness': 'LENGTH', 'dP': 'PRESSURE',
              'Q': 'POWER', 'dH': 'SPECIFICENERGY'}

    _bounds = [-1e5, 1e5]

    @inputParser
    def __init__(self, name, US, DS,
                 w:'MASSFLOW',
                 L:'LENGTH'=None,
                 D:'LENGTH'=None,
                 roughness:'LENGTH'=1e-5,
                 dP:'PRESSURE'=None,
                 Q: 'POWER'=0):

        self.name = name
        self.US = US
        self.DS = DS
        self._D = D
        self._L = L
        self._w = w
        self._roughness = roughness

        # Bounds handled via penalty function
        self.penalty = False

        if dP is None:
            self.update_dP = True
            if (L is None) or (D is None):
                logger.error(f"Connector {self.name} needs to have D and L ",
                             "specified if dP is not specified!")
                raise ValueError("Error encountered in Connector Object "
                                 "Initialization")
        else:
            self.update_dP = False
            self.__dPsetpoint = -dP
            self._dP = -dP


        self._Qobj = None
        if isinstance(Q, (float, int)):
            self._Q = Q
        elif isinstance(Q, flow):
            self._Q = -Q._Q
            self._Qobj = Q
        else:
            from pdb import set_trace
            set_trace()

    def _get_dP(self, US, DS):

        if self.update_dP:
            return dPpipe(US,
                          self._D,
                          self._L,
                          self._w,
                          self._roughness)
        else:
            return self.__dPsetpoint

    def _get_dH(self, US, DS):

        # Q
        if self._w == 0:
            return 0
        else:
            # Get Heat
            self._Q = self._get_Q(US, DS)

            return self._Q/np.abs(self._w)

    def _get_Q(self, US, DS):
        # Update Heat in Pipe

        if self.fixedT_outlet:
            return np.abs(self._w)*(DS._H - US._H)
        elif self._Qobj is not None:
            self._Q = -self._Qobj._Q

        return self._Q

    def initialize(self, model):

        if self.update_dP:
            self._area = np.pi*self._D**2/4
        else:
            self._area = np.inf

        if isinstance(model.nodes[self.DS], (Boundary, TBoundary)):
            self.fixedT_outlet = True
            logger.info(f"Node: '{self.name}' is connected to a TBoundary "
                        "downstream, Q will be calculated to satisfy "
                        "downstream T")
        else:
            self.fixedT_outlet = False

        # Do rest of initialization
        return super().initialize(model)

    @property
    def flowSpeed(self):
        # Calculate Flow Speed of object
        # U = mdot/(rho*A^2)

        US, _ = self._getThermo()

        return self._w/(US._density*self._area)


class statefulHeatNode(Node):


    @property
    def x(self):
        return np.array([self._Q])

    def updateState(self, x):
        self._Q = x[0]
