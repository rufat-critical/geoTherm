from ..thermostate import thermo, addThermoAttributes
from ..units import inputParser
from ..logger import logger
from .node import Node
import numpy as np


@addThermoAttributes
class thermoNode(Node):
    """ Thermodynamic state that has a constant state """

    _displayVars = ['P', 'T', 'phase']

    @inputParser
    def __init__(self, name, fluid,
                 P:'PRESSURE'=None,             # noqa
                 T:'TEMPERATURE'=None,          # noqa
                 H:'SPECIFICENTHALPY'=None,     # noqa
                 S:'SPECIFICENTROPY'=None,      # noqa
                 Q=None,
                 state=None):
        """ Initialize a constant thermodynamic state Boundary via fluid
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
            thermoObj (thermoObj, Optional): instance of thermo Object"""

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

        self.penalty = False

    def initialize(self, model):
        super().initialize(model)

        # Get NodeMap
        nodeMap = self.model.nodeMap[self.name]

        # Add referneces to nodes connecting to this node
        self.US_neighbors = nodeMap['US']
        self.US_nodes = [self.model.nodes[name] for name in nodeMap['US']]
        self.DS_neighbors = nodeMap['DS']
        self.DS_nodes = [self.model.nodes[name] for name in nodeMap['DS']]
        self.hot_neighbors = nodeMap['hot']
        self.hot_nodes = [self.model.nodes[name] for name in nodeMap['hot']]
        self.cool_neighbors = nodeMap['cool']
        self.cool_nodes = [self.model.nodes[name] for name in nodeMap['cool']]

        

class Boundary(thermoNode):
    pass


class POutlet(thermoNode):

    def evaluate(self):

        outlet = self.flowNode.getOutletState()
        try:
            self.thermo._HP = outlet['H'], self.thermo._P
        except:
            print('Failed to update Poutlet')

    def initialize(self, model):

        # Get nodeMap
        nodeMap = model.nodeMap[self.name]

        if (len(nodeMap['US']) != 1
            or len(nodeMap['DS']) != 0):
            from pdb import set_trace
            set_trace()

        self.flowNode = model.nodes[nodeMap['US'][0]]

        super().initialize(model)

class PBoundary(thermoNode):
    """ Thermodynamic state with a specified Pressure but energy
    calculated based on conservation """

    _displayVars = ['P', 'T']


    def updateState(self, x):
        
        # Get the initial state
        X0 = self.x

        try:
            # Update the thermodynamic state
            self.thermo._TP = x[0], self.thermo._P
            self.penalty = False
        except:
            self.thermo._TP = X0, self.thermo._P
            self.penalty = (X0-x)*1e5

    @property
    def x(self):
        # Return the object state (enthalpy)
        return np.array([self.thermo._T])

    @property
    def error(self):

        _,Hnet, Qnet, Wnet = self.model.getFlux(self)

        if self.penalty is not False:
            return self.penalty
        from pdb import set_trace
        set_trace()
        return np.array([Hnet+Qnet+Wnet])


class TBoundary(thermoNode):
    """ Thermodynamic state with a specified Temperature but density
    calculated based on conservation """

    _displayVars = ['P', 'T', 'H', 'phase']


    def updateState(self, x):
        
        # Get the initial state
        X0 = self.x

        try:
            # Update the thermodynamic state
            self.thermo._TD = self.thermo._T, x[0]
            self.penalty = False
        except:
            self.thermo._TD = self.thermo._T, X0
            self.penalty = (X0-x)*1e5


    def updateThermo(self, dsState):

        if 'P' not in dsState:
            from pdb import set_trace
            set_trace()
        else:
            self.thermo._TP = self.thermo._T, dsState['P']

        if self.thermo._P > 1e8:
            from pdb import set_trace
            #set_trace()

    @property
    def x(self):
        # Return the object state (Density)
        return np.array([self.thermo._density])

    @property
    def error(self):

        wNet, _, _,_ = self.model.getFlux(self)

        if self.penalty is not False:
            return self.penalty

        return np.array([wNet])
    
class Outlet(thermoNode):
    """ Outlet Node where the state is determined by outlet properties"""

    def initialize(self, model):
        super().initialize(model)

        # The outlet can only be connected Downstream to another node
        # do some error checking to verify
        if len(self.DS_neighbors) > 0:
            logger.critical(f"Outlet Node '{self.name}' has nodes connected "
                            f"downstream:\n{self.DS_neighbors}\n It should only be "
                            "downstream of 1 flow node!""")

        if len(self.US_neighbors) > 1:
            logger.critical(f"Outlet Node '{self.name}' is connected to "
                            f"multiple upstream nodes:\n{self.US_neighbors}\nIt should "
                            "only be downstream of 1 flow node!")

        if len(self.hot_neighbors) > 0:
            logger.critical(f"Outlet Node '{self.name}' is connected to "
                            f"hot upstream nodes:\n{self.hot_neighbors}\nIt should "
                            "only be downstream of 1 flow node!")

        if len(self.cool_neighbors) > 1:
            logger.critical(f"Outlet Node '{self.name}' is connected to "
                            f"cool upstream nodes:\n{self.cool_neighbors}\nIt should "
                            "only be downstream of 1 flow node!")


    def evaluate(self):

        # Evaluate the upstream node
        self.US_nodes[0].evaluate()

        # Update thermo state to US node outlet state
        outlet_state = self.US_nodes[0].get_outlet_state()

        #Update Outlet State
        self.thermo.updateState(outlet_state)
