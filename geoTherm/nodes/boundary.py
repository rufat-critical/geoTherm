from .baseClasses import ThermoNode
from ..units import inputParser
from ..logger import logger
import numpy as np


class Boundary(ThermoNode):
    pass

class POutlet(ThermoNode):

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


class PBoundary(ThermoNode):
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


class TBoundary(ThermoNode):
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


    def update_thermo(self, dsState):

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
    
class Outlet(ThermoNode):
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
        self.thermo.update_state(outlet_state)
