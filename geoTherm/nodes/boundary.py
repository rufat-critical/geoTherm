from .baseNodes.baseThermo import baseThermo
from .baseNodes.baseFlow import FixedFlow
from ..logger import logger
import numpy as np


class Boundary(baseThermo):
    pass


class PBoundary(baseThermo):
    """ Thermodynamic state with a specified Pressure but enthalpy
        calculated from inlet streams """

    _displayVars = ['P', 'T', 'H']

    _bounds = [-np.inf, np.inf]

    def initialize(self, model):
        super().initialize(model)

        self.state = np.array([self.thermo._H])
        # Original State
        self._state = np.array([self.thermo._H])
    

    def update_state(self, x):

        # Get the initial state
        P0 = self.thermo._P

        self.state = x
        try:
            # Update the thermodynamic state
            self.thermo._HP = x[0], P0
            self.penalty = False
            self._state = x
        except Exception as e:
            logger.warn(f'Failed to update thermo state for {self.name} to:'
                    f'H: {x}, resetting to H0: {self._state}. '
                    f'Error: {e}')       
            self.thermo._HP = self._state, P0
            # Point penalty to the original state
            self.penalty = (self._state-x) * 1e5

    @property
    def x(self):
        return self.state
    
    @property
    def xdot(self):

        if self.penalty is not False:
            return self.penalty
        
        from pdb import set_trace
        #set_trace()
        #return np.array([0])
        return np.array([self.thermo._H - np.sum(self.influx[1:])/self.influx[0]])

        # Get the original state
    def update_thermo(self, state):
        error = self.thermo._update_state(state)

        self.state = np.array([self.thermo._H])
        return error

    def evaluate2(self):
        
        H = 0
        wnet = 0
        # Maybe write a method to have inlet/outlet flux calc
        for node in self.US_nodes:
            node.evaluate()
            upVol = node.US_node.thermo
            w = node._w
            wnet = node._w
            out = self.US_nodes[0].get_outlet_state(upVol, w)
            H += out['H']*w/wnet

        outlet_state = {'P': self.thermo._P, 'H': H}
        try:
            self.thermo.update_state(outlet_state)
        except Exception as e:
            logger.warn(f"Failed to update outlet state to: {outlet_state}")


class Outlet(Boundary):
    """ Outlet Node where the state is determined by outlet properties"""

    def initialize(self, model):
        super().initialize(model)

        # The outlet can only be connected Downstream to another node
        # do some error checking to verify
        if len(self.DS_neighbors) > 0:
            logger.critical(f"Outlet Node '{self.name}' has nodes connected "
                            f"downstream:\n{self.DS_neighbors}\n It should "
                            "only be downstream of 1 flow node!""")

        if len(self.US_neighbors) > 1:
            logger.critical(f"Outlet Node '{self.name}' is connected to "
                            f"multiple upstream nodes:\n{self.US_neighbors}\n"
                            "It should only be downstream of 1 flow node!")

        if len(self.hot_neighbors) > 0:
            logger.critical(f"Outlet Node '{self.name}' is connected to "
                            f"hot upstream nodes:\n{self.hot_neighbors}\nIt "
                            "should only be downstream of 1 flow node!")

        if len(self.cool_neighbors) > 1:
            logger.critical(f"Outlet Node '{self.name}' is connected to "
                            f"cool upstream nodes:\n{self.cool_neighbors}\nIt "
                            "should only be downstream of 1 flow node!")

        if len(self.US_nodes + self.DS_nodes + self.hot_neighbors + self.cool_neighbors) == 0:
            logger.critical(f"Outlet Node '{self.name}' is not connected to any nodes!")


    def evaluate(self):

        # Evaluate the upstream node
        self.US_nodes[0].evaluate()

        # Get the upstream volume node
        upVol = self.US_nodes[0].US_node.thermo
        w = self.US_nodes[0]._w


        if isinstance(self.US_nodes[0], FixedFlow):
            outlet_state = self.US_nodes[0].get_outlet_state(upVol, PR=1)
        else:
            outlet_state = self.US_nodes[0].get_outlet_state(upVol, w=w)


        ### need some refactoring here
        # need to add q to outlet
        # 

        #Update Outlet State
        try:
            self.thermo.update_state(outlet_state)
        except Exception as e:
            logger.warn(f"Failed to update outlet state to: {outlet_state}")


class POutlet(Boundary):
    """ Outlet Node where Pressure is fixed, enthalpy is calculated from inlet streams"""

    def initialize(self, model):
        super().initialize(model)
        # The outlet can only be connected Downstream to another node
        # do some error checking to verify

        if len(self.DS_neighbors) > 0:
            logger.critical(f"Outlet Node '{self.name}' has nodes connected "
                            f"downstream:\n{self.DS_neighbors}\n It should "
                            "only be downstream of 1 flow node!""")


    def evaluate(self):

        if np.sum(self.influx) == 0:
            pass
        else:
            Hmix = np.array([np.sum(self.influx[:-1])/self.influx[0]])
            try:
                # Update the thermo state
                self.thermo.update_state({'P': self.thermo._P, 'H': Hmix})
            except Exception as e:
                logger.warn(f"Failed to update outlet state to: {Hmix}")

