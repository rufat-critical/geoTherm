from .baseClasses import ThermoNode
from ..logger import logger
import numpy as np


class Boundary(ThermoNode):
    pass


class PBoundary(ThermoNode):
    """ Thermodynamic state with a specified Pressure but density
    calculated based on conservation """

    _displayVars = ['P', 'T']

    def update_state(self, x):

        # Get the initial state
        X0 = self.x
        P0 = self.thermo._P

        if x <0:
            self.penalty = (X0-x)*1e5
            return

        try:
            # Update the thermodynamic state
            self.thermo._DP = x[0], P0
            self.penalty = False
        except:
            self.thermo._DP = X0, P0
            self.penalty = (X0-x)*1e5

    @property
    def x(self):
        # Return the object state (enthalpy)
        return np.array([self.thermo._density])

    @property
    def xdot(self):
        return self.error

    @property
    def error(self):

        wnet,Hnet, Qnet, Wnet = self.model.getFlux(self)

        if self.penalty is not False:
            return self.penalty
        from pdb import set_trace
        #set_trace()
        return np.array([wnet])


class TBoundary(ThermoNode):
    """ Thermodynamic state with a specified Temperature but density
    calculated based on conservation """

    _displayVars = ['P', 'T', 'H', 'phase']

    def initialize(self, model):
        super().initialize(model)
        self.penalty = False

    def update_state(self, x):

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

    @property
    def xdot(self):
        return self.error

class Outlet(ThermoNode):
    """ Outlet Node where the state is determined by outlet properties"""

    # @inputParser
    # def __init__(self, name, fluid,
    #              fixedFlow_node,
    #              P: 'PRESSURE'=None,           # noqa
    #              T: 'TEMPERATURE'=None,        # noqa
    #              H: 'SPECIFICENTHALPY'=None,   # noqa
    #              S: 'SPECIFICENTROPY'=None,    # noqa
    #              Q=None,
    #              state=None):
    #     """
    #     Initialize a thermodynamic node with a given fluid and state.

    #     Args:
    #         name (str): Node Name.
    #         fluid (str or Thermo): Fluid name or a Thermo object.
    #         P (float, optional): Pressure.
    #         T (float, optional): Temperature.
    #         H (float, optional): Enthalpy.
    #         S (float, optional): Entropy.
    #         Q (float, optional): Fluid Quality.
    #         state (dict, optional): Dictionary with a predefined
    #                                 thermodynamic state.

    #     Notes:
    #         If `state` is provided, it overrides individual parameters
    #         (P, T, H, S, Q).
    #     """
    #     self.name = name
    #     self.fixedFlow_node = fixedFlow_node

    #     if state is None:
    #         # Generate and trim the state dictionary based on the provided
    #         # parameters
    #         state = {var: val for var, val in {'P': P, 'T': T, 'H': H,
    #                                            'S': S, 'Q': Q}.items()
    #                  if val is not None}
    #         state = state if state else None

    #     # Handle the fluid argument
    #     if isinstance(fluid, str):
    #         # If fluid is a string, create a new thermo object with it
    #         self.thermo = thermo(fluid, state=state)
    #     elif isinstance(fluid, thermo):
    #         # If fluid is a thermo object, use it for calculations
    #         self.thermo = fluid

    #         # Update the thermo object with the provided state, if any
    #         if state is not None:
    #             self.thermo._update_state(state)


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


    def evaluate(self):

        # Evaluate the upstream node
        self.US_nodes[0].evaluate()

        # Update thermo state to US node outlet state
        outlet_state = self.US_nodes[0].get_outlet_state()

        #Update Outlet State
        try:
            self.thermo.update_state(outlet_state)
        except:
            pass
