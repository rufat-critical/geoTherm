import numpy as np
from ...nodes.boundary import Boundary
from ...nodes.heatsistor import Qdot
from ...nodes.volume import lumpedMass, Volume, Station
from ...logger import logger


class Junction:

    _bounds = [-np.inf, np.inf]
    def __init__(self, name, node, network):

        self.name = name
        self.node = node
        self.network = network
        self.model = self.network.model
        self.penalty = False
        self.update_bounds()

    def initialize(self):
        
        net_map = self.network.net_map[self.name]

        self.US_branches = [self.network.branches[name] for name in net_map['US']]
        self.DS_branches = [self.network.branches[name] for name in net_map['DS']]
        self.hot_branches = [self.network.branches[name] for name in net_map['hot']]
        self.cool_branches = [self.network.branches[name] for name in net_map['cool']]


    def update_bounds(self):

        if hasattr(self.node, '_bounds'):
            self._bounds = self.node._bounds

    def evaluate(self):
        self.node.evaluate()

    @property
    def x(self):
        if hasattr(self.node, 'x'):
            return self.node.x
        else:
            return np.array([])

    @property
    def xdot(self):

        if self.penalty is not False:
            return np.array([self.penalty])

        if hasattr(self.node, 'xdot'):
            return self.node.xdot
        else:
            return np.array([])

    def update_state(self, x):
        if self._bounds[0] <= x <= self._bounds[1]:
            self.node.update_state(x)
            self.penalty = False
        else:
            if x < self._bounds[0]:
                self.penalty = (self._bounds[0] - x[0] + 10)*1e5
            else:
                self.penalty = (self._bounds[1] - x[0] - 10)*1e5

    def error_check(self):
        pass


class ThermalJunction(Junction):
    # Temperature Bounds
    _bounds = [50, 5000]

    def initialize(self):

        super().initialize()

        if len(self.network.net_map[self.name]['flow']) != 0:
            # This is associated with a flow branch and state
            # is updated in the branch
            self.stateful = False
        else:
            # Need to update state
            self.stateful = True

    @property
    def Q_flux(self):
        # Get Heat Flux
        Q = 0
        for branch in self.hot_branches:
            Q += branch._Q
        for branch in self.cool_branches:
            Q -= branch._Q

        return Q

    @property
    def x(self):
        if self.stateful:
            return np.array([self.node.thermo._T])
        else:
            return np.array([])

    @property
    def xdot(self):
        if not self.stateful:
            return np.array([])
        if self.penalty is not False:
            return np.array([self.penalty])

        # Calculate flux
        # tuple with (wNet, Hnet, Unet, Qnet)
        flux = self.model.get_flux(self.node)
        return np.array([flux[3]])

    def update_state(self, x):
        if not self.stateful:
            return 
        self.penalty = False
        if self._bounds[0] > x:
            self.penalty = (self._bounds[0] - x[0] + 10)*1e5
            return
        elif self._bounds[1] < x:
            self.penalty = (self._bounds[1] - x[0] - 10)*1e5
            return

        state = {'D': self.node.thermo._density,
                 'T': x[0]}

        error = self.node.update_thermo(state)

        if error:
            from pdb import set_trace
            set_trace()


class BoundaryJunction(Junction):
    @property
    def x(self):
        return np.array([])

    @property
    def xdot(self):
        return np.array([])


class OutletJunction(BoundaryJunction):
    """
    Boundary Junction but Outlet is dependent on inlet branches.
    Should be used with a fixedFlow Branch"""

    def error_check(self):
        super().error_check()

        if len(self.DS_branches) > 0:
            logger.critical(f"The Outlet Boundary Condition '{self.name}' "
                            "cannot have any upstream branches")
        elif len(self.US_branches) != 1:
            logger.critical(f"The Outlet Boundary Condition '{self.name}' "
                            "can only have 1 downstream connection")
        elif not (self.US_branches[0].fixed_flow
                  or self.US_branches[0].fixed_flow_flag):
            logger.critical(f"The Outlet Boundary Condition '{self.name}' "
                            "requires a fixedFlow object to be defined "
                            "upstream")


class FlowJunction(Junction):
    """
    Represents a junction in the geoTherm model network.

    Junctions are points where multiple branches meet, allowing the solver
    to handle mass and energy conservation across these points.
    """

    # These are the bounds on Pressure and Enthalpy
    _bounds = [[1, 1e8], [-np.inf, np.inf]]
    #_bounds = [[-np.inf, np.inf], [-np.inf, np.inf]]
    #_bounds = [[1e-5]]
    def __init__(self, name, node, network):
        """
        Initialize a Junction instance.

        Args:
            node (Node): The node associated with the junction.
            upstream_branches (list): List of upstream branches connected
                                      to the junction.
            downstream_branches (list): List of downstream branches connected
                                        to the junction.
            model (Model): The geoTherm model.
        """
        super().__init__(name, node, network)

        self.initialized = False
        self.solve_energy = True
        self.update_energy = False
        self.constant_density = False
        self.error = 'abs'
        self.state = np.array([])

    def initialize(self):
        """
        Initialize the Junction by adding dynamic properties if necessary.
        """

        super().initialize()

        if not isinstance(self.node, (Volume, Station)):
            from pdb import set_trace
            set_trace()

        
        #if len(self.US_branches) == 1:
            # The energy can be determined solely using
            # US BC
        #    self.solve_energy = False
        #else:
        #    self.solve_energy = True


        self.initialize_state()
        #self.solve_energy = False

    def evaluate(self):

        self.node.evaluate()

        if self.update_energy:
            self.update_thermo({'H': self.mix_H(),
                                'P': self.node.thermo._P})
            
        
        


    def initialize_state(self):

        # Check that update energy is off if solve_energy is on
        if self.solve_energy and self.update_energy:
            from pdb import set_trace
            set_trace()



        if self.solve_energy:
            self.state = np.array([self.node.thermo._P,
                                   self.node.thermo._H])
            self._state = np.array([self.node.thermo._P,
                                   self.node.thermo._H])
        else:
            self.state = np.array([self.node.thermo._P])
            self._state = np.array([self.node.thermo._P])



    @property
    def x(self):
        return self.state


    @property
    def xdot(self):
        if self.penalty is not False:
            return self.penalty

        wNet, Hnet, Wnet, Qnet = self.node.flux

        if self.solve_energy:
            error = np.array([wNet, Hnet+Wnet+Qnet])
        else:
            error = np.array([wNet])

        if self.error == 'relative':
            return error/np.abs(self.state)
        else:
            return error

    def update_state(self, x):

        self.penalty = False
        penalty0 = False
        penalty1 = False
        self.state = x
        if self._bounds[0][0] <= x[0] <= self._bounds[0][1]:
            penalty0 = False
        else:
            if x[0] <= self._bounds[0][0]:
                penalty0 = (self._bounds[0][0] - x[0] + 1)*1e5
            else:
                penalty0 = (self._bounds[0][0] - x[0] - 1)*1e5

            self.penalty = np.array([penalty0])

        if self.solve_energy:
            if self._bounds[1][0] <= x[1] <= self._bounds[1][1]:
                penalty1 = False
            else:
                if x[1] <= self._bounds[1][0]:
                    penalty1 = (self._bounds[1][0] - x[1] + 1)*1e5
                else:
                    penalty1 = (self._bounds[1][0] - x[1] - 1)*1e5

        if penalty0 or penalty1:
            if self.solve_energy:
                self.penalty = np.array([1., 1.])
            if penalty0:
                self.penalty[0] = penalty0
            elif penalty1:
                self.penalty[1] = penalty1

            return


        if self.solve_energy is False:
            if self.constant_density:
                state = {'P': x[0], 'D': self.node.thermo._density}
            else:
                state = {'P': x[0], 'H': self.node.thermo._H}
        else:
            state = {'P': x[0], 'H': x[1]}

        error = self.node.update_thermo(state)

        if error:
            # Point error towards state that worked
            sign = np.sign(self._state-x)
            self.penalty = (self._state-x+10*sign)*1e20
            self.update_state(self._state)
        else:
            self._state = x

    def update_thermo(self, state):


        error = self.node.update_thermo(state)

        self.initialize_state()


    def mix_H(self):

        win = 0
        H= 0
        for node in self.node.US_nodes:
            if node._w > 0:
                win += node._w
                H += node.US_node.thermo._H*node._w

        for node in self.node.DS_nodes:
            if node._w < 0:
                win += -node._w
                H+= node.DS_node.thermo._H*(-node._w)  


        if H == 0:
            Hmix = self.node.thermo._H
        else:
            Hmix = H/(win+1e-10)

        from pdb import set_trace
        #set_trace()

        return Hmix