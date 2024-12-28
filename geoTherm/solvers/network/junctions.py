import numpy as np
from ...nodes.boundary import Boundary
from ...nodes.heatsistor import Qdot
from ...nodes.volume import lumpedMass
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
                self.penalty = (self._bounds[0] -x[0] + 10)*1e5
            else:
                self.penalty = (self._bounds[1] -x[0] - 10)*1e5



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

    def initialize(self):
        super().initialize()

        if len(self.DS_branches) > 0:
            logger.critical("Can't have US for outlet")
        elif len(self.US_branches) != 1:
            logger.critical("Can only have 1 DS for outlet")
        elif not self.US_branches[0].fixed_flow:
            logger.critical("This can only be used with fixedFlow")




class FlowJunction(Junction):
    """
    Represents a junction in the geoTherm model network.

    Junctions are points where multiple branches meet, allowing the solver
    to handle mass and energy conservation across these points.
    """
    _bounds = []
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

        self.thermal_junction=False 
        self.type = None
        
        self.thermal = False
        self.lumped_mass = False

        self.stateless = False
        #self.initialize()
        from pdb import set_trace
        set_trace()


    def initialize(self):
        """
        Initialize the Junction by adding dynamic properties if necessary.
        """
        if isinstance(self.node, (Boundary, Qdot)):
            self.__x = np.array([])
            self.__xdot = np.array([])
            from pdb import set_trace
            set_trace()
            pass
        elif isinstance(self.node, (lumpedMass)):
            self.lumped_mass = True
            self.__x = np.array([self.node.thermo._T])
            self.__xdot = self.node.xdot
            from pdb import set_trace
            set_trace()
        elif len(self.US_flow_branches) == len(self.DS_flow_branches) == 1:
            self.__x = np.array([])
            self.__xdot = np.array([])
            from pdb import set_trace
            set_trace()
            pass
        elif ((len(self.US_flow_branches) == len(self.DS_flow_branches) == 0)
              and self.US_thermal_branches or self.DS_thermal_branches):
            self.thermal = True
            self.__x = np.array([self.node.thermo._T])
            self.__xdot = np.array([self.node.xdot[1]])

        elif hasattr(self.node, 'x'):
            # Add properties and method if self.node has attribute 'x'
            from pdb import set_trace
            set_trace()
            self.__add_dynamic_properties()
        else:
            self.__x = np.array([])
            self.__xdot = np.array([])            

        self.initialized = True

    @property
    def x(self):

        try:
            return self.__x
        except:
            from pdb import set_trace
            set_trace()

    @property
    def xdot(self):
        if self.lumped_mass:
            return self.node.xdot
        else:
            pass
        return self.__xdot
    
    def update_state(self, x):
        
        if self.thermal_junction:
            from pdb import set_trace
            set_trace()
            if x < 50:
                self.penalty = (50-x+10)*1e5
                return
            else:
                state = {'D': self.node.thermo._density,
                        'T': x[0]}
                self.node.update_thermo(state)                

        if self.lumped_mass:
            self.node.update_state(x)


        self.__x = x


    def __add_dynamic_properties(self):
        """
        Dynamically add x, xdot, update_state properties
        """
        def get_x(self):
            return self.node.x

        def get_xdot(self):
            return self.node.xdot

        def update_state(self, x):
            self.node.update_state(x)

        # Dynamically add properties and methods
        setattr(self.__class__, 'x', property(get_x))
        setattr(self.__class__, 'xdot', property(get_xdot))
        setattr(self.__class__, 'update_state', update_state)
