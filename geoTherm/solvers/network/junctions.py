import numpy as np
from ...nodes.boundary import Boundary
from ...nodes.heat import Qdot
from ...nodes.volume import lumpedMass


class Junction:

    _bounds = [-np.inf, np.inf]
    def __init__(self, name, node, model):

        self.name = name
        self.node = node
        self.model = model
        self.penalty = False
        self.update_bounds()

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

    _bounds = [50, 5000]
    def __init__(self, name, node, US_thermal_branches, DS_thermal_branches, stateful, model):
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

        super().__init__(name, node, model)

        self.US_thermal_branches = US_thermal_branches
        self.DS_thermal_branches = DS_thermal_branches
        self.penalty = False
        self.stateful = stateful

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
    def __init__(self, name, node, US_flow_branches, DS_flow_branches, 
                 US_thermal_branches, DS_thermal_branches, model):
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
        super().__init__(name, node, model)
        self.US_flow_branches = US_flow_branches
        self.DS_flow_branches = DS_flow_branches
        self.US_thermal_branches = US_thermal_branches
        self.DS_thermal_branches = DS_thermal_branches

    @property
    def x(self):
        return np.array([])

    @property
    def xdot(self):
        return np.array([])


class FlowJunction(Junction):
    """
    Represents a junction in the geoTherm model network.

    Junctions are points where multiple branches meet, allowing the solver
    to handle mass and energy conservation across these points.
    """
    _bounds = []
    def __init__(self, name, node, US_flow_branches, DS_flow_branches, 
                 US_thermal_branches, DS_thermal_branches, model):
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
        super().__init__(name, node, model)

        self.US_flow_branches = US_flow_branches
        self.DS_flow_branches = DS_flow_branches
        self.US_thermal_branches = US_thermal_branches
        self.DS_thermal_branches = DS_thermal_branches
        self.initialized = False

        self.thermal_junction=False 
        self.type = None
        
        self.thermal = False
        self.lumped_mass = False

        self.stateless = False
        self.initialize()


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
