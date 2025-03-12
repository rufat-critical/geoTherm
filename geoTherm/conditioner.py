
class Conditioner:
    def __init__(self, model, conditioning_type='constant'):
        """
        Initialize the Conditioner.

        Parameters:
        - model: The model object containing state information.
        - conditioning_type: The scaling type ('constant' or 'jacobian').
        """
        self.conditioning_type = conditioning_type
        self.x_scale = None

        if isinstance(model, Model):
            self.model = model
            self.solver = 'nodal'
        elif isinstance(model, Network):
            self.model = model.model
            self.solver = 'network'
        else:
            logger.critical("Model must be of type 'Model' or 'Network'")

        self.initialize()

    def initialize(self):
        """Initialize scaling based on solver type and conditioning type."""
        if self.conditioning_type == 'constant':
            if self.solver == 'nodal':
                self._nodal_scaling()
            elif self.solver == 'network':
                self._network_scaling()
        elif self.conditioning_type == 'None':
            if self.solver == 'nodal':
                self.x_scale = np.ones(len(self.model.x))
            elif self.solver == 'network':
                self.x_scale = np.ones(len(self.model.network.x))
        elif self.conditioning_type == 'jacobian':
            pass

    def _nodal_scaling(self):
        """Compute scaling factors for nodal solver."""
        self.x_scale = np.ones(len(self.model.x))
        for name in self.model.statefuls:
            node = self.model.nodes[name]
            scale = self._determine_nodal_scale(node)
            indx = self.model.istate[name]
            try:
                self.x_scale[indx] = scale
            except:
                from pdb import set_trace
                set_trace()

    def _determine_nodal_scale(self, node):
        """Determine the scale for a nodal solver based on node properties."""
        if (hasattr(node, "_bounds") and
                node._bounds[0] != -np.inf and
                node._bounds[1] != np.inf):
            return 1 / (node._bounds[1] - node._bounds[0])
        elif isinstance(node, gt.lumpedMass):
            return np.array([1e-1])
        elif isinstance(node, gt.PStation):
            return np.array([1e-6])
        elif isinstance(node, gt.Station):
            return np.array([1/10325, 1e-6])
        elif isinstance(node, baseThermo):
            #return np.array([1, 1])
            return np.array([1e-2, 1e-6])
        else:
            return 1

    def _network_scaling(self):
        """Compute scaling factors for network solver."""
        self.x_func = {'x': [], 'xi': {}, 'xdot': {}}
        self.x_scale = np.ones(len(self.model.network.x))

        for name, indx in self.model.network.istate.items():
            scale = self._determine_network_scale(name)
            self.x_scale[indx] *= scale


    def _determine_network_scale(self, name):
        """Determine the scale for a network solver based on network elements.
        """
        if name in self.model.network.flow_branches:
            branch = self.model.network.flow_branches[name]
            if branch._bounds[0] != -np.inf and branch._bounds[1] != np.inf:
                return 1 / (branch._bounds[1] - branch._bounds[0])
            else:
                return 1
        elif name in self.model.network.thermal_branches:
            branch = self.model.network.thermal_branches[name]
            if branch._bounds[0] != -np.inf and branch._bounds[1] != np.inf:
                return 1 / (branch._bounds[1] - branch._bounds[0])
            else:
                print('skiiping')
                return 1
                return 1e-4            
        elif name in self.model.network.junctions:
            node = self.model.nodes[name]
            junction = self.model.network.junctions[name]
            if isinstance(node, gt.Balance):
                if np.isinf(node.knob_max) or np.isinf(node.knob_min):
                    return 1
                else:
                    return 1 / (node.knob_max - node.knob_min)
            elif isinstance(node, gt.Heatsistor):
                return 1e-4
            elif isinstance(junction, FlowJunction):
                if junction.solve_energy:
                    return np.array([1, 1])
                else:
                    return np.array([1e-6])
        return 1

    def _jacobian(self, x):
        """Compute the Jacobian matrix for the current solver."""
        if self.solver == 'network':
            return self.model.network.jacobian(x)
        elif self.solver == 'nodal':
            return self.model.jacobian(x)

    def scale_x(self, x):
        """Scale the input vector x based on the conditioning type."""
        if self.conditioning_type == 'jacobian':
            J = self._jacobian(x)
            return np.linalg.solve(J, x)

        return x * self.x_scale

    def unscale_x(self, x):
        """Unscale the input vector x based on the conditioning type."""
        if self.conditioning_type == 'jacobian':
            J = self._jacobian(x)
            return J @ x

        return x / self.x_scale

    def conditioner(self, func):
        """Wrap a function with scaling and unscaling logic."""
        def wrapper(x):
            x_unscaled = self.unscale_x(x)
            xdot_unscaled = func(x_unscaled)#/abs(x_unscaled)**.25
            return self.scale_x(xdot_unscaled)
        return wrapper