import numpy as np
from scipy.optimize._numdiff import approx_derivative
from .junctions import (
    Junction, ThermalJunction, OutletJunction, BoundaryJunction, FlowJunction
)
from .branches import ThermalBranch, FlowBranch
from ...nodes.boundary import Outlet, Boundary
from ...nodes.heatsistor import Qdot
from ...logger import logger
from ...utils import has_cycle
import geoTherm as gt
from scipy.optimize import fsolve, root


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
        self.xdot_scale = None

        self.model = model.model

        self.initialize()

    def initialize(self):
        """Initialize scaling based on solver type and conditioning type."""
        if self.conditioning_type == 'constant':
            self._network_scaling()
        elif self.conditioning_type == 'None':
            self.x_scale = np.ones(len(self.model.network.x))
        elif self.conditioning_type == 'jacobian':
            pass

    def _network_scaling(self):
        """Compute scaling factors for network solver."""
        self.x_func = {'x': [], 'xi': {}, 'xdot': {}}
        self.x_scale = np.ones(len(self.model.network.x))
        self.xdot_scale = np.ones(len(self.model.network.x))

        for name, indx in self.model.network.istate.items():
            try:
                x_scale, xdot_scale = self._determine_network_scale(name)
            except:
                from pdb import set_trace
                set_trace()

            self.x_scale[indx] *= x_scale
            self.xdot_scale[indx] *= xdot_scale


    def _determine_network_scale(self, name):
        """Determine the scale for a network solver based on network elements.
        """
        if name in self.model.network.flow_branches:
            branch = self.model.network.flow_branches[name]

            x_scale = np.array([1e-1])
            xdot_scale = np.array([1])

            return x_scale, xdot_scale
        elif name in self.model.network.thermal_branches:
            branch = self.model.network.thermal_branches[name]
            if branch._bounds[0] != -np.inf and branch._bounds[1] != np.inf:
                return 1 / (branch._bounds[1] - branch._bounds[0])
            else:
                x_scale = np.array([1e-4])
                xdot_scale = np.array([1e-4])
                return x_scale, xdot_scale
        elif name in self.model.network.junctions:
            node = self.model.nodes[name]
            junction = self.model.network.junctions[name]
            if isinstance(node, gt.Balance):
                if np.isinf(node.knob_max) or np.isinf(node.knob_min):
                    return 1, 1
                else:
                    x_scale = 1 / (node.knob_max - node.knob_min)
                    xdot_scale = 1/x_scale
                    return x_scale, xdot_scale
            elif isinstance(node, gt.Heatsistor):
                return 1e-4
            elif isinstance(junction, FlowJunction):
                if junction.solve_energy:
                    #x_scale = np.array([1e-6, 1e-6])
                    #xdot_scale = np.array([1, 1/1e-6])
                    x_scale = np.array([1e-5, 1e-5])
                    xdot_scale = np.array([1e-5, 1])
                    return x_scale, xdot_scale
                else:
                    #x_scale = np.array([1e-5])
                    #xdot_scale = np.array([1e-2])
                    x_scale = np.array([1e-6])
                    xdot_scale = np.array([1])                   
                    return x_scale, xdot_scale

        return 1, 1

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

    def scale_xdot(self, xdot):
        return xdot * self.xdot_scale


    def conditioner(self, func):
        """Wrap a function with scaling and unscaling logic."""
        def wrapper(x):
            x_unscaled = self.unscale_x(x)
            xdot_unscaled = func(x_unscaled)

            return self.scale_xdot(xdot_unscaled)
        return wrapper


class Network:
    """
    Network Solver for geoTherm.

    This class manages and solves a thermal and fluid network by organizing
    nodes into branches and junctions, updating states, and solving the system.
    """

    def __init__(self, model):
        """
        Initialize the Network solver with a given model.

        Args:
            model (Model): The geoTherm model containing nodes, branches,
                           and junctions.
        """
        self.model = model

        # Flags to control solvers
        self.constant_density = False
        self.solve_energy = False

        # Only if the network is directed
        self.update_energy = False

        # Flag to detect if network has cycle
        self.has_cycle = False

        # Flag to turn on or off backflow
        self.backflow = True

        self.initialize_network()

    def initialize_network(self):
        """
        Initialize the network by setting up branches and junctions.

        This method identifies all the branches and junctions in the model,
        and prepares them for simulation by creating corresponding objects
        and setting their initial states.
        """

        if not self.model.initialized:
            self.model.initialize()

        # Identify branches, junctions, and create the network map
        (flow_branches, thermal_branches, junctions, net_map) = \
            self._identify_branches_and_junctions(self.model.node_map)

        self.net_map = net_map
        self.flow_branches = dict(flow_branches)
        self.thermal_branches = dict(thermal_branches)
        self.thermal_junctions = {}
        self.flow_junctions = {}
        self.junctions = dict.fromkeys(junctions, None)

        # Loop and generate junction objects
        for junction_id in junctions:
            # Get the corresponding node and network map for the current
            # junction
            node = self.model.nodes[junction_id]

            # Extract upstream and downstream flow/thermal branches
            US = self.net_map[junction_id]['US']
            DS = self.net_map[junction_id]['DS']
            hot = self.net_map[junction_id]['hot']
            cool = self.net_map[junction_id]['cool']

            if isinstance(node, Outlet):
                self.junctions[junction_id] = OutletJunction(
                    name=junction_id,
                    node=node,
                    network=self)
            elif isinstance(node, (Boundary, Qdot)):
                self.junctions[junction_id] = BoundaryJunction(
                    name=junction_id,
                    node=node,
                    network=self)
            elif not (US or DS) and (hot or cool):
                # Create a ThermalJunction if there are no associated flow
                # branches but there are thermal branches
                self.junctions[junction_id] = ThermalJunction(
                    name=junction_id,
                    node=node,
                    network=self)
            elif US or DS:
                if isinstance(node, gt.PBoundary):
                    self.junctions[junction_id] = Junction(
                        name=junction_id,
                        node=node,
                        network=self)
                else:
                    # Create a FlowJunction if there are any flow branches
                    self.junctions[junction_id] = FlowJunction(
                        name=junction_id,
                        node=node,
                        network=self)

                self.junctions[junction_id].constant_density = \
                    self.constant_density
                self.junctions[junction_id].solve_energy = \
                    self.solve_energy

            else:
                self.junctions[junction_id] = Junction(
                    name=junction_id,
                    node=node,
                    network=self)

            # Organize the flow and thermal junctions
            if US or DS:
                self.flow_junctions[junction_id] = \
                    self.junctions[junction_id]
            if hot or cool:
                self.thermal_junctions[junction_id] = \
                    self.junctions[junction_id]


        # Initialize branches
        for branch_id, nodes in flow_branches.items():
            US_junction = self.junctions[self.net_map[branch_id]['US']]
            DS_junction = self.junctions[self.net_map[branch_id]['DS']]

            # Create branch objects
            self.flow_branches[branch_id] = FlowBranch(
                name=branch_id,
                nodes=nodes,
                US_junction=US_junction,
                DS_junction=DS_junction,
                network=self
            )

        for branch_id, nodes in thermal_branches.items():
            hot_junction = self.junctions[self.net_map[branch_id]['hot']]
            cold_junction = self.junctions[self.net_map[branch_id]['cool']]

            self.thermal_branches[branch_id] = ThermalBranch(
                name=branch_id,
                nodes=nodes,
                hot_junction=hot_junction,
                cold_junction=cold_junction,
                network=self)

        self.branches = {**self.flow_branches, **self.thermal_branches}

        # Check if the model has recirculation
        if has_cycle(self.model.node_map, self.model.nodes):
            self.has_cycle = True

    def initialize_states(self, flow=True, thermal=True):
        self.istate = {}  # Maps state indices for each network component
        self.state = []  # State vector

        #for junction in self.junctions.values():
        #    junction.initialize()

        #for branch in self.branches.values():
        #    branch.initialize()
        self.initialize_branches_junctions()

        # Initialize state vector for branches
        for obj_id, obj in {**self.branches, **self.junctions}.items():
            # Run error checker
            obj.error_check()

            state_length = len(obj.x)
            if state_length == 0:
                continue

            current_length = len(self.state)

            if (obj in [*self.flow_branches.values(),
                        *self.flow_junctions.values()]
                    and not flow):
                # If flow is disabled
                continue
            elif (obj in [*self.thermal_branches,
                        *self.thermal_junctions] 
                    and not thermal):
                # If thermal is disabled
                continue

            self.istate[obj_id] = np.arange(current_length,
                                            current_length + state_length)
            self.state = np.concatenate((self.state, obj.x))

        # Evaluate Network
        self.evaluate(self.x)

    def initialize_branches_junctions(self, flow=True, thermal=True):

        for obj_id, obj in {**self.branches, **self.junctions}.items():

            if (obj in [*self.flow_branches.values(),
                        *self.flow_junctions.values()]):
                if not flow:
                    # If flow is disabled
                    continue
                elif obj in self.flow_branches.values():
                    obj.backflow = self.backflow
                elif obj in self.flow_junctions.values():
                    obj.solve_energy = self.solve_energy
                    obj.constant_density = self.constant_density
                    self.update_energy = self.update_energy
            elif (obj in [*self.thermal_branches,
                        *self.thermal_junctions]):
                if not thermal:
                    # If thermal is disabled
                    continue
                elif obj in self.thermal_branches.values():
                    obj.backflow = self.backflow
                else:
                    obj.solve_energy = self.solve_energy
                    obj.constant_density = self.constant_density
                    self.update_energy = self.update_energy

            # Intiialize the object
            obj.initialize()



    def initialize_initial_states(self):

        state0 = self.state

        for name, branch in self.flow_branches.items():
            if name in self.istate:
                state0[self.istate[name]] = np.array([1])

        for name, junc in self.junctions.items():
            if isinstance(junc, BoundaryJunction):
                
                HP = junc.node.thermo._HP


                for flow_junc in self.junctions.values():
                    if isinstance(flow_junc, FlowJunction):
                        flow_junc.node.thermo._HP = HP
                        flow_junc.initialize()
                        state0[self.istate[flow_junc.name]] = flow_junc.x

                return state0


    def _identify_branches_and_junctions(self, node_map):
        """
        Identify all flow branches, thermal branches, and junctions in the
        node map.

        Args:
            node_map (dict): A dictionary representing the node map of the
                             geoTherm system.

        Returns:
            tuple: A tuple containing:
                - flow_branches (dict): A dictionary of flow branches.
                - thermal_branches (dict): A dictionary of thermal branches.
                - flow_branch_connections (dict):
                        A dictionary mapping flow branches to their upstream
                        and downstream junctions.
                - thermal_branch_connections (dict):
                        A dictionary mapping thermal branches to their upstream
                        and downstream junctions.
                - other_junctions (dict):
                        A dictionary of junctions not classified as flow or
                        thermal.
        """

        # Initialize data structures
        remaining_nodes = list(node_map.keys())  # Nodes to be processed
        flow_branches = {}        # Dictionary of flow branches
        thermal_branches = {}     # Dictionary of thermal branches
        flow_junctions = []
        thermal_junctions = []
        other_junctions = []
        branch_counter = 0  # Counter for branch identifiers
        net_map = {}


        # Helper functions
        def is_junction(node_name):
            """
            Determine if a node is a junction.

            A junction can be a flow junction or a thermal junction based on
                its connections.

            Args:
                node_name (str): The name of the node.

            Returns:
                tuple: (is_flow_junction, is_thermal_junction,
                        is_other_junction).
            """
            node = self.model.nodes[node_name]

            flow_junction = False
            thermal_junction = False
            other_junction = False

            US = len(node_map[node_name]['US'])
            DS = len(node_map[node_name]['DS'])
            hot = len(node_map[node_name]['hot'])
            cool = len(node_map[node_name]['cool'])

            # Check if there is flow
            has_US = US >= 1
            has_DS = DS >= 1

            # Check if there is thermal flow
            has_hot = hot >= 1
            has_cool = cool >= 1

            # Check flow connections
            has_multiple_flow_connections = US > 1 or DS > 1
            has_unequal_flow_connections = US != DS

            # Check thermal connections
            has_multiple_thermal_connections = hot > 1 or cool > 1
            has_unequal_thermal_connections = hot != cool

            # Determine if node is a junction based on connections
            if (
                has_multiple_flow_connections or
                has_unequal_flow_connections
            ):
                flow_junction = True

            if (
                has_multiple_thermal_connections or
                has_unequal_thermal_connections
            ):
                thermal_junction = True

            if ((has_US or has_DS)
                    and (has_hot or has_cool)):
                thermal_junction = True

            if (not (has_US or has_DS)
                    and not (has_hot or has_cool)):
                other_junction = True

            # Additional condition for boundary nodes
            if isinstance(node, Boundary):
                if has_US and has_DS:
                    flow_junction = True

            return flow_junction, thermal_junction, other_junction

        # Identify all junction nodes
        # Iterate over a copy to allow removal
        for node_name in list(remaining_nodes):
            # Check for different junction types
            flow, thermal, other = is_junction(node_name)

            remaining_nodes.remove(node_name)

            # Initialize junction conectivity map
            net_map[node_name] = {
                'US': [], 'DS': [], 'hot': [], 'cool': [],
                'flow': [], 'thermal': []}

            if flow:
                flow_junctions.append(node_name)

            if thermal:
                # Check if junction is Qdot
                thermal_junctions.append(node_name)
            if other:
                other_junctions.append(node_name)

        # Define branch tracing functions
        def trace_branch(current_node, current_branch, branch_type):
            """
            Trace a branch starting from the current node.

            Args:
                current_node (str): The current node to trace from.
                branch_type (str): Type of branch to trace
                                    ('flow' or 'thermal').

            Returns:
                str: Downstream junction at the end of the branch.
            """

            if branch_type == 'flow':
                DS = 'DS'
                if current_node in flow_junctions:
                    # We reached end of the branch, return the flow junction
                    return current_node
            elif branch_type == 'thermal':
                DS = 'cool'
                if current_node in [*thermal_junctions, *flow_junctions]:
                    # If the branch is at a thermal junction or flow_junction
                    # then end the branch
                    return current_node

            current_branch.append(current_node)
            if current_node in remaining_nodes:
                remaining_nodes.remove(current_node)

            downstream_nodes = node_map[current_node][DS]

            if not downstream_nodes:
                # Downstream junction
                return current_node

            if len(downstream_nodes) > 1:
                from pdb import set_trace
                set_trace()

            downstream_node = downstream_nodes[0]

            return trace_branch(downstream_node, current_branch, branch_type)

        # Identify flow branches and connections
        for junction in flow_junctions:
            downstream_nodes = node_map[junction]['DS']
            for downstream_node in downstream_nodes:
                if (downstream_node in flow_junctions):
                    # No branch between junctions or already processed
                    continue

                current_branch = []
                downstream_junction = trace_branch(downstream_node,
                                                   current_branch,
                                                   'flow')

                flow_branches[branch_counter] = current_branch
                thermal_nodes = []
                for node in current_branch:
                    if node in thermal_junctions:
                        thermal_nodes.append(node)

                net_map[junction]['DS'].append(branch_counter)
                net_map[downstream_junction]['US'].append(branch_counter)

                net_map[branch_counter] = {'US': junction,
                                           'DS': downstream_junction,
                                           'thermal': thermal_nodes}

                for thermal_junc in thermal_nodes:
                    net_map[thermal_junc]['flow'].append(branch_counter)

                branch_counter += 1

        for junction in thermal_junctions:

            cool_node_names = node_map[junction]['cool']
            HOT_junction = self.model.nodes[junction]

            for cool_node_name in cool_node_names:
                COOL_node = self.model.nodes[cool_node_name]

                if cool_node_name in [*flow_junctions, *thermal_junctions]:
                    cool_junction = cool_node_name
                    if isinstance(HOT_junction, Qdot):
                        current_branch = [junction]
                    elif isinstance(COOL_node, Qdot):
                        current_branch = [cool_junction]
                    else:
                        from pdb import set_trace
                        set_trace()
                else:
                    current_branch = []
                    cool_junction = trace_branch(cool_node_name,
                                                 current_branch,
                                                 'thermal')

                    if isinstance(HOT_junction, Qdot):
                        current_branch = [junction, *current_branch]
                    elif isinstance(cool_junction, Qdot):
                        current_branch = [*current_branch, junction]

                thermal_branches[branch_counter] = current_branch

                thermal_nodes = []
                for node in current_branch:
                    if node in thermal_junctions:
                        thermal_nodes.append(node)

                net_map[junction]['cool'].append(branch_counter)
                net_map[cool_junction]['hot'].append(branch_counter)
                net_map[branch_counter] = {'US': [], 'DS': [],
                                           'hot': junction,
                                           'cool': cool_junction}

                branch_counter += 1

        if len(remaining_nodes) != 0:
            from pdb import set_trace
            set_trace()

        if len(remaining_nodes) != 0:
            logger.warn("The following nodes were not recognized in "
                        f"building the network map: {remaining_nodes}")

        return (
            flow_branches, thermal_branches,
            [*flow_junctions, *thermal_junctions, *other_junctions],
            net_map)


    def solve_flow(self, thermal=True):
        # Solves for the flow-field ignoring thermal junctions

        if thermal:
            self._solve_thermal_on()
        else:
            self._solve_thermal_off()

        self._solve_thermal_on()
        #self.initialize_states()        
        self.initialize_states()#flow=False, thermal=False)


        if len(self.x) == 0:
            return

        conditioner = Conditioner(self)
        conditioned = conditioner.conditioner(self.evaluate_flow)
        x_scaled = conditioner.scale_x(self.x)
        sol = fsolve(conditioned, x_scaled, full_output=True)

        x = conditioner.unscale_x(sol[0])
        self.evaluate_flow(x)

        from pdb import set_trace
        set_trace()


    def solve_thermal(self):

        self.initialize_thermal_states()


        conditioner = Conditioner(self)
        conditioned = conditioner.conditioner(self.evaluate_thermal)
        x_scaled = conditioner.scale_x(self.x)
        sol = fsolve(conditioned, x_scaled, full_output=True)

        x = conditioner.unscale_x(sol[0])
        self.evaluate_thermal(x)


    
    def solve_coupled(self):
        from pdb import set_trace
        set_trace()

    
    # For Flow
    # Flow initialize_states(flow=True)
    # Only
    # Evaluate_flow(self, flow_State)
    
    # For thermal
    # Thermal in
    
    # Initialize_network_objects
    # Initialize_flow
    # Solve with no thermal
    # Solve with thermal
    # Flow

    def _solve_thermal_off(self, update_energy=False):
        for junction in self.flow_junctions.values():
            if isinstance(junction, FlowJunction):
                junction.solve_energy = False
                junction.update_energy = update_energy

        for branch in self.flow_branches.values():
            branch.solve_thermal = False

    def _solve_thermal_on(self):
        for junction in self.flow_junctions.values():
            if isinstance(junction, FlowJunction):
                junction.solve_energy = True
                junction.update_energy = False

        for branch in self.flow_branches.values():
            branch.solve_thermal = True


    def check_penalty(self):

        for stateful in self.istate:
            if stateful in self.junctions:
                penalty = self.junctions[stateful].penalty
                if penalty is not False:
                    from pdb import set_trace
                    set_trace()
            else:
                penalty = self.branches[stateful].penalty
                if penalty is not False:
                    from pdb import set_trace
                    set_trace()


    def solve_directed(self, constant_density=True, backflow=True):

        # Solve like this
        self.solve_energy = False
        self.constant_density = constant_density
        self.update_energy = True
        self.backflow = backflow

        self.initialize_states()

        if len(self.x) == 0:
            return self.x

        conditioner = Conditioner(self)

        conditioned = conditioner.conditioner(self.evaluate_flow)

        x_scaled = conditioner.scale_x(self.x)
        #self.flow_branches[1].solve_steady()

        sol = fsolve(conditioned, x_scaled, full_output=True)
        #set_trace()
        x = conditioner.unscale_x(sol[0])
        self.evaluate_flow(x)


        for junction in self.flow_junctions.values():

            if isinstance(junction, FlowJunction):
                Hmix = junction.mix_H()

                state = {'P': junction.node.thermo._P,
                         'H': Hmix}

                junction.update_thermo(state)


    def solve_coupled(self):

        self.solve_energy = True
        self.update_energy = False
        self.constant_density = False

        self.initialize_states()

        if len(self.x) == 0:
            return self.x

        conditioner = Conditioner(self)

        conditioned = conditioner.conditioner(self.evaluate)

        x_scaled = conditioner.scale_x(self.x)

        # solve with root (hybr, lm)
        sol = root(conditioned, x_scaled, method='lm')
        x = conditioner.unscale_x(sol.x)
        # solve with fsolve
        #sol = fsolve(conditioned, x_scaled, full_output=True)      
        #x = conditioner.unscale_x(sol[0])
        self.evaluate(x)



    def solve(self):
        

        if False:
            if not self.has_cycle and not self.thermal_branches:
                # Solve the model as a directed system
                self.solve_directed()
                #from pdb import set_trace
                #set_trace()
                #self.solve_directed(constant_density=False,
                #                    backflow=False)
                #from pdb import set_trace
                #set_trace()
            else:
                self.solve_coupled()

        self.solve_coupled()

        if not self.model.converged:
            logger.warn("Could not converge with network solver, "
                        "attempting with directed solver")
            self.solve_directed()

            self.solve_coupled()


    def update_state(self, x):
        """
        Update the state vector for the network.

        Args:
            x (array): The new state vector to update.
        """
        self.state = x

        # Update Network
        for name, istate in self.istate.items():
            if name in self.flow_branches:
                self.flow_branches[name].update_state(x[istate])
            elif name in self.thermal_branches:
                self.thermal_branches[name].update_state(x[istate])
            elif name in self.junctions:
                self.junctions[name].update_state(x[istate])
            else:
                from pdb import set_trace
                set_trace()

    def evaluate_flow(self, x):

        self.update_state(x)

        for _, branch in self.flow_branches.items():
            branch.evaluate()

        for _, junction in self.flow_junctions.items():
            junction.evaluate()       

        return self.xdot

    def evaluate_thermal(self, x):

        self.update_state(x)

        for _, branch in self.thermal_branches.items():
            branch.evaluate()

        for _, junction in self.thermal_junctions.items():
            junction.evaluate()

        return self.xdot


    def evaluate(self, x):
        """
        Evaluate the network by updating and computing branch and junction
        states.

        Args:
            x (array): The state vector to evaluate.

        Returns:
            np.array: The derivative of the state vector (xdot).
        """
        self.update_state(x)

        for _, junction in self.junctions.items():
            junction.evaluate()

        for _, branch in self.flow_branches.items():
            branch.evaluate()

        for _, branch in self.thermal_branches.items():
            branch.evaluate()

        # Evaluate the junctions again after branches have been updated
        for _, junction in self.junctions.items():
            junction.evaluate()

        print('x:', self.x)
        print('xdot:', self.xdot)
        return self.xdot

    @property
    def xdot(self):
        """
        Compute and return the derivative of the state vector.

        Returns:
            np.array: The derivative of the state vector (xdot).
        """
        xdot = []

        for name, istate in self.istate.items():
            if name in self.flow_branches:
                xdot.append(self.flow_branches[name].xdot)
            elif name in self.thermal_branches:
                xdot.append(self.thermal_branches[name].xdot)
            elif name in self.junctions:
                xdot.append(self.junctions[name].xdot)
            else:
                from pdb import set_trace
                set_trace()

        if len(xdot) > 0:
            return np.concatenate(xdot)
        else:
            return xdot

    @property
    def x(self):
        """
        Get the current state vector.

        Returns:
            np.array: The current state vector.
        """
        return self.state

    def _jacobian(self, x):
        """
        Calculate the Jacobian matrix for the network using numerical
        differentiation.

        Args:
            x (array-like): State vector for which to compute the Jacobian.

        Returns:
            np.ndarray: Jacobian matrix.
        """
        # Save the current state
        original_x = np.copy(self.x)

        # Evaluate the function at x
        f0 = self.evaluate(x)

        # Compute the Jacobian using finite differences
        jacobian_matrix = approx_derivative(
            self.evaluate,
            x,
            rel_step=None,
            abs_step=1e-10,
            method='3-point',
            f0=f0
        )

        # Restore the original state
        self.update_state(original_x)

        return jacobian_matrix

    @property
    def jacobian(self):
        """
        Jacobian matrix at the current state.

        Returns:
            np.ndarray: Jacobian matrix for the current state vector.
        """
        return self._jacobian(self.x)
