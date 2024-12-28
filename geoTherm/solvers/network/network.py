import numpy as np
from scipy.optimize._numdiff import approx_derivative
from .junctions import (
    Junction, ThermalJunction, OutletJunction, BoundaryJunction, FlowJunction
)
from .branches import ThermalBranch, FlowBranch
from ...nodes.boundary import Outlet, Boundary
from ...nodes.heatsistor import Qdot
from ...logger import logger


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
        self.initialize()

    def initialize(self):
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
        self.junctions = dict.fromkeys(junctions, None)

        self.istate = {}  # Maps state indices for each network component
        self.__x = []  # State vector

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
                # Create a FlowJunction if there are any flow branches
                self.junctions[junction_id] = FlowJunction(
                    name=junction_id,
                    node=node,
                    network=self)
            else:
                self.junctions[junction_id] = Junction(
                    name=junction_id,
                    node=node,
                    network=self)

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
            US_junction = self.junctions[self.net_map[branch_id]['hot']]
            DS_junction = self.junctions[self.net_map[branch_id]['cool']]

            self.thermal_branches[branch_id] = ThermalBranch(
                name=branch_id,
                nodes=nodes,
                US_junction=US_junction,
                DS_junction=DS_junction,
                network=self)

        self.branches = {**self.flow_branches, **self.thermal_branches}

        for junction in self.junctions.values():
            junction.initialize()

        # Initialize state vector for branches
        for obj_id, obj in {**self.branches, **self.junctions}.items():
            state_length = len(obj.x)
            if state_length == 0:
                continue

            current_length = len(self.__x)
            self.istate[obj_id] = np.arange(current_length,
                                            current_length + state_length)
            self.__x = np.concatenate((self.__x, obj.x))

        # Evaluate Network
        self.evaluate(self.x)

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
        remaining_nodes = set(node_map.keys())  # Nodes to be processed
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
            US_node = self.model.nodes[junction]
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

            cool_nodes = node_map[junction]['cool']
            for downstream_node in cool_nodes:
                DS_node = self.model.nodes[downstream_node]
                if downstream_node in [flow_junctions, thermal_junctions]:
                    downstream_junction = downstream_node
                    from pdb import set_trace
                    set_trace()
                else:
                    current_branch = []
                    downstream_junction = trace_branch(downstream_node,
                                                       current_branch,
                                                       'thermal')

                    if isinstance(US_node, Qdot):
                        current_branch = [junction, *current_branch]
                    elif isinstance(downstream_junction, Qdot):
                        current_branch = [*current_branch, junction]

                thermal_branches[branch_counter] = current_branch

                thermal_nodes = []
                for node in current_branch:
                    if node in thermal_junctions:
                        thermal_nodes.append(node)

                net_map[junction]['cool'].append(branch_counter)
                net_map[downstream_junction]['hot'].append(branch_counter)
                net_map[branch_counter] = {'US': [],
                                           'DS': [],
                                           'hot': junction,
                                           'cool': downstream_junction}

                branch_counter += 1

        for junction in thermal_junctions:

            downstream_nodes = node_map[junction]['cool']
            US_node = self.model.nodes[junction]

            for downstream_node in downstream_nodes:
                DS_node = self.model.nodes[downstream_node]
                if downstream_node in [*flow_junctions, *thermal_junctions]:
                    downstream_junction = downstream_node
                    if isinstance(US_node, Qdot):
                        current_branch = [junction]
                    elif isinstance(DS_node, Qdot):
                        current_branch = [downstream_junction]
                    else:
                        from pdb import set_trace
                        set_trace()
                else:
                    current_branch = []
                    downstream_junction = trace_branch(downstream_node,
                                                       current_branch,
                                                       'thermal')

                    if isinstance(US_node, Qdot):
                        current_branch = [junction, *current_branch]
                    elif isinstance(downstream_junction, Qdot):
                        current_branch = [*current_branch, junction]

                thermal_branches[branch_counter] = current_branch

                thermal_nodes = []
                for node in current_branch:
                    if node in thermal_junctions:
                        thermal_nodes.append(node)

                net_map[junction]['cool'].append(branch_counter)
                net_map[downstream_junction]['hot'].append(branch_counter)
                net_map[branch_counter] = {'US': [], 'DS': [],
                                           'hot': junction,
                                           'cool': downstream_junction}

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


    def update_state(self, x):
        """
        Update the state vector for the network.

        Args:
            x (array): The new state vector to update.
        """
        self.__x = x

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


        for _, branch in self.thermal_branches.items():
            branch.evaluate()
        
        for _, branch in self.flow_branches.items():
            branch.evaluate()
        
        from pdb import set_trace
        #set_trace()
        for _, junction in self.junctions.items():
            junction.evaluate()
            
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
        return self.__x

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