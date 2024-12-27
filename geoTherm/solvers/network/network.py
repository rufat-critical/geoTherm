# geotherm/folders/solvers/network.py
import numpy as np
from scipy.optimize._numdiff import approx_derivative
from .logger import logger

class Network:
    """
    Network Solver for geoTherm.

    This class is responsible for managing and solving the network of
    branches and junctions in the geoTherm model. It initializes the network
    topology, updates states, and evaluates the model to compute the results.
    """
    
    def __init__(self, model):
        """
        Initialize the network solver with the given geoTherm model.

        Args:
            model (Model): The geoTherm model containing nodes, branches, 
                           and junctions.
        """
        self.model = model
        self.branches = {}
        self.junctions = {}
        self.istate = {}  # Map for state indices in the global state vector
        self.__x = []  # State vector for the network
        self.initialize()

    def initialize(self):
        """
        Initialize the network by identifying branches and junctions, and
        creating corresponding objects for them.
        """
        # Ensure model is initialized
        if not self.model.initialized:
            self.model.initialize()

        # Identify branches and junctions in the node map
        branches, junctions, branch_connections, _ = self._identify_branches_and_junctions()

        # Create Branch and Junction objects
        for branch_id, nodes in branches.items():
            self.branches[branch_id] = Branch(
                name=branch_id,
                nodes=nodes,
                US_junction=branch_connections[branch_id]['US'],
                DS_junction=branch_connections[branch_id]['DS'],
                model=self.model
            )

        for junction_id, junction_map in junctions.items():
            upstream_branches = [self.branches[branch_id] for branch_id in junction_map['US']]
            downstream_branches = [self.branches[branch_id] for branch_id in junction_map['DS']]

            self.junctions[junction_id] = Junction(
                node=self.model.nodes[junction_id],
                US_branches=upstream_branches,
                DS_branches=downstream_branches,
                model=self.model
            )

        # Initialize state vector and indexing for branches and junctions
        self._initialize_state_vector()

    def _initialize_state_vector(self):
        """Initialize the state vector based on branches and junctions."""
        self.__x = []  # Reset the state vector
        for branch_name, branch in self.branches.items():
            if branch.x.size > 0:
                state_length = len(branch.x)
                current_len = len(self.__x)
                self.istate[branch_name] = np.arange(current_len, current_len + state_length)
                self.__x = np.concatenate((self.__x, branch.x))

        for junction_name, junction in self.junctions.items():
            if hasattr(junction, 'x'):
                state_length = len(junction.x)
                current_len = len(self.__x)
                self.istate[junction_name] = np.arange(current_len, current_len + state_length)
                self.__x = np.concatenate((self.__x, junction.x))

        # After initialization, evaluate the network to set the correct state
        self.evaluate(self.__x)

    def _identify_branches_and_junctions(self):
        """
        Identify all branches and junctions in the geoTherm node map.

        Branches are segments that connect two junctions, while junctions are 
        nodes where multiple branches meet. This function parses the model's 
        node map to classify nodes as either branches or junctions.

        Returns:
            tuple: Containing branches, junctions, branch_connections, and 
                   node classification.
        """
        node_map = self.model.node_map

        # Initialize data structures
        remaining_nodes = list(node_map.keys())
        junctions = {}
        branches = {}
        branch_connections = {}
        branch_counter = 0

        def is_junction(node_name):
            """Check if a node is a junction (has multiple inlets/outlets)."""
            node = self.model.nodes[node_name]

            if isinstance(node, Boundary):
                return True
            return isinstance(node, (gt.Boundary)) or \
                   len(node_map[node_name]['US']) != 1 or \
                   len(node_map[node_name]['DS']) != 1

        # Identify junctions
        for node_name in remaining_nodes:
            if is_junction(node_name):
                junctions[node_name] = {'US': [], 'DS': []}

        remaining_nodes = [node for node in remaining_nodes if node not in junctions]

        def trace_branch(current_node, current_branch):
            """Recursively trace a branch starting from the current node."""
            if current_node not in remaining_nodes:
                return current_node

            current_branch.append(current_node)
            remaining_nodes.remove(current_node)

            downstream_node = node_map[current_node]['DS']
            if downstream_node:
                return trace_branch(downstream_node[0], current_branch)

        # Identify branches by tracing downstream from each junction
        for junction_name in junctions:
            downstream_nodes = node_map[junction_name]['DS']
            for node_name in downstream_nodes:
                current_branch = []
                downstream_junction = trace_branch(node_name, current_branch)
                branches[branch_counter] = current_branch
                junctions[junction_name]['DS'].append(branch_counter)
                branch_connections[branch_counter] = {
                    'US': junction_name,
                    'DS': downstream_junction
                }
                branch_counter += 1

        return branches, junctions, branch_connections, None

    def update_state(self, x):
        """
        Update the state vector for branches and junctions.

        Args:
            x (np.array): The new state vector.
        """
        self.__x = x
        for name, istate in self.istate.items():
            if name in self.branches:
                self.branches[name].update_state(x[istate])
            else:
                self.junctions[name].update_state(x[istate])

    def evaluate(self, x):
        """
        Evaluate the network by updating the state of each branch and junction.

        Args:
            x (np.array): The current state vector.

        Returns:
            np.array: The derivative of the state vector (xdot).
        """
        self.update_state(x)
        for branch in self.branches.values():
            branch.evaluate()
        for junction in self.junctions.values():
            junction.evaluate()
        return self.xdot

    @property
    def xdot(self):
        """Return the derivative of the state vector."""
        xdot = []
        for name in self.istate:
            if name in self.branches:
                xdot.append(self.branches[name].xdot)
            else:
                xdot.append(self.junctions[name].xdot)
        return np.concatenate(xdot)

    @property
    def x(self):
        """Get the current state vector."""
        return self.__x

    def jacobian(self, x):
        """
        Calculate the Jacobian matrix for the network using numerical differentiation.

        Args:
            x (np.array): The state vector.

        Returns:
            np.array: The Jacobian matrix.
        """
        f0 = self.evaluate(x)
        return approx_derivative(self.evaluate, x, f0=f0)

