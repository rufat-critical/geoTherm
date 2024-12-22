import numpy as np
import geoTherm as gt
from scipy.optimize import fsolve
from scipy.integrate import BDF
from scipy.optimize._numdiff import approx_derivative
from .nodes.node import modelTable
from .utilities.thermo_plotter import thermoPlotter
from .logger import logger
from .units import addQuantityProperty
from .utils import eps, parse_component_attribute
from .utilities.network_graph import make_dot_diagram, make_graphml_diagram
from .thermostate import thermo
import pandas as pd
from solvers.cvode import CVode_solver
from .utils import thermo_data

class GlobalLimits:
    P = [1, 1e8]


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
            self.x_scale[indx] = scale

    def _determine_nodal_scale(self, node):
        """Determine the scale for a nodal solver based on node properties."""
        if (hasattr(node, "_bounds") and
                node._bounds[0] != -np.inf and
                node._bounds[1] != np.inf):
            return 1 / (node._bounds[1] - node._bounds[0])
        elif isinstance(node, gt.TBoundary):
            return np.array([1e-1])
        elif isinstance(node, gt.PBoundary):
            return np.array([1e-1])
        elif isinstance(node, gt.ThermoNode):
            return np.array([1e-1, 1e-5])
        else:
            from pdb import set_trace
            set_trace()
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
                return 1e-4            
        elif name in self.model.network.junctions:
            node = self.model.nodes[name]
            if isinstance(node, gt.Balance):
                return 1 / (node.knob_max - node.knob_min)
            elif isinstance(node, gt.Heatsistor):
                return 1e-4
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
            xdot_unscaled = func(x_unscaled)
            return self.scale_x(xdot_unscaled)
        return wrapper


class Model(modelTable):
    """ geoTherm System Model """

    def __init__(self, nodes=[]):
        """ Instantiate a geoTherm Model

        Args:
            nodes (list, optional): List of geoTherm nodes
        """
        # Get the names of the nodes and check if they are unique
        names = [node.name for node in nodes]
        unique, count = np.unique(names, return_counts=True)
        # Check for duplicates where count is more than 1
        dup = unique[count > 1]
        # Output error if duplicates are present
        if len(dup) != 0:
            logger.critical("Multiple nodes specified with the same name:\n"
                            ",".join(dup))

        # Create dictionary of nodes
        self.nodes = {node.name: node for node in nodes}

        # Initialization Flag
        self.initialized = False
        # Debug Flag
        self.debug = False

        # Try to initialize Model otherwise output error
        try:
            self.initialize()
        except Exception:
            logger.warn('Failed to Initialize Model')
            self.initialized = False

    def __getitem__(self, nodeName):
        # Return Node if model is indexed by node name
        if nodeName in self.nodes:
            return self.nodes[nodeName]
        else:
            raise ValueError(f'{nodeName} has not been defined')

    def get_node_attribute(self, attribute_string):
        """
        Retrieves the value of a specified attribute from a node in the
        model using the attribute string.

        Args:
            attribute_string (str): The attribute string in the format
                                    'Node.Attribute' or
                                    'Node.SubComponent.Attribute'.

        Returns:
            Any: The value of the specified attribute.

        Raises:
            AttributeError: If the attribute does not exist in the Node.
        """

        # Parse the attribute string to get the node name and attribute chain
        name, attribute_chain = parse_component_attribute(attribute_string)

        # Retrieve the node from the model
        component = self.nodes.get(name)

        if component is None:
            raise AttributeError(f"Node '{component}' not found in the model.")

        # Traverse through the attribute chain to get the final attribute value
        try:
            for attr in attribute_chain.split('.'):
                component = getattr(component, attr)
            return component
        except AttributeError as e:
            raise AttributeError(
                f"Attribute '{attribute_chain}' not found in Node "
                f"'{name}'."
            ) from e

    def initialize(self, t=0):
        """Initialize the model and nodes"""

        # Initialize variables for tracking statefuls
        self.statefuls = []
        self.istate = {}
        self.__nstates = 0
        self.t = t

        # Initialize State Vector (as private variable)
        self.__x = np.array([])

        # Loop thru nodes and add a reference to this model
        for name, node in self.nodes.items():
            node.model = self

        # Generate node_map
        self.node_map = self._generatenode_map()

        # Loop thru nodes and call initialization method if defined
        for name, node in self.nodes.items():
            if hasattr(node, 'initialize'):
                node.initialize(self)

        # Loop thru nodes and call evlauate method
        for name, node in self.nodes.items():
            if hasattr(node, 'evaluate'):
                node.evaluate()

        # Identify stateful components
        for name, node in self.nodes.items():
            if hasattr(node, 'x'):
                self.statefuls.append(name)
                xlen = len(node.x)
                current_len = self.__nstates
                self.istate[name] = np.arange(current_len, current_len + xlen)
                self.__nstates = current_len + xlen

        # Initialize the model state vector
        self.__init_x()

        if len(self.__x) > 0:
            self.__error = np.zeros(len(self.__x))
        else:
            self.__error = []

        # Run error checker
        if self._error_checker():
            logger.critical("Error Checker Found Errors, Fix the Model!")

        self.initialized = True

    def __init_x(self):
        # Initialize the model state vector using the current
        # component states
        self.__x = np.empty(self.__nstates)

        for name, indx in self.istate.items():
            self.__x[indx] = self.nodes[name].x

    def _error_checker(self):
        """Check the model for potential errors"""

        # These caused me some headache in building/debugging
        # so writing it to make development easier

        error = False
        thermoID = {}
        # Error check that the thermo Objects have a unique thermo object
        for name, node in self.nodes.items():
            if hasattr(node, 'thermo'):
                if id(node.thermo) in thermoID:
                    logger.error(f"Node '{name}' is using the same thermo "
                                 f"Object as '{thermoID[id(node.thermo)]}'!")
                    error = True
                else:
                    thermoID[id(node.thermo)] = name

        # Check that outlet has fixed flow upstream

        # Check that boundary and fixed flow are not in series

        return error

    def update_state(self, x):
        """Update the component states in the model"""
        self.__x[:] = x

        # Update Component states
        for name, istate in self.istate.items():
            self.nodes[name].update_state(x[istate])

        for name, node in self.nodes.items():
            node.evaluate()

    @property
    def x(self):
        """Get the state vector"""
        return self.__x

    @property
    def xdot(self):
        """Get the error vector"""
        for name, istate in self.istate.items():
            self.__error[istate] = self.nodes[name].xdot

        return np.copy(self.__error)

    def _generatenode_map(self):
        """ Generate a nodal connectivity map for the defined
            nodes """

        # Initialize empty Node dictionary
        nMap = {name: {'US': [], 'DS': [], 'hot': [], 'cool': []} 
                for name in self.nodes}

        # Loop thru all nodes and find connectivity
        for name, node in self.nodes.items():
            # Check if US defined in node
            if hasattr(node, 'US'):
                if node.US not in nMap:
                    logger.critical(
                        f"'{node.name}' US Node '{node.US}' has not "
                        "been defined in the model"
                    )

                # Check if it already exists in node_map
                if node.US not in nMap[name]['US']:
                    # If not then append to node_map
                    nMap[name]['US'].append(node.US)
                # Check if downstream node exsits in node_map
                if node.DS not in nMap[name]['DS']:
                    # If not then append it too
                    nMap[name]['DS'].append(node.DS)

                # Now check the downstream nodes
                if name not in nMap[node.US]['DS']:
                    nMap[node.US]['DS'].append(name)
                if name not in nMap[node.DS]['US']:
                    nMap[node.DS]['US'].append(name)

            if hasattr(node, 'hot'):
                # Check if it already exists in node_map
                if node.hot not in nMap[name]['hot']:
                    # If not then append to node_map
                    nMap[name]['hot'].append(node.hot)
                # Check if cool node exists in node_map
                if node.hot not in nMap[node.hot]['cool']:
                    # If not then append it too
                    nMap[node.hot]['cool'].append(name)

            if hasattr(node, 'cool'):            
                # Check if it already exists in node_map
                if node.cool not in nMap[name]['cool']:
                    # If not then append to node_map
                    nMap[name]['cool'].append(node.cool)
                # Check if cool node exists in node_map
                if node.cool not in nMap[node.cool]['hot']:
                    # If not then append it too
                    nMap[node.cool]['hot'].append(name)

        return nMap

    def evaluate(self, t, x):
        """Evaluate the model with given state vector x"""

        # Update time
        self.t = t

        # First update the model and component states
        self.update_state(x)

        # Evaluate the nodes
        self.evaluate_nodes()

        # Return the error
        return self.xdot

    def steady_evaluate(self, x, t=0):
        return self.evaluate(t, x)

    def evaluate_nodes(self):
        for name, node in self.nodes.items():
            if hasattr(node, 'evaluate'):
                node.evaluate()

    def addNode(self, nodes):
        """ Add nodes to the model """

        # Loop thru list of nodes
        if isinstance(nodes, list):
            for node in nodes:
                # Call addNode for each node individually
                self.addNode(node)
        elif isinstance(nodes, gt.Node):
            # This is for single node input
            if nodes.name in self.nodes:
                msg = f"'{nodes.name}' already exists in the model - " \
                    "rename this node to add it to the model"
                logger.warn(msg)
                return
            # Add node to the model if the name is unique
            self.nodes[nodes.name] = nodes
            self.initialized = False
        else:
            msg = f'Unknown Node Input Specified: {nodes}'
            logger.warn(msg)
            return

    def removeNode(self, node):
        """ Delete nodes from model"""
        if isinstance(node, str):
            if ',' in node:
                from pdb import set_trace
                set_trace()

            if node in self.nodes:
                msg = f"Deleting '{node}' from Model"
                logger.info(msg)
                self.nodes.pop(node)
                self.initialized = False
            else:
                msg = f" Node '{node}' not found in Model"
                logger.warn(msg)
        elif isinstance(node, gt.Node):
            from pdb import set_trace
            set_trace()
        elif isinstance(node, list):
            from pdb import set_trace
            set_trace()
        else:
            msg = f"removeNode got an unknown Node Input {node}"
            logger.warn(msg)
            return

    def merge(self, otherModel):
        """ Merge with another model"""

        if not isinstance(otherModel, Model):
            logger.warn("Model to merge is not a Model Class")
            return

        for _, node in otherModel.nodes.items():
            self.addNode(node)

    def __iadd__(self, node):
        """ Add nodes to the model"""

        if isinstance(node, (gt.Node, list)):
            self.addNode(node)
        elif isinstance(node, Model):
            self.merge(node)
        else:
            logger.warn(f"{node} is not a Model or Node type")

        return self

    def __isub__(self, node):
        """ Remove nodes from model """
        self.removeNode(node)

        return self

    def thermo_plot(self, plot_type='TS',
                    isolines=None,
                    process_lines=True,
                    x_scale='linear'):

        plot_flag = True
        for name, node in self.nodes.items():
            if isinstance(node, (gt.ThermoNode, gt.Station)):
                if plot_flag:
                    plot = thermoPlotter(node.thermo)
                    plot_flag = False

                plot.add_state_point(name, node.thermo)

                if isolines is not None:
                    plot.add_isoline(isolines, name)

        if process_lines:
            for name in plot.state_points:
                DS_flow = self.node_map[name]['DS'][0]
                DS_node = self.node_map[DS_flow]['DS'][0]
                if isinstance(self.nodes[DS_flow], gt.Turbo):
                    process_type = 'S'
                else:
                    process_type = 'P'
                plot.add_process_line(DS_flow, name, DS_node, process_type,
                                      line_style='-')

        plot.plot(plot_type=plot_type, xscale=x_scale)

    def sim(self, t_span, x0=None, max_step=None, solver='CVODE'):

        # Initialize if not
        if not self.initialized:
            self.initialize()

        self.initialize()
        if x0 is None:
            x0 = np.copy(self.x)

        # Generate Solution Object        
        sol = Solution(self, extras=['t'])
        # Save current time
        sol.save([t_span[0]])
        if solver == 'CVODE':
            # Get t and x from Cvode
            t, x = CVode_solver(self.evaluate, x0, t_span)
            for i, _ in enumerate(t):
                self.evaluate(t[i], x[i])
                sol.save([t[i], self.xdot[0], self.xdot[1], self.x[0]])
        else:
            # Create BDF integrator
            integrator = BDF(self.evaluate, t_span[0],
                             x0, t_span[1],
                             max_step=1e-7,
                             rtol=1e-5)

            while integrator.t < t_span[1]:
                integrator.step()
                sol.save([integrator.t])

        return sol

    def solve_steady(self, netSolver=True):

        # Branch Nodes
        # Set mass flow
        # getOutlet State
        # update
        # Error to update mass flow

        # Initialize the model if it's not initialized
        if not self.initialized:
            try:
                self.initialize()
            except Exception:
                logger.critical("Could not Initialize Model")

        # Check if this is a stateless model
        if len(self.x) == 0:
            # We don't need to solve anything
            # Just evaluate nodes
            self.evaluate_nodes()
            return self.x

        if netSolver:
            # Initialize the network solver
            self.network = Network(self)

            # Check if this is a stateless network model
            if len(self.network.x) == 0:
                # We don't need to solve anything
                return self.network.x

            # Use Network Conditioner
            conditioner = Conditioner(self.network)
            conditioned = conditioner.conditioner(self.network.evaluate)
            # Scale the state vector for fsolve
            x_scaled = conditioner.scale_x(self.network.x)
            # Run fsolve with scaling
            sol = fsolve(conditioned, x_scaled, full_output=True)
            # Unscale/re-scale the solution back to normal
            x = conditioner.unscale_x(sol[0])

            # Update to network state
            self.network.evaluate(x)
            from pdb import set_trace
            #set_trace()
            if not self.converged:
                logger.warn("Failed to converge with Network Solver, "
                            "will try Nodal Solver Next!")
            else:
                return self.x

        conditioner = Conditioner(self)
        conditioned = conditioner.conditioner(self.steady_evaluate)
        # Scale the state vector for fsolve
        x_scaled = conditioner.scale_x(self.x)
        sol = fsolve(conditioned, x_scaled, full_output=True)

        x = conditioner.unscale_x(sol[0])
        self.steady_evaluate(x)
        from pdb import set_trace
        set_trace()
        if not self.converged:
            logger.warn("Failed to converge with Nodal Solver, "
                        "will try Transient Solver Next!")
        else:
            return self.x

        # USE SIM TO TRY AND SOVLE MODEL
        from pdb import set_trace
        set_trace()

    def draw(self, file_path='geoTherm_model_diagram.svg', auto_open=True):
        """
        Generates a DOT plot for the model's node network and saves it as an
        SVG file. Optionally, the plot can be opened automatically in a web
        browser.

        Args:
            model (Model): The geoTherm model containing nodes and their
                        connections.
            file_path (str): The path to save the generated SVG file.
                            Defaults to 'plot.svg'.
            auto_open (bool): Whether to automatically open the SVG file after
                            creation. Defaults to True.
        """
        make_dot_diagram(self, file_path='plot.svg', auto_open=auto_open)

    def make_graphml_diagram(self, file_path='geoTherm_model_diagram.graphml'):
        """
        Generates a GraphML file for the model's node network and saves it to
        the specified file path.

        Args:
            model (Model): The geoTherm model containing nodes and their
                        connections.
            file_path (str): The path to save the generated GraphML file.
                            Defaults to 'plot.graphml'.
        """
        make_graphml_diagram(self, file_path)

    def getFlux(self, node):
        # Calculate mass/energy flux into/out of a node
        # At Steady state mass/energy should be = 0

        # Get node_map for this node
        node_map = self.node_map[node.name]

        wNet = 0
        Hnet = 0
        Wnet = 0
        Qnet = 0

        for name in node_map['US']:

            # Get the Flow Node
            flowNode = self.nodes[name]

            # Get the upstream Station Thermo State
            usNode = self.nodes[self.node_map[name]['US'][0]].thermo

            # Sum Mass Flow
            wNet += self.nodes[name]._w

            if flowNode._w > 0:
                # Inflow Energy
                Hnet += flowNode._w*usNode._H
                # Work from flow Node
                Wnet += flowNode._w*flowNode._dH
            else:
                # Outflow Energy (flowNode w is negative so this subtracts)
                Hnet += flowNode._w*node.thermo._H

        for name in node_map['DS']:

            # Get the Flow Node
            flowNode = self.nodes[name]
            # Get the dowstream Station Thermo State
            dsNode = self.nodes[self.node_map[name]['DS'][0]].thermo

            # Subtract massflow out, if outflow is negative
            # then that means backflow and wnet gets more positive
            wNet += -flowNode._w

            if flowNode._w > 0:
                # Outflow Energy
                Hnet -= node.thermo._H*flowNode._w
            else:
                # Inflow Energy (flowNode is negative so this is positive)
                Hnet -= flowNode._w*dsNode._H
                # Work from flow Node
                Wnet -= flowNode._w*flowNode._dH

        # Sum the heat in/out of the node
        for name in node_map['hot']:
            Qnet += self.nodes[name]._Q

        for name in node_map['cool']:
            Qnet -= self.nodes[name]._Q

        return wNet, Hnet, Wnet, Qnet

    @property
    def converged(self):

        # Reinitialize model x using component states
        self.__init_x()

        self.update_state(self.x)

        # Get the error
        if all(abs(self.xdot/(self.x + eps)) < 1e-2):
            return True
        else:
            return False

    @property
    def performance(self):
        # Calculate Power, Qin 

        Qnet = 0.
        Wnet = 0.
        Qin = 0.
        for name, node in self.nodes.items():
            if isinstance(node, (gt.ThermoNode, gt.Station)):
                continue
            
            if isinstance(node, (gt.Heat, gt.simpleHEX)):
                if hasattr(node, 'Q'):
                    Qnet += node.Q
                    if node.Q > 0:
                        Qin += node.Q

            if hasattr(node, 'W'):
                Wnet += node.W

        if Qin == 0:
            eta = 0
        else:
            eta = Wnet/Qin*1e2

        return np.array([Wnet, Qin, eta])

    # Performance Calc
        # Loop thru all components with W and add positive/negative power
        # Loop thru heat transfer nodes and add all Qin
        # Eta = Qin
        # dP loss in pipe/heat exchanger

    def Carnot_eta(self):
        # Calculate Carnot Efficiency
        pass


class Network:
    """
    Network Solver for geoTherm.

    This class is responsible for managing and solving the network of
    branches and junctions in the geoTherm model. It initializes the network
    topology, updates states, and evaluates the model to compute the results.
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

        # Ensure model is initialized
        if not self.model.initialized:
            self.model.initialize()

        # Get the branches and junctions from the model's node map
        #(branches, junctions, branch_connections,
        ## node_classification) = self._identify_branches_and_junctions(
        #    self.model.node_map
        #)
        
        ((flow_branches, flow_branch_connections),
        (thermal_branches, thermal_branch_connections),
        junctions)= self._identify_branches_and_junctions(
            self.model.node_map)
        
        

        # Store branches and junctions
        self.flow_branches = dict(flow_branches)
        self.thermal_branches = dict(thermal_branches)
        self.junctions = junctions
        self.istate = {}  # Index for state variables
        self.__x = []  # State vector


        # Initialize branches
        for branch_id, nodes in flow_branches.items():
            # Create branch objects
            self.flow_branches[branch_id] = FlowBranch(
                name=branch_id,
                nodes=nodes,
                US_junction=flow_branch_connections[branch_id]['US'],
                DS_junction=flow_branch_connections[branch_id]['DS'],
                model=self.model
            )

        for branch_id, nodes in thermal_branches.items():
            self.thermal_branches[branch_id] = ThermalBranch(
                name=branch_id,
                nodes=nodes,
                US_junction=thermal_branch_connections[branch_id]['hot'],
                DS_junction=thermal_branch_connections[branch_id]['cool'],
                model=self.model)

        # Initialize junctions
        for junction_id, junction_map in junctions.items():
            US_flow_branches = [self.flow_branches[branch_id] for branch_id
                                 in junction_map['US']]
            DS_flow_branches = [self.flow_branches[branch_id] for branch_id
                                 in junction_map['DS']]
                                 
            US_thermal_branches = [self.thermal_branches[branch_id] for branch_id
                                   in junction_map['hot']]
            DS_thermal_branches = [self.thermal_branches[branch_id] for branch_id
                                   in junction_map['cool']]
                                   

            self.junctions[junction_id] = Junction(
                node=self.model.nodes[junction_id],
                US_flow_branches=US_flow_branches,
                DS_flow_branches=DS_flow_branches,
                US_thermal_branches=US_thermal_branches,
                DS_thermal_branches=DS_thermal_branches,
                model=self.model
            )

        # Initialize Branches and Junctions if not initialized
        #for branch_name, branch in self.branches.items():
        #    if not branch.initialized:
        #        from pdb import set_trace
        #        set_trace()

        #for junction_name, junction in self.junctions.items():
        #    if not junction.initialized:
        #        from pdb import set_trace
        #        set_trace()

        # Initialize state vector for branches
        for branch_id, branch in self.flow_branches.items():
            if len(branch.x) == 0:
                continue

            state_length = len(branch.x)
            current_length = len(self.__x)
            self.istate[branch_id] = np.arange(current_length,
                                                 current_length + state_length)
            self.__x = np.concatenate((self.__x, branch.x))

        for branch_id, branch in self.thermal_branches.items():
            if len(branch.x) == 0:
                continue

            state_length = len(branch.x)
            current_length = len(self.__x)
            self.istate[branch_id] = np.arange(current_length,
                                                 current_length + state_length)
            
            self.__x = np.concatenate((self.__x, branch.x))


        # Initialize state vector for junctions
        for junc_name, junction in self.junctions.items():
            if not hasattr(junction, 'x'):
                continue

            state_length = len(junction.x)
            current_length = len(self.__x)
            self.istate[junc_name] = np.arange(current_length,
                                               current_length + state_length)
            self.__x = np.concatenate((self.__x, junction.x))

        # Evaluate Network
        self.evaluate(self.x)

    def _identify_branches_and_junctions(self, node_map):
        """
        Identify all branches and junctions in the geoTherm node map.

        Args:
            node_map (dict): A dictionary representing the node map of the
                             geoTherm system.

        Returns:
            tuple: A tuple containing:
                - branches (dict): A dictionary of branches.
                - junctions (dict): A dictionary of junctions.
                - branch_connections (dict): A dictionary mapping branches
                  to their upstream and downstream junctions.
                - node_classification (dict): A dictionary mapping nodes to
                  their corresponding branch or junction.
        """

        # Initialize data structures
        remaining_nodes = list(node_map.keys())  # Nodes to be processed
        junctions = {}  # Dictionary of junctions
        branches = {}  # Dictionary of branches
        branch_counter = 0  # Identifier for branches
        branch_connections = {}  # Map branches to their connections
        node_classification = {}  # Map nodes to their classification

        # Helper function to determine if a node is a junction
        def is_junction(node_name):
            """
            Determine if a node is a junction.

            Junctions are boundary nodes, nodes with more than one inlet or
            outlet, or nodes with a heat connection (hot or cool).

            Args:
                node_name (str): The name of the node.

            Returns:
                bool: True if the node is a junction, False otherwise.
            """
            node = self.model.nodes[node_name]

            if isinstance(node, (gt.Boundary)):
                return True

            if node_map[node_name]['hot'] or node_map[node_name]['cool']:
                from pdb import set_trace
                set_trace()
                if (len(node_map[node_name]['US']) == 0 and
                        len(node_map[node_name]['DS']) == 0):
                    return True
                return False

            return (len(node_map[node_name]['US']) != 1 or
                    len(node_map[node_name]['DS']) != 1 or
                    len(node_map[node_name]['hot']) != 0 or
                    len(node_map[node_name]['cool']) != 0)

        # Identify all junction nodes
        for node_name in remaining_nodes:
            if is_junction(node_name):
                junctions[node_name] = {'US': [], 'DS': [],
                                        'hot': [], 'cool': []}
                node_classification[node_name] = node_name
            else:
                from pdb import set_trace
                set_trace()

        # Remove junction nodes from remaining_nodes list
        remaining_nodes = [node for node in remaining_nodes
                           if node not in junctions]

        # Helper function to trace branches starting from a node
        def trace_branch(node_map, current_node, current_branch,
                         remaining_nodes):
            """
            Recursively trace a branch starting from the current node.

            Args:
                node_map (dict): The node map of the geoTherm system.
                current_node (str): The current node to trace from.
                current_branch (list): The list to store the nodes in the
                                      current branch.
                remaining_nodes (list): The list of all remaining nodes to
                                       process.

            Returns:
                str: The name of the downstream junction node.
            """
            if current_node not in remaining_nodes:
                return current_node

            current_branch.append(current_node)
            remaining_nodes.remove(current_node)
            node_classification[current_node] = branch_counter

            downstream_node = node_map[current_node]['DS']
            if downstream_node:
                return trace_branch(node_map, downstream_node[0],
                                    current_branch,
                                    remaining_nodes)

        # Identify branches and their connections
        for junction_name in junctions:
            downstream_nodes = node_map[junction_name]['DS']
            for node_name in downstream_nodes:
                current_branch = []
                if is_junction(node_name):
                    from pdb import set_trace
                    set_trace()  # This should not occur
                else:
                    downstream_junction = trace_branch(node_map, node_name,
                                                       current_branch,
                                                       remaining_nodes)
                    branches[branch_counter] = current_branch
                    junctions[junction_name]['DS'].append(branch_counter)
                    try:
                        junctions[downstream_junction]['US'].append(
                            branch_counter)
                        branch_connections[branch_counter] = {
                            'US': junction_name,
                            'DS': downstream_junction,
                            'hot': [], 'cool': []
                        }
                        branch_counter += 1
                    except Exception:
                        from pdb import set_trace
                        set_trace()

        if len(remaining_nodes) != 0:
            from pdb import set_trace
            set_trace()  # Some nodes were not classified correctly

        return branches, junctions, branch_connections, node_classification


    def _identify_branches_and_junctions(self, node_map):
        """
        Identify all flow branches, thermal branches, and junctions in the node map.

        Args:
            node_map (dict): A dictionary representing the node map of the
                            geoTherm system.

        Returns:
            tuple: A tuple containing:
                - flow_branches (dict): A dictionary of flow branches.
                - thermal_branches (dict): A dictionary of thermal branches.
                - flow_branch_connections (dict): A dictionary mapping flow
                                                branches to their upstream and downstream junctions.
                - thermal_branch_connections (dict): A dictionary mapping
                                                    thermal branches to their upstream and downstream junctions.
                - junctions (dict): A dictionary of junctions with their connections.
        """

        # Initialize data structures
        remaining_nodes = set(node_map.keys())  # Nodes to be processed
        flow_branches = {}        # Dictionary of flow branches
        thermal_branches = {}     # Dictionary of thermal branches
        flow_junctions = []
        thermal_junctions = []
        junctions = {}            # Dictionary of junctions
        # Map flow branches to their connections
        flow_branch_connections = {}
        # Map thermal branches to their connections
        thermal_branch_connections = {}
        branch_counter = 0     # Identifier for flow branches

        # Helper functions
        def is_junction(node_name):
            """
            Determine if a node is a junction.

            A junction can be a flow junction or a thermal junction based on
                its connections.

            Args:
                node_name (str): The name of the node.

            Returns:
                bool: True if the node is a junction, False otherwise.
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
                and not thermal_junction):
                other_junction = True


            # Additional condition for boundary nodes
            if isinstance(node, gt.Boundary):
                if has_US and has_DS:
                    flow_junction = True

            return flow_junction, thermal_junction, other_junction

        # Identify all junction nodes
        # Iterate over a copy to allow removal
        for node_name in list(remaining_nodes):

            flow, thermal, other = is_junction(node_name)

            if flow:
                flow_junctions.append(node_name)

            if thermal:
                thermal_junctions.append(node_name)

            if flow or thermal or other:
                junctions[node_name] = {'US': [],
                                        'DS': [],
                                        'hot': [],
                                        'cool': []}
                
                
                remaining_nodes.remove(node_name)


        # Define branch tracing functions
        def trace_branch(current_node, current_branch, branch_type):
            """
            Trace a branch starting from the current node.

            Args:
                current_node (str): The current node to trace from.
                branch_type (str): Type of branch to trace
                                    ('flow' or 'thermal').

            Returns:
                tuple: (branch_nodes (list), downstream_junction (str))
            """

            if branch_type == 'flow':
                DS = 'DS'
                if current_node in flow_junctions:
                    return current_node
            elif branch_type == 'thermal':
                DS = 'cool'
                if current_node in [*thermal_junctions, *flow_junctions]:
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

            if branch_type == 'flow' and downstream_node in flow_junctions:
                # Return downstream junction
                return downstream_node
            elif (branch_type == 'thermal' and downstream_node 
                  in [*thermal_junctions, *flow_junctions]):
                return downstream_node
            else:
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

                # Record connections
                junctions[junction]['DS'].append(branch_counter)
                if downstream_junction in junctions:
                    junctions[downstream_junction]['US'].append(
                        branch_counter)
                else:
                    from pdb import set_trace
                    set_trace()
                flow_branch_connections[branch_counter] = {
                    'US': junction,
                    'DS': downstream_junction
                }
                branch_counter += 1

        for junction in thermal_junctions:
            downstream_nodes = node_map[junction]['cool']
            for downstream_node in downstream_nodes:
                
                if (downstream_node in junctions):
                    # No branch between junctions or already processed
                    continue

                current_branch = []
                downstream_junction = trace_branch(downstream_node,
                                                   current_branch,
                                                   'thermal')

                thermal_branches[branch_counter] = current_branch
                junctions[junction]['cool'].append(branch_counter)
                if downstream_junction in junctions:
                    junctions[downstream_junction]['hot'].append(
                        branch_counter)
                else:
                    from pdb import set_trace
                    set_trace()
                thermal_branch_connections[branch_counter] = {
                    'hot': junction,
                    'cool': downstream_junction
                }
                branch_counter+=1

        if len(remaining_nodes) != 0:
            logger.warn("The following nodes were not recognized in "
                        f"building the network map: {remaining_nodes}")
            from pdb import set_trace
            set_trace()

        return ((flow_branches, flow_branch_connections),
                (thermal_branches, thermal_branch_connections),
                junctions)


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


class Junction:
    """
    Represents a junction in the geoTherm model network.

    Junctions are points where multiple branches meet, allowing the solver
    to handle mass and energy conservation across these points.
    """

    def __init__(self, node, US_flow_branches, DS_flow_branches, 
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
        self.node = node
        self.US_flow_branches = US_flow_branches
        self.DS_flow_branches = DS_flow_branches
        self.US_thermal_branches = US_thermal_branches
        self.DS_thermal_branches = DS_thermal_branches
        self.model = model
        self.initialized = False
        self.initialize()

    def initialize(self):
        """
        Initialize the Junction by adding dynamic properties if necessary.
        """
        if isinstance(self.node, (gt.Boundary)):
            pass
        elif len(self.US_flow_branches) == len(self.DS_flow_branches) == 1:
            pass
        elif hasattr(self.node, 'x'):
            # Add properties and method if self.node has attribute 'x'
            self.__add_dynamic_properties()

        self.initialized = True

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

    def evaluate(self):
        """
        Evaluate the junction by calling its associated node's evaluate method.
        """
        self.node.evaluate()




@addQuantityProperty
class FlowBranch:
    """
    Represents a branch in the geoTherm model network.

    Branches are segments that connect different nodes in the network,
    facilitating flow and heat transfer between them.
    """

    _units = {'w': 'MASSFLOW'}
    _bounds = [-np.inf, np.inf]

    def __init__(self, name, nodes, US_junction, DS_junction, model, w=None):
        """
        Initialize a Branch instance.

        Args:
            nodes (list): List of node names or instances in sequential order.
            USJunc (str or instance): Upstream junction name or instance.
            DSJunc (str or instance): Downstream junction name or instance.
            model (object): The model containing all nodes and junctions.
            w (float, optional): Branch Mass Flow Rate with a default
                                 value of 0.
        """
        self.name = name

        # Convert node names to instances if necessary
        if isinstance(nodes[0], str):
            self.nodes = [model.nodes[name] for name in nodes]
        else:
            self.nodes = nodes

        if isinstance(US_junction, str):
            self.USJunc = model.nodes[US_junction]
            self.US_junction = model.nodes[US_junction]
        else:
            self.USJunc = US_junction
            self.US_junction = US_junction

        if isinstance(DS_junction, str):
            self.DSJunc = model.nodes[DS_junction]
            self.DS_junction = model.nodes[DS_junction]
        else:
            self.DSJunc = DS_junction
            self.DS_junction = DS_junction

        self.model = model
        # Mass Flow for all the nodes in the branch
        self._w = w
        # Flag to specify if mass flow is constant
        self.fixedFlow = False
        # Backflow flag
        self.backFlow = False
        # Node for controlling mass flow
        self.fixed_flow_node = None
        self.penalty = False
        self.solver = 'forward'

        self.initialized = False

        self.initialize()

    @property
    def average_w(self):
        """
        Calculate and return the average mass flow in all components.

        Returns:
            float: The average mass flow rate.
        """

        Wnet = 0
        flow_elements = 0

        for node in self.nodes:
            if isinstance(node, (gt.flowNode,
                          gt.BoundaryConnector)):
                
                node.evaluate()
                Wnet += node._w
                flow_elements += 1

        if flow_elements == 0:
            from pdb import set_trace
            set_trace()

        return float(Wnet/flow_elements)

    @property
    def xdot(self):
        
        if self.solver == 'forward':
            return self.xdot_forward
        elif self.solver == 'reverse':
            return self.xdot_reverse

    @property
    def xdot_forward(self):

        if self.penalty is not False:
            return np.array([self.penalty])

        if self._w < 0:
            Pout = self.USJunc.thermo._P
        else:
            Pout = self.DSJunc.thermo._P

        if self.fixedFlow:
            if isinstance(self.fixed_flow_node, gt.Turbine):
                self.error = (self.DS_target['P']/Pout - 1)*Pout
            elif isinstance(self.fixed_flow_node, gt.Pump):
                self.error = (Pout/self.DS_target['P'] - 1)*self.DS_target['P']*1e2

            else:
                from pdb import set_trace
                set_trace()

        else:
        #if self.fixedFlow:
            self.error = (np.sign(self.DS_target['P']-Pout)
                        * np.abs(self.DS_target['P']-Pout)**1.3
                        + (self.DS_target['P']-Pout)*10)

            self.error = (self.DS_target['P']-Pout)

        return np.array([self.error])

    @property
    def xdot_reverse(self):

        if self.penalty is not False:
            return np.array([self.penalty])

        if self._w < 0:
            Pin = self.DSJunc.thermo._P
        else:
            Pin = self.USJunc.thermo._P

        if self.fixedFlow:
            from pdb import set_trace
            set_trace()
        else:
            self.error = self.US_target['P']**2/Pin - Pin
        return np.array([self.error])


    def evaluate(self):

        if self.solver == 'forward':
            self.evaluate_forward()
        elif self.solver == 'reverse':
            self.evaluate_reverse()

    def evaluate_reverse(self):

        nodes = self.nodes

        if self.fixedFlow:
            self._w = self.fixed_flow_node._w

        if self.penalty:
            # If penalty was triggered then return
            return

        if self._w < 0:
            US_junction = self.DS_junction
            DS_junction = self.US_junction
        else:
            # Reverse node order
            nodes = nodes[::-1]
            US_junction = self.US_junction
            DS_junction = self.DS_junction

        DS_thermo = thermo_data(H=US_junction.thermo._H,
                            P=DS_junction.thermo._P,
                            fluid=US_junction.thermo.Ydict,
                            model=US_junction.thermo.model)

        Q = 0
        for name, hot_nodes in self.hot_nodes.items():
            for hot in hot_nodes:
                Q +=self.model.nodes[hot]._Q


        # I should put a try except state for updating DS thermo
        # If fail then heat flux is too high and need to increase
        # mass flow
        DS_thermo._H += Q/(self._w+1e-8)

        for inode, node in enumerate(nodes):

            if node.name in self.hot_nodes:
                DS_thermo._H -= Q/(self._w+1e-8)

            if isinstance(node, gt.ThermoNode):
                if not inode & 0x1:
                    from pdb import set_trace
                    set_trace()

                continue

            try:
                dsThermo = gt.thermo.from_state(DS_thermo.state)
            except:
                # Flow is too hot and mass flow needs to go up
                self.penalty = (self._w+10)*1e3
                return

            US_node, US_state = node.get_US_state(self._w, DS_thermo)

            if US_state['P'] > 1e9:
                self.penalty = -(US_state['P'])*1e5
                return

            DS_thermo.update(US_state)

            if US_state is None:
                from pdb import set_trace
                set_trace()

            if inode == len(nodes) - 1:
                node.evaluate()
                self.US_target = US_state

                if US_node != US_junction.name:
                    from pdb import set_trace
                    set_trace()
                break

            if nodes[inode+1].name != US_node:
                from pdb import set_trace
                set_trace()

            US_state = self._get_US_thermo(US_node, US_state)


            error = self.model.nodes[US_node].update_thermo(US_state)
            
            if error:
                from pdb import set_trace
                set_trace()

            try:
                node.evaluate()
            except:
                from pdb import set_trace
                set_trace()

            if node._dH !=0:
                from pdb import set_trace
                set_trace()

            if np.abs(node._w - self._w) > 1e-3:
                from pdb import set_trace
                set_trace()


    def evaluate_forward(self):
        """
        Evaluate the branch by updating nodes and calculating errors.

        This method checks the flow and state of each node in the branch,
        reverses order for backflow, and computes errors if constraints
        are violated.
        """
        nodes = self.nodes

        # Update mass flow if this is a fixed flow branch
        if self.fixedFlow:
            try:
                self._w = self.fixed_flow_node._w
            except:
                from pdb import set_trace
                set_trace()

        if self.penalty:
            # If penalty was triggered then return
            return

        # Reverse order if w is negative
        if self._w < 0:
            # Reverse node order
            nodes = nodes[::-1]
            DS_junction = self.US_junction.name
        else:
            DS_junction = self.DS_junction.name

        # Loop thru Branch nodes
        for inode, node in enumerate(nodes):
            # Evaluate the node
            node.evaluate()

            if isinstance(node, (gt.ThermoNode)):
                # We're working with either a thermoNode or Station node.
                # The expected pattern of connections is:
                # flowNode => Station => flowNode
                # The following logic checks and enforces this pattern.

                # Perform a bitwise AND operation to check if 'inode' is even.
                # '0x1' is the hexadecimal representation of the binary value
                # '0001'. If 'inode & 0x1' results in 0, 'inode' is even,
                # otherwise it's odd. If 'inode' is even, trigger the debugger
                # to inspect the state.
                if not inode & 0x1:
                    from pdb import set_trace
                    set_trace()

                # Skip the rest of the loop iteration for this node
                continue

            if isinstance(node, gt.TBoundary):
                continue

            if isinstance(node, gt.fixedFlowNode):
                pass
            else:
                # Update the downstream state
                error = node._set_flow(self._w)

                if error:
                    self.penalty = error
                    return

            DS_node, DS_state = node.get_DS_state()

            if DS_state is None:
                # If this is none then there was an error
                # with setting the flow, so lets apply penalty
                logger.warn(f"Error trying to set node {node.name} to "
                            f"{self._w} in branch {self.name}")

                if self.fixedFlow:
                    if isinstance(self.fixed_flow_node, gt.Pump):
                        from pdb import set_trace
                        set_trace()
                    else:
                        from pdb import set_trace
                        set_trace()   
                else:
                    # This is negative so use error to decrease mass flow
                    self.penalty = (-self._w-10*np.sign(self._w))*1e5
                    if self.penalty > 0:
                        from pdb import set_trace
                        set_trace()
                return

            if DS_state['P'] < 0:
                # Pressure drop is too high because massflow too high,
                # lets decrease mass flow by applying penalty
                # Maybe set this as seperate method for different cases
                logger.warn("Pressure <0 detected in Branch "
                            f"'{self.name}' for branch state: "
                            f"{self.x}")

                if self.fixedFlow:
                    # Calculate penalty based on pressure
                    if isinstance(self.fixed_flow_node, gt.Pump):
                        # Increase Pressure Ratio
                        self.penalty = (-DS_state['P']+10)*1e5
                    else:
                        from pdb import set_trace
                        set_trace()
                else:
                    # This is negative so we need to decrease mass flow
                    self.penalty = (DS_state['P']-10)*1e5
                return

            # Last node error check
            if inode == len(nodes) - 1:
                # What the DS Junction Node state should be
                # for steady state
                self.DS_target = DS_state
                if DS_node != DS_junction:
                    # error check
                    from pdb import set_trace
                    set_trace()
                return

            if isinstance(node, (gt.flowNode, gt.Turbo)):

                if nodes[inode+1].name != DS_node:
                    # Error checking, dsNode should be the next
                    # node in the list
                    from pdb import set_trace
                    set_trace()

                DS_state = self._getDSThermo(DS_node, DS_state)
                # Update thermo for dsNode

                if isinstance(nodes[inode+1], (gt.TBoundary, gt.PBoundary)):
                    error = self.model.nodes[DS_node].update_thermo(DS_state)
                else:
                    error = self.model.nodes[DS_node].update_thermo(DS_state)

                if error:
                    logger.warn("Failed to update thermostate in Branch "
                                f" evaluate call for '{DS_node}' to state: "
                                f"{DS_state}")

                    # Reduce Mass Flow
                    if self.fixedFlow:
                        # Point error back to x0
                        self.penalty = (self._x0[0] - self.x[0]-1e2)*1e5


    def _getDSThermo(self, dsNode, dsState):
        # THis method should calculate the downstream thermo based on mix of inputs

        # Pressure for downstream thermo should be set by flowNode object. This 
        # Calculation handles energy transfer
        
        node_map = self.model.node_map[dsNode]

        if len(node_map['cool']) > 0:
            # There shouldn't be any cool nodes
            # debug if there are
            from pdb import set_trace
            #set_trace()
            print('Check Line 1037ish')

        for hotNode in node_map['hot']:
            node = self.model.nodes[hotNode]

            if hotNode in node_map['US']:
                continue

            if 'H' in dsState:
                try:
                    dsState['H'] += node._Q/abs(self._w)
                except:
                    from pdb import set_trace
                    set_trace()
            else:
                set_trace()

        for coolNode in node_map['cool']:
            node = self.model.nodes[coolNode]

            dsState['H'] -= node._Q/abs(self._w)
            #from pdb import set_trace
            #set_trace()

        return dsState

    def _get_US_thermo(self, US_node, US_state):

        # If Q are added then use this to sum stuff up
        return US_state


    def initialize(self):

        # Define the states defining this dict

        # If PR Turbine then set PR as state controller
        # Otherwise massflow as controller => For that mass flow needs to be fixed

        if self._w is None:
            self._w = self.average_w

        self.istate = {}
        self.x = np.array([])
        self.hot_nodes = {}

        self.pressure_gain = False

        for inode, node in enumerate(self.nodes):
            # Something to add, bounds from each class 
            # If mass flow is above/below bounds then apply penalty
            nMap = self.model.node_map[node.name]


            if isinstance(node, (gt.fixedFlowNode,
                                 gt.Pump)):
                self.pressure_gain = True

            if nMap['hot'] or nMap['cool']:
                self.hot_nodes[node.name] = nMap['hot'] + nMap['cool']

            if isinstance(node, (gt.Station)):
                # Check the node_map if there are heat nodes
                pass

            elif isinstance(node, gt.fixedFlowNode):
                if self.fixedFlow:
                    if node._w == self._w:
                        logger.warn(f"'{self.fixed_flow_node.name}' is in "
                                    "series with another fixedFlow object: "
                                    f"'{node.name}'. Setting the flow rate to "
                                    f"{self._w} kg/s")
                    else:
                        logger.critical(
                            f"'{self.fixed_flow_node.name}' is in series with "
                            f"another fixedFlow object: '{node.name}' and has "
                            "different flow rates specified. Double check the "
                            "model!"
                            )
                    continue

                # Turn on fixedFlow flag for this branch
                self.fixedFlow = True
                # Set Branch Mass Flow to this component flow
                self._w = node._w
                # Node that sets the mass flow
                self.fixed_flow_node = node

                if isinstance(node, gt.fixedFlowTurbo):
                    # Get the node bounds
                    if hasattr(node, '_bounds'):
                        self._bounds = node._bounds

                    if len(self.x) != 0:
                        # There shouldn't be any other states
                        from pdb import set_trace
                        set_trace()

                    if hasattr(node, 'x'):
                        self.x = node.x

            if isinstance(node, gt.statefulFlowNode):
                if not self.fixedFlow:
                    self._bounds[0] = np.max([self._bounds[0],
                                              node._bounds[0]])
                    self._bounds[1] = np.min([self._bounds[1],
                                              node._bounds[1]])

        if isinstance(self.DS_junction, gt.Outlet):
            # If the downstream junction is an outlet then
            # it can be any state and not dependent on branch mass
            # mass flow, so this is a stateless fixed flow object
            if self.fixedFlow:
                if len(self.x) != 0:
                    from pdb import set_trace
                    set_trace()

            self.fixedFlow = True

        if self.fixedFlow:
            pass
        else:
            # This is if no other state present, then state is massflow 
            self.x = np.array([self._w])

        if len(self.x) > 1:
            # There should only be 1 state
            from pdb import set_trace
            set_trace()

        self._x0 = np.copy(self.x)
        self.initialized = True

    def update_state(self, x):
        """
        Update the Branch state with a new state vector.

        Args:
            x (array): The new state vector.
        """

        # Store the original state
        # We may need to revert if penalty are triggered

        self._x0 = np.copy(self.x)
        self.x = x
        self.penalty = False

        if x < self._bounds[0]:
            self.penalty = (self._bounds[0] - x[0] + 10)*1e5
            self.x = self._x0
            return
        elif x > self._bounds[1]:
            self.penalty = (self._bounds[1] - x[0] - 10)*1e5
            self.x = self._x0
            return

        if self.fixedFlow:
            if isinstance(self.fixed_flow_node, gt.fixedFlow):
                from pdb import set_trace
                set_trace()
                return

            self.fixed_flow_node.update_state(x)
            # Get the penalty from the setter object
            self.penalty = self.fixed_flow_node.penalty
        else:
            if x[0] < 0 and not self.backFlow:
                # If backflow is not enabled apply penalty
                self.penalty = (10 - x[0])*1e5
                return

            # Update branch mass flow
            self._w = x[0]

    @property
    def xdot_copy(self):
        if isinstance(self.error, float):
            return np.array([self.error])
        else:
            return self.error

        
    

@addQuantityProperty
class BaseBranch:
    """
    Represents a generic branch in the geoTherm model network.

    This base class encapsulates common properties and methods for both fluid and thermal branches.
    """

    _units = {}
    _bounds = [-np.inf, np.inf]

    def __init__(self, name, nodes, US_junction, DS_junction, model, x=0):
        """
        Initialize a BaseBranch instance.

        Args:
            name (str): Unique identifier for the Branch.
            nodes (list): List of node names or instances in sequential order.
            US_junction (str or instance): Upstream junction name or instance.
            DS_junction (str or instance): Downstream junction name or instance.
            model (object): The model containing all nodes and junctions.
            x (float, optional): Flow-related property (mass flow rate or heat flow rate).
        """
        self.name = name

        # Convert node names to instances if necessary
        if isinstance(nodes[0], str):
            self.nodes = [model.nodes[name] for name in nodes]
        else:
            self.nodes = nodes

        # Assign upstream junction
        if isinstance(US_junction, str):
            self.USJunc = model.nodes[US_junction]
            self.US_junction = model.nodes[US_junction]
        else:
            self.USJunc = US_junction
            self.US_junction = US_junction

        # Assign downstream junction
        if isinstance(DS_junction, str):
            self.DSJunc = model.nodes[DS_junction]
            self.DS_junction = model.nodes[DS_junction]
        else:
            self.DSJunc = DS_junction
            self.DS_junction = DS_junction

        self.model = model
        self._x = x

        self.fixed_flow = False
        self.backFlow = False

        # Node for controlling flow or heat
        self.fixed_flow_node = None
        self.penalty = False
        self.solver = 'forward'

        self.initialized = False

        #self.initialize()

    @property
    def x(self):
        return self._x
    
    @x.setter
    def x(self, x):
        self._x = x

    def __str__(self):
        """
        Return a string representation of the Flow Branch in a map-like format.
        """
        
        US_junc_name = self.US_junction.name
        DS_junc_name = self.DS_junction.name
        node_names = [node.name for node in self.nodes]
        
        branch_path = f"{US_junc_name} => " + " => ".join(node_names) + f" => {DS_junc_name}"
        return f"Flow Branch: {self.name}\n{branch_path}\nx: {self.x}"

    def update_state(self, x):
        """
        Update the Branch state with a new state vector.

        Args:
            x (array): The new state vector.
        """

        # Store the original state
        # We may need to revert if penalty are triggered

        self._x0 = np.copy(self._x)
        self._x = x
        self.penalty = False

        if x < self._bounds[0]:
            self.penalty = (self._bounds[0] - x[0] + 10)*1e5
            self.x = self._x0
            return
        elif x > self._bounds[1]:
            self.penalty = (self._bounds[1] - x[0] - 10)*1e5
            self.x = self._x0
            return

        if self.fixedFlow:
            self.fixed_flow_node.update_state(x)
            # Get the penalty from the setter object
            self.penalty = self.fixed_flow_node.penalty
        else:
            if x[0] < 0 and not self.backFlow:
                # If backflow is not enabled apply penalty
                self.penalty = (10 - x[0])*1e5
                return



@addQuantityProperty
class ThermalBranch(BaseBranch):
    """
    Represents a thermal branch in the geoTherm model network.

    ThermalBranches are segments with thermal resistors in series
    """

    _units = {'Q': 'HEATFLOW'}
    _bounds = [-np.inf, np.inf]
    
    def __init__(self, name, nodes, US_junction, DS_junction, model, Q=None):
        """
        Initialize a ThermalBranch instance.

        Args:
            name (str): Unique identifier for the ThermalBranch.
            nodes (list): List of thermal node names or instances in sequential order.
            US_junction (str or instance): Upstream thermal junction name or instance.
            DS_junction (str or instance): Downstream thermal junction name or instance.
            model (object): The model containing all thermal nodes and junctions.
            Q (float, optional): Heat flow rate with a default value of 0.
        """
        super().__init__(name, nodes, US_junction, DS_junction, model, x=Q)

        self.initialize()

    @property
    def _Q(self):
        if self.fixed_flow:
            return self.fixed_flow_node._Q
        else:
            return self._x[0]

    @property
    def average_Q(self):
        
        Qnet = 0
        flow_elements = 0

        if self.fixed_flow:
            return self.fixed_flow_node._Q

        for node in self.nodes:
            if isinstance(node, (gt.Heatsistor)):
                node.evaluate()
                Qnet += node._Q
                flow_elements +=1

        if flow_elements == 0:
            from pdb import set_trace
            set_trace()

        return float(Qnet/flow_elements)

    def initialize(self):

        
        self.istate = {}
        self.x = np.array([])

        for inode, node in enumerate(self.nodes):
            nMap = self.model.node_map[node.name]

            if isinstance(node, (gt.Qdot)):
                self.fixed_flow = True
                self.fixed_flow_node = node
                self.fixedFlow = True
                self.initialized = True
                return

        self.x = np.array([self.average_Q])
        self._x0 = np.copy(self.x)
        self.initialized = True

        self.fixedFlow = self.fixed_flow


    def evaluate(self):
        
        nodes = self.nodes
        
        if self.fixed_flow:
            if len(self.nodes) == 1:
                return
            from pdb import set_trace
            set_trace()
            self._x = self.fixed_flow_node
        
        if self.penalty:
            return
            
        if self._x < 0:
            nodes = nodes[::-1]
            US_junction = self.DS_junction
            DS_junction - self.US_junction
        else:
            US_junction = self.US_junction
            DS_junction = self.DS_junction
        
        if US_junction.thermo._T > DS_junction.thermo._T:
            if self._x > 0:
                pass
            else:
                from pdb import set_trace
                set_trace()
        
        elif US_junction.thermo._T < DS_junction.thermo._T:
            if self._x < 0:
                pass
            else:
                from pdb import set_trace
                set_trace()    
        else:
            if self._x !=0:        
                # Check Temperatures
                from pdb import set_trace
                set_trace()

        
        # Loop thru Branch nodes
        for inode, node in enumerate(nodes):
            # Evaluate the node
            node.evaluate()

            if isinstance(node, (gt.ThermoNode)):
                # We're working with either a thermoNode or Station node.
                # The expected pattern of connections is:
                # flowNode => Station => flowNode
                # The following logic checks and enforces this pattern.

                # Perform a bitwise AND operation to check if 'inode' is even.
                # '0x1' is the hexadecimal representation of the binary value
                # '0001'. If 'inode & 0x1' results in 0, 'inode' is even,
                # otherwise it's odd. If 'inode' is even, trigger the debugger
                # to inspect the state.
                if not inode & 0x1:
                    from pdb import set_trace
                    set_trace()

                # Skip the rest of the loop iteration for this node
                continue

            elif isinstance(node, (gt.Heatsistor)):
                # Update the downstream state
                error = node._set_heat(self._Q)

                if error:
                    self.penalty = error
                    return
                
            DS_node, DS_state = node.get_DS_state()

            if DS_state is None:
                from pdb import set_trace
                set_trace()
            
            if DS_state['T']<0:
                self.penalty = (DS_state['T']-10)*1e5

            if inode == len(nodes) - 1:
                self.DS_target = DS_state
                if DS_node != DS_junction.name:
                    from pdb import set_trace
                    set_trace()
                return
            
            self.model.nodes[DS_node].update_thermo(DS_state)



        from pdb import set_trace
        set_trace()
    

    @property
    def xdot(self):

        if self.penalty is not False:
            return np.array([self.penalty])
        
        if self._x < 0:
            Tout = self.US_junction.thermo._T
        else:
            Tout = self.DS_junction.thermo._T
        
        if self.fixed_flow:
            from pdb import set_trace
            set_trace()
        else:
            self.error = (self.DS_target['T'] - Tout)

        return np.array([self.error])





class Solution:

    """
    Solution class for storing geoTherm Model data in a pandas DataFrame.

    Attributes:
        model (object): The model object containing nodes with attributes.
        extras (list): List of additional data to store.
        df (pd.DataFrame): DataFrame containing the model data.
    """

    def __init__(self, model, extras=None):
        """
        Initializes the Solution object and prepares the DataFrame for
        storing data.

        Args:
            model (object): The model containing nodes with attributes.
            extras (list, optional): List of additional data to store.
                                     Defaults to an empty list.
        """
        self.model = model
        self.extras = extras if extras is not None else []
        self.initialize()

    def get_column_units(self):
        """
        Retrieves the units for each column in the DataFrame based on the
        attributes of the nodes and predefined units for performance metrics.

        Returns:
            units (dict): A dictionary mapping column names to their units.
        """
        output_units = gt.units.output_units
        units = {}

        for column in self.df.columns:
            # Column represents a node attribute
            if '.' in column:
                name, attr = column.split('.')
                node = self.model.nodes[name]
                # Special handling for thermo units
                if attr in ['P', 'T', 'H', 'U', 'S', 'Q', 'density']:
                    quantity = thermo._units[attr]
                    units[column] = output_units.get(quantity, '')
                elif hasattr(node, '_units') and attr in node._units:
                    quantity = node._units[attr]
                    units[column] = output_units.get(quantity, '')
                else:
                    # Fallback for attributes without associated units
                    units[column] = ''
            elif column in ['Wnet', 'Qin']:
                units[column] = output_units['POWER']
            else:
                units[column] = ''

        return units

    def initialize(self):
        """
        Initializes the node attributes, prepares the DataFrame with the
        necessary columns, and pre-allocates resources for efficient data
        handling.
        """

        attributes_to_check = [
            'P', 'T', 'H', 'U', 'density', 'phase', 'w', 'W', 'area', 'Q_in',
            'Q_out', 'PR', 'N', 'Ns', 'Ds', 'phi', 'psi', 'x', 'xdot',
        ]

        dtype = {}
        self.node_attrs = {}

        # Iterate over each node in the model and get the attributes
        for name, node in self.model.nodes.items():
            for attr in attributes_to_check:
                if not hasattr(node, attr):
                    continue

                attr_val = getattr(node, attr)
                column = f"{name}.{attr}"

                if isinstance(attr_val, (float, int)):
                    dtype[column] = 'float64'
                    self.node_attrs[column] = {
                        'name': name, 'attr': attr, 'length': None,
                        'index': None
                    }
                elif isinstance(attr_val, str):
                    dtype[column] = 'str'
                    self.node_attrs[column] = {
                        'name': name, 'attr': attr, 'length': None,
                        'index': None
                    }
                elif isinstance(attr_val, np.ndarray):
                    length = len(attr_val)
                    for i in range(length):
                        col_name = f"{column}[{i}]"
                        dtype[col_name] = 'float64'
                        self.node_attrs[col_name] = {
                            'name': name, 'attr': attr, 'index': i
                        }
                else:
                    raise TypeError(
                        f"Unsupported data type for attribute {attr}"
                        )

        # Performance metric columns
        for col in ['Wnet', 'Qin', 'eta']:
            dtype[col] = 'float64'

        # Store state and xdot for the model
        for i in range(0, len(self.model.x)):
            dtype[f'x[{i}]'] = 'float64'
            dtype[f'xdot[{i}]'] = 'float64'

        for col in self.extras:
            dtype[col] = 'float64'

        # Initialize the DataFrame and pre-allocate the row data
        self.df = pd.DataFrame({col: pd.Series(dtype=d)
                                for col, d in dtype.items()})
        self.__row = {col: None for col in self.df.columns}
        # Check what unit system data is being saved in
        self.unit_system = gt.units.output_units

    def save(self, extras=None):
        """
        Saves the current state of the model's nodes, performance metrics,
        and extras into the DataFrame.

        Args:
            extras (list, optional): List of extra values to save in the
                                     DataFrame. Must correspond to
                                     `self.extras`.
        """
        extras = extras if extras is not None else []

        # Reset the pre-allocated row to NaN values
        self.__row = {key: np.nan for key in self.__row.keys()}

        # Extract data for each node using the pre-determined attributes
        for column, attr_properties in self.node_attrs.items():
            node = self.model.nodes[attr_properties['name']]
            if attr_properties['index']:
                # Handle array data
                self.__row[column] = getattr(
                    node, attr_properties['attr']
                    )[attr_properties['index']]
            else:
                # Handle scalar and string data
                self.__row[column] = getattr(
                    node, attr_properties['attr'], np.nan
                    )

        self.__row['Wnet'] = self.model.performance[0]
        self.__row['Qin'] = self.model.performance[1]
        self.__row['eta'] = self.model.performance[2]

        # Store state and xdot for the model
        x = self.model.x
        xdot = self.model.xdot

        for i, xi in enumerate(x):
            self.__row[f'x[{i}]'] = xi
            self.__row[f'xdot[{i}]'] = xdot[i]

        # Fill in the extras data by index
        for i, extra in enumerate(self.extras):
            self.__row[extra] = extras[i]

        self.df = pd.concat([
            self.df, pd.DataFrame([self.__row], columns=self.df.columns)
            ],
            ignore_index=True
        )

    def __getitem__(self, item):
        """
        Allows slicing of the Solution object by column name using the
        DataFrame's slicing.

        Args:
            item (str or list of str): The column name(s) to slice.

        Returns:
            pd.DataFrame or pd.Series: The sliced DataFrame or Series.
        """
        return self.df[item]

    def save_csv(self, file_path):
        """
        Saves the DataFrame to a CSV file with headers modified to include
        units.

        Args:
            file_path (str): The file path where the CSV file will be saved.
        """
        # Get the units for each column
        column_units = self.get_column_units()

        # Modify the headers to include units
        modified_headers = []
        for column in self.df.columns:
            unit = column_units.get(column, '')
            if unit:
                modified_header = f"{column} [{unit}]"
            else:
                modified_header = column
            modified_headers.append(modified_header)

        # Save the DataFrame to CSV with modified headers
        self.df.to_csv(file_path, header=modified_headers, index=False)
