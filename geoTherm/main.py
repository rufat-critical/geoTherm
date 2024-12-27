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
from .nodes.baseNodes.baseThermo import baseThermo
from .nodes.baseNodes.baseNode import Node
from .nodes.baseNodes.baseThermal import baseThermal
from .solvers.network.junctions import Junction, ThermalJunction, BoundaryJunction, FlowJunction
from .solvers.network.branches import ThermalBranch, FlowBranch

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
        elif isinstance(node, baseThermo):#gt.ThermoNode):
            return np.array([1e-1, 1e-5])
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
            if isinstance(node, gt.Balance):
                if np.isinf(node.knob_max) or np.isinf(node.knob_min):
                    return 1
                else:
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

        # Update Component states
        for name, istate in self.istate.items():
            self.nodes[name].update_state(x[istate])

        for name, node in self.nodes.items():
            node.evaluate()

    @property
    def x(self):
        """Get the state vector"""
        for name, istate in self.istate.items():
            self.__x[istate] = self.nodes[name].x
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
                if node.hot is not None:
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
                if node.cool is not None:
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
        elif isinstance(nodes, Node):
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
        elif isinstance(node, Node):
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

        if isinstance(node, (Node, list)):
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

            if not self.converged:
                logger.warn("Failed to converge with Network Solver, "
                            "will try Nodal Solver Next!")
            else:
                return self.x

        self.initialize()
        conditioner = Conditioner(self)
        conditioned = conditioner.conditioner(self.steady_evaluate)
        # Scale the state vector for fsolve
        x_scaled = conditioner.scale_x(self.x)
        conditioned(x_scaled)
        sol = fsolve(conditioned, x_scaled, full_output=True)

        x = conditioner.unscale_x(sol[0])
        self.steady_evaluate(x)
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
        return self.get_flux(node)

    def get_flux(self, node):
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
            if isinstance(node, (baseThermo)):
                continue
            
            if isinstance(node, (baseThermal)):
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
        
        ((flow_branches, flow_branch_connections, flow_junctions),
        (thermal_branches, thermal_branch_connections, thermal_junctions),
        other_junctions)= self._identify_branches_and_junctions(
            self.model.node_map)
        

        junctions = {**flow_junctions, **thermal_junctions, **other_junctions}
        flow_nodes = [node for nodes in flow_branches.values() for node in nodes]
        flow_nodes = [*flow_nodes, *flow_junctions.keys()]
        # Remove thermal_junctions nodes that are in flow_nodes
        #for name in list(thermal_junctions.keys()):
        #    if name in flow_nodes:
        #        del thermal_junctions[name]
        # Store branches and junctions

        stateless_thermal = []
        for junc,jmap in dict(junctions).items():
            if junc in flow_nodes:
                if jmap['US'] or jmap['DS']:
                    pass
                else:
                    print('deleting')
                    stateless_thermal.append(junc)


        self.flow_branches = dict(flow_branches)
        self.thermal_branches = dict(thermal_branches)
        self.junctions = junctions
        self.istate = {}  # Index for state variables
        self.__x = []  # State vector


        for junction_id, junction_map in junctions.items():
            US_flow_branches = [flow_branch_connections[branch_id] for branch_id
                                 in junction_map['US']]
            DS_flow_branches = [flow_branch_connections[branch_id] for branch_id
                                 in junction_map['DS']]
                                 
            US_thermal_branches = [thermal_branch_connections[branch_id] for branch_id
                                   in junction_map['hot']]
            DS_thermal_branches = [thermal_branch_connections[branch_id] for branch_id
                                   in junction_map['cool']]
            node = self.model.nodes[junction_id]

            n_branches = (len(US_flow_branches) + len(DS_flow_branches)
                          + len(US_thermal_branches) + len(DS_thermal_branches)) 

            if isinstance(node, (gt.Boundary, gt.Qdot)):
                self.junctions[junction_id] = BoundaryJunction(
                    name=junction_id,
                    node=self.model.nodes[junction_id],
                    US_flow_branches=US_flow_branches,
                    DS_flow_branches=DS_flow_branches,
                    US_thermal_branches=US_thermal_branches,
                    DS_thermal_branches=DS_thermal_branches,
                    model=self.model)
            elif n_branches == 0:
                self.junctions[junction_id] = Junction(name=junction_id,
                                                       node=node,
                                                       model=self.model)
            elif len(US_flow_branches) == len(DS_flow_branches) == 0:        
                if junction_id in stateless_thermal:
                    stateful = False
                else:
                    stateful = True

                self.junctions[junction_id] = ThermalJunction(
                name=junction_id,
                node=self.model.nodes[junction_id],
                US_thermal_branches=US_thermal_branches,
                DS_thermal_branches=DS_thermal_branches,
                stateful = stateful,
                model=self.model)
            else:
                self.junctions[junction_id] = FlowJunction(
                    name=junction_id,
                    node=self.model.nodes[junction_id],
                    US_flow_branches=US_flow_branches,
                    DS_flow_branches=DS_flow_branches,
                    US_thermal_branches=US_thermal_branches,
                    DS_thermal_branches=DS_thermal_branches,
                    model=self.model)            
        
        # Initialize branches
        for branch_id, nodes in flow_branches.items():
            US_junction=self.junctions[flow_branch_connections[branch_id]['US']]
            DS_junction=self.junctions[flow_branch_connections[branch_id]['DS']]

            thermal_juncs = flow_branch_connections[branch_id]['thermal']
            thermal = {}
            for junc in thermal_juncs:
                j_map = thermal_junctions[junc]

                thermal[junc] = j_map

            # Create branch objects
            self.flow_branches[branch_id] = FlowBranch(
                name=branch_id,
                nodes=nodes,
                US_junction=US_junction,
                DS_junction=DS_junction,
                thermal=thermal,
                network=self
            )

        for branch_id, nodes in thermal_branches.items():
            US_junction = self.junctions[thermal_branch_connections[branch_id]['hot']]
            DS_junction = self.junctions[thermal_branch_connections[branch_id]['cool']]

            self.thermal_branches[branch_id] = ThermalBranch(
                name=branch_id,
                nodes=nodes,
                US_junction=US_junction,
                DS_junction=DS_junction,
                network=self)

        # Initialize junctions
        for junction_id, junction_map in junctions.items():
            continue
            US_flow_branches = [self.flow_branches[branch_id] for branch_id
                                 in junction_map['US']]
            DS_flow_branches = [self.flow_branches[branch_id] for branch_id
                                 in junction_map['DS']]
                                 
            US_thermal_branches = [self.thermal_branches[branch_id] for branch_id
                                   in junction_map['hot']]
            DS_thermal_branches = [self.thermal_branches[branch_id] for branch_id
                                   in junction_map['cool']]

            from pdb import set_trace
            set_trace()
            if len(US_flow_branches) == len(DS_flow_branches) == 0:

                self.junctions[junction_id] = ThermalJunction(
                node=self.model.nodes[junction_id],
                US_thermal_branches=US_thermal_branches,
                DS_thermal_branches=DS_thermal_branches,
                model=self.model)
            else:
                from pdb import set_trace
                set_trace()
                self.junctions[junction_id] = Junction(
                    node=self.model.nodes[junction_id],
                    US_flow_branches=US_flow_branches,
                    DS_flow_branches=DS_flow_branches,
                    US_thermal_branches=US_thermal_branches,
                    DS_thermal_branches=DS_thermal_branches,
                    model=self.model)

        #from pdb import set_trace
        #set_trace()

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
            state_length = len(junction.x)
            if state_length == 0:
                continue

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
            node_map (dict): A dictionary representing the node map of the geoTherm system.

        Returns:
            tuple: A tuple containing:
                - flow_branches (dict): A dictionary of flow branches.
                - thermal_branches (dict): A dictionary of thermal branches.
                - flow_branch_connections (dict): A dictionary mapping flow
                                                branches to their upstream and downstream junctions.
                - thermal_branch_connections (dict): A dictionary mapping
                                                    thermal branches to their upstream and downstream junctions.
                - other_junctions (dict): A dictionary of junctions not classified as flow or thermal.
        """

        # Initialize data structures
        remaining_nodes = set(node_map.keys())  # Nodes to be processed
        flow_branches = {}        # Dictionary of flow branches
        thermal_branches = {}     # Dictionary of thermal branches
        flow_junctions = {}
        thermal_junctions = {}
        other_junctions = {}
        flow_branch_connections = {}  # Flow branch connectivity
        thermal_branch_connections = {}  # Thermal branch connectivity
        flow_nodes = []  # Nodes involved in flow branches
        branch_counter = 0  # Counter for branch identifiers

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
            connections = node_map[node_name]

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
                and not (has_hot, has_cool)):
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

            junction_map = {'US': [],
                            'DS': [],
                            'hot': [],
                            'cool': []}

            if flow:
                flow_junctions[node_name] = junction_map
                flow_nodes.append(node_name)
                remaining_nodes.remove(node_name)

            if thermal:
                # Check if junction is Qdot
                thermal_junctions[node_name] = junction_map
            if other:
                other_junctions[node_name] = junction_map
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
                str: Downstream junction at the end of the branch.
            """

            if branch_type == 'flow':
                DS = 'DS'
                if current_node in flow_junctions:
                    # We reached end of the branch, return the flow junction
                    return current_node
                else:
                    flow_nodes.append(current_node)
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

            #if branch_type == 'flow' and downstream_node in flow_junctions:
                # Return downstream junction
            #    return downstream_node
            #elif (branch_type == 'thermal' and downstream_node 
            #      in [*thermal_junctions, *flow_junctions]):
            #    return downstream_node
            #else:
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

                # Record connections
                flow_junctions[junction]['DS'].append(branch_counter)
                if downstream_junction in [*flow_junctions, *thermal_junctions]:
                    flow_junctions[downstream_junction]['US'].append(
                        branch_counter)
                else:
                    from pdb import set_trace
                    set_trace()

                flow_branch_connections[branch_counter] = {
                    'US': junction,
                    'DS': downstream_junction,
                    'thermal': thermal_nodes
                }
                branch_counter += 1

        for junction in thermal_junctions:
            downstream_nodes = node_map[junction]['cool']
            US_node = self.model.nodes[junction]

            for downstream_node in downstream_nodes:
                DS_node = self.model.nodes[downstream_node]

                if (downstream_node in [*flow_junctions, *thermal_junctions]):
                    # No branch between junctions or already processed
                    # This happens with Qdot connected to a branch
                    downstream_junction = downstream_node

                    if isinstance(US_node, gt.Qdot):
                        current_branch = [junction]
                    elif isinstance(DS_node, gt.Qdot):
                        current_branch = [downstream_junction]
                    else:
                        from pdb import set_trace
                        set_trace()
                    #if downstream_node in flow_nodes:
                    #    current_branch = [junction]
                    #else:
                    #    current_branch = [downstream_junction]
                else:
                    current_branch = []
                    downstream_junction = trace_branch(downstream_node,
                                                       current_branch,
                                                       'thermal')
                    if isinstance(US_node, gt.Qdot):
                        current_branch = [junction, *current_branch]
                    elif isinstance(downstream_junction, gt.Qdot):
                        current_branch = [*current_branch, junction]

                thermal_branches[branch_counter] = current_branch
                thermal_junctions[junction]['cool'].append(branch_counter)

                if downstream_junction in [*flow_junctions, *thermal_junctions]:
                    thermal_junctions[downstream_junction]['hot'].append(
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
            junction_map = {'US': [],
                            'DS': [],
                            'hot': [],
                            'cool': []}

            for node in remaining_nodes:
                if node not in {**thermal_junctions, **flow_junctions}:
                    other_junctions[node] = junction_map

        if len(remaining_nodes) != 0:
            logger.warn("The following nodes were not recognized in "
            f"building the network map: {remaining_nodes}")

        return ((flow_branches, flow_branch_connections, flow_junctions),
                (thermal_branches, thermal_branch_connections, thermal_junctions),
                other_junctions)


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
