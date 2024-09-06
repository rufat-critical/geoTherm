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


class Conditioner:

    def __init__(self, model, solver, conditioning_type='constant'):
        # Initialize Model Conditioner

        self.model = model
        self.solver = solver

        self.conditioning_type = conditioning_type

        self.initialize()

    def initialize(self):

        #if self.solver =='network':
        
        self.get_scaling()

    # Scalings
    def get_scaling(self):
        
        if self.solver == 'nodal':
            self._nodal_scaling()
        elif self.solver == 'network':
            self._network_scaling()

    def _nodal_scaling(self):
        # Initialize
        self.x_func = {'x':[], 'xi': {}, 'xdot':{}}
        self.x_scale = np.ones(len(self.model.x))
        # Do loopy for statefuls
        for name in self.model.statefuls:
            # Get node object
            node = self.model.nodes[name]

            # Get state variable index
            indx = self.model.istate[name]

            # Scalings
            self.x_scale[indx] = 1

    def _network_scaling(self):
        self.x_func = {'x':[], 'xi': {}, 'xdot':{}}
        self.x_scale = np.ones(len(self.model.network.x))

        # Do loopy for statefuls
        for name, indx in self.model.network.istate.items():
            
            if name in self.model.network.branches:
                self.x_scale[indx] *=1
            else:
                from pdb import set_trace
                set_trace()

    def scale_x(self, x):
        #if self.conditioning_type == 'jacobian':
        #    J = self.model._netJac(x)+eps
        #    return x/J

        return x*self.x_scale

    def unscale_x(self, x):
        #if self.conditioning_type == 'jacobian':
        #    J = self.model._netJac(x)+eps
        #    return x*J

        return x/self.x_scale

    def conditioner(self, func):

        if self.conditioning_type == 'constant':
            def wrapper(x):
                x_unscaled = self.unscale_x(x)
                xdot_unscaled = func(x_unscaled)
                return self.scale_x(xdot_unscaled)
            return wrapper
        elif self.conditioning_type =='jacobian':
            def wrapper(x):
                x_unscaled = self.unscale_x(x)
                xdot_unscaled = func(x_unscaled)
                return self.scale_x(xdot_unscaled)
            return wrapper            
        
# Bounds
# min-max 
# 


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
        dup = unique[count>1]
        # Output error if duplicates are present
        if len(dup) != 0:
            msg = 'Multiple nodes specified with the same name:\n'
            msg += ','.join(dup)
            raise ValueError(msg)

        # Create dictionary of nodes
        self.nodes = {node.name: node for node in nodes}

        # Initialization Flag
        self.initialized = False
        # Debug Flag
        self.debug = False

        # Try to initialize Model otherwise output error
        try:
            self.initialize()
        except:
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

        # Run error checker
        if self._error_checker():
            raise ValueError("Errors Found, Fix the Model!")

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
        for name, node in self.nodes.items():
            if hasattr(node, 'evaluate'):
                node.evaluate()

        # Return the error
        return self.xdot

    def steady_evaluate(self, x, t=0):
        return self.evaluate(t, x)

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

    # Initialize

        # Generate node_map

    # solveSystem

    # Print results

    # Plot performance data

    # Save results

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
                plot.add_process_line(DS_flow, name, DS_node, process_type, line_style='-')    
                #from pdb import set_trace
                #set_trace()


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
            integrator = BDF(self.evaluate, t_span[0], x0, t_span[1],max_step=1e-7, rtol=1e-5)

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
            self.initialize()

        # Check if this is a stateless model
        if len(self.x) == 0:
            # We don't need to solve anything
            return self.x

        if netSolver:
            # Initialize the network solver
            #self.initialize_network()

            self.network = Network(self)

            conditioner = Conditioner(self, 'network')
            #conditioned = conditioner.conditioner(self.evaluate_network)
            conditioned = conditioner.conditioner(self.network.evaluate)
            # Try Fsolve with Network Solver
            #sol = fsolve(self._evaluate_network, self._netX, full_output=True)
            x_scaled = conditioner.scale_x(self.network.x)
            sol = fsolve(conditioned, x_scaled, full_output=True)
            #sol = fsolve(self.evaluate_network, self.net['x'], full_output=True)
            x = conditioner.unscale_x(sol[0])
            # Update to network state
            #self.evaluate_network(sol[0])
            self.network.evaluate(sol[0])
            
            from pdb import set_trace
            #set_trace()
            if not self.converged:
                from scipy.optimize import root_scalar

                from pdb import set_trace
                set_trace()
                def func(x):
                    return self.evaluate_network(np.array([x]))[0]
                try:
                    print('TRYING Root Scalar')
                    sol2 = root_scalar(func,x0=sol[0][0])
                    self.evaluate_network(np.array([sol2.root]))
                    
                    if not self.converged:
                        # Try bounded problem
                        try:
                            print('Trying Bisection')
                            sol3 = root_scalar(func,bracket=[0, 1])
                            self.evaluateNet(np.array([sol3.root]))
                        except:
                            print('failed bisection')
                            self.debug=True
                            self.evaluateNet(self._netX)
                            from pdb import set_trace
                            #set_trace()
                
                except:
                    print('Failed 1st conv')
                    from pdb import set_trace
                    #set_trace()
            else:
                return x

        from pdb import set_trace
        set_trace()
        conditioner = Conditioner(self, 'nodal')
        conditioned = conditioner.conditioner(self.steady_evaluate)
        sol = fsolve(conditioned, self.x, full_output=True)

        x = conditioner.unscale_x(sol[0])
        self.steady_evaluate(x)
        if not self.converged:
            from pdb import set_trace
            set_trace()
        else:
            self.steady_evaluate(x)


    def draw(self, file_path='geoTherm_model_diagram.svg', auto_open=True):
        """
        Generates a DOT plot for the model's node network and saves it as an SVG
        file. Optionally, the plot can be opened automatically in a web browser.

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

            if flowNode._w >0:
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

            if flowNode._w >0:
                # Outflow Energy
                Hnet -= node.thermo._H*flowNode._w
            else:
                # Inflow Energy (flowNode is negative so this is positive)
                Hnet -= flowNode._w*dsNode._H
                # Work from flow Node
                Wnet -= flowNode._w*flowNode._dH

        # Sum the heat in/out of the node
        for name in node_map['hot']:
            try:
                Qnet += self.nodes[name]._Q
            except:
                from pdb import set_trace
                set_trace()

        for name in node_map['cool']:
            Qnet -= self.nodes[name]._Q

        return wNet, Hnet, Wnet, Qnet


    @property
    def converged(self):

        # Reinitialize model x using component states
        self.__init_x()

        self.update_state(self.x)

        # Get the error
        if all(abs(self.xdot/(self.x + eps))<1e-2):
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
        (branches, junctions, branch_connections,
         node_classification) = self._identify_branches_and_junctions(
            self.model.node_map
        )
        # Store branches and junctions
        self.branches = dict(branches)
        self.junctions = dict(junctions)
        self.istate = {}  # Index for state variables
        self.__x = []  # State vector

        # Initialize branches
        for branch_id, nodes in branches.items():
            # Create branch objects
            self.branches[branch_id] = Branch(
                name=branch_id,
                nodes=nodes,
                US_junction=branch_connections[branch_id]['US'],
                DS_junction=branch_connections[branch_id]['DS'],
                model=self.model
            )

        # Initialize junctions
        for junction_id, junction_map in junctions.items():
            upstream_branches = [self.branches[branch_id] for branch_id
                                 in junction_map['US']]
            downstream_branches = [self.branches[branch_id] for branch_id
                                   in junction_map['DS']]

            self.junctions[junction_id] = Junction(
                node=self.model.nodes[junction_id],
                US_branches=upstream_branches,
                DS_branches=downstream_branches,
                model=self.model
            )

        # Initialize Branches and Junctions if not initialized
        for branch_name, branch in self.branches.items():
            if not branch.initialized:
                from pdb import set_trace
                set_trace()

        for junction_name, junction in self.junctions.items():
            if not junction.initialized:
                from pdb import set_trace
                set_trace()

        # Initialize state vector for branches
        for branch_name, branch in self.branches.items():
            if len(branch.x) == 0:
                continue

            state_length = len(branch.x)
            current_length = len(self.__x)
            self.istate[branch_name] = np.arange(current_length,
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

    def update_state(self, x):
        """
        Update the state vector for the network.

        Args:
            x (array): The new state vector to update.
        """
        self.__x = x

        # Update Network
        for name, istate in self.istate.items():
            if name in self.branches:
                self.branches[name].update_state(x[istate])
            else:
                self.junctions[name].update_state(x[istate])

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

        # Evaluate branches and junctions
        for _, branch in self.branches.items():
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

        for name, _ in self.istate.items():
            if name in self.branches:
                xdot.append(self.branches[name].error)
            else:
                xdot.append(self.junctions[name].xdot)

        return np.array(xdot)

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

    def __init__(self, node, US_branches, DS_branches, model):
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
        self.US_branches = US_branches
        self.DS_branches = DS_branches
        self.model = model
        self.initialized = False

        self.initialize()

    def initialize(self):
        """
        Initialize the Junction by adding dynamic properties if necessary.
        """
        if isinstance(self.node, (gt.Boundary)):
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
class Branch:
    """
    Represents a branch in the geoTherm model network.

    Branches are segments that connect different nodes in the network,
    facilitating flow and heat transfer between them.
    """

    _units = {'w': 'MASSFLOW'}

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
        self.branch_name = name

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
        self.flow_setter_node = None
        self.penalty = False

        self.initialized = False
        # Bounds (for massflow if not fixedFlow)
        self.bounds = [-np.inf, np.inf]

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
            if isinstance(node, gt.flowNode):
                Wnet += node._w
                flow_elements += 1

        return float(Wnet/flow_elements)

    def evaluate(self):
        """
        Evaluate the branch by updating nodes and calculating errors.

        This method checks the flow and state of each node in the branch,
        reverses order for backflow, and computes errors if constraints
        are violated.
        """
        nodes = self.nodes

        # Update mass flow if fixed flow is enabled
        if self.fixedFlow:
            self._w = self.flow_setter_node._w

        # Reverse order if w is negative
        if self._w < 0:
            if self.backFlow:
                # Reverse node order
                nodes = nodes[::-1]
                # dP will be calculated and compared to this Pressure
                Pout = self.USJunc.thermo._P
                # Downstream Junction
                DSJunc = self.USJunc.name
                DS_junction = self.US_junction.name
            else:
                self.error = (-self._w + 1)*1e5
                return
        else:
            Pout = self.DSJunc.thermo._P
            DSJunc = self.DSJunc.name
            DS_junction = self.DS_junction.name

        # Check for penalty
        if self.penalty is not False:
            self.error = self.penalty
            return


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

            # Update the downstream state
            try:
                dsNode, dsState = node._setFlow(self._w)
            except Exception:
                from pdb import set_trace
                set_trace()

            if dsState['P'] < 0:
                # Pressure drop is too high because massflow too high,
                # lets decrease mass flow by applying penalty
                # Maybe set this as seperate method for different cases

                if self.fixedFlow:
                    # Calculate Error based on pressure
                    if isinstance(self.flow_setter_node, gt.Pump):
                        self.error = (0-dsState['P'])*1e5
                    else:
                        from pdb import set_trace
                        set_trace()
                else:
                    logger.info("Pressure <0 detected in Branch "
                                f"'{self.branch_name}' for mass flow of: "
                                f"{self._w}")
                    # This is negative so use error to decrease mass flow
                    self.error = (dsState['P'])*1e5
                return

            # Last node error check
            if inode == len(nodes) - 1:
                if dsNode != DS_junction:
                    # dsNode is not dsJunc for some reason
                    # error check
                    from pdb import set_trace
                    set_trace()

                if self.fixedFlow:
                    if self.penalty is not False:
                        self.error = self.penalty
                        return

                    if isinstance(self.flow_setter_node, gt.Turbine):
                        self.error = (dsState['P']/Pout - 1)*Pout
                    elif isinstance(self.flow_setter_node, gt.Pump):
                        #self._error = (Pout/dsState['P'] - 1)*dsState['P']*np.sign(self.x)
                        usState = node.US_node.thermo
                        self.error = (Pout/usState._P - self.x)#*1e6
                        self.error = (Pout/dsState['P'] - 1)*dsState['P']*1e2

                else:                    
                    self.error = (np.sign(dsState['P']-Pout)
                                  * np.abs(dsState['P']-Pout)**1.3
                                  + (dsState['P']-Pout)*10)

                if self.model.debug:
                    from pdb import set_trace
                    set_trace()
                return

            if isinstance(node, (gt.flowNode, gt.Turbo)):

                if nodes[inode+1].name != dsNode:
                    # Error checking, dsNode should be the next
                    # node in the list 
                    from pdb import set_trace
                    set_trace()
                
                dsState = self._getDSThermo(dsNode, dsState)
                # Update thermo for dsNode

                if isinstance(nodes[inode+1], (gt.TBoundary, gt.PBoundary)):
                    #print('REFACTOR AND MAKE THIS NEATER!')
                    error = self.model.nodes[dsNode].update_thermo(dsState)
                else:
                    error = self.model.nodes[dsNode].update_thermo(dsState)

                if error:
                    logger.warn("Failed to update thermostate in Branch "
                                f" evaluate call for '{dsNode}' to state: "
                                f"{dsState}")

                    # Reduce Mass Flow
                    # What if PR
                    if self.fixedFlow:
                        # Point error back to x0
                        self.error = (self._x0 - self.x)*1e5
                    else:
                        self.error = (-self._w - np.sign(self._w+eps))*1e5
                    return

    def _getDSThermo(self, dsNode, dsState):
        # THis method should calculate the downstream thermo based on mix of inputs

        # Pressure for downstream thermo should be set by flowNode object. This 
        # Calculation handles energy transfer
        
        node_map = self.model.node_map[dsNode]

        if len(node_map['cool']) > 0:
            # There shouldn't be any cool nodes
            # debug if there are
            from pdb import set_trace
            set_trace()
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

    def initialize(self):
        
        # Define the states defining this dict

        # If PR Turbine then set PR as state controller
        # Otherwise massflow as controller => For that mass flow needs to be fixed

        if self._w is None:
            self._w = self.average_w

        self.istate = {}
        self.x = np.array([])

        for inode, node in enumerate(self.nodes):
            # Something to add, bounds from each class 
            # If mass flow is above/below bounds then apply penalty

            if isinstance(node, (gt.Station)):
                # Check the node_map if there are heat nodes
                nMap = self.model.node_map[node.name]

                if len(nMap['cool']) > 0:
                    print('Check Line 1073ish')
                    #from pdb import set_trace
                    #set_trace()

            if isinstance(node, gt.fixedFlowNode):
                if self.fixedFlow:
                    # Fixed Flow already activated somewhere
                    from pdb import set_trace
                    set_trace()

                # Turn on fixedFlow flag for this branch
                self.fixedFlow = True
                # Set Branch Mass Flow to this component flow
                self._w = node._w
                # Node that sets the mass flow
                self.flow_setter_node = node
                # Get the node bounds
                self.bounds = node.bounds

                if len(self.x) != 0:
                    # There shouldn't be any other states
                    from pdb import set_trace
                    set_trace()

                #self.istate[node.name] = np.arange(current_len,
                #                                   current_len + len(node.x))
                #self._x = np.concatenate((self.x, node.x))
                self.x = node.x

        #if len(self._x) == 0:
        if self.fixedFlow:
            pass
        else:
            # This is if no other state present, then state is massflow 
            self.x = np.array([self._w])

        if len(self.x) > 1:
            # There should only be 1 state
            from pdb import set_trace
            set_trace()

        self.initialized = True

    def update_state(self, x):
        """
        Update the Branch state with a new state vector.

        Args:
            x (array): The new state vector.
        """

        # Store the original state
        # We may need to revert if penalty are triggered
        self._x0 = self.x
        self.x = x
        if self.fixedFlow:
            self.flow_setter_node.update_state(x)
            # Get the penalty from the setter object
            self.penalty = self.flow_setter_node.penalty
        else:
            # Update branch mass flow
            self._w = x[0]


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
