import numpy as np
import geoTherm as gt
from scipy.optimize import fsolve
from scipy.integrate import BDF
from scipy.optimize._numdiff import approx_derivative
from .nodes.node import modelTable
from .utilities.thermo_plotter import thermoPlotter
from .logger import logger
from .utils import eps, parse_component_attribute
from .utilities.network_graph import make_dot_diagram, make_graphml_diagram, generate_dot_code
from .thermostate import thermo
import pandas as pd
from .nodes.baseNodes.baseThermo import baseThermo
from .nodes.baseNodes.baseNode import Node
from .nodes.baseNodes.baseThermal import baseThermal
from .solvers.network.network import Network
from .solvers.network.junctions import FlowJunction
from .utils import yaml_loader, yaml_writer, parse_dimension

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
        else:
            logger.critical("Model must be of type 'Model' '")

        self.initialize()

    def initialize(self):
        """Initialize scaling based on solver type and conditioning type."""
        if self.conditioning_type == 'constant':
            if self.solver == 'nodal':
                self._nodal_scaling()
        elif self.conditioning_type == 'None':
            if self.solver == 'nodal':
                self.x_scale = np.ones(len(self.model.x))
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
            return np.array([1, 1e-6])
        elif isinstance(node, baseThermo):
            #return np.array([1, 1])
            return np.array([1e-2, 1e-6])
        else:
            return 1

    def _jacobian(self, x):
        """Compute the Jacobian matrix for the current solver."""
        if self.solver == 'nodal':
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
            logger.critical("Multiple nodes specified with the same name: "
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

    @staticmethod
    def load(state_dict):
        """Load a state dictionary into the model"""
        model = Model()
        for name, node in state_dict['Nodes'].items():
            node['name'] = name
            model.addNode(node)
        return model

    @property
    def state(self):
        """Return a dictionary of the model state organized with Models and Nodes sections"""
        organized_dict = {
            'Model': [node.name for node in self.nodes.values()],
            'Nodes': {}
        }

        for name, node in self.nodes.items():
            node_state = node._state_dict.copy()  # Create a copy to avoid modifying original
            organized_dict['Nodes'][name] = node_state

        return organized_dict

    def save(self, yaml_path='model.yaml'):
        """Save the model to a YAML file"""
        yaml_writer(yaml_path, self.state)

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

        #v1 = self.evaluate(t, x)

        #v2 = self.evaluate(t, x)

        #if all(v1 != v2):
        ##    from pdb import set_trace
         #   set_trace()
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
        elif isinstance(nodes, dict):
            NodeType = getattr(gt.nodes, nodes['Node_Type'])
            node = NodeType(nodes['name'], **nodes['config'])

            if 'x' in nodes['config']:
                node.update_state(nodes['x'])
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
            if isinstance(node, (gt.Boundary, gt.Station)):
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
            # Need to move this to solvers folder
            from solvers.cvode import CVode_solver
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

    def solve_steady(self, netSolver=True, try_steady=False):

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
            return self.x, True

        if netSolver:
            # Initialize the network solver
            self.network = Network(self)

            self.network.solve()

            if not self.converged:
                from pdb import set_trace
                #set_trace()
                logger.warn("Failed to converge with Network Solver, "
                            "will try Nodal Solver Next!")
            else:
                return self.x, True


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
            if try_steady:
                logger.warn("Failed to converge with Nodal Solver, "
                            "will try Transient Solver Next!")
                # Try to use sim to solve model
                sol = self.sim([0, 1e3])

            if not self.converged:
                logger.warn("Could not converge")
                return self.x, False


        else:
            logger.info('CONVERGED!')
            return self.x, True


    def draw(self, file_path='geoTherm_model_diagram.svg', auto_open=True,
             display_in_notebook=False):
        """
        Generates a DOT plot for the model's node network.
        
        Args:
            file_path (str): The path to save the generated SVG file.
            auto_open (bool): Whether to automatically open the SVG file.
            display_in_notebook (bool): Whether to display in Jupyter notebook.
        
        Returns:
            IPython.display.SVG if display_in_notebook=True, None otherwise.
        """
        if display_in_notebook:
            from IPython.display import SVG, display
            dot_code = generate_dot_code(self)
            from plantuml import PlantUML
            plantuml = PlantUML(url='http://www.plantuml.com/plantuml/svg/')
            svg_content = plantuml.processes(dot_code)
            return display(SVG(svg_content))
        else:
            make_dot_diagram(self, file_path=file_path, auto_open=auto_open)

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

    def get_in_flux(self, node):
        """Calculate incoming mass and energy fluxes to a control volume.
        
        Args:
            node: Node object representing the control volume
            
        Returns:
            tuple: (w_in, H_in, W_in, Q_in)
                w_in: Net incoming mass flow [kg/s]
                H_in: Net incoming enthalpy flux [W]
                W_in: Net incoming work [W]
                Q_in: Net incoming heat [W]
        """
        # Get node connections
        node_map = self.node_map[node.name]
        
        w_in = 0.0
        H_in = 0.0
        W_in = 0.0
        Q_in = 0.0
        
        # Check upstream branches for inflow
        for name in node_map['US']:
            flow_node = self.nodes[name]
            if flow_node._w > 0:  # Positive flow is inflow
                w_in += flow_node._w
                us_thermo = self.nodes[self.node_map[name]['US'][0]].thermo
                H_in += flow_node._w * us_thermo._H
                W_in += flow_node._w * flow_node._dH

        # Check downstream branches for backflow
        for name in node_map['DS']:
            flow_node = self.nodes[name]
            if flow_node._w < 0:  # Negative flow is backflow (inflow)
                w_in += -flow_node._w  # Convert to positive inflow
                ds_thermo = self.nodes[self.node_map[name]['DS'][0]].thermo
                H_in += -flow_node._w * ds_thermo._H
                W_in += -flow_node._w * flow_node._dH
        
        # Add heat input
        for name in node_map['hot']:
            Q_node = self.nodes[name]
            if Q_node._Q > 0:  # Positive Q is heat input
                Q_in += Q_node._Q
                
        return np.array([w_in, H_in, W_in, Q_in])

    def get_out_flux(self, node):
        """Calculate outgoing mass and energy fluxes from a control volume.
        
        Args:
            node: Node object representing the control volume
            
        Returns:
            tuple: (w_out, H_out, W_out, Q_out)
                w_out: Net outgoing mass flow [kg/s]
                H_out: Net outgoing enthalpy flux [W]
                W_out: Net outgoing work [W]
                Q_out: Net outgoing heat [W]
        """
        # Get node connections
        node_map = self.node_map[node.name]
        
        w_out = 0.0
        H_out = 0.0
        W_out = 0.0
        Q_out = 0.0
        
        # Check downstream branches for outflow
        for name in node_map['DS']:
            flow_node = self.nodes[name]
            if flow_node._w > 0:  # Positive flow is outflow
                w_out += flow_node._w
                H_out += flow_node._w * node.thermo._H
                W_out += flow_node._w * flow_node._dH
                
        # Check upstream branches for backflow
        for name in node_map['US']:
            flow_node = self.nodes[name]
            if flow_node._w < 0:  # Negative flow is backflow (outflow)
                w_out += -flow_node._w  # Convert to positive outflow
                H_out += -flow_node._w * node.thermo._H
                W_out += -flow_node._w * flow_node._dH
        
        # Add heat output
        for name in node_map['cool']:
            Q_node = self.nodes[name]
            if Q_node._Q > 0:  # Positive Q on cool side is heat output
                Q_out += Q_node._Q
                
        return np.array([w_out, H_out, W_out, Q_out])

    def outlet_flux(self, node):
        from pdb import set_trace
        set_trace()

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

        return np.array([wNet, Hnet, Wnet, Qnet])

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

def load(yaml_path):
    config = yaml_loader(yaml_path)
    geometry = initialize_geometry_groups(config['GeometryGroups'])

    from pdb import set_trace
    set_trace()


def initialize_geometry_groups(geometry_groups):
    """
    Initialize the geometry groups from the model configuration.
    """
    geoGroups = {}

    for name, geometry_group in geometry_groups.items():
        try:
            geoGroups[name] = gt.GeometryGroup.from_dict(geometry_group)
        except Exception as e:
            logger.critical(f"Error initializing geometry group {name}: {e}")

    return geoGroups


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


