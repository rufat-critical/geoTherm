from .junctions import (Junction, BoundaryJunction,
                        ThermalJunction, FlowJunction)
from geoTherm.units import addQuantityProperty
from ...nodes.baseNodes.baseThermal import baseThermal
from ...nodes.baseNodes.baseThermo import baseThermo
from ...nodes.baseNodes.baseFlow import baseFlow, baseInertantFlow, FixedFlow
from ...nodes.baseNodes.baseTurbo import Turbo
from ...nodes.flowDevices import fixedFlow
from ...nodes.baseNodes.baseFlowResistor import baseFlowResistor
from ...nodes.cycleCloser import cycleCloser
from ...logger import logger
from ...utils import thermo_data
import numpy as np
import geoTherm as gt
from ...thermostate import thermo
from ...nodes.heatsistor import Qdot, ConvectiveResistor
from ...nodes.pump import Pump, basePump
from ...nodes.turbine import Turbine, simpleTurbine, chokedTurbine
from ...utils import eps
from scipy.optimize import fsolve
from .junctions import OutletJunction
from scipy.optimize import root_scalar


@addQuantityProperty
class baseBranch:
    """
    Represents a generic branch in the geoTherm model network.

    This base class encapsulates common properties and methods for both fluid and thermal branches.
    """

    _units = {}
    _bounds = [-np.inf, np.inf]

    def __init__(self, name, nodes, US_junction, DS_junction, network):
        """
        Initialize a BaseBranch instance.

        Args:
            name (str): Unique identifier for the Branch.
            nodes (list): List of node names or instances in sequential order.
            US_junction (str or instance): Upstream junction name or instance.
            DS_junction (str or instance): Downstream junction name or
                instance.
            model (object): The model containing all nodes and junctions.
        """
        self.name = name

        #if not isinstance(network, gt.Network)
        self.network = network
        self.model = self.network.model

        # Convert node names to instances if necessary
        if isinstance(nodes[0], str):
            self.nodes = [self.model.nodes[name] for name in nodes]
        else:
            self.nodes = nodes

        if not isinstance(US_junction, Junction):
            logger.critical(f"US_junction for branch {self.name} needs to "
                            "be a Junction classType")
        if not isinstance(DS_junction, Junction):
            logger.critical(f"DS_junction for branch {self.name} needs to "
                            "be a Junction classType")

        self.US_junction = US_junction
        self.DS_junction = DS_junction

        # Flags
        self.fixed_flow = False
        self.fixed_flow_flag = False
        self.backflow = True

        # Flow value (w for flowBranch, Q for thermalBranch)
        self._flow_value = 0

        # Node for controlling flow or heat
        self.fixed_flow_node = None
        self.penalty = False
        self.solver = 'forward'
        self.stateful = True
        self.initialized = False
        self.linearly_independent = False

        self.US_branch_error = 0
        self.DS_branch_error = 0

        # Create temporary thermo object for intermediate calcs
        if isinstance(self.US_junction.node, baseThermo):
            self._thermo = thermo.from_state(self.US_junction.node.thermo.state)
        else:
            if isinstance(self.DS_junction.node, baseThermo):
                self._thermo = thermo.from_state(self.DS_junction.node.thermo.state)
            elif isinstance(self.DS_junction.node, baseFlow):
                self._thermo = thermo.from_state(self.DS_junction.node.US_node.thermo.state)
            else:
                if isinstance(self.US_junction.node, baseFlow):
                    self._thermo = thermo.from_state(self.US_junction.node.DS_node.thermo.state)
                else:
                    from pdb import set_trace
                    set_trace()



    def reset_flags(self):
        self.fixed_flow = False
        self.fixed_flow_node = None
        self.stateful = True
        self.initialized = False

    @property
    def is_linear_independent(self):
        # Check if upstream and downstream are boundaries

        # Have to check there are no thermal junctions present
        from pdb import set_trace
        set_trace()

        if (isinstance(self.US_junction, BoundaryJunction)
            and isinstance(self.DS_junction, BoundaryJunction)):
            self.linearly_independent = True
        else:
            self.linearly_independent = False

        return self.linearly_independent

    @property
    def x(self):
        return self.state

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

        self.penalty = False
        #self._state = np.copy(self.state)
        self.state = x

        if self.backflow is False:
            if self._flow_value >= 0 and x[0] < 0:
                self.penalty = (-x[0] + 10)*1e5
                return
            elif self._flow_value < 0 and x[0] > 0:
                self.penalty = (-x[0] - 10)*1e5
                return

        if self._bounds[0] <= x[0] <= self._bounds[1]:
            if not self.fixed_flow:
                self._flow_value = x[0]
        else:
            if x[0] < self._bounds[0]:
                self.penalty = (self._bounds[0] - x[0] + 10)*1e5
                self.state = self._state
                logger.warn(f"Flow Branch {self.name} has a flow value of "
                                f" {x[0]} which is below the lower bound of "
                                f"{self._bounds[0]}")
            else:
                self.state = self._state
                self.penalty = (self._bounds[1] - x[0] - 10)*1e5
                logger.warn(f"Flow Branch {self.name} has a flow value of "
                                f" {x[0]} which is above the upper bound of "
                                f"{self._bounds[1]}")
        
    def solve(self):
        from pdb import set_trace
        set_trace()

    def evaluate(self):

        if not self.stateful:
            if len(self.nodes) == 1:
                self.nodes[0].evaluate()
                return
            else:
                from pdb import set_trace
                set_trace()

        if self.solver == 'reverse':
            from pdb import set_trace
            set_trace()
            self.evaluate_reverse()


        if self._w < 0:
            nodes = self.nodes[::-1]
            US_thermo = self.DS_junction.node.thermo
            DS_junction = self.US_junction
        else:
            nodes = self.nodes
            US_thermo = self.US_junction.node.thermo
            DS_junction = self.DS_junction

        if self._w == 0:
            self.DS_target = {'P': US_thermo._P, 'H': US_thermo._H}
            return
        elif US_thermo._P < DS_junction._P:
            # Check only if there are no pressure gain components
            # and no fixed flow components
            if not self.pressure_gain and not self.fixed_flow:
                if self._w < 0:
                    self.penalty = (-self.state[0] + 10)*1e20
                else:
                    self.penalty = (-self.state[0] - 10)*1e20
                logger.warn(f"Flow Branch {self.name} has flow from low to high "
                            "pressure")
                return

        self.evaluate_forward(US_thermo, DS_junction, nodes)


    def error_check(self):
        pass


@addQuantityProperty
class FlowBranch(baseBranch):
    """
    Represents a branch in the geoTherm model network.

    Branches are segments that connect different nodes in the network,
    facilitating flow and heat transfer between them.
    """

    _units = {'w': 'MASSFLOW'}
    _bounds = [-np.inf, np.inf]

    def __init__(self, *args, **kwargs):

        super().__init__(*args, **kwargs)

        self.solve_thermal = True

        # Flag to check for pressure gain
        # If flow is from low pressure to high pressure
        # Then this would force error except if there are
        # pressure gain devices in line like a pump
        self.pressure_gain = False

        self.debug = False

        # I need to come up with a better variable name
        # This is for ndoe that is not fixed flow but
        # with a fixed flow flag attached to component
        # like pipe
        self.fixed_flow_flag = False

        # Flag to check if branch contains a compressible fluid
        self.compressible = False
        self.iResistors = []
        self._iResistors = {}


    @property
    def _w(self):
        return self._flow_value

    @_w.setter
    def _w(self, w):
        self._flow_value = w

    def initialize(self):
        # Define the states defining this dict

        # If PR Turbine then set PR as state controller
        # Otherwise massflow as controller => For that mass flow needs to be fixed

        self.reset_flags()
        # These are all the associated thermal junctions
        self.thermal_junctions = {
            name: self.network.junctions[name] for name
            in self.network.net_map[self.name]['thermal']
        }

        self.istate = {}
        self.state = np.array([])
        self.node_types = []
        self.update_downstream_energy = False

        if isinstance(self.DS_junction, FlowJunction):
            self.update_downstream_energy = self.DS_junction.update_energy


        if len(self.nodes) == 1 and False:
            if isinstance(self.nodes[0], baseFlowResistor):
                self.stateful = False
                self._w = self.nodes[0]._w
                return

        for inode, node in enumerate(self.nodes):
            # Something to add, bounds from each class 
            # If mass flow is above/below bounds then apply penalty
            nMap = self.model.node_map[node.name]

            if isinstance(node, (fixedFlow)):
                self.node_types.append(fixedFlow)
            elif isinstance(node, baseThermo):
                self.node_types.append(baseThermo)
            elif isinstance(node, baseFlowResistor):
                self.node_types.append(baseFlowResistor)
                self.iResistors.append(inode)
            elif isinstance(node, cycleCloser):
                self.node_types.append(cycleCloser)
            elif isinstance(node, baseInertantFlow):
                self.node_types.append(baseInertantFlow)
            elif isinstance(node, Turbo):
                self.node_types.append(Turbo)
            elif isinstance(node, simpleTurbine):
                self.node_types.append(simpleTurbine)
            elif isinstance(node, chokedTurbine):
                self.node_types.append(chokedTurbine)
                self.cnode = inode
                self.iResistors.append(inode)
                self._iResistors[inode] = node
                self.compressible = True
                #from pdb import set_trace
                #set_trace()
                #self.node_ds = inode + 1


            if isinstance(node, basePump):
                # Turn Pressure Gain flag on if branch contains a pump
                logger.info("Setting pressure gain flag to true for branch "
                            f"{self.name} that contains a pump: {node.name}")
                self.pressure_gain = True

            # Check last node
            if inode == len(self.nodes)-1:
                if inode == 0:
                    if isinstance(node, baseFlowResistor):
                        # If the branch is a single node resistor
                        # set the stateful to false and set the flow
                        # to the resistor flow
                        self.stateful = False
                        self._w = node._w
                        self.solver = 'forward'


            if isinstance(node, baseFlow):

                # Check if node has a fixed flow flag
                if hasattr(node, 'fixed_flow'):
                    if self.fixed_flow:
                        from pdb import set_trace
                        set_trace()

                    if node.fixed_flow:
                        logger.info(f"Flow is flow for branch {self.name} "
                                    f"to {node.name} w: {node._w}")
                        self.fixed_flow_flag = True
                        self.fixed_flow_node = node
                        self._w = node._w

                if False:#node._unidirectional:
                    if self._w >= 0:
                        self._bounds[0] = np.max([self._bounds[0], 0])
                    else:
                        self._bounds[1] = np.min([self._bounds[1], 0])

            if isinstance(node, (Pump, cycleCloser, fixedFlow, FixedFlow)):
                # Check for pressure gain components
                self.pressure_gain = True

           # print('Check for pressure gain components')

            if isinstance(node, (fixedFlow, FixedFlow)):
                if self.fixed_flow_flag:
                    from pdb import set_trace
                    set_trace()

                if self.fixed_flow:
                    from pdb import set_trace
                    set_trace()
                    # There can only be 1 fixed flow object in series
                    logger.critical(
                        f"'{self.fixed_flow_node.name}' is in series with "
                        f"another fixedFlow object: '{node.name}' and has "
                        "different flow rates specified. Double check the "
                        "model!"
                        )

                # Turn on fixedFlow flag for this branch
                self.fixed_flow = True

                # Fixed flow node requires iterating on DS boundary to ensure
                # outlet BC can be properly satisfied
                if inode == len(self.nodes)-1:
                    # If this is the last node in the branch then no iteration is
                    # required
                    self.stateful = False
                    self.solver = 'forward'
                elif inode == 0:
                    self.stateful = True
                    self.solver = 'forward'#'reverse'
                else:
                    self.stateful = True
                    self.solver = 'forward'

                # Set Branch Mass Flow to this component flow
                self._w = node._w
                # Node that sets the mass flow
                self.fixed_flow_node = node
                self._bounds = node._bounds

                if isinstance(node, gt.fixedFlowTurbo):
                    # Get the node bounds
                    if hasattr(node, '_bounds'):
                        self._bounds = node._bounds

                    if len(self.x) != 0:
                        # There shouldn't be any other states
                        from pdb import set_trace
                        set_trace()

            elif isinstance(node, cycleCloser):
                # This calculates dH, dP imbalance

                if not self.fixed_flow:
                    from pdb import set_trace
                    set_trace()

                self.stateful = False

                if inode != len(self.nodes) - 1:
                    # This needs to be at end
                    from pdb import set_trace
                    set_trace()

            elif isinstance(node, baseInertantFlow):

                self.stateful = True
                # Get Bounds
                if not self.fixed_flow:
                    self._bounds[0] = np.max([self._bounds[0],
                                              node._bounds[0]])
                    self._bounds[1] = np.min([self._bounds[1],
                                              node._bounds[1]])

        if isinstance(self.DS_junction.node, gt.Outlet):
            # If the downstream junction is an outlet then
            # it can be any state and not dependent on branch mass
            # mass flow, so this is a stateless fixed flow object
            if not (self.fixed_flow or self.fixed_flow_flag):
                logger.critical(
                    "An outlet Boundary Condition can only be specified if "
                    "a fixedFlow object is defined upstream. Define a "
                    "fixedFlow object upstream of Outlet "
                    f"'{self.DS_junction.node.name}'")

            self.stateful = False

        if self.fixed_flow or self.fixed_flow_flag:
            self._w = self.fixed_flow_node._w
        else:
            self._w = self.average_w

        if self.stateful:
            if self.fixed_flow:
                self.fixed_flow_node.evaluate()

                x = self.fixed_flow_node.PR

                if self._bounds[0] < x < self._bounds[1]:
                    self.update_state([self.fixed_flow_node.PR])
                else:
                    x = self._bounds[0] + np.diff(self._bounds)*.05
                    self.update_state(x)
            else:
                #if self.compressible:
                #    PR = self.nodes[self.node_ds].thermo._P/self.nodes[self.node_ds-2].thermo._P
                #    self.state = np.array([self._w, PR])
                #    from pdb import set_trace
                #    set_trace()
                #else:
                self.state = np.array([self._w])

        if len(self.x) > 1:
            # There should only be 1 state
            from pdb import set_trace
            set_trace()

        self._state = np.copy(self.x)
        self.initialized = True


    @property
    def average_w(self):
        """
        Calculate and return the average mass flow in all components.

        Returns:
            float: The average mass flow rate.
        """

        Wnet = 0
        flow_elements = 0

        W_element = []

        for node in self.nodes:
            if isinstance(node, (baseFlow,
                                 baseInertantFlow)):

                node.evaluate()
                Wnet += node._w
                W_element.append(node._w)
                flow_elements += 1

        if flow_elements == 0:
            from pdb import set_trace
            set_trace()

        W = min(W_element, key=abs)

        #return W
        from pdb import set_trace
        #set_trace()

        return float(Wnet/flow_elements)

    @property
    def xdot(self):

        
        if self.penalty is False:
            self._state = np.copy(self.state)

        if self.solver == 'forward':
            return self.xdot_forward
        elif self.solver == 'reverse':
            return self.xdot_reverse

    @property
    def xdot_forward(self):

        if self.penalty is not False:
            return np.array([self.penalty])

        if self._w < 0:
            Pout = self.US_junction.node.thermo._P
        else:
            Pout = self.DS_junction.node.thermo._P

        self.error = (self.DS_target['P']/Pout - 1)*np.sign(self._w)


        return np.array([self.error])

    @property
    def xdot_reverse(self):

        if self.penalty is not False:
            return np.array([self.penalty])

        if self._w < 0:
            Pin = self.DSJunc.thermo._P
        else:
            Pin = self.USJunc.thermo._P

        if self.fixed_flow:
            from pdb import set_trace
            set_trace()
        else:
            self.error = self.US_target['P']**2/Pin - Pin
        return np.array([self.error])


    def evaluate(self):
        

        if len(self.state) == 0:
            if self.fixed_flow:
                from pdb import set_trace
                #set_trace()
            else:
                for node in self.nodes:
                    node.evaluate()

                self._w = self.nodes[-1]._w

                if self.update_downstream_energy:# and False:
                    H_up = self.US_junction.node.thermo._H
                    #self.DS_junction.node.thermo._HP = H_up, self.DS_junction.node.thermo._P

                return
        

        if self._w < 0:
            nodes = self.nodes[::-1]
            US_node = self.DS_junction.node
            DS_junction = self.US_junction
        else:
            nodes = self.nodes
            US_node = self.US_junction.node
            DS_junction = self.DS_junction

        if self._w == 0:
            self.DS_target = {'P': US_node.thermo._P, 'H': US_node.thermo._H}
            return
        elif US_node.thermo._P < DS_junction.node.thermo._P:
            # Check only if there are no pressure gain components
            # and no fixed flow components
            if not self.pressure_gain and not self.fixed_flow:
                if self._w < 0:
                    self.penalty = (-self.state[0] + 10)*1e20
                else:
                    self.penalty = (-self.state[0] - 10)*1e20
                logger.warn(f"Flow Branch {self.name} has flow from low to high "
                            "pressure")
                return

        self.evaluate_forward(US_node, DS_junction, nodes)


    def update_energy(self):

        if self._w >= 0:
            nodes = self.nodes
        else:
            print('here')
            from pdb import set_trace
            set_trace()

        self.H = np.zeros(len(self.nodes))

        self.H[0] = self.US_junction.node.thermo._H
        for inode, node in enumerate(nodes):

            node.evaluate()

            if isinstance(node, baseThermo):
                self.H[inode] = self.H[0] + 0
                node.thermo._HP = self.H[inode], node.thermo._P


    def evaluate_forward(self, US_node, DS_junction, nodes, choked=False, debug=False):
        """
        Evaluate the branch by updating nodes and calculating errors.

        This method checks the flow and state of each node in the branch,
        reverses order for backflow, and computes errors if constraints
        are violated.
        """
        

        US_thermo = US_node.thermo

        # Update mass flow if this is a fixed flow branch
        if self.fixed_flow and len(self.nodes) > 1:
            if isinstance(nodes[0], FixedFlow):
                nodes[0].get_outlet_state(US_thermo, self.x[0])
                self._w = self.fixed_flow_node._w
            else:
                from pdb import set_trace
                #set_trace()

        if self.penalty:
            # If penalty was triggered then return
            return



        # Track Qin to this branch
        Qin = 0


        #US_j, DS_j, flow, thermo, nodes2 = self.get_nodes_junctions()


        from pdb import set_trace
        #set_trace()

        

        # Loop thru Branch nodes
        for inode, node in enumerate(nodes):

            if debug:
                from pdb import set_trace
                set_trace()

            if inode == len(nodes) - 1:
                # Set node flow to branch mass flow
                node._set_flow(self._w)
                # Update DS_target

                if isinstance(node, (fixedFlow, FixedFlow)):
                    # It doesn't matter what the outlet state is
                    # because the flow is fixed
                    
                    if self.stateful:
                        from pdb import set_trace
                        set_trace()
                    return
                
                if isinstance(node, gt.chokedTurbine):
                    node.evaluate()
                    from pdb import set_trace
                    #set_trace()
                    #if node._w > self._w:
                    #    from pdb import set_trace
                        #set_trace()
                    #    self.penalty = (node._w/self._w + 10)*10#*1e5
                    #else:
                   #     self.penalty = -(self._w - node._w + 10)*10#*1e5
                    self.penalty = (node._w/self._w-1)*10#*1e5
                    
                    return


                self.DS_target = node._get_outlet_state(US_thermo, self._w)
                if not self.stateful:
                    # Evaluate node
                    node.evaluate()
                return
            else:
                # Evaluate the node
                node.evaluate()
                DS_node = nodes[inode+1]

            if isinstance(node, (baseThermo)):
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

                US_thermo = node.thermo
                # Skip the rest of the loop iteration for this node
                continue
            
            # Update the node mass flow to branch object
            error = node._set_flow(self._w)


            if error:
                self.penalty = error
                from pdb import set_trace
                set_trace()
                return

            if isinstance(node, (fixedFlow, FixedFlow)):
                if not self.stateful:
                    # This condition shouldnt occur I think?
                    DS_state = node._get_outlet_state(US_thermo, 1)
                    if not isinstance(self.DS_junction, OutletJunction):
                        from pdb import set_trace
                        #set_trace()
                else:
                    DS_state = node._get_outlet_state(US_thermo, self.x[0])

                if isinstance(node, gt.fixedFlowPump):
                    PR_bounds = np.array(
                        [1, np.max([100, 2*DS_junction.node.thermo._P/US_thermo._P])]
                    )
                    if choked:
                        from pdb import set_trace
                        set_trace()
                    #self.evaluate_fixed_turbo(inode, US_thermo, PR_bounds=PR_bounds)
                    self.evaluate_choked(US_thermo, DS_junction, nodes[inode:], PR_bounds=PR_bounds)
                    from pdb import set_trace
                    #set_trace()
                    return
                else:
                    PR_bounds = [DS_junction.node.thermo._P/US_thermo._P, 1]
                    
                    if choked:
                        from pdb import set_trace
                        set_trace()
                    self.evaluate_choked(US_thermo, DS_junction, nodes[inode:], PR_bounds=PR_bounds)
                    from pdb import set_trace
                    #set_trace()
                    return

                #from pdb import set_trace
                #set_trace()
                #self.evaluate_choked(inode, US_thermo, PR_bounds=PR_bounds)
                #return

            elif isinstance(node, baseFlowResistor):
                w_max = node._w_max(US_thermo)
                
                CHOKED_TOL = 0.999
                BLEND_TOL = 0.99

                if self._w > w_max*BLEND_TOL:
                    if self._w > w_max:#*(2-CHOKED_TOL):
                        logger.info(f"Flow in Branch {self.name} is higher than max flow for"
                                    f" {node.name} of {w_max} by with a flow of {self._w}")
                        self.penalty = (w_max*(2-CHOKED_TOL) - self._w)*1e5
                        return
                    elif self._w > w_max*CHOKED_TOL:
                        logger.info(f"Choking detected in {node.name} for branch {self.name}")
                        
                        if choked:
                            from pdb import set_trace
                            #set_trace()
                        
                        from pdb import set_trace
                        set_trace()
                        
                        self.evaluate_choked(US_thermo, DS_junction, nodes[inode:])
                        #self.evaluate_choked(inode, US_thermo)
                        thermoNode = nodes[-2]
                        flowNode = nodes[-1]    
                        self.DS_target = flowNode._get_outlet_state(thermoNode.thermo, self._w)
                        return
                        # Choked state
                        # Get choked state
                    
                    PR_crit = node.flow.PR_crit(US_thermo)

                    DS_state = node._get_outlet_state(US_thermo, self._w)

                    blend_factor = (self._w/w_max - BLEND_TOL)/(CHOKED_TOL - BLEND_TOL)

                    DS_state['P'] = (1-blend_factor)*DS_state['P'] + blend_factor*PR_crit*US_thermo._P


                DS_state = node._get_outlet_state(US_thermo, self._w)
                # 2 paths
                # Get max mass flow rate
                # If above max then penalty to reduce

                # If equal then evaluate choked
            else:
                DS_state = node._get_outlet_state(US_thermo, self._w)
        

            if DS_state is None:
                # If this is none then there was an error
                # with setting the flow, so lets apply penalty
                logger.warn(f"Error trying to set node {node.name} to "
                            f"{self._w} in branch {self.name}")
                if self.fixed_flow:
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

                if self.fixed_flow:
                    # Calculate penalty based on pressure
                    if isinstance(self.fixed_flow_node, gt.FixedFlow):
                        # Pressure ratio needs to be increased
                        self.penalty = (-DS_state['P']+10)*1e5
                    else:
                        # Add Error to upstream branch
                        # Since this is a fixed flow node
                        # The upstream pressure needs to be increased
                        # to achieve the desired flow
                        from pdb import set_trace
                        set_trace()
                else:
                    #self.DS_target = DS_state
                    if self._w >= 0:
                        # Mass flow needs to get smaller
                        self.penalty = (DS_state['P']-10)*1e5
                    else:
                        # Mass flow needs to get less negative
                        self.penalty = (-DS_state['P']+10)*1e5

                    logger.warn(f"Setting penalty to {self.penalty}")
                    # This is negative so we need to decrease mass flow
                    #self.penalty = (DS_state['P'])*1e5 - 10
                #from pdb import set_trace
                #set_trace()
                return

            if DS_node.name in self.thermal_junctions and self.solve_thermal:
                self.thermal_junctions[DS_node.name].penalty = False
                Qin += (self.thermal_junctions[DS_node.name].Q_flux / (abs(self._w) + eps))
                DS_state['H'] += Qin


            # Last node error check
            if inode == len(nodes) - 1:
                # What the DS Junction Node state should be
                # for steady state
                self.DS_target = DS_state
                return

            if isinstance(node, (baseFlow, baseFlowResistor)):
                # Update thermo for dsNode
                error = DS_node.update_thermo(DS_state)
                if error:
                    logger.warn("Failed to update thermostate in Branch '{self.name}'"
                                f" evaluate call for '{DS_node.name}' to state: "
                                f"{DS_state}")

                    if Qin != 0:
                        DS_state['H'] -= Qin
                        error = DS_node.update_thermo(DS_state)

                        if error is False:
                            logger.warn(f"Heat flow into node '{DS_node.name}' is too high")


                    self.penalty = (self._state[0] - self.x[0]-1e2)*1e4
                    return


                    # Reduce Mass Flow
                    if self.fixed_flow:
                        try:
                        # Point error back to x0
                            self.penalty = (self._state[0] - self.x[0]-1e2)*1e5
                            return
                        except:
                            from pdb import set_trace
                            #set_trace()
                node.evaluate()

            else:
                from pdb import set_trace
                set_trace()


    def get_nodes_junctions(self, inode=0):

        if self._w < 0:
            nodes = self.nodes[::-1]
            US_junction = self.DS_junction
            DS_junction = self.US_junction
        else:
            nodes = self.nodes
            US_junction = self.US_junction
            DS_junction = self.DS_junction
            flow = self.nodes[inode]
            thermo = self.nodes[inode+1]
            nodes = self.nodes[inode+2:]

        return US_junction, DS_junction, flow, thermo, nodes

    def evaluate_choked(self, US_thermo, DS_junction, nodes, PR_bounds=None):
        # If branch gets choked then DS pressure does not affect US and we need to do stuff


        US = US_thermo.from_state(US_thermo.state)
        def find_PR(PR):

            if PR < PR_min:
                return PR_min - PR + 1
            elif PR > PR_hi:
                return PR_hi - PR - 1
            

            DS_state = nodes[0].get_outlet_state(US_thermo, PR)
            US._HP = DS_state['H'], US_thermo._P * PR

            if nodes[1].name in self.thermal_junctions:
                Q = self.thermal_junctions[nodes[1].name].Q_flux/(abs(self._w) + eps)
                DS_state['H'] += Q

            error = nodes[1].update_thermo(DS_state)
            if error:
                from pdb import set_trace
                set_trace() 


            self.evaluate_forward(nodes[1], DS_junction, nodes[2:], choked=True, debug=False)


            if self.penalty:
                return -self.penalty
            else:
                return DS_junction.node.thermo._P - self.DS_target['P']


        if PR_bounds is None:
            PR_hi = nodes[0].flow.PR_crit(US_thermo)
            PR_min = self.DS_junction.thermo._P/US_thermo._P
        else:
            PR_hi = PR_bounds[1]
            PR_min = PR_bounds[0]

        self.penalty = False
        # Figure out PR downstream of choked node
        try:
            sol = root_scalar(find_PR, bracket=[PR_min, PR_hi], method='brentq')
        except:
            from pdb import set_trace
            set_trace()
        #sol = fsolve(find_PR, PR_min, full_output=True)

        PR = sol.root

        if isinstance(nodes[0], gt.FixedFlow):
            DS_state = nodes[0].get_outlet_state(US_thermo, PR)
        else:
            DS_state = {'P': US_thermo._P * PR, 'H': US_thermo._H}

        if nodes[1].name in self.thermal_junctions:
            Q = self.thermal_junctions[nodes[1].name].Q_flux
            DS_state['H'] += Q/(abs(self._w) + eps)


        nodes[1].update_thermo(DS_state)

    def update_choked_thermo(self, US, DS, res, PR):
        # Update the thermo for the choked node
        # US Resistor


        # Get choked node
        DS_state = {'P': US._P*PR, 'H': US._H}

        error = DS.update_thermo(DS_state)

        if error:
            from pdb import set_trace
            set_trace()

        res.evaluate()
        DS_state = {'P': US._P*PR, 'H':US._H + res._dH}
        error = DS.update_thermo(DS_state)

        if error:
            from pdb import set_trace
            set_trace()


    def solve_steady(self):

        def evaluate(w):

            self.update_state(w)
            self.evaluate()
            return self.xdot

        
        self.debug = True

        evaluate(np.array([-246]))


        xdot = []
        w_vec = np.arange(-500,-50 ,.5)
        for w in w_vec:
            xdot.append(evaluate(np.array([w]))[0])
        
        from matplotlib import pyplot as plt
        #plt.plot(w_vec,xdot)

        sol = fsolve(evaluate, self._w, full_output=True)





        from pdb import set_trace
        set_trace()


@addQuantityProperty
class ThermalBranch(baseBranch):
    """
    Represents a thermal branch in the geoTherm model network.

    ThermalBranches are segments with thermal resistors in series
    """

    _units = {'Q': 'POWER'}
    _bounds = [-np.inf, np.inf]
    

    def __init__(self, name, nodes, hot_junction, cold_junction, network):

        super().__init__(name, nodes, hot_junction, cold_junction, network)

        self.hot_junction = hot_junction
        self.cold_junction = cold_junction


    @property
    def _Q(self):
        if self.fixed_flow:
            return self.fixed_flow_node._Q
        elif not self.stateful:
            return self.nodes[0]._Q
        else:
            return self.state[0]

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
        self.state = np.array([])

        if len(self.nodes) == 1:
            if isinstance(self.nodes[0], (Qdot, ConvectiveResistor)):
                # The heat is specified via the Qdot object
                self.stateful = False
                return
            elif (isinstance(self.US_junction, BoundaryJunction) and
                  isinstance(self.DS_junction, BoundaryJunction)):
                self.stateful = False
                return


        from pdb import set_trace
        set_trace()
        for inode, node in enumerate(self.nodes):
            nMap = self.model.node_map[node.name]

            if isinstance(node, (gt.Qdot)):
                self.fixed_flow = True
                self.fixed_flow_node = node
                self.initialized = True
                self.stateful = False

                if self._Q >= 0:
                    self.solver = 'reverse'
                else:
                    self.solver = 'forward'

                return

        self.state = np.array([self.average_Q])
        self._state = np.copy(self.x)
        self.initialized = True


    def evaluate_reverse(self):

        nodes = self.nodes

        if self._Q < 0:
            nodes = nodes
            US_junction = self.DS_junction.node
            DS_junction = self.US_junction.node
        else:
            nodes = nodes[::-1]
            US_junction = self.US_junction.node
            DS_junction = self.DS_junction.node

        
        from pdb import set_trace
        set_trace()
        for inode, node in enumerate(nodes):
            # Evaluate the node
            node.evaluate()

            if isinstance(node, (baseThermo)):
                continue
            elif isinstance(node, (gt.Heatsistor)):
                node._set_heat(self._Q)
            elif isinstance(node, gt.Qdot):
                continue
                from pdb import set_trace
                set_trace()   
            
            US_state, US_node = node.get_US_state()

            if inode == len(nodes) - 1:
                self.US_target = US_state

            if US_node == 'TurbOut2':
                from pdb import set_trace
                set_trace()            
            self.model.nodes[US_node].update_thermo(US_state)

            from pdb import set_trace
            #set_trace()

    def evaluate_forward(self):

        nodes = self.nodes

        if self.fixed_flow:
            if len(self.nodes) == 1:
                return
            else:
                self.nodes[-1]._Q = self._Q
                return

            from pdb import set_trace
            set_trace()
            #self.state = self.fixed_flow_node

        if self.penalty:
            return

        if self._Q < 0:
            # Reverse nodes if heat flux is negative
            nodes = nodes[::-1]
            US_junction = self.DS_junction.node
            DS_junction = self.US_junction.node
            hot_junction = self.DS_junction#.node
            cool_junction = self.US_junction#.node

            if isinstance(hot_junction, baseFlow):
                from pdb import set_trace
                set_trace()
            if isinstance(cool_junction, baseFlow):
                from pdb import set_trace
                set_trace()
                cool_junction = cool_junction.US_node
        else:
            US_junction = self.US_junction.node
            DS_junction = self.DS_junction.node
            hot_junction = self.US_junction#.node
            cool_junction = self.DS_junction#.node

            #if isinstance(hot_junction, baseFlow):
            #    from pdb import set_trace
            #    set_trace()
            #if isinstance(cool_junction, baseFlow):

             #   _,_, i = cool_junction.thermostates()

             #   if i > 1:
             #       cool_junction = cool_junction.US_node
             #   else:
             #       cool_junction = cool_junction.DS_node

            #from pdb import set_trace
            #set_trace()

        #US_junction = self.US_junction.node
        #DS_junction = self.DS_junction.node

        if not self.fixed_flow:
            if hot_junction._T < cool_junction._T:
                if self._Q < 0:
                    self.penalty = (-self.state[0] + 10)*1e10
                else:
                    self.penalty = (-self.state[0] - 10)*1e10

                logger.warn(f"Thermal Branch {self.name} has heat flowing from "
                            "cold to hot")
                return


        hot_thermo = hot_junction.thermo
        for inode, node in enumerate(nodes):
            # Evaluate the node
            node.evaluate()

            if isinstance(node, (baseThermo)):
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

                hot_node = node.thermo
                # Skip the rest of the loop iteration for this node
                continue
            
            elif isinstance(node, gt.Qdot):
                from pdb import set_trace
                set_trace()

            elif isinstance(node, (gt.Heatsistor)):
                
                # Update the downstream state
                error = node._set_heat(self._Q)

                if error:
                    self.penalty = error
                    from pdb import set_trace
                    set_trace()
                    return
            
            #node.get_cool_state()

            cool_state = node.cool_state(hot_thermo, self._Q)

            if cool_state is None:
                from pdb import set_trace
                set_trace()

            if cool_state['T']<0:
                self.penalty = (cool_state['T']-10)*1e5

                logger.warn(f"Thermal Branch {self.name} heat flow: {self._Q} "
                            "is too high and results in negative temperature")

                return

            if inode == len(nodes) - 1:
                self.cool_target = cool_state
                return

            from pdb import set_trace
            set_trace()
            self.model.nodes[DS_node].update_thermo(DS_state)



        from pdb import set_trace
        set_trace()
    
    @property
    def x(self):
        if not self.stateful:
            return np.array([])
        else:
            return self.state 
            from pdb import set_trace
            set_trace()

    @property
    def xdot(self):
        if not self.stateful:
            return np.array([])

        if self.penalty is not False:
            return np.array([self.penalty])

        if self._Q < 0:
            Tout = self.US_junction._T
        else:
            Tout = self.DS_junction._T
            from pdb import set_trace
            set_trace()

        if self.fixed_flow:
            from pdb import set_trace
            set_trace()
        else:
            # Error is proportional to Q
            # Maybe in the future implement relative and absolute error?
            error = (self.cool_target['T'] - Tout)/np.abs(self._Q+eps)**.25

        return np.array([error])
