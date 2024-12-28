from .junctions import (Junction, BoundaryJunction,
                        ThermalJunction, FlowJunction)
from geoTherm.units import addQuantityProperty
from ...nodes.baseNodes.baseThermal import baseThermal
from ...nodes.baseNodes.baseThermo import baseThermo
from ...nodes.baseNodes.baseFlow import baseFlow, baseInertantFlow
from ...nodes.baseNodes.baseTurbo import Turbo, fixedFlowTurbo
from ...nodes.flowDevices import fixedFlow
from ...nodes.baseNodes.baseFlowResistor import baseFlowResistor
from ...nodes.cycleCloser import cycleCloser
from ...logger import logger
from ...utils import thermo_data
import numpy as np
import geoTherm as gt
from ...thermostate import thermo
from ...nodes.pump import Pump
from ...nodes.turbine import Turbine

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
            DS_junction (str or instance): Downstream junction name or instance.
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
        self._back_flow = False

        # Node for controlling flow or heat
        self.fixed_flow_node = None
        self.penalty = False
        self.solver = 'forward'
        self.stateful = True
        self.initialized = False
        self.linearly_independent = False
        # Create temporary thermo object for intermediate calcs
        if isinstance(self.US_junction.node, baseThermo):
            self._thermo = thermo.from_state(self.US_junction.node.thermo.state)
        else:
            self._thermo = thermo.from_state(self.DS_junction.node.thermo.state)

        self.initialize()

    @property
    def back_flow(self):
        return self._back_flow

    @back_flow.setter
    def back_flow(self, flag):

        self._back_flow = flag

        if self._back_flow:
            self._bounds = [-self._bounds[1],
                            self._bounds[1]]

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
        return self._x

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

        if self.fixed_flow:
            self.fixed_flow_node.update_state(x)
            # Get the penalty from the setter object
            self.penalty = self.fixed_flow_node.penalty
        else:
            if x[0] < 0 and not self.back_flow:
                # If backflow is not enabled apply penalty
                self.penalty = (10 - x[0])*1e5
                return

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



        if self.solver == 'forward':
            self.evaluate_forward()
        elif self.solver == 'reverse':
            self.evaluate_reverse()


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

   
        self.state = 'massflow'
    # Checks
    # Check if cycleCloser is True
    #   Check if fixedFlow specified = then can't use
    #   fixedFlowPump
    #   Should use fixedPump

    def get_Q(self, node):

        if node.name in self.thermal_junctions:
            return self.thermal_junctions[node.name].Q_flux
        else:
            return 0


    def initialize(self):

        # Define the states defining this dict

        # If PR Turbine then set PR as state controller
        # Otherwise massflow as controller => For that mass flow needs to be fixed
        
        thermal_junctions = self.network.net_map[self.name]['thermal']

        self.thermal_junctions = {name:self.network.junctions[name] for name in thermal_junctions}


        self.istate = {}
        self._x = np.array([])
        self.hot_nodes = {}
        self.node_types = []

        # Check if fixed_flow
        # If outlet or dP downstream

        # If outlet then update Pout
            # We can solve forward

        # If BC
        # Have to solve backwards


        for inode, node in enumerate(self.nodes):
            # Something to add, bounds from each class 
            # If mass flow is above/below bounds then apply penalty
            nMap = self.model.node_map[node.name]

            from pdb import set_trace
            #set_trace()
            # FixedflowTurbo
            if isinstance(node, (fixedFlow)):
                self.node_types.append(fixedFlow)
            elif isinstance(node, (fixedFlowTurbo)):
                self.node_types.append(fixedFlowTurbo)
            elif isinstance(node, baseThermo):
                self.node_types.append(baseThermo)
            elif isinstance(node, baseFlowResistor):
                self.node_types.append(baseFlowResistor)
            elif isinstance(node, cycleCloser):
                self.node_types.append(cycleCloser)
            elif isinstance(node, baseInertantFlow):
                self.node_types.append(baseInertantFlow)
            elif isinstance(node, Turbo):
                self.node_types.append(Turbo)
            else:
                from pdb import set_trace
                set_trace()

            if isinstance(node, (fixedFlowTurbo, fixedFlow)):
                if self.fixed_flow:
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
                    self.state = 'pressure'
                else:
                    self.stateful = True
                    self.solver = 'forward'
                    self.state = 'pressure'

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
                        from pdb import set_trace
                        #set_trace()
                        #self._x = node.x

            elif isinstance(node, cycleCloser):
                # This calculates dH, dP imbalance

                if not self.fixed_flow:
                    from pdb import set_trace
                    set_trace()

                from pdb import set_trace
                #set_trace()
                self.stateful = False

                if inode != len(self.nodes) - 1:
                    # This needs to be at end
                    from pdb import set_trace
                    set_trace()

            elif isinstance(node, baseInertantFlow):
                self.stateful = True
                #from pdb import set_trace
                #set_trace()

            elif isinstance(node, gt.statefulFlowNode):
                from pdb import set_trace
                set_trace()
                if not self.fixed_flow:
                    self._bounds[0] = np.max([self._bounds[0],
                                              node._bounds[0]])
                    self._bounds[1] = np.min([self._bounds[1],
                                              node._bounds[1]])

        if isinstance(self.DS_junction.node, gt.Outlet):
            # If the downstream junction is an outlet then
            # it can be any state and not dependent on branch mass
            # mass flow, so this is a stateless fixed flow object
            if self.fixed_flow:
                if len(self.x) != 0:
                    from pdb import set_trace
                    set_trace()

            self.stateful = False



        #if self.fixed_flow:
        #    if self.stateful:
        #        pass
        #    else
        #elif not self.stateful:
        #    pass
        #else:
        if self.fixed_flow:
            self._w = self.fixed_flow_node._w
        else:
            self._w = self.average_w


        if self.stateful:
            if self.fixed_flow:
                self.fixed_flow_node.evaluate()

                if isinstance(self.fixed_flow_node, Pump):
                    self._bounds = [1, 500]
                else:
                    from pdb import set_trace
                    set_trace()

                x = self.fixed_flow_node.PR

                if self._bounds[0] < x < self._bounds[1]:
                    self.update_state([self.fixed_flow_node.PR])
                else:
                    x = self._bounds[0] + np.diff(self._bounds)*.05
                    self.update_state(x)
                
            else:
                self._x = np.array([self._w])

        if len(self.x) > 1:
            # There should only be 1 state
            from pdb import set_trace
            set_trace()

        self._x0 = np.copy(self.x)
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

        for node in self.nodes:
            if isinstance(node, (baseFlow,
                                 baseInertantFlow)):
                          #gt.BoundaryConnector,
                          #baseFlowResistor)):
                
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
            Pout = self.US_junction.node.thermo._P
        else:
            Pout = self.DS_junction.node.thermo._P

        if self.fixed_flow:
            if isinstance(self.fixed_flow_node, gt.Turbine):
                self.error = (self.DS_target['P']/Pout - 1)*Pout
            elif isinstance(self.fixed_flow_node, gt.Pump):
                self.error = (self.DS_target['P']/Pout - 1)
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

        if self.fixed_flow:
            from pdb import set_trace
            set_trace()
        else:
            self.error = self.US_target['P']**2/Pin - Pin
        return np.array([self.error])

    def evaluate(self):


        self.evaluate_forward()
        return

        # CHECK IF FF is 1st Node, if not then need more work

        if self.fixed_flow and not self.stateful:
            self._w = self.fixed_flow_node._w

            if self._w >= 0 and self.solver == 'forward':
                self.evaluate_forward()
            else:
                self.evaluate_reverse()

            return

        if self.solver == 'forward':
            self.evaluate_forward()
        elif self.solver == 'reverse':
            self.evaluate_reverse()


    def _evaluate_reverse(self, H_outlet, US_junction, nodes):

        DS_thermo = thermo_data(H=H_outlet,
                    P=self.DS_junction.node.thermo._P,
                    fluid=US_junction.thermo.Ydict,
                    model=US_junction.thermo.model)


        Q = np.sum([self.get_Q(node) for node in self.nodes])

    
        # I should put a try except state for updating DS thermo
        # If fail then heat flux is too high and need to increase
        # mass flow


        DS_thermo = self._thermo

        DS_thermo._HP = self.DS_junction.node.thermo._HP

        if self._w != 0:
            self._thermo._HP += Q/(self._w), self._thermo._P

        for inode, node in enumerate(nodes):
            # Evaluate the node
            node.evaluate()

        
            if isinstance(node, baseThermo):
                if not inode & 0x1:
                    print('Here')
                    from pdb import set_trace
                    set_trace()

                DS_thermo = node.thermo
                continue
            
            if isinstance(node, fixedFlow):
                US_node = nodes[inode+2]
                US_vol = nodes[inode+1]
                US_node._w = self._w
                from pdb import set_trace
                set_trace()
                _, DS_state = US_node.get_DS_state()

                US_vol.update_thermo(DS_state)
                continue
                from pdb import set_trace
                set_trace()

            if inode == len(nodes) - 1:
                print("here")
                from pdb import set_trace
                set_trace()
                return

            # Update Temporary Thermo Object
            #DS_thermo._HP = H_outlet, DS_thermo._P
            try:
                US_state = node.get_US_state(self._thermo, self._w)
            except:
                from pdb import set_trace
                set_trace()
            #DS_thermo.update(US_state)1
            
            US_node = nodes[inode+1]


            if US_state is None:
                from pdb import set_trace
                set_trace()

            if US_state['P'] > 1e9:
                self.penalty = -(US_state['P'])*1e5
                return

            from pdb import set_trace
            set_trace()


            error =US_node.update_thermo(US_state)        

            if error:
                    

                from pdb import set_trace
                set_trace()
                #self.DS_junction.node.thermo._HP = HP0
                #from pdb import set_trace
                #set_trace()
            
            node.evaluate()
            
            # DS State
            # Update DS state to be isenthalpic
            # get US State

            # For Pump/Turbine
            # Vol => [Pump] => Vol
            #         +dQ      Vol+dH


            # Forward appraoch
            # Vol H  pump dH Vol H
            # H1      dH      H1+dH
            
            # H1      dH   H1+dH 
            # Iterate to update energy



            


            #DS_thermo.update(US_state)
            if inode == len(nodes) - 1:
                from pdb import set_trace
                set_trace()
                node.evaluate()
                self.US_target = US_state

                if US_node != US_junction.name:
                    from pdb import set_trace
                    set_trace()
                break

            if nodes[inode+1].name != US_node:
                from pdb import set_trace
                set_trace()

            #error = self.model.nodes[US_node].update_thermo(US_state)
            
            if error:
                from pdb import set_trace
                set_trace()


            if np.abs(node._w - self._w) > 1e-3:
                from pdb import set_trace
                set_trace()

        from pdb import set_trace
        set_trace()

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




    def evaluate_reverse(self):

        nodes = self.nodes

        if self.fixed_flow:
            self._w = self.fixed_flow_node._w

        if self.penalty:
            # If penalty was triggered then return
            return

        if self._w < 0:
            US_junction = self.DS_junction.node
            DS_junction = self.US_junction.node
        else:
            # Reverse node order
            nodes = nodes[::-1]
            US_junction = self.US_junction.node
            DS_junction = self.DS_junction.node

        H_guess = US_junction.thermo._H

        self.update_energy()
        self._evaluate_reverse(H_guess, US_junction, nodes)

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
        if self.fixed_flow:
            self._w = self.fixed_flow_node._w

        if self.penalty:
            # If penalty was triggered then return
            return

        try:
            # Reverse order if w is negative
            if self._w < 0:
                # Reverse node order
                nodes = nodes[::-1]
                DS_junction = self.US_junction.name
            else:
                DS_junction = self.DS_junction.name
        except:
            from pdb import set_trace
            set_trace()

        US_thermo = self.US_junction.node.thermo
        # Loop thru Branch nodes
        from pdb import set_trace
        #set_trace()
        for inode, node in enumerate(nodes):
            # Evaluate the node
            node.evaluate()

            if inode == len(nodes) -1:
                
                if not self.stateful:
                    return
                else:
                    node._set_flow(self._w)
                    node.evaluate()
                    self.DS_target = node.get_outlet_state(US_thermo, self._w)
                    return
            else:
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

            

            if isinstance(node, gt.fixedFlowNode):
                from pdb import set_trace
                set_trace()
                if node._w == self._w:
                    pass
                else:
                    from pdb import set_trace
                    set_trace()
            else:
                try:
                    # Update the downstream state
                    error = node._set_flow(self._w)
                except:
                    from pdb import set_trace
                    set_trace()

                if error:
                    self.penalty = error
                    return

            if isinstance(node, (fixedFlow, fixedFlowTurbo)):
                DS_state = node.get_DS_state(US_thermo, self.x)
            else:
                DS_state = node.get_DS_state(US_thermo, self._w)

            if self._w != 0:
                Q = self.get_Q(DS_node)
                DS_state['H'] += Q/abs(self._w)


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
            try:
                if DS_state['P'] < 0:
                    # Pressure drop is too high because massflow too high,
                    # lets decrease mass flow by applying penalty
                    # Maybe set this as seperate method for different cases
                    logger.warn("Pressure <0 detected in Branch "
                                f"'{self.name}' for branch state: "
                                f"{self.x}")

                    if self.fixed_flow:
                        # Calculate penalty based on pressure
                        if isinstance(self.fixed_flow_node, gt.Pump):
                            # Increase Pressure Ratio
                            self.penalty = (-DS_state['P']+10)*1e5
                        else:
                            from pdb import set_trace
                            set_trace()
                    else:
                        # This is negative so we need to decrease mass flow
                        self.penalty = (DS_state['P'])*1e5 - 10
                    return
            except:
                from pdb import set_trace
                set_trace()

            # Last node error check
            if inode == len(nodes) - 1:
                # What the DS Junction Node state should be
                # for steady state
                self.DS_target = DS_state
                return

            if isinstance(node, (baseFlow, baseFlowResistor)):


                #DS_state = self._getDSThermo(DS_node, DS_state)
                # Update thermo for dsNode
                error = DS_node.update_thermo(DS_state)

                if error:
                    logger.warn("Failed to update thermostate in Branch "
                                f" evaluate call for '{DS_node.name}' to state: "
                                f"{DS_state}")

                    # Reduce Mass Flow
                    if self.fixed_flow:

                        # Point error back to x0
                        self.penalty = (self._x0[0] - self.x[0]-1e2)*1e5
            else:
                from pdb import set_trace
                set_trace()

    def update_state(self, x):
        """
        Update the Branch state with a new state vector.

        Args:
            x (array): The new state vector.
        """

        # Store the original state
        # We may need to revert if penalty are triggered

        self._x0 = np.copy(self.x)
        self._x = x
        self.penalty = False

        if self._bounds[0] <= x[0] <= self._bounds[1]:
            if not self.fixed_flow:
                self._w = x[0]
        else:
            if x[0] < self._bounds[0]:
                self.penalty = (self._bounds[0] - x[0])*1e5 + 10
            else:
                self.penalty = (self._bounds[0] - x[0])*1e5 - 10

    @property
    def xdot_copy(self):
        if isinstance(self.error, float):
            return np.array([self.error])
        else:
            return self.error

@addQuantityProperty
class ThermalBranch(baseBranch):
    """
    Represents a thermal branch in the geoTherm model network.

    ThermalBranches are segments with thermal resistors in series
    """

    _units = {'Q': 'POWER'}
    _bounds = [-np.inf, np.inf]
    
    @property
    def _Q(self):
        if self.fixed_flow:
            return self.fixed_flow_node._Q
        elif not self.stateful:
            return self.nodes[0]._Q
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
        self._x = np.array([])

        if (isinstance(self.US_junction, BoundaryJunction) and
            isinstance(self.DS_junction, BoundaryJunction)):
            if len(self.nodes) == 1:
                self.stateful = False
        

        if len(self.nodes) == 1:
            if (isinstance(self.US_junction, BoundaryJunction)
                and isinstance(self.DS_junction, ThermalJunction)):
                self.stateful = False
                return
            elif (isinstance(self.DS_junction, BoundaryJunction)
                  and isinstance(self.US_junction, ThermalJunction)):
                self.stateful = False
                return

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

        self._x = np.array([self.average_Q])
        self._x0 = np.copy(self.x)
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
            #self._x = self.fixed_flow_node

        if self.penalty:
            return

        if self._Q <= 0:
            # Reverse nodes if heat flux is negative
            nodes = nodes[::-1]
            US_junction = self.DS_junction.node
            DS_junction = self.US_junction.node
        else:
            US_junction = self.US_junction.node
            DS_junction = self.DS_junction.node
    
        if not self.fixed_flow:
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
                    # Heat can't flow from cold to hot
                    self.penalty = (-self._x[0]-1)*1e5
                    return
            else:
                if self._x !=0:        
                    # Check Temperatures
                    from pdb import set_trace
                    set_trace()

        # Loop thru Branch nodes
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
                    return
                
            DS_node, DS_state = node.get_DS_state()

            if DS_state is None:
                from pdb import set_trace
                set_trace()

            if DS_state['T']<0:
                self.penalty = (DS_state['T']-10)*1e5
                return

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
    def x(self):
        if not self.stateful:
            return np.array([])
        else:
            return self._x 
            from pdb import set_trace
            set_trace()

    @property
    def xdot(self):
        if not self.stateful:
            return np.array([])

        if self.penalty is not False:
            return np.array([self.penalty])
        
        if self._x < 0:
            Tout = self.US_junction.node.thermo._T
        else:
            Tout = self.DS_junction.node.thermo._T
        
        if self.fixed_flow:
            from pdb import set_trace
            set_trace()
        else:
            self.error = (self.DS_target['T'] - Tout)

        return np.array([self.error])
