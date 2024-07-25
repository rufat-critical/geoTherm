import numpy as np
import geoTherm as gt
from scipy.optimize import fsolve
from scipy.integrate import BDF
from scipy.optimize._numdiff import approx_derivative
from .nodes.node import modelTable
from .logger import logger
from geoTherm.units import addQuantityProperty
from geoTherm.utils import eps
from pdb import set_trace

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

    def initialize(self):
        """Initialize the model and nodes"""

        # Initialize variables for tracking statefuls
        self.statefuls = []
        self.istate = {}
        self.__nstates = 0

        # Initialize State Vector (as private variable)
        self.__x = np.array([])

        # Loop thru nodes and add a reference to this model
        for name, node in self.nodes.items():
            node.model = self

        # Generate nodeMap
        self.nodeMap = self._generateNodeMap()

        # Loop thru nodes and call initialization method if defined
        for name, node in self.nodes.items():
            if hasattr(node, 'initialize'):
                node.initialize(self)

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
        if self.errorChecker():
            raise ValueError("Errors Found, Fix the Model!")

        self.initialized = True

    def __init_x(self):
        # Initialize the model state vector using the current
        # component states
        self.__x = np.empty(self.__nstates)

        for name, indx in self.istate.items():
            self.__x[indx] = self.nodes[name].x

    def errorChecker(self):
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

        return error

    def updateState(self, x):
        """Update the component states in the model"""
        self.__x[:] = x

        # Update Component states
        for name, istate in self.istate.items():
            self.nodes[name].updateState(x[istate])

    @property                
    def x(self):
        """Get the state vector"""
        return self.__x

    @property
    def error(self):
        """Get the error vector"""
        for name, istate in self.istate.items():
            self.__error[istate] = self.nodes[name].error

        return self.__error

    def _generateNodeMap(self):
        """ Generate a nodal connectivity map for the defined
            nodes """

        # Initialize empty Node dictionary
        nMap = {name: {'US': [], 'DS': [], 'hot': [], 'cool': []} 
                for name in self.nodes}

        # Loop thru all nodes and find connectivity
        for name, node in self.nodes.items():
            # Check if US defined in node
            if hasattr(node, 'US'):
                # Check if it already exists in nodeMap
                if node.US not in nMap[name]['US']:
                    # If not then append to nodeMap
                    nMap[name]['US'].append(node.US)
                # Check if downstream node exsits in nodeMap
                if node.DS not in nMap[name]['DS']:
                    # If not then append it too
                    nMap[name]['DS'].append(node.DS)

                # Now check the downstream nodes 
                if name not in nMap[node.US]['DS']:
                    nMap[node.US]['DS'].append(name)
                if name not in nMap[node.DS]['US']:
                    nMap[node.DS]['US'].append(name)

            if hasattr(node, 'hot'):
                # Check if it already exists in nodeMap
                if node.hot not in nMap[name]['hot']:
                    # If not then append to nodeMap
                    nMap[name]['hot'].append(node.hot)                
                # Check if cool node exists in nodeMap
                if node.hot not in nMap[node.hot]['cool']:
                    # If not then append it too
                    nMap[node.hot]['cool'].append(name)     

            if hasattr(node, 'cool'):
                # Check if it already exists in nodeMap
                if node.cool not in nMap[name]['cool']:
                    # If not then append to nodeMap
                    nMap[name]['cool'].append(node.cool)                
                # Check if cool node exists in nodeMap
                if node.cool not in nMap[node.cool]['hot']:
                    # If not then append it too
                    nMap[node.cool]['hot'].append(name)                         

        return nMap


    def evaluate(self, x):
        """Evaluate the model with given state vector x"""

        # First update the model and component states
        self.updateState(x)

        # Evaluate the nodes
        for name, node in self.nodes.items():
            if hasattr(node, 'evaluate'):
                node.evaluate()

        # Return the error
        return self.error


    def _netJac(self, x):
        # Evaluate the network Jacobian

        x0 = np.copy(self._netX)
        f0 = self.evaluateNet(x0)

        J = approx_derivative(self.evaluateNet, x,
                              rel_step=None,
                              abs_step=1e-10,
                              method='3-point',
                              f0=f0)

        # Revert state back to OG state
        self._updateNet(x0)

        return J

    def _generateBranchMap(self):
        # Generate Branch Map for 

        # Currently being used for 1 loop
        # THIS SHOULD BE UPDATED IN THE FUTURE IF BRANCHING
        # IS ADDED TO THE CODE

        from pdb import set_trace
        set_trace()


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

        # Generate nodeMap

    # solveSystem

    # Print results

    # Plot performance data

    # Save results

    def evaluateNet(self, x):

        self._updateNet(x)


        for _,branch in self.branches.items():
            branch.evaluate()

        for _,junc in self.junctions.items():
            junc.evaluate()

        return self._netError
  

    def solve(self, netSolver=True):

        # Branch Nodes
        # Set mass flow
        # getOutlet State
        # update
        # Error to update mass flow

        if netSolver:
            self.initializeSteady()
            
            sol = fsolve(self.evaluateNet, self._netX, full_output=True)
            
            self.debug = True
            self.evaluateNet(sol[0])
            if not self.converged:
                from pdb import set_trace
                set_trace()
            else:
                return



        sol = fsolve(self.evaluate, self.x, full_output=True)
        # Update State to Sol
        # Add Update State Method and update that way
        self.evaluate(sol[0])
        #from pdb import set_trace
        #set_trace()


    def sim(self):
        
        integrator = BDF(lambda t,x: self.evaluate(x), 0, self.x, 1e5)

        while integrator.t < .1:
            try:
                integrator.step()
            except:
                from pdb import set_trace
                set_trace()
        from pdb import set_trace
        set_trace()



    def evaluateNetwork(self, x):
        # Evaluate the network 
        # (with nodes grouped into branches and junctions)

        for bname, branch in self.branches.items():
            branch.updateState(x)


        branch.evaluate()

        return branch.error
        

    def evaluate_old(self, x):
        # Evaluate each node individually

        
        #x[1] = np.abs(x[0])
        #x[1] = np.max([1, x[0]])
        # Update the system state
        for name, istate in self.istate.items():
            self.nodes[name].updateState(x[istate])
        
        # I need to initialize and check # of states
        # Then check what error to output

        # This needs to be updated, write code to find branches
        # and then loop for branches, but for now this works,
        # may not in the future
        for i, (name, node) in enumerate(self.nodes.items()):

            if isinstance(node, (gt.PBoundary,
                                 gt.Station,
                                 gt.Boundary)):
                continue

            if i == len(self.nodes)-1:
                outletState = node.getOutletState()

                #error = [(outletState['T'] - self.nodes['P'].thermo._T),
                #            outletState['P'] - self.nodes['P'].thermo._P]
                error = [outletState['P'] - self.nodes['P'].thermo._P]
                return np.array(error)


            outletState = node.getOutletState()

            if outletState['P'] < 0:
                from pdb import set_trace
                set_trace()

            # Get the DS node
            DS = self.nodes[self.nodeMap[name]['DS'][0]]
            DS.updateThermo(outletState)

    def _getBranchesAndJunctions(self, nodeMap):
        """
        Find all the branches and junctions in a geoTherm node map.

        Args:
            nodeMap (dict): A dictionary representing the node map of the geoTherm system.

        Returns:
            tuple: A tuple containing:
                - branches (dict): A dictionary of branches.
                - junctions (dict): A dictionary of junctions.
                - branchConnections (dict): A dictionary mapping branches to their upstream and downstream junctions.
                - nodeClassification (dict): A dictionary mapping nodes to their corresponding branch or junction.
        """

        # List of all nodes to be processed
        remainingNodes = list(nodeMap.keys())
        # Dictionaries to store junctions and branches
        junctions = {}
        branches = {}
        # Branch identifier
        branchCounter = 0
        # Dictionary to map branches to their upstream and downstream junctions
        branchConnections = {}
        # Dictionary to map nodes to their corresponding branch or junction
        nodeClassification = {}

        def isJunction(nodeName):
            """
            Determine if a node is a junction.

            Junctions are boundary nodes, nodes with more than one inlet or outlet, 
            or nodes with a heat connection (hot or cool).

            Args:
                nodeName (str): The name of the node.

            Returns:
                bool: True if the node is a junction, False otherwise.
            """
            
            # Get the node instance 
            node = self.nodes[nodeName]
            
            if isinstance(node, (gt.Boundary, gt.hexVolume)):
                return True
            if nodeMap[nodeName]['hot'] or nodeMap[nodeName]['cool']:
                if (len(nodeMap[nodeName]['US']) == 0 and
                    len(nodeMap[nodeName]['DS']) == 0):
                    return True
                
                # Check these cases
                # hot or cool only has hot/cool connector and no flow
                print('CHECK JUNCTION CLASSIFIER WHEN SOBER!')
                # If hot or cool in upstream then we ok
                return False
            
            # If these conditions are false then this is branch node
            return (len(nodeMap[nodeName]['US']) !=1 or
                    len(nodeMap[nodeName]['DS']) !=1 or
                    len(nodeMap[nodeName]['hot']) !=0 or
                    len(nodeMap[nodeName]['cool']) !=0)

        # Identify all junction nodes
        for nodeName in remainingNodes:
            if isJunction(nodeName):
                junctions[nodeName] = {'US': [], 'DS': [], 'hot': [], 'cool': []}
                nodeClassification[nodeName] = nodeName

        # Remove junction nodes from remainingNodes list
        remainingNodes = [node for node in remainingNodes if node not in junctions]

        def traceBranch(nodeMap, currentNode, currentBranch, remainingNodes):
            """
            Recursively trace a branch starting from the current node.

            Args:
                nodeMap (dict): The node map of the geoTherm system.
                currentNode (str): The current node to trace from.
                currentBranch (list): The list to store the nodes in the current branch.
                remainingNodes (list): The list of all remaining nodes to process.

            Returns:
                str: The name of the downstream junction node.
            """
            if currentNode not in remainingNodes:
                return currentNode
                
            currentBranch.append(currentNode)
            remainingNodes.remove(currentNode)
            nodeClassification[currentNode] = branchCounter

            DSnode = nodeMap[currentNode]['DS']
            if DSnode:
                return traceBranch(nodeMap, DSnode[0], currentBranch, remainingNodes)



        # Identify all branches and their connections to junctions
        for junctionName in junctions:
            downstreamNodes = nodeMap[junctionName]['DS']
            for nodeName in downstreamNodes:
                currentBranch = []
                if isJunction(nodeName):
                    # This condition should not occur because we start tracing from junctions
                    from pdb import set_trace
                    set_trace()
                else:
                    downstreamJunction = traceBranch(nodeMap, nodeName, currentBranch, remainingNodes)
                    branches[branchCounter] = currentBranch
                    junctions[junctionName]['DS'].append(branchCounter)
                    try:
                        junctions[downstreamJunction]['US'].append(branchCounter)
                        branchConnections[branchCounter] = {'US': junctionName, 'DS': downstreamJunction, 'hot': [], 'cool': []}
                        branchCounter += 1
                    except:
                        from pdb import set_trace
                        set_trace()

        if len(remainingNodes) != 0:
            # Some node wasn't classified properly
            from pdb import set_trace
            set_trace()

        return branches, junctions, branchConnections, nodeClassification

    def initializeSteady(self):


        if not self.initialized:
            self.initialize()

        (branch, junction,
         branchConnections,
         nodeClassification) = self._getBranchesAndJunctions(self.nodeMap)

        self.branches = dict(branch)
        self.junctions = dict(junction)

        for B, nodes in branch.items():
            self.branches[B] = Branch.create(nodes=nodes,
                          USJunc=branchConnections[B]['US'],
                          DSJunc=branchConnections[B]['DS'],
                          model=self)
        
        for J, JMap in junction.items():
            usBranches = [self.branches[ibranch] for ibranch in JMap['US']]
            dsBranches = [self.branches[ibranch] for ibranch in JMap['DS']]

            self.junctions[J] = Junction(node=self.nodes[J],
                                         usBranches=usBranches,
                                         dsBranches=dsBranches,
                                         model=self)

        # Initialize Branches and Junctions if not initialized
        for bname, branch in self.branches.items():
            if not branch.initialized:
                from pdb import set_trace
                set_trace()

        for jname, junction in self.junctions.items():
            if not junction.initialized:
                from pdb import set_trace
                set_trace()

        self._netState = {}
        self.__netX = np.array([])

        for bname, branch in self.branches.items():
            if len(branch.x) == 0:
                continue

            xlen = len(branch.x)

            current_len = len(self.__netX)

            self._netState[bname] = np.arange(current_len,
                                              current_len+xlen)

            self.__netX = np.concatenate((self.__netX, branch.x))

        for jname, junc in self.junctions.items():
            if not hasattr(junc, 'x'):
                continue
            
            xlen = len(junc.x)
            
            current_len = len(self.__netX)

            self._netState[jname] = np.arange(current_len,
                                           current_len+xlen)

            self.__netX = np.concatenate((self.__netX, junc.x))       


        # Initialize Network Error Variable
        self.__netError = np.zeros(len(self.__netX))


    @property
    def _netX(self):
        return self.__netX


    @property
    def _netError(self):
        # I need to evaluate nodes first maybe
        # do this a more efficient way
        for name, istate in self._netState.items():
            if name in self.branches:
                self.__netError[istate] = self.branches[name].error
            else:
                self.__netError[istate] = self.junctions[name].error
        
        return self.__netError


    def _updateNet(self, x):
        for net, indx in self._netState.items():
            if net in self.branches:
                self.branches[net].updateState(x[indx])
            else:
                self.junctions[net].updateState(x[indx])


        


    def getFlux(self, node):
        # Calculate mass/energy flux into/out of a node
        # At Steady state mass/energy should be = 0

        # Get nodeMap for this node
        nodeMap = self.nodeMap[node.name]

        wNet = 0
        Hnet = 0
        Wnet = 0
        Qnet = 0

        for name in nodeMap['US']:

            # Get the Flow Node
            flowNode = self.nodes[name]

            # Get the upstream Station Thermo State
            usNode = self.nodes[self.nodeMap[name]['US'][0]].thermo

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

        for name in nodeMap['DS']:

            # Get the Flow Node
            flowNode = self.nodes[name]
            # Get the dowstream Station Thermo State
            dsNode = self.nodes[self.nodeMap[name]['DS'][0]].thermo

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
        for name in nodeMap['hot']:
            try:
                Qnet += self.nodes[name]._Q
            except:
                from pdb import set_trace
                set_trace()

        for name in nodeMap['cool']:
            Qnet -= self.nodes[name]._Q

        return wNet, Hnet, Wnet, Qnet


    @property
    def converged(self):

        # Reinitialize model x using component states
        self.__init_x()

        # Get the error
        if all(abs(self.error/(self.x + eps)<1e-3)):
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
            if isinstance(node, (gt.thermoNode, gt.Station)):
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

        return Wnet, Qin, eta

    # Performance Calc
        # Loop thru all components with W and add positive/negative power
        # Loop thru heat transfer nodes and add all Qin
        # Eta = Qin
        # dP loss in pipe/heat exchanger


    def Carnot_eta(self):
        # Calculate Carnot Efficiency
        pass


class Junction:
    # Junction 

    def __init__(self, node, usBranches, dsBranches, model):
        self.node = node
        self.usBranches = usBranches
        self.dsBranches = dsBranches
        self.model = model
        self.initialized = False

        try:
            self.initialize()
        except:
            from pdb import set_trace
            set_trace()
    
    @staticmethod
    def create(name, usBranches, dsBranches, model):
        return Junction(name=name,
                        usBranches=usBranches,
                        dsBranches=dsBranches,
                        model=model)


    def initialize(self):
        if isinstance(self.node, (gt.Boundary)):
            pass
        elif hasattr(self.node, 'x'):
            # Add properties and method if self.node has attribute 'x'
            self.__add_dynamic_properties()

        self.initialized = True

    def __add_dynamic_properties(self):
        # Add property 'x' to the instance
        def get_x(self):
            return self.node.x

        def get_error(self):
            return self.node.error

        def update_state(self, x):
            self.node.updateState(x)

        # Dynamically add properties and methods
        setattr(self.__class__, 'x', property(get_x))
        setattr(self.__class__, 'error', property(get_error))
        setattr(self.__class__, 'updateState', update_state)


    def evaluate(self):
        self.node.evaluate()


@addQuantityProperty
class Branch:

    _units = {'w': 'MASSFLOW'}
    
    def __init__(self, nodes, USJunc, DSJunc, model, w=None):
        """
        Initialize a Branch instance.

        Args:
            nodes (list): List of node names or instances in sequential order.
            USJunc (str or instance): Upstream junction name or instance.
            DSJunc (str or instance): Downstream junction name or instance.
            model (object): The model containing all nodes and junctions.
            w (float, optional): Branch Mass Flow Rate with a default value of 0.
        """
        self.nodes = nodes
        self.USJunc = USJunc
        self.DSJunc = DSJunc
        self.model = model
        self._w = w
        self.initialized = False
        # Flag to specify if mass flow is constant
        self.fixedFlow = False
        self.backFlow = True
        # Node for controlling mass flow
        self.wController = None
        self.penalty = False

        # Convert node names to instances if necessary
        if isinstance(self.nodes[0], str):
            self.nodes = [model.nodes[name] for name in self.nodes]

        # Convert junction names to instances if necessary
        if isinstance(self.USJunc, str):
            self.USJunc = model.nodes[self.USJunc]
        if isinstance(self.DSJunc, str):
            self.DSJunc = model.nodes[self.DSJunc]

        if isinstance(self.nodes[0], str):
            # Get Node Instance
            self.nodes = [model.nodes[name] for name in self.nodes]

        if isinstance(self.USJunc, str):
            # Get Upstream Junction instance
            self.USJunc = model.nodes[USJunc]

        if isinstance(self.DSJunc, str):
            # Get Downstream Junction instance
            self.DSJunc = model.nodes[DSJunc]
         
        # Try to initialize the Branch
        #try:
        self.initialize()
        #except:
        #    set_trace()

    @property
    def average_w(self):
        # Get the average w in all components

        Wnet = 0
        n_w = 0

        for node in self.nodes:
            if isinstance(node, gt.flowNode):
                Wnet += node._w
                n_w += 1

        return float(Wnet/n_w)

    def evaluate(self):
        # Loop and update nodes in branch

        # Get nodes
        nodes = self.nodes

        # If we have fixedflow then update mass flow
        if self.fixedFlow:
            self._w = self.wController._w

        self.backFlow = False

        # Reverse order if w is negative
        if self._w < 0:
            # Check if backflow is allowed
            if self.backFlow:
                nodes = nodes[::-1]
                Pout = self.USJunc.thermo._P
                DSJunc = self.USJunc.name
            else:
                self._error = (-self._w - np.sign(self._w+eps))*1e5
                return
        else:
            Pout = self.DSJunc.thermo._P
            DSJunc = self.DSJunc.name


        # Loop thru Branch nodes
        for inode, node in enumerate(nodes):
            # Evaluate the node
            node.evaluate()

            if isinstance(node, gt.Station):
                # We are setting the station nodes using flow
                # nodes
                # The branch should be organized as:
                # flowNode => Station => flowNode
                # Check this pattern 

                # Bitwise comparison
                # ChatGPT told me about this!
                if not inode & 0x1:
                    from pdb import set_trace
                    set_trace()

                continue

            if isinstance(node, gt.TBoundary):
                #print('NEED TO REFACTOR')
                continue
            
            # The next nodes should be flowNodes

            # Set the mass flow and get the downstream node 
            # and downstream state
            dsNode, dsState = node._setFlow(self._w)

            if dsState['P'] < 0:
                # Apply penalty
                # Maybe set this as seperate method for different cases
                # Like PR or w, for now do this
                #self._error = (self._x0 - self.x)*1e5
                #self._error = (dsState['P'])*1e5*np.sign(self._w)
                # Point error to opposite side of mass flow
                self._error = (-self._w - np.sign(self._w+eps))*1e5
                return

            # This is the last node, lets calculate error
            if inode == len(nodes) - 1:

                if dsNode != DSJunc:
                    # dsNode is not dsJunc for some reason
                    # error check
                    from pdb import set_trace
                    set_trace()


                if dsState['P'] < 0:
                    # Pressure drop is too high because massflow too high, lets decrease mass flow
                    #self._error = (dsState['P'])*1e5*np.sign(self._w)
                    self._error = (-self._w - np.sign(self._w+eps))*1e5
                    print('TRIGGERED THIS', dsState['P'])
                    return


                if self.fixedFlow:
                    if self.penalty is not False:
                        self._error = self.penalty
                        return
                    if self.fixedObject == 'Turbine':
                        self._error = (dsState['P']/Pout - 1)*Pout
                    else:
                        self._error = (Pout/dsState['P'] - 1)*dsState['P']
                else:
                    self._error = (dsState['P']/Pout - 1)*np.sign(self._w)*Pout
                    if self.model.debug == True:
                        from pdb import set_trace
                        #set_trace()
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
                    error = self.model.nodes[dsNode].updateThermo(dsState)
                else:
                    error = self.model.nodes[dsNode].updateThermo(dsState)

                if error:
                    logger.warn("Failed to update thermostate in Branch "
                                f" evaluate call for '{dsNode}' to state: "
                                f"{dsState}")
                    
                    # Reduce Mass Flow
                    # What if PR
                    self._error = (-self._w - np.sign(self._w+eps))*1e5
                    return

    def _getDSThermo(self, dsNode, dsState):
        # THis method should calculate the downstream thermo based on mix of inputs

        # Pressure for downstream thermo should be set by flowNode object. This 
        # Calculation handles energy transfer
        
        nodeMap = self.model.nodeMap[dsNode]

        if len(nodeMap['cool']) > 0:
            # There shouldn't be any cool nodes
            # debug if there are
            from pdb import set_trace
            set_trace()
            print('Check Line 1037ish')

        for hotNode in nodeMap['hot']:
            node = self.model.nodes[hotNode]
            
            if hotNode in nodeMap['US']:
                continue
            
            if 'H' in dsState:
                try:
                    dsState['H'] += node._Q/abs(self._w)
                except:
                    from pdb import set_trace
                    set_trace()
            else:
                set_trace()

        for coolNode in nodeMap['cool']:
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
        self._x = np.array([])

        for inode, node in enumerate(self.nodes):
            # Something to add, bounds from each class 
            # If mass flow is above/below bounds then apply penalty

            if isinstance(node, (gt.Station)):
                # Check the nodeMap if there are heat nodes
                nMap = self.model.nodeMap[node.name]

                if len(nMap['cool']) > 0:
                    print('Check Line 1073ish')
                    #from pdb import set_trace
                    #set_trace()

            if isinstance(node, (gt.fixedWTurbine, gt.fixedWPump)):
                if self.fixedFlow:
                    # Fixed Flow already activated somewhere
                    from pdb import set_trace
                    set_trace()

                if isinstance(node, gt.fixedWTurbine):
                    self.fixedObject = 'Turbine'
                else:
                    self.fixedObject = 'Pump'

                self.fixedFlow = True
                # Set Branch Mass Flow to this component flow
                self._w = node._w
                self.wController = node

                current_len = len(self.x)
                self.istate[node.name] = np.arange(current_len,
                                                   current_len + len(node.x))

                self._x = np.concatenate((self.x, node.x))

            elif isinstance(node, gt.fixedFlow):
                if self.fixedFlow:
                    from pdb import set_trace
                    set_trace()

                self.fixedFlow = True
                # Set Branch Mass Flow to this component flow
                self._w = node._w
                self.wController = node
            

        if len(self._x) == 0:
            if self.fixedFlow:
                pass
            else:
                # This is if no other state present, then state is massflow 
                self._x = np.array([self._w])

        self.initialized = True

    @property
    def x(self):
        return self._x

    @property
    def error(self):

        return self._error

        # Check if US Junc = DSJunc
        # Check if last node is heat flux or T node
        # If T Node then DSJunc needs to be a Pressure Node
        # Check if Tout != Tout

        # IF StatefulTurbine then error is Turbine PR 

    @staticmethod
    def create(nodes, USJunc, DSJunc, model):
        """
        Generate a Branch instance from a list of nodes and model.

        Args:
            nodes (list): List of node names or instances in sequential order.
            USJunc (str or instance): Upstream junction name or instance.
            DSJunc (str or instance): Downstream junction name or instance.
            model (object): The model containing all nodes and junctions.

        Returns:
            Branch: A new Branch instance.
        """
        return Branch(nodes, USJunc, DSJunc, model)

    def updateState(self, x):
        # Update Branch State

        self._x0 = self.x
        self._x = x

        if len(self.istate) == 0:
            self._w = x[0]
        else:
            for name in self.istate:
                self.model.nodes[name].updateState(x[self.istate[name]])
                self.penalty = self.model.nodes[name].penalty

        #if len(self.istate) == 0:
        #self._state0 = self._w
        
        #self._w = x[0]


        # Update Tout
        # Qin - Qout = 0

    def evaluate_old(self):
        """
        Evaluate the nodes in the branch.

        This method updates the states of downstream nodes based on the outlet states of upstream nodes.
        """
        
        for i, node in enumerate(self.nodes):

            # Check if are on the last node
            if i == len(self.nodes) - 1:
                # Get outlet state
                outletState = node.getOutletState(self.model)

                self.error = self.DSJunc.thermo._P - outletState['P']
                return


            if isinstance(node, (gt.PBoundary,
                                 gt.Station,
                                 gt.Boundary)):
                continue

            node.updateState(self._w)
            

            # Get outlet state
            outletState = node.getOutletState(self.model)

            if outletState['P']<0:
                from pdb import set_trace
                set_trace()

            from pdb import set_trace
            set_trace()
            # Get the Downstream node Station
            # Update the thermo state
            DS = self.nodes[i+1]
            DS.updateThermo(outletState)




    # Collection of Nodes

    # Evaluate

    # Evaluate nodes in state


# Organize
# self._network[names] 

# Error
# Pout - Pnode => w'
#
# Tcond =T


# Heat Transfer Junc
# Specify H Out
# 

# Junc Downstream
# Tout = Junc out T

# HEX T: If DS Junc:
# Set T
# If Branch:
# Update State

# Put in Branch Initialize
# Check Outlet BC
# If Pressure then Change to Boundary
# If Boundary and T!= Tout, then output error
# 

# If W = 0 and error <0:
# Raise ValueError (Heat in or Qout error)



# Junc
# mh_out - mh_in = error 
