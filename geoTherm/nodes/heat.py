from .node import Node
from .baseClass import statefulHeatNode, flow
from geoTherm.units import inputParser, addQuantityProperty
from geoTherm import logger
import numpy as np

# The heat stuff needs some serious reorg

class fixedQdot(Node):
    """ Specified Qdot that is added to a station"""

    def __init__(self, name, qdot):
        pass

@addQuantityProperty
class HEXConnector(statefulHeatNode):
    """ Connector that connects outlet T to Qin HEX """
    """ Qdot is calculated from outlet T Hex"""

    _displayVars = ['Q']
    _units = {'Q': 'POWER'}

    def __init__(self, name, hot, cool):
        self.name = name
        self.hot = hot
        self.cool = cool
        self._Q = 0

    def initialize(self, model):

        # Check the types of hot and cool
        # If Hot and Cool do stuff with signs
        hot = self.model.nodes[self.hot]
        cool = self.model.nodes[self.cool]

        # Check instances
        hot_hexT = isinstance(hot, HEX_T)
        hot_hexQ = isinstance(hot, HEX_Q)
        cool_hexT = isinstance(cool, HEX_T)
        cool_hexQ = isinstance(cool, HEX_Q)

        # Identify proper nodes
        if (hot_hexT and cool_hexQ) or (hot_hexQ and cool_hexT):
            if hot_hexT:
                self._hexT = hot
                self._hexQ = cool
            else:
                self._hexT = cool
                self._hexQ = hot

            # Q is negative of hexT
            self._Q = -self._hexT._Q

        else:
            msg = "Incorrect HEX types specified to HEXConnector " \
                f"'{self.name}', There needs to be 1 HEX_T and 1 HEX_Q Type"
            logger.error(msg)
            raise ValueError(msg)

        # Do rest of initialization 
        return super().initialize(model)

    def updateState(self, x):
        self._Q = x[0]

        self._hexQ._Q = x[0]

    @property
    def error(self):
        return self._Q + self._hexT._Q


@addQuantityProperty
class HEX(flow):
    pass


class QController(statefulHeatNode):
    
    @inputParser
    def __init__(self, name, node, T_setpoint):
        self.name = name
        self.node = node
        self.cool = node
        self._T_setpoint = T_setpoint

    @ property
    def error(self):

        return self.model.nodes[self.node].thermo._T - self._T_setpoint

@addQuantityProperty
class Qdot(statefulHeatNode):

    _displayVars = ['Q']
    _units = {'Q': 'POWER'}

    @inputParser
    def __init__(self, name, hotNode, coolNode, Q:'POWER'=0):
        self.name = name
        self.node = hotNode
        self.cool = coolNode
        self._Q = Q

    def initialize(self, model):
        from pdb import set_trace
        set_trace()
        return super().initialize(model)



class discretizedHeat:


    @inputParser
    def __init__(self, name, Inlet, A:'AREA', L:'LENGTH', Nsections=2):

        self.name = name
        self.Inlet = Inlet
        self._A = A
        self._L = L


# This connects 2 heat exchangers together
# Tin and Q
# Check for only 1 condition
# Can vary T to get pinch Temperature
# Source T => Get Q, Use for State
# Update state after evaluating Q to get upstream state