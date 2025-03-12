from .baseNodes.baseThermal import baseThermal, baseHeatsistor
from ..units import inputParser, addQuantityProperty


@addQuantityProperty
class Heatsistor(baseHeatsistor):

    _displayVars = ['Q', 'R', 'hot', 'cool']
    _units = {'Q': 'POWER', 'R': 'THERMALRESISTANCE'}

    @inputParser
    def __init__(self, name, hot, cool, R:'THERMALRESISTANCE'):
        self.name = name
        self.hot = hot
        self.cool = cool
        self._R = R

    def evaluate(self):
        T_hot = self.model[self.hot].thermo._T
        T_cold = self.model[self.cool].thermo._T
        self._Q = (T_hot-T_cold)/self._R

    def get_outlet_state(self):

        # Check Temp
        if self._Q > 0:
            T_hot = self.model[self.hot].thermo._T
            D = self.model[self.cool].thermo._density
        else:
            T_hot = self.model[self.cool].thermo._T
            D = self.model[self.hot].thermo._density

        return {'T': T_hot - self._Q*self._R,
                'D': D}

    def get_inlet_state(self):

        if self._Q >0:
            T_cold = self.model[self.cool].thermo._T
            D = self.model[self.hot].thermo._density
        else:
            T_cold = self.model[self.hot].thermo._T
            D = self.model[self.cool].thermo._density

        return {'T': T_cold + self._Q*self._R,
                'D': D}

    def get_cool_state(self, hot_thermo, Q):

        return {'T': hot_thermo._T - Q*self._R,
                'D': hot_thermo._density}



class ConvectiveResistor(Heatsistor):

    @inputParser
    def __init__(self, name, flow, boundary, HTC):

        self.name = name
        self.flow = flow
        self.boundary = boundary

        self.cool = flow
        self.hot = boundary

        self._H = 1e10

    def evaluate(self):

        self._Q = (self.hot_node.thermo._T - self.cool_node.thermo._T)*self._H
        from pdb import set_trace
        #set_trace()

    def get_cool_state(self, hot_thermo, Q):

        return {'T': hot_thermo._T -Q*self._H,
                'D': hot_thermo._density}

        from pdb import set_trace
        set_trace()


    @property
    def Q2(self):
        from pdb import set_trace
        set_trace()
        return 100



@addQuantityProperty
class Qdot(baseThermal):

    _displayVars = ['Q', 'hot', 'cool']
    _units = {'Q': 'POWER'}

    @inputParser
    def __init__(self, name, hot=None, cool=None,
                 Q:'POWER'=0):
        self.name = name

        self.hot = hot
        self.cool = cool

        self._Q = Q


