from .baseNodes.baseThermal import baseThermal
from ..units import inputParser, addQuantityProperty

@addQuantityProperty
class Heatsistor(baseThermal):

    _displayVars = ['Q', 'R', 'hot', 'cool']
    _units = {'Q': 'POWER', 'R': 'THERMALRESISTANCE'}

    @inputParser
    def __init__(self, name, hot, cool, R:'THERMALRESISTANCE'):
        # Specify either Q or H
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