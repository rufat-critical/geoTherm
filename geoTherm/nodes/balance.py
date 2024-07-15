from .node import Node
from .boundary import TBoundary
import numpy as np
import re

class Balance(Node):

    _displayVars = ['control', 'x']

    def __init__(self, name, control, feedback, setpoint, delta=0):

        self.name = name
        self.control = control
        self.feedback = self._parseVariable(feedback)
        self.setpoint = self._parseVariable(setpoint)
        self.delta = delta

        self.penalty = False

    def _parseVariable(self, var):
        # Parse the variable string into a component name and variable

        if isinstance(var, str):

            # Regular expression to match the pattern: Node.Variable
            pattern = r"(\w+)\.(\w+)"
            # Search for the pattern in the input string
            match = re.search(pattern, var)

            if match:
                # If a match is found, create a dictionary from the groups
                return [match.group(1), match.group(2)]
            else:
                from pdb import set_trace
                set_trace()

        else:
            from pdb import set_trace
            set_trace()

    @property
    def error(self):

        if self.penalty is not False:
            return np.array([self.penalty])

        feedback = getattr(self.model.nodes[self.feedback[0]],
                            self.feedback[1])
        setpoint = getattr(self.model.nodes[self.setpoint[0]],
                            self.setpoint[1])

        return (feedback - (setpoint + self.delta))*1e3



class wBalance(Balance):

    _bounds = [-1e3, 1e3]

    def updateState(self, x):
        if self._bounds[0] < x[0] < self._bounds[1]:
            setattr(self.model.nodes[self.control], '_w', x[0])
            self.penalty = False
        else:
            if x < self._bounds[0]:
                self.penalty = (self._bounds[0] - x[0] + 10)*1e5
            else:
                self.penalty = (x[0] - self._bounds[1] - 10)*1e5

    @property
    def x(self):
        return np.array([getattr(self.model.nodes[self.control], '_w')])


class TBalance(Balance):
    """" Temperature Balance"""

    _bounds = [0, 2000]

    def updateState(self, x):
        #updateThermo(new x)

        if self._bounds[0] < x[0] < self._bounds[1]:
            # update thermo
            try:
                thermo = getattr(self.model.nodes[self.control], 'thermo')
                T0 = thermo._T
                thermo._TP = x[0], thermo._P
                self.penalty = False
            except:
                thermo._TP = T0, thermo._P
                self.penalty = (T0 - x[0])*1e5
        else:
            if x < self._bounds[0]:
                self.penalty = (self._bounds[0] - x[0] + 10)*1e5
            else:
                self.penalty = (x[0] - self._bounds[1] - 10)*1e5

    @property
    def x(self):
        return np.array([getattr(self.model.nodes[self.control].thermo, '_T')])





class massBalance(Node):

    def __init__(self, name, mass_node, HEXSource, QcontrolNode, deltaT):

        self.name = name
        self.mass_node = mass_node
        self.HEXSource = HEXSource
        self.QcontrolNode = QcontrolNode
        self.delta = deltaT
        self.penalty = 0

    def initialize(self, model):

        self.model = model

        self.mnode = self.model.nodes[self.mass_node]
        self.Qcool = self.model.nodes[self.HEXSource]
        self.Qhot = self.model.nodes[self.QcontrolNode]

    @property
    def x(self):
        return np.array([self.mnode._w])

    def updateState(self, x):

        if x<0:
            self.penalty = (0 -x[0]+10)*1e5
            return
        elif x>200:
            self.penalty = (200 -x[0]-10)*1e5
            return
        else:
            self.penalty = 0
            
        self.mnode._w = x[0]

    @property
    def error(self):
        if self.penalty != 0:

            return self.penalty
        return -(self.Qhot.thermo._T - self.Qcool.thermo._T - self.delta)*1e3
