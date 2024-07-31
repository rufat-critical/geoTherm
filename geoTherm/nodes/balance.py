from .node import Node
from .boundary import Boundary, TBoundary
import numpy as np
import re

class Balance(Node):

    _displayVars = ['knob', 'x', 'error']

    def __init__(self, name, knob, feedback, setpoint, gain=0.2,
                 knob_min=-np.inf, knob_max=np.inf):

        self.name = name
        self.knob = knob
        self.feedback = feedback
        self.setpoint = setpoint
        self.gain = gain

        self.knob_min = knob_min
        self.knob_max = knob_max

        self.penalty = False
        self.scale = 1

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

    def initialize(self, model):

        self.model = model
        name, var = self._parseVariable(self.knob)

        self.knob_node = model.nodes[name]
        self.knob_var = var

        # Get Feedback Value
        self.scale = 1
        self.scale = 1/self.knob_val


        name, var = self._parseVariable(self.feedback)

        self.feedback_node = model.nodes[name]
        self.feedback_var = var


    @property
    def feedback_val(self):
        return getattr(self.feedback_node, self.feedback_var)

    @property
    def knob_val(self):
        return getattr(self.knob_node, self.knob_var)*self.scale

    @property
    def error(self):

        if (self.knob_node.penalty is not False
            or self.penalty is not False):
            return np.array([self.penalty])

        return (self.setpoint - self.feedback_val)*self.gain

    @property
    def x(self):
        return np.array([self.knob_val])

    def updateState(self, x):

        self._x = x

        if x[0]<self.knob_min*self.scale:
            self.penalty = (self.knob_min - x[0] + 1)*1e8
            return
        elif x[0]>self.knob_max*self.scale:
            self.penalty = (self.knob_max - x[0] + 1)*1e8
            return
        else:
            self.penalty = False 


        if isinstance(self.knob_node, Boundary):
            if self.knob_var == 'T':
                self.knob_node.thermo._TP = x[0], self.knob_node.thermo._P
            elif self.knob_var == 'P':
                self.knob_node.thermo._TP = self.knob_node.thermo._T, x[0]/self.scale
            else:
                from pdb import set_trace
                set_trace()
        else:
            from pdb import set_trace
            set_trace()


class wBalance(Balance):

    _bounds = [0, 1e3]

    def initialize(self, model):
        self._x = 0
        return super().initialize(model)
    
    def updateState(self, x):

        self._x = np.copy(x)
        
        x = x*np.diff(self._bounds)

        if self._bounds[0] < x < self._bounds[1]:
            setattr(self.model.nodes[self.control], '_w', x[0])
            self.penalty = False
        else:
            self.penalty = (self._bounds[0] - x[0] -10*np.sign(x))*1e8


    @property
    def x(self):
        x = getattr(self.model.nodes[self.control], '_w')

        return x/np.diff(self._bounds)


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
