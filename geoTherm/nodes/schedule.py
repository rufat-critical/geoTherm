from .node import Node
from ..utils import parse_knob_string
import numpy as np

class Schedule(Node):
    """Define schedule where parameter is function of time, 
    chatGTP help me out with description"""

    def __init__(self, name, knob, t_points, y_points):
        self.name = name
        self.knob = knob
        self.t_points = np.array(t_points)
        self.y_points = np.array(y_points)

    def initialize(self, model):
        super().initialize(model)

        name, var = parse_knob_string(self.knob)

        self.feedback_node = model.nodes[name]
        self.feedback_var = var


    def evaluate(self):

        val = np.interp(self.model.t,
                        self.t_points,
                        self.y_points)

        setattr(self.feedback_node,
                self.feedback_var,
                val)