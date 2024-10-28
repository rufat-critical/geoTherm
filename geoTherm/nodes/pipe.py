from .node import Node
from .flow import flow
from ..units import inputParser, addQuantityProperty
from ..utils import dP_pipe
from ..logger import logger
import numpy as np


@addQuantityProperty
class Pipe(flow):
    pass


class LumpedPipe(Node):

    def __init__(self, name):
        pass

class discretePipe(Node):

    def __init__(self, name):
        pass

    # Solve for pressure drop with pressure drop

    # Check for choked flow condition
    # sqrt(gam*R*T)

# ESTIMATE Q LOSS FOR PIPE SECTION
