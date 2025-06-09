from geoTherm.common import inputParser, addQuantityProperty, logger
from .baseNodes.baseTurbo import baseTurbo
from .flowDevices import fixedFlow
from ..flow_funcs import _dH_isentropic


class Fan(baseTurbo):
    def __init__(self, name, US, DS, eta, w):
        self.name = name
        self.US = US
        self.DS = DS
        self.eta = eta
        self.w = w

    def evaluate(self):
        from pdb import set_trace
        set_trace()

    def get_output(self, US, w):
        from pdb import set_trace
        set_trace()

    @property
    def _dH_is(self):
        from pdb import set_trace
        set_trace()



class fixedFlowFan(Fan, fixedFlow):
    pass