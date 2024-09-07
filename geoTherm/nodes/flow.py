from .baseClasses import statefulFlowNode
from ..units import inputParser, addQuantityProperty
from ..logger import logger
from ..utils import dP_pipe
import numpy as np
from .boundary import Boundary, TBoundary


@addQuantityProperty
class flow(statefulFlowNode):
    """ Flow Object """

    _displayVars = ['w', 'dP', 'dH', 'Q']
    _units = {'D': 'LENGTH', 'L': 'LENGTH', 'w': 'MASSFLOW',
              'roughness': 'LENGTH', 'dP': 'PRESSURE',
              'Q': 'POWER', 'dH': 'SPECIFICENERGY',
              'U': 'VELOCITY'}

    _bounds = [-1e5, 1e5]

    @inputParser
    def __init__(self, name, US, DS,
                 w:'MASSFLOW',                  # noqa
                 L:'LENGTH'=None,               # noqa
                 D:'LENGTH'=None,               # noqa
                 roughness:'LENGTH'=1e-5,       # noqa
                 dP:'PRESSURE'=None):            # noqa

        self.name = name
        self.US = US
        self.DS = DS
        self._D = D
        self._L = L
        self._w = w
        self._roughness = roughness

        # Bounds handled via penalty function
        self.penalty = False

        if dP is None:
            self.update_dP = True
            if (L is None) or (D is None):
                logger.critical(f"Connector {self.name} needs to have D and L "
                                "specified if dP is not specified!")

        else:
            self.update_dP = False
            self._dP = -dP

    def _get_dP(self):

        US, _ = self._get_thermo()

        if self.update_dP:
            return dP_pipe(US,
                           self._U,
                           self._D,
                           self._L,
                           self._roughness)
        else:
            return self._dP

    def _get_dH(self):

        US, DS = self._get_thermo()

        # Q
        if self._w == 0:
            return 0
        else:
            # Get Heat
            #self._Q = self._get_Q(US, DS)

            return self._Q/np.abs(self._w)

    #def _get_Q(self, US, DS):
    #    # Update Heat in Pipe

    #    if self._Qobj is not None:
    #        self._Q = -self._Qobj._Q
    #
    #   return self._Q

    def initialize(self, model):

        if self.update_dP:
            self._area = np.pi*self._D**2/4
        else:
            self._area = np.inf

        if isinstance(model.nodes[self.DS], (Boundary, TBoundary)):
            self.fixedT_outlet = True
            logger.info(f"Node: '{self.name}' is connected to a Boundary "
                        "downstream, Q will be calculated to satisfy "
                        "downstream T")
        else:
            self.fixedT_outlet = False

        # Do rest of initialization
        return super().initialize(model)

    @property
    def _U(self):
        # Calculate Flow Velocity of object
        # U = mdot/(rho*A^2)

        US, _ = self._get_thermo()

        return self._w/(US._density*self._area)

    @property
    def _cdA(self):
        dP = np.abs(self.US_node.thermo._P - self.DS_node.thermo._P)
        return self._w/np.sqrt(2*self.US_node.thermo._density*dP)