from .node import Node
#from .geometry import GeometryProperties, GeometryGroup
from .baseNodes.baseFlow import baseInertantFlow
from ..units import inputParser, addQuantityProperty
from ..utils import UserDefinedFunction
from ..logger import logger
import numpy as np
from ..decorators import state_dict
from geoTherm.geometry.internal import Cylinder
from geoTherm.geometry.geometry import GeometryProperties, GeometryGroup
from geoTherm.geometry.internal.cylinder import InternalCylinder, InternalCylinderBend
from geoTherm.pressure_drop.internal.pipe import StraightLoss as SL2
from geoTherm.pressure_drop.internal.pipe import PipeLoss
from geoTherm.pressure_drop.pressure_drop import PressureDrop


@addQuantityProperty
class BasePipe(baseInertantFlow, GeometryProperties):

    _displayVars = ['w', 'dP', 'dH']
    _units = {**baseInertantFlow._units,
              **GeometryProperties._units,
              **{
                  'U': 'VELOCITY', 'dP': 'PRESSURE'}}
    _bounds = [-1e5, 1e5]

    @inputParser
    def __init__(self, name, US, DS, w, Z, geometry=None):
        super().__init__(name, US, DS, w, Z)
        self.geometry = geometry

    @property
    def _dP(self):
        US, _, _ = self.thermostates()

        return self.loss.evaluate(US, self._w)

    @property
    def _U(self):
        US, _, _ = self.thermostates()

        return self._w/(US._density*self.geometry._area)


    def get_outlet_state(self, US, *, w=None, PR=None):

        if w is not None:
            dP = self.loss.evaluate(US, w)
        elif PR is not None:
            from pdb import set_trace
            set_trace()
            dP = -US._P * (1 - PR)
        else:
            logger.critical(
                "Either 'w' (mass flow rate) or 'PR' (pressure ratio) "
                "must be provided"
            )
        return {'H': US._H,
                'P': US._P + dP}


class FixedDP(BasePipe):
    """
    A pipe component with a fixed pressure drop.

    This class represents a pipe in a fluid system where the pressure drop
    is either constant or calculated based on a predefined function.
    """

    _units = BasePipe._units | {'dP': 'PRESSURE'}


    @inputParser
    def __init__(self, name, US, DS, dP:'PRESSURE', Z=1, w=0):
        """
        Initialize the FixedDP object.

        Args:
            name: Name of the pipe
            US: Upstream connection
            DS: Downstream connection
            dP: Pressure drop (constant or callable)
            Z: Number of parallel pipes (default: 1)
        """
        super().__init__(name, US, DS, Z=Z, w=w)
        self.loss = UserDefinedFunction(dP)

    def get_outlet_state(self, US, *, w=None, PR=None):
        dP = self.loss.evaluate(US, w, self.model)
        return {'H': US._H,
                'P': US._P + dP}

    @state_dict
    def _state_dict(self):
        return {'dP': self.loss._state}

    @property
    def _dP(self):
        """
        Calculate the pressure drop across the pipe.

        Returns:
            float: The pressure drop value
        """
        US, _, _ = self.thermostates()
        return self.loss.evaluate(US, self._w, self.model)

    @_dP.setter
    def _dP(self, value):
        """
        Set a new pressure drop value.

        Args:
            value: New pressure drop value
        """
        self.loss.set_func(value)


class Pipe(BasePipe):
    _displayVars = ['w', 'dP', 'dH', 'geometry']
    _units = {**BasePipe._units,
              **{
                  'dP_head': 'PRESSURE'}}

    _bounds = [-1e5, 1e5]

    @inputParser
    def __init__(self, name, US, DS,
                 L=None,
                 D=None,
                 dz=0,
                 roughness=1e-4,
                 geometry=None,
                 w=0,
                 Z=None):

        super().__init__(name, US, DS, Z=Z, w=w, geometry=geometry)

        if geometry is None:
            if L is None or D is None:
                logger.critical("Either geometry or L and D must be provided")

            self.geometry = InternalCylinder(D=D, L=L, dz=dz, roughness=roughness)

        if Z is None:
            self._Z = self.geometry._L/self.geometry._area**2

        self.loss = PipeLoss(self.geometry)

    @state_dict
    def _state_dict(self):
        """
        Get the state dictionary containing the node's current state
        information.

        This property extends the parent class's state dictionary by adding the
        current state vector 'x' to it. The state vector typically contains
        enthalpy and pressure values for the node.
        """

        if self.geometry is None:
            geometry_state = None
        else:
            geometry_state = self.geometry._state

        # Add the current state vector to the dictionary
        return {'geometry': geometry_state}

    @property
    def _dP_z(self):
        US, _, _ = self.thermostates()
        return self.loss._dP_z(US)


@addQuantityProperty
class LumpedPipe(Pipe):

    _displayVars = ['w;.3f', 'dP;.3f', 'dH;.3f', 'geometry']
    _units = {**Pipe._units, **{
        'dP_split': 'PRESSURE'
    }}

    @inputParser
    def __init__(self, name, US, DS, geometry,
                 w:'MASSFLOW'=0,
                 Z:'INERTANCE'=None,
                 dP:'PRESSURE'=None):

        self.name = name
        self.US = US
        self.DS = DS
        self._w = w
        self.penalty = False

        if isinstance(geometry, GeometryGroup):
            self.geometry = geometry
        else:
            self.geometry = GeometryGroup(geometry)
        
        if Z is None:
            self._Z = self.geometry._L / self.geometry._area_avg**2
        else:
            self._Z = Z

    def initialize(self, model):
        super().initialize(model)

        self.loss = []
        for geometry in self.geometry.geometries:
            self.loss.append(PipeLoss(geometry))

    @property
    def _dP_split(self):
        US, _, _ = self.thermostates()
        return [loss.evaluate(US, self._w) for loss in self.loss]

    @property
    def _dP(self):
        return np.sum(self._dP_split)

    @property
    def dP_sections(self):
        geometry = [geometry for geometry in self.geometry.geometries]
        dP_split = self.dP_split
        dP_total = sum(dP_split)

        # Create dictionary with total dP and individual section values
        result = {'dP': dP_total}

        # Add individual section values
        for geom, dp in zip(geometry, dP_split):
            result[geom._type] = dp

        return result

    def evaluate_losses(self, US, w):
        dP = 0
        for loss in self.loss:
            dP += loss.evaluate(US, w)
        return dP

    def get_outlet_state(self, US, *, w=None, PR=None):
        dP = self.evaluate_losses(US, w)

        return {'H': US._H,
                'P': US._P + dP}



class discretePipe(Node):

    def __init__(self, name):
        pass

    # Solve for pressure drop with pressure drop

    # Check for choked flow condition
    # sqrt(gam*R*T)

# ESTIMATE Q LOSS FOR PIPE SECTION


#class PipeBend(Node):

