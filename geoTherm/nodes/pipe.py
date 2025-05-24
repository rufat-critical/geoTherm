from .node import Node
#from .geometry import GeometryProperties, GeometryGroup
from .baseNodes.baseFlow import baseInertantFlow
from ..units import inputParser, addQuantityProperty
from ..utils import dP_pipe
from ..logger import logger
import numpy as np
from ..resistance_models.flow import PipeLoss#, PipeLossModel, BendLoss
from ..resistance_models.flow import ConstantPhysics, CustomPhysics, PipePhysics, load_physics
from rich.console import Console
from rich.table import Table
from rich.box import SIMPLE
from ..decorators import state_dict
from geoTherm.geometry.internal import Cylinder
from geoTherm.geometry.geometry import GeometryProperties, GeometryGroup


@addQuantityProperty
class Pipe(baseInertantFlow, GeometryProperties):
    _displayVars = ['w', 'dP', 'dH', 'geometry']
    _units = {**GeometryProperties._units, **{
        'w': 'MASSFLOW', 'U': 'VELOCITY'
    }
    }
    #_units = {'D': 'LENGTH', 'L': 'LENGTH', 'w': 'MASSFLOW',
    #          'roughness': 'LENGTH', 'dP': 'PRESSURE',
    #          'Q': 'POWER', 'dH': 'SPECIFICENERGY',
    #          'U': 'VELOCITY'}

    _bounds = [-1e5, 1e5]

    @inputParser
    def __init__(self, name, US, DS,
                 L=None,
                 D=None,
                 dz=0,
                 roughness=1e-4,
                 geometry=None,
                 physics=None,
                 w:'MASSFLOW'=0,
                 dP:'PRESSURE'=None,
                 Z:'INERTANCE'=None):


        super().__init__(name, US, DS)
        #self.name = name
        #self.US = US
        #self.DS = DS
        self._w = w

        if geometry is not None:
            self.geometry = geometry
            self._Z = self.geometry._L/self.geometry._area**2
        else:
            # Initialize Geometry
            if L is not None and D is not None:
                self.geometry = Cylinder(D=D, L=L, dz=dz, roughness=roughness)
                if Z is None:
                    self._Z = self.geometry._L/self.geometry._area**2
                else:
                    self._Z = Z
            else:
                self.geometry = None
                if Z is None:
                    self._Z = 1
                else:
                    self._Z = Z

        if physics is not None:
            self.physics = load_physics(physics)
        else:
            if dP is not None:
                if isinstance(dP, (float, int)):
                    self.physics = ConstantPhysics(dP=dP, node=self)
            elif self.geometry is not None:
                self.physics = PipePhysics(node=self)
            else:
                from pdb import set_trace
                set_trace()

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

        physics_state = self.physics._state

        # Add the current state vector to the dictionary
        return {'geometry': geometry_state,
                'physics': physics_state}

    @property
    def _dP(self):
        US, DS, _ = self.thermostates()

        return self.physics.evaluate(self._w, US, DS)['dP']

    @_dP.setter
    def _dP(self, value):

        if callable(value):
            self.physics = CustomPhysics(dP_func=value)
        else:
            self.physics = ConstantPhysics(dP=value, node=self)
            logger.info(f"Setting dP for {self.name} to {value}")

        if value is None:
            # Check if geometry is specified
            if self.geometry is None:
                logger.warn("Can't set dP to none because geometry is not "
                            f"specified for {self.name} node")
                return

            self.physics = PipePhysics(node=self)

    @property
    def _U(self):
        # Incompressible velocity
        US, _, _ = self.thermostates()
        return self._w/(US._density*self.geometry._area)

    @property
    def _cdA(self):
        # Incompressible cdA
        US, DS, _ = self.thermostates()
        dP = np.abs(US.thermo._P - DS.thermo._P)
        return self._w/np.sqrt(2*US._density*dP)

    def get_outlet_state(self, US, w):

        # Evaluate loss
        dProp = self.physics.evaluate(w, US, None)

        return {'H': US._H + dProp['dH'],
                'P': US._P + dProp['dP']}


    @property
    def xdot2(self):
        if self.penalty is not False:
            return np.array([self.penalty])

        #if self._w >= 0:
        #    US, DS = self.US_node.thermo, self.DS_node.thermo
        #else:
        #    US, DS = self.DS_node.thermo, self.US_node.thermo
        US, DS, _ = self.thermostates()

        DS_target = self.get_outlet_state(US, self._w)

        return np.array([DS_target['P'] - DS._P])*np.sign(self._w)        


@addQuantityProperty
class LumpedPipe(Pipe):

    _displayVars = ['w;.3f', 'dP;.3f', 'dH;.3f', 'geometry', "dP_sections;.3f"]
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

        #self.loss = PipeLossModel(self, self.geometry)
        
        if Z is None:
            self._Z = self.geometry._L / self.geometry._area_avg**2
        else:
            self._Z = Z

    def initialize(self, model):
        super().initialize(model)

        self.loss = []
        for geometry in self.geometry.geometries:
            if geometry._type == 'Cylinder':
                self.loss.append(PipeLoss(self, geometry, loss_type='straight'))
            elif geometry._type == 'CylinderBend':
                self.loss.append(PipeLoss(self, geometry, loss_type='bend'))
            else:
                from pdb import set_trace
                set_trace()

    @property
    def _dP(self):
        US, _, _ = self.thermostates()
        return self.evaluate_losses(self._w, US)

    @property
    def _dP_split(self):
        US, _, _ = self.thermostates()
        return [loss.evaluate(self._w, US) for loss in self.loss]
    
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

    def evaluate_losses(self, w, US):
        dP = 0
        for loss in self.loss:
            dP += loss.evaluate(w, US)
        return dP


    def get_outlet_state(self, US, w):
        dP = self.evaluate_losses(np.abs(w), US)
        H = US._H + self._dH

        return {'H': H,
                'P': US._P + dP}



class discretePipe(Node):

    def __init__(self, name):
        pass

    # Solve for pressure drop with pressure drop

    # Check for choked flow condition
    # sqrt(gam*R*T)

# ESTIMATE Q LOSS FOR PIPE SECTION


#class PipeBend(Node):

