from .node import Node
from .geometry import Cylinder, GeometryProperties, GeometryGroup
from .baseNodes.baseFlow import baseInertantFlow
from ..units import inputParser, addQuantityProperty
from ..utils import dP_pipe
from ..logger import logger
import numpy as np
from ..resistance_models.flow import PipeLoss#, PipeLossModel, BendLoss
from rich.console import Console
from rich.table import Table
from rich.box import SIMPLE


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
                 t=0,
                 geometry=None,
                 w:'MASSFLOW'=0,
                 dP:'PRESSURE'=None):


        #super().__init__(name, US, DS)
        self.name = name
        self.US = US
        self.DS = DS
        self._w = w
        self.penalty = False

        if geometry is not None:
            self.geometry = geometry
            self._Z = self.geometry._L/self.geometry._area**2
        else:
            # Initialize Geometry
            if L is not None and D is not None:
                self.geometry = Cylinder(D=D, L=L, dz=dz, t=t, roughness=roughness)
                self._Z = self.geometry._L/self.geometry._area**2
            else:
                self.geometry = None
                self._Z = 1

        # Initialize Loss Model
        self.loss = PipeLoss(self, self.geometry, dP=dP)


    @property
    def _dP(self):
        US, _, _ = self.thermostates()
        return self.loss.evaluate(self._w, US)


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

    def get_outlet_state2(self, US, w):

        # Evaluate loss
        dP = self.loss.evaluate(np.abs(w), US)

        return {'H': US._H + self._dH,
                'P': US._P + dP}


    def _get_dP(self, US, w):
        return self.loss.evaluate(np.abs(w), US)

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

    def __init__(self, name, US, DS, geometry,
                 w:'MASSFLOW'=0):
        
        
        self.name = name
        self.US = US
        self.DS = DS
        self._w = w

        if isinstance(geometry, GeometryGroup):
            self.geometry = geometry
        else:
            self.geometry = GeometryGroup(geometry)

        #self.loss = PipeLossModel(self, self.geometry)
        
        self._Z = self.geometry._L / self.geometry._area_avg**2


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

    def _get_dP(self, US, w):
        return self.evaluate_losses(np.abs(w), US)


    def get_outlet_state2(self, US, w):
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

