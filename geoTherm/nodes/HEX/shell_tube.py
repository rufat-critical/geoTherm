from geoTherm.nodes.HEX.baseHEX import BaseHEX
from geoTherm.nodes.baseNodes.baseFlow import baseInertantFlow
from geoTherm.geometry.simple import Cylinder
from geoTherm.geometry.internal import Cylinder as InnerCylinder
from geoTherm.common import logger
from geoTherm.pressure_drop.internal.pipe import StraightLoss


class ShellTube(BaseHEX):
    pass


class Tube(baseInertantFlow):

    def __init__(self, name, US, DS, L=None, D=None, dz=0, roughness=1e-4, geometry=None, w=0, dP=0, Z=None):
        super().__init__(name, US, DS, w, Z)

        if geometry is not None:
            self.geometry = geometry

        self._Z =1


    def initialize(self, model):
        super().initialize(model)

        if isinstance(self.geometry, Cylinder):
            self.geometry = self.geometry.inner
        elif isinstance(self.geometry, InnerCylinder):
            pass
        else:
            logger.critical("Geometry in Tube for Shell and Tube must be a "
                            "Cylinder, the specified geometry is "
                            f"{type(self.geometry)}")

        self.loss = StraightLoss(self.geometry)
        self._ref_thermo = self.US_node.thermo.copy()

    def get_outlet_state(self, US, w):



        if w == 0:
            return {'P': US._P, 'H': US._H}

        dP = self.loss.evaluate(US, w)
        # Get the outlet state


        return {'P': US._P + dP, 'H': US._H}
