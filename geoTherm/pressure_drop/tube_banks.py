from ..geometry.simple import TubeBank
from ..logger import logger


class ESDU:

    def __init__(self, geometry: TubeBank):

        if not isinstance(geometry, TubeBank):
            logger.error("geometry must be a TubeBank object")

        self.geometry = geometry

        self.check_geometry_validity()

    def check_Re_validity(self, Re=None):
        if Re < 1000 or Re > 80000:
            logger.warn(f"Re: {Re} is out of range for ESDU correlation: 1000 < Re < 80000")

    def check_geometry_validity(self):
        if self.geometry._N_f*.0254 < 11 or self.geometry._N_f*.0254 > 32:
            logger.warn(f"N_f: {self.geometry._N_f*.0254} 1/in is out of range for ESDU correlation: 11 1/in < N_f < 32 1/in")

        if self.geometry._Do/.0254 < 0.5 or self.geometry._Do/.0254 > 1.25:
            logger.warn(f"Do: {self.geometry._Do/.0254} in is out of range for ESDU correlation: 0.5 in < Do < 1.25 in")

        if self.geometry._h_fin/.0254 < 0.03 or self.geometry._h_fin/.0254 > 0.1:
            logger.warn(f"h_fin: {self.geometry._h_fin/.0254} in is out of range for ESDU correlation: 0.03 in < h_fin < 0.1 in")


    def evaluate(self, thermo, w, V_max=None):

        if V_max is None:
            V_max = w/(self.geometry._area_flow_min * thermo._density)

        Re = V_max * self.geometry._Do * thermo._density / thermo._viscosity

        self.check_Re_validity(Re)

        Kf = (4.567*Re**-0.242*(self.geometry._surface_outer/self.geometry._surface_outer_bare)**0.504
              *(self.geometry._S_t/self.geometry._Do)**-0.376
              *(self.geometry._S_l/self.geometry._Do)**-0.546)

        Ka = 1 + self.geometry.flow_area_contraction_ratio**2

        return (Ka + self.geometry.N_L*Kf) * (thermo._density*V_max**2/2)
