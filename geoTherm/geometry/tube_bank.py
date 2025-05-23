from .internal.cylinder import InternalCylinder
from .external.tube_bank import ExternalTubeBank, ExternalCircularFinnedTubeBank
from .geometry import Geometry, register_geometry
from geoTherm.common import inputParser, addQuantityProperty, units


@register_geometry
@addQuantityProperty
class TubeBank(Geometry):
    """
    A bundle of tubes in a rectangular array.
    """

    _units = {'L': 'LENGTH', 'volume': 'VOLUME'}

    def __init__(self, Di: 'LENGTH', L: 'LENGTH', 
                 Do: 'LENGTH',
                 th: 'LENGTH',
                 S_t: 'LENGTH', S_l: 'LENGTH',
                 N_l, N_t,
                 N_l_passes=1,
                 roughness=1e-5):
        """
        Initialize a tube bundle geometry.

        Args:
            Di: Inner diameter of the tubes (LENGTH)
            L: Length of the tubes (LENGTH)
            S_t: Transverse spacing between tubes (LENGTH)
            S_l: Longitudinal spacing between tubes (LENGTH)
            N_l: Number of tubes in the longitudinal direction
            N_t: Number of tubes in the transverse direction
            th: Wall thickness of the tubes (LENGTH)
            N_l_passes: Number of passes
        """

        if Do is None:
            Do = Di + 2 * th
        else:
            th = (Do - Di) / 2


        # Multiple passes extend internal cylinder length
        self.inner = InternalCylinder(D=Di, L=L*N_L_passes, roughness=roughness)
        self.outer = ExternalTubeBank(D=Do, L=L, S_t=S_t, S_l=S_l, N_L=N_L, N_T=N_T)

        self._L = L
        self.N_l = N_L
        self.N_t = N_T
        self.N_l_passes = N_L_passes

    @property
    def _volume(self):
        return self.inner.volume*self.N_l*self.N_t/self.N_l_passes

    @property
    def _area_inner(self):
        return self.inner.area*self.N_l*self.N_t/self.N_l_passes


@register_geometry
@addQuantityProperty
class FinnedTubeBank(TubeBank):
    """
    A bundle of tubes in a rectangular array.
    """

    _units = {'D': 'LENGTH', 'L': 'LENGTH', 't': 'LENGTH',
              'Di': 'LENGTH', 'Do': 'LENGTH', 'th': 'LENGTH',
              'surface_outer': 'AREA', 'surface_inner': 'AREA',
              'area_t': 'AREA', 'Dh': 'LENGTH', 'S_t': 'LENGTH',
              'S_l': 'LENGTH'}

    def __init__(self, Di: 'LENGTH', L: 'LENGTH', 
                 S_t: 'LENGTH', S_l: 'LENGTH',
                 N_l, N_t,
                 N_fin,
                 D_fin: 'LENGTH',
                 th_fin: 'LENGTH',
                 Do: 'LENGTH' = None,
                 th: 'LENGTH' = 0,
                 N_l_passes=1,
                 roughness=1e-5):
        """
        Initialize a tube bundle geometry.

        Args:
            Di: Inner diameter of the tubes (LENGTH)
            L: Length of the tubes (LENGTH)
            S_t: Transverse spacing between tubes (LENGTH)
            S_l: Longitudinal spacing between tubes (LENGTH)
            N_l: Number of tubes in the longitudinal direction
            N_t: Number of tubes in the transverse direction
            N_fin: Number of fins
            D_fin: Diameter of the fins (LENGTH)
            th_fin: Thickness of the fins (LENGTH)
            Do: Outer diameter of the tubes (LENGTH)
            th: Wall thickness of the tubes (LENGTH)
            N_passes: Number of passes
        """

        if Do is None:
            Do = Di + 2 * th
        else:
            th = (Do - Di) / 2


        # Multiple passes extend internal cylinder length
        self.inner = InternalCylinder(D=Di, L=L*N_l_passes, roughness=roughness)
        self.outer = ExternalCircularFinnedTubeBank(D=Do, L=L, S_t=S_t, S_l=S_l,
                                                    N_l=N_l, N_t=N_t, N_fin=N_fin,
                                                    D_fin=D_fin, th_fin=th_fin)

        self._L = L
        self.N_l_passes = N_l_passes
        self.N_l = N_l
        self.N_t = N_t
        self.N_l_passes = N_l_passes
