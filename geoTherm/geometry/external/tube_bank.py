from .base_external import ExternalGeometry
from geoTherm.common import addQuantityProperty, inputParser, units
import numpy as np


@addQuantityProperty
class ExternalTubeBank(ExternalGeometry):
    """
    A bundle of tubes in a rectangular array.

    This class represents a bundle of tubes arranged in a rectangular array pattern.
    It provides methods to calculate geometric properties and visualize the tube bank.

    Attributes
    ----------
    D : float
        Outer diameter of the tubes [m]
    L : float
        Length of the tubes [m]
    S_t : float
        Transverse spacing between tubes [m]
    S_l : float
        Longitudinal spacing between tubes [m]
    N_l : int
        Number of tubes in the longitudinal direction
    N_t : int
        Number of tubes in the transverse direction
    """

    _units = {'D': 'LENGTH', 'L': 'LENGTH', 
              'surface_outer': 'AREA', 'surface_inner': 'AREA',
              'area_t': 'AREA', 'Dh': 'LENGTH', 'S_t': 'LENGTH',
              'S_l': 'LENGTH', 'area_flow': 'AREA'}

    @inputParser
    def __init__(self, D: 'LENGTH', L: 'LENGTH', S_t: 'LENGTH', S_l: 'LENGTH', N_l, N_t):
        """
        Initialize a tube bundle geometry.

        Parameters
        ----------
        D : float
            Outer diameter of the tubes [m]
        L : float
            Length of the tubes [m]
        S_t : float
            Transverse spacing between tubes [m]
        S_l : float
            Longitudinal spacing between tubes [m]
        N_l : int
            Number of tubes in the longitudinal direction
        N_t : int
            Number of tubes in the transverse direction
        """
        self._D = D
        self._L = L
        self._S_t = S_t
        self._S_l = S_l
        self.N_l = N_l
        self.N_t = N_t

    @property
    def N_tubes(self):  
        """Total number of tubes in the bundle."""
        return self.N_l * self.N_t

    @property
    def _area(self):
        """Cross-sectional area of all tubes."""
        return self.N_tubes * np.pi * self._D**2/4

    @property
    def _surface(self):
        """Total external surface area."""
        return self._surface_bare

    @property
    def _surface_bare(self):
        """Bare tube surface area."""
        return self.N_tubes * np.pi * self._D * self._L

    @property
    def _area_flow(self):
        """Flow area across the tube bank."""
        return (self._S_t - self._D) * self.N_t * self._L
    
    @property
    def _Dh(self):
        return self._D


@addQuantityProperty
class ExternalCircularFinnedTubeBank(ExternalTubeBank):
    """
    A finned tube bank geometry.

    This class extends ExternalTubeBank to add circular fins to each tube.
    The fins increase the heat transfer surface area while maintaining
    the basic tube bank structure.

    Attributes
    ----------
    D : float
        Outer diameter of the tubes [m]
    L : float
        Length of the tubes [m]
    S_t : float
        Transverse spacing between tubes [m]
    S_l : float
        Longitudinal spacing between tubes [m]
    N_l : int
        Number of tubes in the longitudinal direction
    N_t : int
        Number of tubes in the transverse direction
    N_fin : float
        Number of fins per unit length [1/m]
    D_fin : float
        Outer diameter of the fins [m]
    th_fin : float
        Thickness of the fins [m]
    """

    _units = ExternalTubeBank._units | {'D_fin': 'LENGTH', 'th_fin': 'LENGTH', 
                                       'N_fin': '1/LENGTH'}

    @inputParser
    def __init__(self, D: 'LENGTH', L: 'LENGTH', S_t: 'LENGTH', S_l: 'LENGTH', 
                 N_l, N_t, N_fin:'1/LENGTH', D_fin: 'LENGTH', th_fin: 'LENGTH'):
        """
        Initialize a finned tube bank geometry.

        Parameters
        ----------
        D : float
            Outer diameter of the tubes [m]
        L : float
            Length of the tubes [m]
        S_t : float
            Transverse spacing between tubes [m]
        S_l : float
            Longitudinal spacing between tubes [m]
        N_l : int
            Number of tubes in the longitudinal direction
        N_t : int
            Number of tubes in the transverse direction
        N_fin : float
            Number of fins per unit length [1/m]
        D_fin : float
            Outer diameter of the fins [m]
        th_fin : float
            Thickness of the fins [m]
        """
        super().__init__(D, L, S_t, S_l, N_l, N_t)
        self._N_fin = N_fin
        self._D_fin = D_fin
        self._th_fin = th_fin

    @property
    def _state(self):
        return {
            'D': (self._D, 'm'),
            'L': (self._L, 'm'),
            'S_t': (self._S_t, 'm'),
            'S_l': (self._S_l, 'm'),
            'N_l': self.N_l,
            'N_t': self.N_t,
            'N_fin': (self._N_fin, '1/m'),
            'D_fin': (self._D_fin, 'm'),
            'th_fin': (self._th_fin, 'm'),
        }

    @property
    def _h_fin(self):
        return (self._D_fin - self._D)/2

    @property
    def _surface_fin_single(self):
        """Surface area of a single fin."""

        return ((2 * np.pi *(self._D_fin**2 - self._D**2)/4)
                + np.pi*self._D_fin*self._th_fin)

    @property
    def _surface_fin_tube(self):
        """Surface area of fins per tube."""
        return self._surface_fin_single * self._N_fin * self._L

    @property
    def _surface_fin_longitudinal(self):
        """Surface area of fins per longitudinal row."""
        return self._surface_fin_tube * self._N_l

    @property
    def _surface_fin_transverse(self):
        """Surface area of fins per transverse row."""
        return self._surface_fin_tube * self._N_t

    @property
    def _surface_fin(self):
        """Surface area of all fins."""
        return self.N_tubes * self._surface_fin_tube

    @property
    def _surface_fin_total(self):
        """Total surface area of all fins."""
        return self._surface_fin_longitudinal + self._surface_fin_transverse

    @property
    def _L_tube_exposed(self):
        return self._L*(1-self._N_fin*self._th_fin)

    @property
    def _dL_tube_exposed(self):
        return 1/self._N_fin - self._th_fin

    @property
    def _surface_exposed(self):
        return self.N_tubes*np.pi*self._D*self._L_tube_exposed

    @property
    def _surface(self):
        return self._surface_fin + self._surface_exposed

    @property
    def area_increase(self):
        return self._surface/self._surface_bare

    @property
    def _area_flow_min(self):
        return self._area_flow_transverse

    @property
    def _area_flow_transverse(self):
        return (self._S_t - self._D - 2*self._th_fin*self._h_fin*self._N_fin)*self._L*self.N_t

    @property
    def _area_face(self):
        return self._S_t * self._L * (self.N_t + 1)

    @property
    def flow_area_contraction_ratio(self):
        return self._area_flow_min/self._area_face
