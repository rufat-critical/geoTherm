import numpy as np
from scipy.special import iv as I  # Modified Bessel function of the first kind
from scipy.special import kv as K  # Modified Bessel function of the second kind


def Kern_Kraus_eta(h, k_fin, t_fin, D_fin, D_o):
    """
    Calculates annular fin efficiency using Kern and Kraus from
    Heat Exchanger Design Handbook, 2nd Ed. Kuppan, Thulukkanam,

    Parameters
    ----------
    h : float
        Convective heat transfer coefficient (W/m^2·K)
    k_fin : float
        Thermal conductivity of fin material (W/m·K)
    t_fin : float
        Fin thickness (m)
    D_o : float
        Outer diameter of tube (m)
    D_fin : float
        Outer diameter of fin (m)

    Returns
    -------
    eta_f : float
        Fin efficiency
    """
    r_o = D_o / 2
    r_e = D_fin / 2
    m = np.sqrt(2 * h / (k_fin * t_fin))

    mr_o = m * r_o
    mr_e = m * r_e

    bessel_top = I(1, mr_e) * K(1, mr_o) - K(1, mr_e) * I(1, mr_o)
    bessel_bottom = I(0, mr_o) * K(1, mr_e) + I(1, mr_e) * K(0, mr_o)

    return  (2.*r_o)/(m*(r_e**2 - r_o**2))*(bessel_top/bessel_bottom)

