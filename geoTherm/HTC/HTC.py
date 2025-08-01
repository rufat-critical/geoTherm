from ..logger import logger
from ..thermostate import thermo
import numpy as np
from ..utils import Re_, Boiling_, Bond_, Re_square
from abc import ABC, abstractmethod


class BaseHTC(ABC):

    def __init__(self, geometry, **kwargs):
        self.geometry = geometry

    @abstractmethod
    def evaluate(self, thermo, w):
        pass

    def Re(self, thermo, w):
        return w*self.geometry._Dh/(thermo._viscosity*self.geometry._area)


class Bringer_Smith(BaseHTC):

    def Nu(self, thermo, w):

        Re = self.Re(thermo, w)
        Pr = thermo.prandtl

        return 0.0266*Re**(0.77)*Pr**(0.55)

    def evaluate(self, thermo, w):
        Nu = self.Nu(thermo, w)
        return Nu*thermo._conductivity/self.params['Dh']


def Nu_Khan_Khan(Re, Pr, beta):
    # For plate HEX
    return (0.0161*beta/90+0.1298)*Re**(0.198*beta/90+0.6398)*Pr**(0.35)


def Nu_Kumar(Re, Pr, beta, mu_ratio=1):
    # For Plate Heat Exchanger
    kumar_constants = {
        30: [(10, 0.718, 0.349), (100, 0.348, 0.663)],
        45: [(10, 0.718, 0.349), (100, 0.400, 0.598)],
        50: [(20, 0.630, 0.333), (300, 0.291, 0.591)],
        60: [(20, 0.562, 0.326), (400, 0.306, 0.529)],
        65: [(20, 0.562, 0.326), (500, 0.331, 0.503)],
    }

    # Select closest beta value
    beta = min(kumar_constants.keys(), key=lambda b: abs(b - beta))

    # Determine C1 and m based on Re
    for Re_limit, C1, m in kumar_constants[beta]:
        if Re <= Re_limit:
            break

    # Calculate Nusselt number
    Nu = C1 * (Re ** m) * (Pr ** 0.33) * (mu_ratio ** 0.17)

    return Nu


def Nu_Maslov(Re, Pr):
    # For plate hex
    return 0.78*Re**(0.5)*Pr**(1/3)

def h_Amalfi(w, D_h, q, A_channel, fluid):
    
    beta = 45
    beta_max = 90
    x = fluid.Q
    temp_thermo = thermo.from_state(fluid.state)
    temp_thermo.PQ = fluid.P, 0

    Re_l = Re_(temp_thermo, D_h, w)*(1-x)
    Re_l = Re_square(temp_thermo, D_h, w/A_channel)*(1-x)


    rho_l = temp_thermo._density
    mu_l = temp_thermo._viscosity
    k_l = temp_thermo._conductivity
    h_l = temp_thermo._H
    I = temp_thermo.pObj.cpObj.surface_tension()
    temp_thermo.PQ = fluid.P, 1
    Re_g = Re_(temp_thermo, D_h, w)*x
    Re_g = Re_square(temp_thermo, D_h, w/A_channel)*x

    rho_g = temp_thermo._density
    mu_g = temp_thermo._viscosity
    k_g = temp_thermo._conductivity
    h_g = temp_thermo._H

    H_vap = h_g - h_l

    G = w/A_channel

    Bd = Bond_(fluid, D_h)
    Bo = q/(G*H_vap)


    chevron_angle = beta
    chevron_angle_max = 45.
    beta_s = chevron_angle/chevron_angle_max
    rho_s = (rho_l/rho_g)
    import fluids
    Bd2 = fluids.core.Bond(rhol=rho_l, rhog=rho_g, sigma=I, L=D_h)
    x = fluid.Q
    rho_h = 1./(x/rho_g + (1-x)/rho_l)
    Re_lo = G*D_h/mu_l*(1-x)
    Re_go = G*x*D_h/mu_g


    h = 18.495*(k_l/D_h)*(beta/beta_max)**0.248*Re_g**0.135*Re_l**0.351*(rho_l*(1-x)/(rho_g*x))**(-.223)*Bd**(0.235)*Bo*(0.198)
    import ht

    h_am = ht.boiling_plate.h_boiling_Amalfi

    h2= h_am(w,.01, D_h, rho_l, rho_g,mu_l, mu_g, k_l, H_vap, I, q, A_channel, 45)

    from pdb import set_trace
    set_trace()



def Nu_Gungor_Chen(q, G, D, thermo):


    P_red = thermo._P/thermo._P_crit

    K_bo = (55*P_red**(0.12)*(-np.log10(P_red))**(-0.55)
            *thermo._molecular_weight*q**(0.67))


    x = thermo.Q
    temp_thermo = thermo.from_state(thermo.state)
    
    temp_thermo._PQ = temp_thermo._P, 0
    hf = temp_thermo._H
    mu_l = temp_thermo._viscosity
    rho_l = temp_thermo._density
    temp_thermo._PQ = temp_thermo._P, 1
    hg = temp_thermo._H
    mu_g = temp_thermo._viscosity
    rho_g = temp_thermo._density

    hfg = hg-hf

    Bo = q/(hfg*G)

    Re_l = G*D*(1-x)/mu_l

    Xtt = ((1-x)/x)**(0.9)*(rho_g/rho_l)**0.5*(mu_l/mu_g)**0.1

    E = 1+24000*Bo**(1.16) + 1.37*(1/(Xtt))**0.86

    S = 1/(1+1.15e-6*E**2*Re_l**1.17)




    from pdb import set_trace
    set_trace()



    phi_w = Q/A_w

    P_red = P_sat/P_c

    # Boiling
    Kbo =  55*P_red**(0.12)*(-np.log10(P_red))**(-0.55)*M**(-.5)*phi_w**(0.67)
    
    Kev = E*K1 + S*Kbo


def gungor_chen_evaporation(G, D, P, x, fluid='Water'):
    """
    Calculate the two-phase evaporation heat transfer coefficient using the Gungor-Chen correlation.

    Parameters:
    - G: Mass flux (kg/m²s)
    - D: Tube diameter (m)
    - P: Pressure (Pa)
    - x: Vapor quality (0 to 1)
    - fluid: Working fluid (default is 'Water')

    Returns:
    - K_ev: Two-phase heat transfer coefficient (W/m²K)
    """

    # Fluid properties at saturation
    T_sat = CP.PropsSI('T', 'P', P, 'Q', 0, fluid)  # Saturation temperature (K)
    rho_l = CP.PropsSI('D', 'P', P, 'Q', 0, fluid)  # Liquid density (kg/m^3)
    rho_v = CP.PropsSI('D', 'P', P, 'Q', 1, fluid)  # Vapor density (kg/m^3)
    mu_l = CP.PropsSI('V', 'P', P, 'Q', 0, fluid)   # Liquid viscosity (Pa.s)
    mu_v = CP.PropsSI('V', 'P', P, 'Q', 1, fluid)   # Vapor viscosity (Pa.s)
    k_l = CP.PropsSI('L', 'P', P, 'Q', 0, fluid)    # Liquid thermal conductivity (W/mK)
    cp_l = CP.PropsSI('C', 'P', P, 'Q', 0, fluid)   # Liquid specific heat (J/kgK)
    h_fg = CP.PropsSI('H', 'P', P, 'Q', 1, fluid) - CP.PropsSI('H', 'P', P, 'Q', 0, fluid)  # Latent heat of vaporization (J/kg)
    P_crit = CP.PropsSI('Pcrit', fluid)  # Critical pressure (Pa)

    # Reduced pressure
    Pred = P / P_crit  

    # Reynolds numbers
    Re_l = G * D * (1 - x) / mu_l
    Re_v = G * D * x / mu_v

    # Prandtl numbers
    Pr_l = mu_l * cp_l / k_l
    Pr_v = mu_v * CP.PropsSI('C', 'P', P, 'Q', 1, fluid) / CP.PropsSI('L', 'P', P, 'Q', 1, fluid)

    # Boiling number
    Bo = (G * h_fg)**-1  # Needs q_flux (to be included in an implicit calculation)

    # Martinelli parameter (Xtt)
    Xtt = ((1 - x) / x)**0.9 * (rho_v / rho_l)**0.5 * (mu_l / mu_v)**0.1

    # Boiling heat transfer coefficient (K_bo)
    K_bo = 55 * Pred**0.12 * np.log10(Pred)**0.55 * G**0.67

    # Enhancement factor (E)
    E = 1 + 24000 * Bo**1.16 + 1.37 * (1 / Xtt)**0.86

    # Suppression factor (S)
    S = 1 / (1 + 1.15e6 * E**2 * Re_l**-1.17)

    # Evaporation heat transfer coefficient (K_ev)
    K_ev = E * K_bo + S * K_bo

    return K_ev


class HTC:
    def __init__(self, HTC, flow_node, h=0):
        """Initialize Heat Transfer Coefficient calculator
        
        Args:
            HTC (str): Type of HTC correlation ('Dittus-Boelter' or 'constant')
            geometry: Geometry object containing flow parameters
            h (float): Initial/constant heat transfer coefficient value
        """
        self.HTC = HTC
        self.flow_node = flow_node

    def evaluate(self, thermo): #, w, thermo):
        """Calculate heat transfer coefficient
        
        Args:
            w: Mass flow rate
            thermo: Thermodynamic properties object
        
        Returns:
            float: Heat transfer coefficient value
        """
        if self.HTC == 'Dittus-Boelter':
            from pdb import set_trace
            #set_trace()
            Re = self.Re(thermo)
            Pr = thermo.prandtl
            self._Nu = Nu_Dittus_Boelter(Re, Pr)
            return self._Nu
        elif self.HTC == 'constant':
            return self._h
        else:
            from pdb import set_trace
            set_trace()

    def Re(self, thermo):
        return Re_(thermo, self.flow_node.geometry._Di, self.flow_node._w)


class Convection:

    def __init__(self, htc_model):

        if htc_model == 'Dittus-Boelter':
            self.htc_model = Nu_Dittus_Boelter
        elif htc_model == 'Gungor-Chen':
            self.htc_model = gungor_chen_evaporation

    def evaluate(self, w, thermo):
        pass



class Conduction:

    def __init__(self, geometry, material):
        self.geometry = geometry
        self.material = material

    @property
    def _R(self):
        from pdb import set_trace
        set_trace()
