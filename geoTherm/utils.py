import numpy as np
import re
from scipy.optimize import root_scalar
from .logger import logger
from .thermostate import thermo

# Get Machine precision
eps = np.finfo(float).eps

## Various Utilities/Helper Functions for geoTherm or standalone


def dP_pipe(thermo, U, Dh, L, roughness=2.5e-6):
    """ Estimate pressure drop for pipe flow 

    Args:
        thermo (thermostate): geoTherm thermostate object
        U (float): Flow Speed in m/s
        Dh (float): Hydraulic Diameter in m
        L (float): Pipe length in m
        roughness (float, optional): Pipe roughness in m (Default = 1e-5 m)

    Returns:
        Pipe pressure drop in Pa """

    # Calculate friction factor
    f = friction_factor(thermo, U, Dh, roughness=roughness)

    # Calculate Friction Pressure loss
    return -f*L/Dh*(.5*thermo._density*U**2)


def Re_calc(thermo, U, L):
    """ Calculate the Reynolds number

    Args:
        thermo (thermostate): geoTherm thermo Object
        U (float): Flow Velocity in m/s
        L (float): Characteristic Length in m

    Returns:
        Reynolds Number """

    return thermo._density*U*L/thermo._viscosity


def friction_factor(thermo, U, Dh, roughness=2.5e-6):
    """ Get the friction factor for a pipe. 
    
    Args:
        thermo (thermostate): geoTherm thermo Object
        U (float): Flow Speed in m/s
        Dh (float): Hydraulic Diameter in m

    Returns:
        friction factor """
    

    # Calculate Reynolds number
    Re = Re_calc(thermo, U, Dh)

    # friction factor for laminar/turbulent
    if Re<2100:
        # Laminar flow friction factor
        f = 64/Re
    else:
        # Use colebrook
        
        # Calculate relative roughness
        k = roughness/Dh
        f = colebrook(k, Re)

    return f

def colebrook(k, Re):
    """ Get the friction factor using Colebook Equation 
    
    Args:
        k (float): Relative Roughness
        Re (float): Reynolds Number 
    
    Returns:
        friction factor """
    
    def res(f):
        # Residual function
        return 1/np.sqrt(f) + 2*np.log10(k/3.7 + 2.51/(Re*np.sqrt(f)))
    
    # Use root scalar to find root
    f = root_scalar(res, method='brentq', bracket=[1e-10, 1e4]).root

    return f


def dittus_boelter(Re, Pr, heating=True):
    # Dittus Boelter heat transfer correlation
    # output is Nusselt #

    # Check Applicability
    if 0.6 <= Pr <= 160:
        pass
    else:
        logger.warn("Dittus-Boelter relation is outside valid range"
                    "of 0.6<=Pr<=160, "
                    f"current {Pr}")

    if Re >= 1e4:
        pass
    else:
        logger.warn("Dittus-Boelter relation is outside valid range"
                    f"Re>1e4, current {Re}")

    # Check what Exponent to use for Nusselt #
    if heating:
        n = 0.4
    else:
        n = 0.3

    return 0.023*Re*Pr**n


def dH_isentropic(inlet_thermo, Pout):
    # Calculate isentropic enthalpy change for isentropic change in pressure

    try:
        isentropic_outlet = thermo(fluid=inlet_thermo.Ydict,
                                   state={'S': inlet_thermo._S,
                                          'P': Pout},
                                    model=inlet_thermo.thermo_model)
    except:
        # Check state and try isentropic or incompressible form
        if inlet_thermo.phase == 'liquid':
            return dH_isentropic_incompressible(inlet_thermo, Pout)
        else:
            return dH_isentropic_perfect(inlet_thermo, Pout)
            from pdb import set_trace
            set_trace()
            # Check state and try isentropic or incompressible form
            thermo(inlet_thermo.Ydict, state={'S': inlet_thermo._S, 'P': Pout}, model=inlet_thermo.thermo_model)

    return isentropic_outlet._H - inlet_thermo._H

def dH_isentropic_perfect(inlet_thermo, Pout):
    # Calculate dH assuming incompressible fluid

    gamma = inlet_thermo.gamma

    cp = inlet_thermo._cp
    P0 = inlet_thermo._P

    return cp*P0*(1-(Pout/P0)**((gamma-1)/gamma))

def dH_isentropic_incompressible(inlet_thermo, Pout):
    # Isentropic incompressible dH
    # dH = dP/rho

    return (Pout-inlet_thermo._P)/inlet_thermo._density


def pump_eta(phi):
    # Pump efficiency from Claudio
    eta_max = 0.83
    phi_max = 1.75
    k1 = 27
    k2 = 5000
    k3 = 10
    # Why is this 0 and why
    delta_eta_p = 0

    if phi <= 0.08:
        eta = eta_max*(1-k1*(phi_max*phi -phi)**2 - k2*(phi_max*phi-phi)**4) + delta_eta_p
    elif phi > 0.08:
        eta = eta_max*(1-k3*(phi_max*phi-phi)**2) + delta_eta_p

    return eta

#def turb_axial_eta(phi):

    