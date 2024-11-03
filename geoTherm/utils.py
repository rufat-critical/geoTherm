import numpy as np
import re
from scipy.optimize import root_scalar
from .logger import logger

from .thermostate import thermo

# Get Machine precision
eps = np.finfo(float).eps


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

def turb_axial_eta(phi, psi, psi_opt):

    eta_opt = 0.913 + 0.103*psi-0.0854*psi**2 + 0.0154*psi**3

    phi_opt = 0.375 + 0.25*psi_opt

    K = 0.375 - 0.125*psi

    return eta_opt - K*(phi-phi_opt)**2

def turb_radial_eta(ns):
    return 0.87 - 1.07*(ns-0.55)**2-0.5*(ns-0.55)**3

def parse_knob_string(knob_string):
    # Parse the knob string into a component name and variable
    
    if isinstance(knob_string, str):

        # Regular expression to match the pattern: Node.Variable
        pattern = r"(\w+)\.(\w+)"
        # Search for the pattern in the input string
        match = re.search(pattern, knob_string)

        if match:
            # If a match is found, create a dictionary from the groups
            return [match.group(1), match.group(2)]
        else:
            from pdb import set_trace
            set_trace()

    else:
        from pdb import set_trace
        set_trace()


    from pdb import set_trace
    set_trace()


def parse_component_attribute(attribute_string):
    """
    Parses an attribute string to extract the component name and the attribute
    chain.

    Args:
        attribute_string (str): The attribute string in the format 
                                'Component.Attribute' or 
                                'Component.SubComponent.Attribute'.

    Returns:
        list: A list where the first element is the component name and the 
              second element is the attribute chain.
    """
    if isinstance(attribute_string, str):
        # Regular expression to match the pattern: Component.Attribute or 
        # Component.SubComponent.Attribute
        pattern = r"(\w+)\.(.+)"
        
        # Search for the pattern in the input string
        match = re.search(pattern, attribute_string)

        if match:
            # Return a list where the first element is the component name 
            # and the second is the attribute chain
            return [match.group(1), match.group(2)]
        else:
            raise ValueError("Invalid attribute string format.")
    else:
        raise TypeError("Input should be a string.")


class thermo_data:

    def __init__(self, H, P, fluid, model):
        self._H = H
        self._P = P
        self.fluid = fluid
        self.model = model

    def update(self, state):

        self._H = state['H']
        self._P = state['P']


    @property
    def state(self):

        return {
            'fluid': self.fluid,
            'state': {'H': (self._H, 'J/kg'),
                      'P': (self._P, 'Pa')},
            'model': self.model
        }


def _extend_bounds(f, bounds, max_iter=10, factor=2):
    """
    Check and extend bounds to ensure that they bracket a root.

    Args:
        f (callable): Function for which the root is sought.
        bounds (list): Initial bounds as a list [lower, upper].
        max_iter (int, optional): Maximum number of iterations to extend
                                  bounds. Default is 10.
        factor (float, optional): Factor by which to extend the bounds.
                                  Default is 2.

    Returns:
        list: Adjusted bounds that bracket the root.
    """

    lower, upper = bounds
    f_lower = f(lower)
    f_upper = f(upper)

    iteration = 0

    while np.sign(f_lower) == np.sign(f_upper) and iteration < max_iter:
        # Extend bounds by multiplying with the factor
        if f_lower < 0 and f_upper < 0:
            # If both are negative, extend upper bound
            lower = upper
            upper *= factor
        elif f_lower > 0 and f_upper > 0:
            # If both are positive, reduce lower bound
            upper = lower
            lower /= factor
        else:
            # In case the signs are mixed or zero, we have proper bounds
            break

        # Re-evaluate function at the new bounds
        f_lower = f(lower)
        f_upper = f(upper)
        iteration += 1

    # Return adjusted bounds
    return [lower, upper]