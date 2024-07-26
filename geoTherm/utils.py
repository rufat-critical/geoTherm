import numpy as np
import re
from scipy.optimize import root_scalar
from .logger import logger

# Get Machine precision
eps = np.finfo(float).eps

## Various Utilities/Helper Functions for geoTherm or standalone
def parseState(stateDict):
    """Check the stateDictionary for quantity code

    Input:
        stateDict: Dictionary containing 2 state variables

    Return:
        State Variable code """

    if stateDict is None:
        return None

    # Creates a set using dict keys from stateDict, we'll use this to
    # compare with quanities defined below
    stateVars = set(stateDict)

    if len(stateVars) != 2:
        msg = 'Error: The thermodynamic state needs to be defined with exactly '
        msg += f'2 state variables, the current stateDict is : {stateDict}'
        raise ValueError(msg)
    else:
        stateVars = set(stateDict)

    # These are all the property inputs that have been programmed so far
    # These are refered to in UpdateState method in thermostate.py
    quantities = {'TP': {'T', 'P'}, 
                  'TS': {'T', 'S'},
                  'HP': {'H', 'P'},
                  'SP': {'S', 'P'},
                  'DU': {'D', 'U'},
                  'PU': {'P', 'U'},
                  'DP': {'D', 'P'},
                  'HS': {'H', 'S'},
                  'DH': {'D', 'H'},
                  'TQ': {'T', 'Q'},
                  'PQ': {'P', 'Q'},
                  'TD': {'T', 'D'}}

    # Check if the set of stateVars matches any of the sets in quantities
    for code, vars in quantities.items():
        if stateVars == vars:
            return code

    # If we reached the end of the loop without returning then the quantity
    # code hasn't been coded yet
    raise ValueError(f'Invalid Thermostate Variables specified: {stateDict}')
    
def parseComposition(composition):
    """Parse the composition into a dictionary containing species 
        and quantities 
        
    Input:
        composition: String, Array, Dictionary
    
    Returns:
        Composition Dictionary"""
    
    if composition is None:
        return None

    if isinstance(composition, str):
        # Use Regular Expression to parse composition string

        # Search for composition strings in the form of:
        # species:quantity, species: quantity
        cRe = re.compile(r'([A-Z0-9]*):([\.0-9]*)', re.IGNORECASE)

        comp = cRe.findall(composition)

        # Check if length of this is 0
        # If it is then maybe single name was specified for fluid
        if len(comp) == 0:
            composition = composition.split()
            if len(composition) == 1:
                comp = [(composition[0], 1.0)]
            else:
                from pdb import set_trace
                # Something is wrong with the composition string
                set_trace()

        # Make Dictionary containing composition
        composition = {species[0]: float(species[1])
                       for species in comp}
        
    elif isinstance(composition, (np.ndarray, list)):
        # If composition is specified as array of values then loop thru species
        # and generate composition
        composition = {name: composition[i] for i, name
                        in enumerate(self.species_names)}
    
    elif isinstance(composition, dict):
        # If input is a dictionary then we guchi
        pass

    # Normalize all the quantities
    tot = sum(composition.values())
    # Normalize by total sum
    composition = {species:q/tot for species, q in composition.items()}

    return composition

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


def pump_eta(phi):
    # Pump efficiency from Claudio
    eta_max = 0.83
    phi_max = 0.08
    k1 = 27
    k2 = 5000
    k3 = 10
    # Why is this 0 and why
    delta_eta_p = 0

    if phi <= 0.08:
        eta = eta_max*(1-k1*(phi_max -phi)**2 - k2*(phi_max-phi)**4) + delta_eta_p
    elif phi > 0.08:
        eta = eta_max*(1-k3*(phi_max-phi)**2) + delta_eta_p

    return eta

#def turb_axial_eta(phi):

    