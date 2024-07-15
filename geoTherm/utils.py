import numpy as np
import re
from scipy.optimize import root_scalar

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

def dPpipe(thermo, D, L, w, roughness=1e-5):
    """ Estimate pressure drop for pipe flow 
    
    Args:
        thermo (thermostate): geoTherm thermostate object
        D (float): Pipe Diameter in m
        L (float): Pipe length in m
        roughness (float, optional): Pipe roughness in m (Default = 1e-5 m)
    
    Returns:
        Pipe pressure drop in Pa """


    w = np.abs(w)

    # Calculate friction factor
    f = frictionFactor(thermo, D, w, roughness=roughness)
    # Calculate Flow Area
    A = np.pi*D**2/4

    # Calculate Pressure drop
    dP = f*L/D*(.5/thermo._density*(w/A)**2)

    return dP


def ReCalc(thermo, D, w):
    """ Get the Reynolds number for pipe flow 
    
    Args:
        thermo (thermostate): geoTherm thermo Object
        D (float): Pipe Diameter in m
        w (float): Flow rate in kg/s
    
    Returns:
        Reynolds Number """

    return (4*w)/(np.pi*D*thermo._viscosity)


def frictionFactor(thermo, D, w, roughness=1e-5):
    """ Get the friction factor for a pipe. 
    
    Args:
        thermo (thermostate): geoTherm thermo Object
        D (float): Pipe Diameter in m
        w (float): Flow rate in kg/s

    Returns:
        friction factor """
    

    # Calculate Reynolds number
    Re = ReCalc(thermo, D, w)

    # friction factor for laminar/turbulent
    if Re<2100:
        # Laminar flow friction factor
        f = 64/Re
    else:
        # Use colebrook
        
        # Calculate relative roughness
        k = roughness/D
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
