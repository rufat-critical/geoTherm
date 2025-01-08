import numpy as np
import re
from scipy.optimize import root_scalar
from .logger import logger
import geoTherm as gt

# Get Machine precision
eps = np.finfo(float).eps


def dP_pipe(thermo, Dh, L, w, roughness=2.5e-6):
    """ Estimate pressure drop for pipe flow using mass flow.

    Args:
        thermo (thermostate): geoTherm thermostat object
        Dh (float): Hydraulic Diameter in m
        L (float): Pipe length in m
        w (float): Mass flow rate in kg/s
        roughness (float, optional): Pipe roughness in m (Default = 2.5e-6 m)

    Returns:
        float: Pipe pressure drop in Pa
    """
    K_pipe = pipe_K(thermo, L, Dh, w, roughness)
    A = np.pi / 4 * Dh ** 2
    return -K_pipe * (w / A) ** 2 / (2 * thermo._density)


def pipe_K(thermo, L, Dh, w, roughness=2.5e-6):
    """ Calculate the loss coefficient for pipe flow.

    Args:
        thermo (thermostate): geoTherm thermostat object
        L (float): Pipe length in m
        Dh (float): Hydraulic Diameter in m
        w (float): Mass flow rate in kg/s
        roughness (float, optional): Pipe roughness in m (Default = 2.5e-6 m)

    Returns:
        float: Loss coefficient
    """
    Re = Re_(thermo, Dh, w)
    f = friction_factor(Re, roughness / Dh)
    return f * L / Dh


def Re_(thermo, Dh, w):
    """ Calculate Reynolds number based on mass flow.

    Args:
        thermo (thermostate): geoTherm thermostat object
        Dh (float): Hydraulic Diameter in m
        w (float): Mass flow rate in kg/s

    Returns:
        float: Reynolds Number
    """
    return 4 * np.abs(w) / (np.pi * Dh * thermo._viscosity)


def friction_factor(Re, rel_roughness):
    """ Calculate the friction factor based on Reynolds number and 
        relative roughness.

    Args:
        Re (float): Reynolds Number
        rel_roughness (float): Relative roughness (roughness/Dh)

    Returns:
        float: Friction factor
    """
    if Re < 2100:
        return 64 / Re
    else:
        return colebrook(rel_roughness, Re)

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


def find_bounds(f, bounds, upper_limit=np.inf, lower_limit=-np.inf,
                max_iter=10, factor=2):
    """
    Expand bounds to ensure they bracket a root, returning
    bounds as [lower, upper] where lower < upper and within specified limits.

    Args:
        f (callable): Function for which the root is sought.
        bounds (list): Initial bounds as a list [lower, upper].
        upper_limit (float, optional): Maximum allowable value for the upper
                                       bound.
        lower_limit (float, optional): Minimum allowable value for the lower
                                       bound.
        max_iter (int, optional): Maximum number of iterations to extend
                                  bounds. Default is 10.
        factor (float, optional): Factor by which to extend the bounds.
        Default is 2.

    Returns:
        list: Adjusted bounds that bracket the root, ordered as [lower, upper].
    """
    lower, upper = min(bounds), max(bounds)

    if factor < 1:
        factor = 1/factor

    # Ensure initial bounds are within the specified limits

    lower = max(lower, lower_limit)
    upper = min(upper, upper_limit)

    f_lower_sign = np.sign(f(lower))
    f_upper_sign = np.sign(f(upper))

    if f_lower_sign != f_upper_sign:
        return [lower, upper]

    iteration = 0

    # Expand bounds until a root is bracketed, limits are reached, or
    # max iterations
    while f_lower_sign == f_upper_sign and iteration < max_iter:
        # Check if both bounds have reached their respective limits
        if ((upper_limit is not None and upper >= upper_limit) and
                (lower_limit is not None and lower <= lower_limit)):

            # Both limits reached, cannot extend further
            logger.info("Upper and lower bound limits reached in bound search")
            break

        if lower_limit < upper < upper_limit:
            upper = min(upper_limit, upper*factor)
            if np.sign(f(upper)) != f_upper_sign:
                from pdb import set_trace
                set_trace()

        if lower_limit < lower < upper_limit:
            lower = max(lower_limit, lower/factor)
            if np.sign(f(lower)) != f_lower_sign:
                return lower, lower*factor

        iteration += 1

    logger.info(f"Unable to find bounds after {iteration} iterations")
    # Return adjusted bounds
    return [lower, upper]


def has_cycle(node_map, node_list):
    # Detect if node_map has recirculation or not

    # Helper function to perform DFS
    def dfs(node, visited, stack):
        visited.add(node)
        stack.add(node)

        # Traverse all downstream nodes
        for neighbor in node_map[node]['DS']:
            if neighbor not in visited:
                # Recur for unvisited nodes
                if dfs(neighbor, visited, stack):
                    return True
            elif neighbor in stack:
                # If neighbor is in stack, we found a cycle
                return True

        # Remove the node from the stack after visiting all its neighbors
        stack.remove(node)
        return False

    visited = set()
    stack = set()

    # Check each node in the graph
    for node in node_map:
        if node not in visited:
            if dfs(node, visited, stack):
                if isinstance(node_list[node], gt.Boundary):
                    continue
                else:
                    return True
    return False
