from .units import inputParser, output_converter
from .utils import find_bounds, eps
import numpy as np
from .thermostate import thermo
from scipy.optimize import root_scalar
from .logger import logger


def _hunt_static_PR_M(PR, M, total, static):
    # Objective function to find PR corresponding to specific Mach
    if PR == 1:
        # Updating static._SP sometimes causes coolprop to go crazy
        # but U = 0 if PR=1 so avoid having to evaluate thermostates
        U = 0
    else:
        static._SP = total._S, PR*total._P
        U = np.sqrt(2 * np.max([0, (total._H - static._H)]))

    return M - U/static._sound

def _hunt_total_PR_M(PR, M, static, total):
    # Objective function to find PR corresponding to specific Mach
    total._SP = static._S, PR*static._P
    U = np.sqrt(2 * np.max([0, (total._H - static._H)]))
    return M - U/static._sound


def critial_static_pressure_isentropic(total, static=None):

    if static is None:
        # Temperory thermo object used for intermediate calcs
        static = thermo.from_state(total.state)

    if total.phase in ['gas', 'supercritical_gas', 'supercritical']:
        PR_crit = root_scalar(_hunt_static_PR_M,
                            args=(1, total, static),
                            bracket=[0.1, 1.0],
                            method='brentq').root
    else:
        PR_crit = total._Pvap/total._P

    return PR_crit

def critial_total_pressure_isentropic(static, total=None):

    if total is None:
        total = thermo.from_state(total.state)

    if static.phase in ['gas', 'supercritical_gas']:
        PR_crit = root_scalar(_hunt_total_PR_M,
                              args=(1, total, static),
                              bracket=[0.1, 1.0],
                              method='brentq').root
    else:
        PR_crit = 0.5


def sonic_isentropic_state(total, static=None):

    if static is None:
        # Temperory thermo object used for intermediate calcs
        static = thermo.from_state(total.state)

    PR_crit = critial_static_pressure_isentropic(total, static=static)

    static._SP = total._S, PR_crit*total._P

    return static


def _total_to_static(total, w_flux, supersonic=False, static=None):
    """
    Convert total (stagnation) state to static state given mass flux
    in SI units.

    Args:
        total (thermo): Total (stagnation) thermodynamic state.
        w_flux (float): Mass flux (kg/s/m^2).
        supersonic (bool): Indicates if the flow is supersonic.
        static (thermo, optional): Pre-computed static thermodynamic state.

    Returns:
        thermo: Static thermodynamic state.
    """
    if static is None:
        static = thermo.from_state(total.state)

    if w_flux == 0:
        return static

    # Calculate critical pressure ratio for Mach 1
    PR_crit = critial_static_pressure_isentropic(total, static)

    def hunt_PR(PR):
        """
        Function to solve for pressure ratio (PR) that balances
        calculated mass flux with the given mass flux.

        Args:
            PR (float): Pressure ratio to adjust to balance mass flux.

        Returns:
            float: Difference between target and calculated mass flux.
        """
        # Constrain PR within a valid range
        if PR < 0:
            return (-PR + 10) * 1e5
        elif PR > 1:
            return (1 - PR - 10) * 1e5

        # Set static state based on entropy and PR-adjusted pressure
        static._SP = total._S, PR * total._P

        # Offset slightly for numerical stability near choking
        return (w_flux - _w_isen(total, static)) - 1e-8

    if supersonic:
        P_bounds = [1e-2, PR_crit]
    else:
        P_bounds = [PR_crit, 1]

    #P_bounds = _extend_bounds(hunt_PR, P_bounds, max_iter=20, factor=1.5)
    try:
        # Solve for pressure ratio (PR) using subsonic or supersonic bounds
        PR = root_scalar(hunt_PR, method='brentq', bracket=P_bounds).root
    except:
        logger.warn(f"No static state possible with a mass flux: {w_flux} "
                    f"and Total P: {total._P}, Total T: {total._T}")
        return None

    static._SP = total._S, PR * total._P
    return static

def _static_to_total(static, w_flux, total=None, PR_bounds=None):
    """
    Convert static state to total (stagnation) state given area and flow rate
    in SI units.

    Args:
        static (thermo): Static thermodynamic state.
        w_flux (float): Mass flow rate per unit area (kg/s/m^2).
        supersonic (bool): Flag to indicate supersonic flow conditions.
        total (thermo, optional): Pre-computed total thermodynamic state.

    Returns:
        thermo: The total (stagnation) thermodynamic state.
    """
    if total is None:
        total = thermo.from_state(static.state)

    if w_flux == 0:
        return total

    def hunt_PR(PR):
        """
        Function to solve for pressure ratio (PR) that balances
        calculated mass flux with the given mass flux.

        Args:
            PR (float): Pressure ratio to adjust to balance mass flux.

        Returns:
            float: Difference between target and calculated mass flux.
        """
        # Constrain PR within a valid range
        if PR < 0:
            return (-PR + 10) * 1e5
        elif PR > 1:
            return (1 - PR - 10) * 1e5

        # Set static state based on entropy and PR-adjusted pressure
        total._SP = static._S, static._P/PR

        # Offset slightly for numerical stability near choking
        return (w_flux - _w_isen(total, static)) - 1e-8

    
    PR_bounds = find_bounds(hunt_PR, bounds=[.5, 1], upper_limit=1)

    # Solve for pressure ratio (PR) using subsonic or supersonic bounds
    PR = root_scalar(hunt_PR, method='brentq', bracket=PR_bounds).root

    total._SP = static._S, static._P/PR
    return total


def total_to_static_Mach(total, M=1, static=None):
    """
    Calculate isentropic outlet condition based on Mach number.

    Args:
        total (thermo): Upstream thermodynamic state.
        M (float): Desired Mach number.
        static (thermo, optional): Pre-computed static thermodynamic state.

    Returns:
        thermo: Thermodynamic state at the desired Mach number.
    """
    if static is None:
        static = thermo.from_state(total.state)

    # Calculate critical pressure ratio for Mach 1
    PR_crit = perfect_ratio_from_Mach(1, total.gamma, 'PR')

    def hunt_PR(PR):
        # Function to find outlet pressure that results in sonic velocity
        static._SP = total._S, PR*total._P
        U = np.sqrt(2 * np.max([0, (total._H - static._H)]))
        return M-U/static._sound

    if M <= 1:
        P_bounds = [PR_crit, 1]
    else:
        P_bounds = [1e-2, PR_crit]

    from pdb import set_trace
    set_trace()
    # Extend bounds if the sign is the same
    #P_bounds = _extend_bounds(hunt_PR, P_bounds, max_iter=10, factor=1.5)
    PR = root_scalar(hunt_PR, method='brentq', bracket=P_bounds).root

    static._SP = total._S, PR * total._P
    return static


def static_to_total_Mach(static, M, total=None, PR_bounds=None):
    """
    Calculate total state for perfect

    Args:
        static (thermo): Upstream thermodynamic state.
        M (float): Desired Mach number.
        total (thermo, optional): Pre-computed total thermodynamic state.

    Returns:
        thermo: Thermodynamic state at the desired Mach number.
    """
    if total is None:
        total = thermo.from_state(static.state)

    def hunt_PR(PR):
        total._SP = static._S, static._P / PR
        U = np.sqrt(2 * abs(total._H - static._H))
        return M - U / static._sound

    # Solve for pressure ratio (PR) using subsonic or supersonic bounds
    if PR_bounds is None:
        PR_bounds = [1e-2, 1]

    # Extend bounds if the sign is the same
    PR = root_scalar(hunt_PR, method='brentq', bracket=PR_bounds).root

    total._SP = static._S, static._P/PR
    return total


@output_converter('SPECIFICENERGY')
@inputParser
def dH_isentropic(inlet_thermo, Pout: 'PRESSURE'):  # noqa
    """
    Calculate isentropic enthalpy change.

    Args:
        inlet_thermo (thermo): Inlet thermodynamic state.
        Pout (float): Outlet pressure.

    Returns:
        float: Isentropic enthalpy change.
    """
    return _dH_isentropic(inlet_thermo, Pout)


def _dH_isentropic(inlet_thermo, Pout):
    """
    Calculate isentropic enthalpy change for a pressure change in SI units.

    Args:
        inlet_thermo (thermo): Inlet thermodynamic state.
        Pout (float): Outlet pressure (Pa).

    Returns:
        float: Isentropic enthalpy change (in J).
    """
    try:
        # Calculate the isentropic outlet state at the given outlet pressure
        isentropic_outlet = thermo(
            fluid=inlet_thermo.Ydict,
            state={'S': inlet_thermo._S, 'P': Pout},
            model=inlet_thermo.model
        )
    except Exception as e:
        logger.info(f"Couldn't calculate Isentropic dH: {e}")
        # Check state and try isentropic or incompressible form
        if inlet_thermo.phase == 'liquid':
            logger.info("Falling back on incompressible assumption")
            return _dH_incompressible(inlet_thermo, Pout)
        else:
            logger.info("Falling back on compressible assumption")
            return _dH_isentropic_perfect(inlet_thermo, Pout)

    return isentropic_outlet._H - inlet_thermo._H


@output_converter('SPECIFICENERGY')
@inputParser
def dH_isentropic_perfect(inlet_thermo, Pout: 'PRESSURE'):  # noqa
    """
    Calculate enthalpy change assuming a perfect gas model.

    Args:
        inlet_thermo (thermo): Inlet thermodynamic state.
        Pout (float): Outlet pressure.

    Returns:
        float: Enthalpy change for perfect gas assumption.
    """
    return _dH_isentropic_perfect(inlet_thermo, Pout)


def _dH_isentropic_perfect(inlet_thermo, Pout):
    """
    Calculate enthalpy change assuming a perfect gas model in SI Units.

    Args:
        inlet_thermo (thermo): Thermodynamic state at the inlet.
        Pout (float): Outlet pressure (Pa).

    Returns:
        float: Enthalpy change (dH) for perfect gas assumption (in J).
    """
    gamma = inlet_thermo.gamma  # Specific heat ratio
    P0 = inlet_thermo._P  # Inlet pressure

    # Isentropic enthalpy change for perfect gas
    return -inlet_thermo._H * (1 - (Pout / P0) ** ((gamma - 1) / gamma))


@output_converter('SPECIFICENERGY')
@inputParser
def dH_incompressible(inlet_thermo, Pout: 'PRESSURE'):  # noqa
    """
    Calculate isentropic enthalpy change assuming incompressible flow.

    Args:
        inlet_thermo (thermo): Inlet thermodynamic state.
        Pout (float): Outlet pressure.

    Returns:
        float: Enthalpy change for incompressible flow.
    """
    return _dH_incompressible(inlet_thermo, Pout)


def _dH_incompressible(inlet_thermo, Pout):
    """
    Calculate isentropic enthalpy change assuming incompressible flow
    in SI Units.

    Args:
        inlet_thermo (thermo): Inlet thermodynamic state.
        Pout (float): Outlet pressure (in Pa).

    Returns:
        float: Enthalpy change for incompressible flow (in J).
    """
    # dH_inc = dP/rho
    return (Pout - inlet_thermo._P) / inlet_thermo._density


@output_converter('MASSFLUX')
@inputParser
def w_incomp(US_thermo, DS_thermo) -> float: # noqa
    """
    Calculate incompressible orifice flow rate.

    Args:
        US_thermo (thermo): Upstream thermodynamic state.
        DS_thermo (thermo): Downstream thermodynamic state.

    Returns:
        float: Mass Flux in geoTherm outout units.
    """
    return _w_incomp(US_thermo, DS_thermo)


def _w_incomp(US_thermo, DS_thermo) -> float:
    """
    Calculate incompressible orifice flow rate in SI units.

    Args:
        US_thermo (thermo): Upstream thermodynamic state.
        DS_thermo (thermo): Downstream thermodynamic state.

    Returns:
        float: Mass Flux (kg/m^2/s).
    """
    # Determine the direction of flow based on the pressure difference
    if US_thermo._P > DS_thermo._P:
        US, DS, flow_sign = US_thermo, DS_thermo, 1
    else:
        US, DS, flow_sign = DS_thermo, US_thermo, -1

    # Calculate the pressure drop (dP) across the orifice
    Pvc = DS._P  # Vena Contracta Pressure
    dP = US._P - Pvc

    # Orifice flow equation: w/cdA = sqrt(2 * rho * dP)
    return flow_sign * np.sqrt(2 * US._density * dP)


@output_converter('PRESSURE')
@inputParser
def dP_incomp(US_thermo, w_flux: 'MASSFLUX') -> float:  # noqa
    """
    Calculate incompressible orifice flow pressure drop in SI units.

    Args:
        US_thermo (thermo): Upstream thermodynamic state.
        w_flux (float): Mass flux.

    Returns:
        float: Pressure drop.
    """
    return _dP_incomp(US_thermo, w_flux)


def _dP_incomp(US_thermo, w_flux) -> float:
    """
    Calculate incompressible orifice flow pressure drop.

    Args:
        US_thermo (thermo): Upstream thermodynamic state.
        w_flux (float): Mass flux (kg/s/m^2).

    Returns:
        float: Pressure drop (Pa).
    """
    # Pressure drop equation: dP = (w / A)^2 / (2 * rho)
    return -(w_flux) ** 2 / (2 * US_thermo._density)

def _dP_incomp_reverse(DS_thermo, w_flux, US_thermo=None) -> float:
    
    if US_thermo is None:
        US_thermo = thermo.from_state(DS_thermo.state)
    
    # Calculate w_flux using US_thermo density
    dP = (w_flux) ** 2 / (2 * US_thermo._density)
    
    US_thermo._HP = DS_thermo._H, DS_thermo._P + dP

    _w_incomp(US_thermo, DS_thermo)
    
    def hunt_dP(dP):
        US_thermo._HP = DS_thermo._H, DS_thermo._P + dP

        return w_flux -_w_incomp(US_thermo, DS_thermo) 

    bounds = find_bounds(hunt_dP, [dP, dP*5],
                        upper_limit=dP*100,
                        lower_limit=0,
                        max_iter=10,
                        factor=2)

    dP = root_scalar(hunt_dP, method='brentq',
                        bracket=bounds).root
    
    return dP


@output_converter('MASSFLOW')
@inputParser
def w_comp(US_thermo, DS_thermo, cdA:'AREA') -> float:  # noqa
    """
    Calculate compressible orifice flow rate using isentropic approximation.

    Args:
        US_thermo (thermo): Upstream thermodynamic state.
        DS_thermo (thermo): Downstream thermodynamic state.
        cdA (float): Effective flow area.

    Returns:
        float: Flow rate.
    """
    return _w_comp(US_thermo, DS_thermo, cdA)


def _w_comp(US_thermo, DS_thermo):
    """
    Calculate compressible orifice flow rate using isentropic approximation
    in SI units.

    Args:
        US_thermo (thermo): Upstream thermodynamic state.
        DS_thermo (thermo): Downstream thermodynamic state.

    Returns:
        w_flux (float): Mass flux (kg/s/m^2).

    Derivation:
    Starting with the mass flow rate equation:
        mdot = rho * U * A

    For compressible, isentropic flow:
        U^2 = 2 * (h0 - h)  (from energy conservation)
    And using the definition of enthalpy:
        h = cp * T
    For isentropic flow of an ideal gas, the relationship between temperature
    and pressure can be derived from:
        T2 / T1 = (P2 / P1)^((gamma - 1) / gamma)

    Substituting into the equation for U^2 and using the definition of speed
    of sound, a, the critical pressure ratio (PR_crit) is defined as:
        PR_crit = (2 / (gamma + 1))^(gamma / (gamma - 1))

    The mass flow rate then becomes:
        mdot = cdA * sqrt(2 * gamma / (gamma - 1) * rho * P1 *
               (PR^(2 / gamma) - PR^((gamma + 1) / gamma)))

    Where PR is the pressure ratio across the orifice (Pthroat / P1).
    """
    # Determine direction of flow based on pressure difference
    if US_thermo._P >= DS_thermo._P:
        US, DS, flow_sign = US_thermo, DS_thermo, 1
    else:
        US, DS, flow_sign = DS_thermo, US_thermo, -1

    gamma = US.gamma

    if gamma <= 1:
        gamma = 1 + eps
        logger.warn("Compressible flow function is not valid for US "
                    f"thermostate:\nT: {US._T}, P: {US._P}"
                    f"phase: {US.phase}")

    elif US.phase not in ['gas', 'supercritical_gas', 'supercritical']:
        logger.warn("Compressible flow function is not valid for US "
            f"thermostate:\nT: {US._T}, P: {US._P}"
            f"phase: {US.phase}")

    # Calculate critical pressure ratio
    PR_crit = (2. / (gamma + 1.)) ** (gamma / (gamma - 1.))
    PR = max(DS._P / US._P, PR_crit)

    # Compressible orifice flow equation:
    return flow_sign * np.sqrt(
        2 * gamma / (gamma - 1) * US._density * US._P *
        (PR ** (2. / gamma) - PR ** ((gamma + 1.) / gamma))
    )


@output_converter('PRESSURE')
@inputParser
def dP_comp(US_thermo, w_flux: 'MASSFLUX') -> float:    # noqa
    """
    Calculate pressure drop for compressible flow through an orifice.

    Args:
        US_thermo (thermo): Upstream thermodynamic state.
        w_flux (float): Mass flux.

    Returns:
        float: Pressure drop.
    """
    return _dP_comp(US_thermo, w_flux)


def _dP_comp(US_thermo, w_flux) -> float:
    """
    Calculate pressure drop for compressible flow through an orifice
    in SI units.

    Args:
        US_thermo (thermo): Upstream thermodynamic state.
        w_flux (float): Mass flux (kg/s/m^2).

    Returns:
        float: Pressure drop (Pa).
    """

    M = _Mach_total_perfect(US_thermo, w_flux)

    if M is None:
        return None

    return US_thermo._P*(perfect_ratio_from_Mach(M, US_thermo.gamma, 'PR') - 1)


@output_converter('MASSFLUX')
@inputParser
def w_isen(US_thermo, DS_thermo) -> float:     # noqa
    """
    Calculate the isentropic mass flow rate.

    Args:
        US_thermo (thermo): Upstream thermodynamic state.
        DS_thermo (thermo): Downstream thermodynamic state.
        cdA (float): Effective flow area.

    Returns:
        float: Mass flow rate in the geoTherm output units.
    """

    return _w_isen(US_thermo, DS_thermo)


def _w_isen(US_thermo, DS_thermo) -> float:
    """
    Calculate isentropic mass flux between two states.

    Args:
        US_thermo (thermo): Upstream thermodynamic state.
        DS_thermo (thermo): Downstream thermodynamic state.

    Returns:
        float: Mass flux (kg/s/m^2).
    """

    if US_thermo._P >= DS_thermo._P:
        US, DS, flow_sign = US_thermo, DS_thermo, 1
    else:
        US, DS, flow_sign = DS_thermo, US_thermo, -1

    # Get the outlet thermo state
    outlet = thermo.from_state(US.state)
    outlet._SP = US._S, DS._P

    U = np.sqrt(2*np.max([(US._H - outlet._H), 0]))

    # Check if the flow is sonic
    if U > (outlet.sound*1.001):
        return _w_isen_max(US, outlet) * flow_sign


    return outlet._density * U * flow_sign

def _w_isen_max(total, static=None) -> float:
    # Calculate maximum isentropic mass flux

    if static is None:
        # Temperory thermo object used for intermediate calcs
        static = thermo.from_state(total.state)

    PR_crit = critial_static_pressure_isentropic(total, static)
    static._SP = total._S, total._P*PR_crit

    if static.phase in ['gas', 'supercritical_gas']:
        return static._sound*static._density

    return _w_isen(total, static)


@output_converter('PRESSURE')
@inputParser
def dP_isen(US_thermo, w_flux:'MASSFLUX') -> float:     # noqa
    return _dP_isen(US_thermo, w_flux)


def _dP_isen(US_thermo, w_flux) -> float:
    """
    Calculate the pressure drop for isentropic flow through an orifice.

    Args:
        US_thermo (thermo): Upstream thermodynamic state.
        cdA (float): Flow area in square meters (m^2).
        w (float): Mass flow rate in kilograms per second (kg/s).

    Returns:
        float: The pressure drop (dP) in Pascals (Pa).
    """

    static = _total_to_static(US_thermo, w_flux)

    if static:
        return static._P - US_thermo._P
    else:
        # If no solution is found because mass flux is too high
        # then _total_to_static will output None, return None
        return static

def _dP_isen_reverse(DS_thermo, w_flux) -> float:
    """
    Calculate the required dP pressure for isentropic flow through an orifice
    """

    total = _static_to_total(DS_thermo, w_flux)

    if total:
        return total._P - DS_thermo._P
    else:
        return total


def _dP_isenthalpic_reverse(DS_thermo, w_flux,
                            total=None,
                            static=None) -> float:

    if total is None:
        total = thermo.from_state(DS_thermo.state)

    if static is None:
        static = thermo.from_state(DS_thermo.state)

    if w_flux == 0:
        return total

    def hunt_PR(PR):
        total._HP = DS_thermo._H, DS_thermo._P/PR

        static._SP = total._S, DS_thermo._P

        return _w_isen(total, static) - w_flux

    bounds = find_bounds(hunt_PR, [.5, 1],
                         upper_limit=1,
                         max_iter=10,
                         factor=2)

    if DS_thermo._P/bounds[0] > 1e8:
        return 1e15

    PR = root_scalar(hunt_PR, method='brentq',
                     bracket=bounds).root

    total._HP = DS_thermo._H, DS_thermo._P/PR

    return total._P - DS_thermo._P


def Mach_isentropic(US_thermo, DS_thermo) -> float:
    """
    Calculate the Mach number for isentropic expansion.

    Args:
        US_thermo (thermo): Upstream thermodynamic state.
        DS_thermo (thermo): Downstream thermodynamic state.

    Returns:
        float: Mach number.
    """

    outlet = thermo.from_state(US_thermo.state)
    outlet._SP = US_thermo._S, DS_thermo._P
    U = np.sqrt(2 * (US_thermo._H - outlet._H))

    return U / outlet._sound


def Mach_perfect(US_thermo, DS_thermo) -> float:
    """
    Calculate the Mach number assuming a perfect gas model.

    Args:
        US_thermo (thermo): Upstream thermodynamic state.
        DS_thermo (thermo): Downstream thermodynamic state.

    Returns:
        float: Mach number.
    """

    gamma = US_thermo.gamma
    Ps = DS_thermo._P
    Pt = US_thermo._P

    # Perfect gas Mach number
    # Ps/Pt = (1+(gam-1)/2*M^2)^(-gam/(gam-1)
    return np.sqrt(((Ps / Pt) ** (-(gamma - 1) / gamma) - 1)
                   * (2 / (gamma - 1)))


def _Mach_total_perfect(total, w_flux, supersonic=False):

    def hunt_Mach(M):
        gamma = total.gamma
        rho_s = total._density*perfect_ratio_from_Mach(M, gamma, 'DR')
        a_t = np.sqrt(gamma*total._P/total._density)
        a_s = perfect_ratio_from_Mach(M, gamma, 'soundR')*a_t

        return w_flux - rho_s*M*a_s

    if supersonic:
        M_bounds = [1, 30]
    else:
        M_bounds = [0, 1]

    if np.sign(hunt_Mach(M_bounds[0])) == np.sign(hunt_Mach(M_bounds[1])):
        # If no solution is found because mass flux is too high
        # then _total_to_static will output None, return None   
        return None

    M = root_scalar(hunt_Mach, method='brentq',
                    bracket=M_bounds).root

    return M


def _Mach_static_perfect(static, w_flux, supersonic=False):
    def hunt_Mach(M):
        gamma = static.gamma
        rho_s = static._density*perfect_ratio_from_Mach(M, gamma, 'DR')
        a_t = np.sqrt(gamma*static._P/static._density)
        a_s = perfect_ratio_from_Mach(M, gamma, 'soundR')*a_t

        return w_flux - rho_s*M*a_s

    if supersonic:
        M_bounds = [1, 30]
    else:
        M_bounds = [0, 1]

    M = root_scalar(hunt_Mach, method='brentq', bracket=M_bounds).root

    return M


def perfect_ratio_from_Mach(M, gamma, property):
    """
    Compute isentropic property changes for a given Mach number.

    Args:
        M (float): Mach number.
        gamma (float): Specific heat ratio (cp/cv).
        property (str): Type of property to compute:
            'PR' - Pressure ratio (P/P0)
            'TR' - Temperature ratio (T/T0)
            'DR' - Density ratio (rho/rho0)
            'HR' - Enthalpy ratio (H/H0)
            'soundR' - Ratio of speeds of sound (a/a0)
            'AR' - Area ratio (A/A*)

    Returns:
        float: Calculated property ratio based on the input Mach number.
    """
    if property == 'PR':  # Pressure ratio
        return (1 + (gamma - 1) / 2 * M ** 2) ** (-gamma / (gamma - 1))
    elif property == 'TR':  # Temperature ratio
        return 1 / (1 + (gamma - 1) / 2 * M ** 2)
    elif property == 'DR':  # Density ratio
        return (1 + (gamma - 1) / 2 * M ** 2) ** (-1 / (gamma - 1))
    elif property == 'HR':  # Enthalpy ratio
        # HR is equivalent to TR for isentropic processes
        return perfect_ratio_from_Mach(M, gamma, 'TR')
    elif property == 'soundR':  # Ratio of speeds of sound
        return 1/np.sqrt(1 + (gamma - 1) / 2 * M ** 2)
    elif property == 'AR':  # Area ratio
        return (1 / M) * (((2 / (gamma + 1)) * (1 + (gamma - 1) / 2 * M ** 2))
                          ** ((gamma + 1) / (2 * (gamma - 1))))
    else:
        logger.critical(f"Invalid property type '{property}' specified\n"
                        "Supported properties are: PR, TR, DR, HR, soundR, AR")


def perfect_Mach_from_ratio(ratio, gamma, property, supersonic=False):
    """
    Find the Mach number from a given property ratio.

    Args:
        ratio (float): Property ratio (e.g., P/P0, T/T0).
        gamma (float): Specific heat ratio (cp/cv).
        property (str): Type of property ratio:
            'PR' - Pressure ratio (P/P0)
            'TR' - Temperature ratio (T/T0)
            'DR' - Density ratio (rho/rho0)
            'HR' - Enthalpy ratio (H/H0)
            'soundR' - Ratio of speeds of sound (a/a0)
            'AR' - Area ratio (A/A*)
        supersonic (bool): Flag indicating if supersonic solution is desired.

    Returns:
        float: Calculated Mach number.
    """
    if property == 'PR':  # Pressure ratio
        return np.sqrt(2/(gamma-1)*(ratio**(-(gamma-1)/gamma)-1))
    elif property == 'TR':  # Temperature ratio
        return np.sqrt(2 / (gamma - 1) * (1 / ratio - 1))
    elif property == 'DR':  # Density ratio
        return np.sqrt((ratio ** (gamma - 1) - 1) * (2 / (gamma - 1)))
    elif property == 'HR':  # Enthalpy ratio
        # HR is equivalent to TR for isentropic processes
        return perfect_Mach_from_ratio(ratio, gamma, 'TR')
    elif property == 'soundR':  # Ratio of speeds of sound
        return np.sqrt((1 / ratio ** 2 - 1) * 2 / (gamma - 1))
    elif property == 'AR':
        # Define a function to find the root for Mach number using the
        # area ratio
        def hunt_Mach(M):
            return ratio - perfect_ratio_from_Mach(M, gamma, 'AR')

        if supersonic:
            M_bounds = [1, 10]
        else:
            M_bounds = [0.01, 1]

        return root_scalar(hunt_Mach, method='brentq', bracket=M_bounds).root
    else:
        logger.critical(f"Invalid property type '{property}' specified\n"
                        "Supported properties are: PR, TR, DR, HR, soundR, AR")


class flow_func:

    def __init__(self, flow_func):
        self.flow_func = flow_func

    @property
    def flow_func(self):
        return self._flow_func

    @flow_func.setter
    def flow_func(self, flow_func):

        if flow_func not in ['isentropic', 'incomp', 'comp']:
            logger.critical("Invalid flow function. Support flows are: "
                            "'isentropic', 'incomp', 'comp'")

        self._flow_func = flow_func

    def _dP(self, US_thermo, w_flux):

        if self.flow_func == 'isentropic':
            return _dP_isen(US_thermo, w_flux)
        elif self.flow_func == 'incomp':
            return _dP_incomp(US_thermo, w_flux)
        else:
            from pdb import set_trace
            set_trace()

    def _dP_reverse(self, DS_thermo, w_flux, US_thermo=None):

        if self.flow_func == 'isentropic':
            return _dP_isenthalpic_reverse(DS_thermo, w_flux)
        elif self.flow_func == 'incomp':
            return _dP_incomp_reverse(DS_thermo, w_flux, US_thermo=None)
        else:
            from pdb import set_trace
            set_trace()

    def _w_flux(self, US_thermo, DS_thermo):
        if self.flow_func == 'isentropic':
            return _w_isen(US_thermo, DS_thermo)
        elif self.flow_func == 'incomp':
            return _w_incomp(US_thermo, DS_thermo)
        elif self.flow_func == 'comp':
            return _w_comp(US_thermo, DS_thermo)

class OneDflow:

    def __init__(self, flow_type):
        self.flow_type = flow_type

    
    def totalInlet_mass(self, total_state, w_flux, dP, dHt):
        
        inlet = _total_to_static(total_state, w_flux)
        outlet = total_state.from_state(total_state.state)

        Pe = inlet._P - dP
        Hte = total_state._H - dHt

        def find_outlet(S):
            outlet._SP = inlet._S*S, Pe
            
            return Hte - (outlet._H+.5*(w_flux/outlet._density)**2)

        
        S = root_scalar(find_outlet, method='brentq', bracket = [.9, 1.5]).root
        outlet._SP = inlet._S*S, Pe

        from pdb import set_trace
        set_trace()
    




    def totalInlet_mass2(self, total_state, Min, dP, dHt):
        

        #static_inlet = total_to_static_Mach(total_state, Min)

        #PR = perfect_ratio_from_Mach(.6, 1.4, 'PR')

        static_inlet = total_state.from_state(total_state.state)
        static_inlet._SP = total_state._S, PR*total_state._P
       # U = static_inlet.sound_speed*.6

        #w_flux = static_inlet._density*U

        #dP = 45/3*.02*static_inlet.
        
        from pdb import set_trace
        set_trace()


        dP2 = static_inlet._P - dP
        dHT2 = total_state._H - dHt

        outlet = total_state.from_state(total_state.state)
        S1 = outlet._S
        H2 = total_state._H - dHt

        outlet._SP = S1*1.007035, dP2

        outlet_total = _static_to_total(outlet, w_flux)

        print(outlet_total._H -H2)

        from pdb import set_trace
        set_trace()


def cdA(total_state, w, flow_func):

    if flow_func == 'isentropic':
        return _cdA_isen(total_state, w)
    elif flow_func == 'incomp':
        return _cdA_incomp(total_state, w)
    else:
        from pdb import set_trace
        set_trace()


def _cdA_isen(total_state, w):
    return 1

def _cdA_incomp(total_state, w):
    return 1

def _cdA_comp(total_state, static_state, w):
    """Calculate discharge coefficient * area for compressible flow.

    Args:
        total_state: Upstream total (stagnation) state
        static_state: Downstream static state
        w: Mass flow rate [kg/s]

    Returns:
        float: Discharge coefficient * area [m²]
    """
    # Get fluid properties
    R = total_state.gas_constant
    gamma = total_state.gamma
    P_t = total_state._P
    T_t = total_state._T
    P_s = static_state._P

    # Calculate pressure ratio and critical pressure ratio
    PR = P_s/P_t
    PR_crit = (2/(gamma + 1))**(gamma/(gamma-1))

    if PR > PR_crit:
        # Subsonic flow
        G = P_t * np.sqrt(
            (2*gamma/(R*T_t*(gamma-1))) * 
            (PR**(2/gamma) - PR**((gamma+1)/gamma))
        )
    else:
        # Flow is choked
        G = P_t * np.sqrt(
            gamma/(R*T_t) * 
            (2/(gamma+1))**((gamma+1)/(gamma-1))
        )

    # Calculate required cdA
    cdA = w/G

    return cdA


class baseFlow:
    """Base class for all flow types with unit conversion decorators."""

    @classmethod
    @output_converter('AREA')
    @inputParser
    def cdA(cls, total_state, static_state, w: 'MASSFLOW') -> 'AREA':
        """Calculate discharge coefficient * area with unit conversion."""
        return cls._cdA(total_state, static_state, w)

    @classmethod
    def _cdA(cls, total_state, static_state, w):
        """Raw discharge coefficient * area in m²."""
        mass_flux = cls._w_flux(total_state, static_state)
        return w/mass_flux

    @classmethod
    @output_converter('MASSFLUX')
    @inputParser
    def w_flux(cls, total_state, static_state) -> 'MASSFLUX':
        """Calculate mass flux with unit conversion."""
        return cls._w_flux(total_state, static_state)

    @classmethod
    @output_converter('SPECIFICENERGY')
    @inputParser
    def dH(cls, total, Pout):
        """Calculate enthalpy change with unit conversion."""
        return cls._dH(total, Pout)


class IncompressibleFlow(baseFlow):

    @classmethod
    def _w_flux(cls, total, static):
        """
        Calculate incompressible orifice flow rate in SI units.

        Args:
            US_thermo (thermo): Upstream thermodynamic state.
            DS_thermo (thermo): Downstream thermodynamic state.

        Returns:
            float: Mass Flux (kg/m^2/s).
        """
        # Determine the direction of flow based on the pressure difference
        if total._P > static._P:
            US, DS, flow_sign = total, static, 1
        else:
            US, DS, flow_sign = static, total, -1

        # Calculate the pressure drop (dP) across the orifice
        Pvc = DS._P  # Vena Contracta Pressure
        dP = US._P - Pvc

        # Orifice flow equation: w/cdA = sqrt(2 * rho * dP)
        return flow_sign * np.sqrt(2 * US._density * dP)

    @classmethod
    def _dH(cls, total, Pout):
        """
        Calculate isentropic enthalpy change assuming incompressible flow
        in SI Units.

        Args:
            total (thermo): Total thermodynamic state.
            Pout (float): Outlet pressure (in Pa).

        Returns:
            float: Enthalpy change for incompressible flow (in J).
        """
        # dH_inc = dP/rho
        return (Pout - total._P) / total._density

    @classmethod
    def _dP(cls, total, w_flux, static=None):
        """Calculate pressure drop for incompressible flow.

        Args:
            total (thermo): Total state
            w_flux (float): Mass flux [kg/s/m²]
            static (thermo, optional): Static state

        Returns:
            float: Pressure drop [Pa]
        """
        return -w_flux**2/(2*total._density), None


class PerfectGasFlow(baseFlow):

    @staticmethod
    def critical_PR(gamma):
        """Get critical pressure ratio."""
        return (2/(gamma + 1))**(gamma/(gamma-1))

    @classmethod
    def is_choked(cls, total_state, PR):
        """Check if flow is choked."""
        return PR <= cls.critical_PR(total_state.gamma)

    @classmethod
    def _w_flux(cls, US_thermo, DS_thermo):
        """
        Calculate compressible orifice flow rate using isentropic approximation
        in SI units.

        Args:
            US_thermo (thermo): Upstream thermodynamic state.
            DS_thermo (thermo): Downstream thermodynamic state.

        Returns:
            w_flux (float): Mass flux (kg/s/m^2).

        Derivation:
        Starting with the mass flow rate equation:
            mdot = rho * U * A

        For compressible, isentropic flow:
            U^2 = 2 * (h0 - h)  (from energy conservation)
        And using the definition of enthalpy:
            h = cp * T
        For isentropic flow of an ideal gas, the relationship between temperature
        and pressure can be derived from:
            T2 / T1 = (P2 / P1)^((gamma - 1) / gamma)

        Substituting into the equation for U^2 and using the definition of speed
        of sound, a, the critical pressure ratio (PR_crit) is defined as:
            PR_crit = (2 / (gamma + 1))^(gamma / (gamma - 1))

        The mass flow rate then becomes:
            mdot = cdA * sqrt(2 * gamma / (gamma - 1) * rho * P1 *
                (PR^(2 / gamma) - PR^((gamma + 1) / gamma)))

        Where PR is the pressure ratio across the orifice (Pthroat / P1).
        """
        # Determine direction of flow based on pressure difference
        if US_thermo._P >= DS_thermo._P:
            US, DS, flow_sign = US_thermo, DS_thermo, 1
        else:
            US, DS, flow_sign = DS_thermo, US_thermo, -1

        gamma = US.gamma

        if gamma <= 1:
            gamma = 1 + eps
            logger.warn("Compressible flow function is not valid for US "
                        f"thermostate:\nT: {US._T}, P: {US._P}"
                        f"phase: {US.phase}")

        elif US.phase not in ['gas', 'supercritical_gas', 'supercritical']:
            logger.warn("Compressible flow function is not valid for US "
                f"thermostate:\nT: {US._T}, P: {US._P}"
                f"phase: {US.phase}")

        # Calculate critical pressure ratio
        PR_crit = cls.critical_PR(gamma)
        PR = max(DS._P / US._P, PR_crit)

        # Compressible orifice flow equation:
        return flow_sign * np.sqrt(
            2 * gamma / (gamma - 1) * US._density * US._P *
            (PR ** (2. / gamma) - PR ** ((gamma + 1.) / gamma))
        )

    @classmethod
    def _dH(cls, total_state, Pout):
        """Calculate enthalpy change for compressible flow.

        Args:
            total_state: Upstream total (stagnation) state
            Pout: Outlet pressure [Pa]
        """

        gamma = total_state.gamma  # Specific heat ratio
        P0 = total_state._P  # Inlet pressure

        # Isentropic enthalpy change for perfect gas
        return -total_state._H * (1 - (Pout / P0) ** ((gamma - 1) / gamma))

    @classmethod
    def _dP(cls, total, w_flux, static=None):
        """Calculate pressure drop for compressible flow.

        Args:
            total (thermo): Total state
            w_flux (float): Mass flux [kg/s/m²]

        Returns:
            float: Pressure drop [Pa]
        """

        M = cls.Mach_from_total_w(total, w_flux)

        if M is None:
            error = {'w_flux_max': cls._w_flux_max(total)}
            return None, error

        return total._P*(cls.P_ratio(M, total.gamma) - 1), None

    @classmethod
    def _w_flux_max(cls, total):
        """Calculate maximum mass flux for compressible flow.

        Args:
            total (thermo): Total state

        Returns:
            float: Maximum mass flux [kg/s/m²]
        """ 

        # Get Static Density
        D = cls.D_ratio(1, total.gamma)*total._density
        # Get Sound Ratio
        a = cls.sound_ratio(1, total.gamma)*total._sound

        return D*a

    @classmethod
    def P_ratio(cls, M, gamma):
        return (1 + (gamma - 1) / 2 * M ** 2) ** (-gamma / (gamma - 1))

    @classmethod
    def D_ratio(cls, M, gamma):
        return (1 + (gamma - 1) / 2 * M ** 2) ** (-1 / (gamma - 1))

    @classmethod
    def sound_ratio(cls, M, gamma):
        return 1/np.sqrt(1 + (gamma - 1) / 2 * M ** 2)

    @classmethod
    def Mach_from_total_w(cls, total, w_flux, supersonic=False):

        def hunt_Mach(M):
            gamma = total.gamma
            rho_s = total._density*cls.D_ratio(M, gamma)
            a_t = np.sqrt(gamma*total._P/total._density)
            a_s = cls.sound_ratio(M, gamma)*a_t

            return w_flux - rho_s*M*a_s

        if supersonic:
            M_bounds = [1, 30]
        else:
            M_bounds = [0, 1]

        if np.sign(hunt_Mach(M_bounds[0])) == np.sign(hunt_Mach(M_bounds[1])):
            # If no solution is found because mass flux is too high
            # then _total_to_static will output None, return None   
            return None

        M = root_scalar(hunt_Mach, method='brentq',
                        bracket=M_bounds).root

        return M

class IsentropicFlow(baseFlow):
    """Class for isentropic compressible flow calculations."""


    @classmethod
    def critical_PR_from_total(cls, total, static=None):
        """Calculate critical pressure ratio from total conditions.
        
        Args:
            total (thermo): Total (stagnation) state
            static (thermo, optional): Static state for calculations
            
        Returns:
            float: Critical pressure ratio
        """
        if static is None:
            static = thermo.from_state(total.state)

        if total.phase in ['gas', 'supercritical_gas', 'supercritical']:
            # Find PR where Mach number = 1
            PR_crit = root_scalar(
                cls.__hunt_static_PR_M,
                args=(1, total, static),
                bracket=[0.1, 1.0],
                method='brentq'
            ).root
        else:
            logger.warn("Critical pressure ratio is not valid for US "
                        f"thermostate:\nT: {total._T}, P: {total._P}"
                        f"phase: {total.phase}, using vapor pressure ratio")
            # For non-gas phases, use vapor pressure ratio
            PR_crit = total._Pvap/total._P

        return PR_crit

    @classmethod
    def critical_PR_from_static(cls, static, total=None):
        """Calculate critical pressure ratio from static conditions.

        Args:
            static (thermo): Static state
            total (thermo, optional): Total state for calculations

        Returns:
            float: Critical pressure ratio
        """
        if total is None:
            total = thermo.from_state(static.state)

        if static.phase in ['gas', 'supercritical_gas']:
            PR_crit = root_scalar(_hunt_total_PR_M,
                                args=(1, total, static),
                                bracket=[0.1, 1.0],
                                method='brentq').root
        else:
            logger.warn("Critical pressure ratio is not valid for US "
                        f"thermostate:\nT: {static._T}, P: {static._P}"
                        f"phase: {static.phase}, using vapor pressure ratio")
            PR_crit = static._Pvap/static._P

        return PR_crit

    @classmethod
    def sonic_isentropic_state(cls, total, static=None):
        """Calculate sonic isentropic state.

        Args:
            total (thermo): Total state
            static (thermo, optional): Static state for calculations    

        Returns:
            thermo: Sonic isentropic state
        """
        if static is None:
            static = thermo.from_state(total.state)

        PR_crit = cls.critical_PR_from_total(total, static)

        static._SP = total._S, total._P * PR_crit

        return static

    @classmethod
    def _w_flux(cls, total, static):
        """Calculate isentropic mass flux between two states.
        
        Args:
            total (thermo): Total (stagnation) state
            static (thermo): Static state
            
        Returns:
            float: Mass flux [kg/s/m²]
        """
        # Determine flow direction
        if total._P >= static._P:
            upstream, downstream, flow_sign = total, static, 1
        else:
            upstream, downstream, flow_sign = static, total, -1

        # Calculate isentropic outlet state
        outlet = thermo.from_state(upstream.state)
        outlet._SP = upstream._S, downstream._P

        # Calculate velocity from enthalpy change
        dH = upstream._H - outlet._H
        velocity = np.sqrt(2 * np.max([dH, 0]))

        # Check for sonic conditions
        if velocity > (outlet.sound * 1.001):
            return cls._w_flux_max(upstream, outlet) * flow_sign

        return outlet._density * velocity * flow_sign

    @classmethod
    def _w_flux_max(cls, total, static=None):
        """Calculate maximum (choked) mass flux.
        
        Args:
            total (thermo): Total state
            static (thermo, optional): Static state for calculations
            
        Returns:
            float: Maximum mass flux [kg/s/m²]
        """
        if static is None:
            static = thermo.from_state(total.state)

        # Get critical conditions
        PR_crit = cls.critical_PR_from_total(total, static)
        static._SP = total._S, total._P * PR_crit

        # For gases, use sonic flow equation
        if static.phase in ['gas', 'supercritical_gas']:
            return static._sound * static._density

        # For other phases, calculate using isentropic relations
        return cls._w_flux(total, static)

    @classmethod
    def _total_to_static(cls, total, w_flux, supersonic=False, static=None):
        """Convert total state to static state for given mass flux.

        Args:
            total (thermo): Total state
            w_flux (float): Mass flux [kg/s/m²]
            supersonic (bool): If True, find supersonic solution
            static (thermo, optional): Static state for calculations

        Returns:
            thermo: Calculated static state, or None if no solution
        """
        if static is None:
            static = thermo.from_state(total.state)

        if w_flux == 0:
            return static

        # Get critical pressure ratio
        PR_crit = cls.critical_PR_from_total(total, static)

        # Set pressure ratio bounds based on flow regime
        P_bounds = [1e-2, PR_crit] if supersonic else [PR_crit, 1]

        try:
            # Find pressure ratio that gives target mass flux
            PR = root_scalar(
                lambda PR: cls.__hunt_PR(PR, total, static, w_flux),
                method='brentq',
                bracket=P_bounds
            ).root

            static._SP = total._S, PR * total._P

            return static, None

        except:
            logger.warn(
                "No static state solution found:\n"
                f"  Mass flux: {w_flux:.2f} kg/s/m²\n"
                f"  Total P: {total._P/1e5:.2f} bar\n"
                f"  Total T: {total._T:.2f} K"
            )

            # Return with error
            error = {'w_flux_max': cls._w_flux_max(total, static)}

            return None, error

    @classmethod
    def _dP(cls, total, w_flux, static=None):
        """Calculate isentropic pressure drop for given mass flux.

        Args:
            total (thermo): Total state
            w_flux (float): Mass flux [kg/s/m²]
            static (thermo, optional): Static state for calculations

        Returns:
            float: Pressure drop [Pa], or None if no solution
        """
        static, error = cls._total_to_static(total, w_flux, static=static)

        if error is None:
            return static._P - total._P, error
        else:
            return None, error

    @classmethod
    def _dH(cls, inlet_thermo, Pout):
        """
        Calculate isentropic enthalpy change for a pressure change in SI units.

        Args:
            inlet_thermo (thermo): Inlet thermodynamic state.
            Pout (float): Outlet pressure (Pa).

        Returns:
            float: Isentropic enthalpy change (in J).
        """
        try:
            # Calculate the isentropic outlet state at the given outlet
            # pressure
            isentropic_outlet = thermo(
                fluid=inlet_thermo.Ydict,
                state={'S': inlet_thermo._S, 'P': Pout},
                model=inlet_thermo.model
            )
        except Exception as e:
            logger.warn(f"Error calculating isentropic outlet state: {e}")
            if inlet_thermo.phase == 'liquid':
                logger.info("Falling back on incompressible assumption")
                return IncompressibleFlow._dH(inlet_thermo, Pout)
            else:
                logger.info("Falling back on compressible assumption")
                return PerfectGasFlow._dH(inlet_thermo, Pout)

        return isentropic_outlet._H - inlet_thermo._H

    @classmethod
    def __hunt_PR(cls, PR, total, static, w_flux):
        """Find pressure ratio for target mass flux.

        Args:
            PR (float): Trial pressure ratio
            total (thermo): Total state
            static (thermo): Static state to update
            w_flux (float): Target mass flux [kg/s/m²]

        Returns:
            float: Mass flux error
        """
        if PR < 0:
            return (-PR + 10) * 1e5
        elif PR > 1:
            return (1 - PR - 10) * 1e5

        static._SP = total._S, PR * total._P
        return (w_flux - cls._w_flux(total, static)) - 1e-8

    @classmethod
    def __hunt_static_PR_M(cls, PR, M, total, static):
        """Find pressure ratio for target Mach number.

        Args:
            PR (float): Trial pressure ratio
            M (float): Target Mach number
            total (thermo): Total state
            static (thermo): Static state to update

        Returns:
            float: Mach number error
        """
        if PR == 1:
            # Special case: PR=1 means zero velocity
            return M - 0

        static._SP = total._S, PR * total._P
        dH = total._H - static._H
        velocity = np.sqrt(2 * np.max([0, dH]))

        return M - velocity/static.sound


class FlowModel:
    """Model for calculating flow properties using different flow functions.

    This class provides a high-level interface for flow calculations by
    combining flow functions (isentropic, incompressible, compressible) with
    a discharge coefficient * area (cdA) value. It handles mass flow and
    pressure drop calculations for a specific flow geometry.

    Attributes:
        flow_func: Flow calculation class (IsentropicFlow, IncompressibleFlow,
                  or PerfectGasFlow)
        _cdA (float): Discharge coefficient * area [m²]

    Args:
        flow_func (str): Type of flow calculation ('isen',
                        'incomp', or 'comp')
        cdA (float): Discharge coefficient * area [m²]

    Example:
        >>> flow = FlowModel('isentropic', cdA=1e-4)
        >>> mass_flow = flow._w(total_state, static_state)
        >>> pressure_drop = flow._dP(total_state, mass_flow)
    """

    def __init__(self, flow_func: str, cdA: float):
        """Initialize flow model with specified flow function and cdA.

        Args:
            flow_func (str): Type of flow calculation ('isen',
                           'incomp', or 'comp')
            cdA (float): Discharge coefficient * area [m²]

        Raises:
            ValueError: If flow_func is not one of the supported types
        """
        self._cdA = cdA

        # Map string to flow function class
        flow_map = {
            'isen': IsentropicFlow,
            'incomp': IncompressibleFlow,
            'comp': PerfectGasFlow
        }

        if flow_func not in flow_map:
            raise ValueError(
                f"Invalid flow_func: {flow_func}. "
                f"Must be one of: {list(flow_map.keys())}"
            )

        self.flow_func = flow_map[flow_func]

    def PR_crit(self, total, static=None):
        return self.flow_func.critical_PR_from_total(total, static)

    def _w(self, total, static):
        """Calculate mass flow rate.

        Args:
            total (thermo): Total (stagnation) state
            static (thermo): Static state

        Returns:
            float: Mass flow rate [kg/s]

        Note:
            Mass flow = mass_flux * cdA
        """
        return self.flow_func._w_flux(total, static) * self._cdA

    def _w_max(self, total, static=None):

        return self.flow_func._w_flux_max(total, static) * self._cdA

    def _dP(self, total, w, static=None):
        """Calculate pressure drop for given mass flow.

        Args:
            total (thermo): Total (stagnation) state
            w (float): Mass flow rate [kg/s]
            static (thermo, optional): Static state for calculations

        Returns:
            float: Pressure drop [Pa]

        Note:
            Converts mass flow to mass flux using cdA before calculation
        """

        dP, error = self.flow_func._dP(total, w/self._cdA, static=static)

        if error is None:
            return dP, error
        else:
            error = {'w_max': error['w_flux_max']*self._cdA}
            return None, error

    def _dH(self, inlet_thermo, Pout):
        """
        Calculate isentropic enthalpy change for a pressure change in SI units.

        Args:
            inlet_thermo (thermo): Inlet thermodynamic state.
            Pout (float): Outlet pressure (Pa).

        Returns:
            float: Isentropic enthalpy change (in J).
        """
        return self.flow_func._dH(inlet_thermo, Pout)
