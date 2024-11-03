from .units import inputParser, output_converter
import numpy as np
from .thermostate import thermo
from scipy.optimize import root_scalar
from .logger import logger
from .logger import logger


def _hunt_static_PR_M(PR, M, total, static):
    # Objective function to find PR corresponding to specific Mach
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

    if total.phase in ['gas', 'supercritical_gas']:
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
        # Calculate velocity using energy balance (difference in enthalpy)

        # Offset slightly for numerical stability near choking
        return (w_flux - w_isen(total, static)) - 1e-8

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
        # Enforce PR bounds for valid range
        if PR < 0:
            return (-PR + 10) * 1e5
        elif PR > 1:
            return (1 - PR - 10) * 1e5

        # Function to find outlet pressure that results in mass flow
        total._SP = static._S, static._P/PR
        U = np.sqrt(2 * np.max([0, (total._H - static._H)]))
        return w_flux - static._density * U

    if PR_bounds is None:
        PR_bounds = [1e-2, 1]

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

    from pdb import set_trace
    set_trace()
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

    # Extend bounds if the sign is the same
    #P_bounds = _extend_bounds(hunt_PR, P_bounds, max_iter=10, factor=1.5)
    from pdb import set_trace
    set_trace()
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
    except Exception:
        logger.info("Couldn't calculate Isentropic dH")
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

    U = np.sqrt(2*np.max([(US._H - outlet._H),0]))

    # Check if the flow is sonic
    if U > outlet.sound:
        return _w_isen_max(US, outlet) * flow_sign

    # rho*U*A
    return outlet._density * U * flow_sign

def _w_isen_max(US_thermo, static=None) -> float:
    # Calculate maximum isentropic mass flux

    outlet = sonic_isentropic_state(US_thermo, static)

    return outlet.sound*outlet.density

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


    PR = .5
    PR_fac = .5
    i = 0
    while True:

        if DS_thermo._P/PR>1e8:
            return 1e15

        total._HP = DS_thermo._H, DS_thermo._P/PR
        static._SP = total._S, DS_thermo._P

        w_max = _w_isen(total, static)

        if w_max > w_flux:
            break
        else:
            PR*=PR_fac

        if i>15:
            from pdb import set_trace
            set_trace()

    def hunt_PR(PR):
        total._HP = DS_thermo._H, DS_thermo._P/PR

        static._SP = total._S, DS_thermo._P

        return _w_isen(total, static) - w_flux

    PR = root_scalar(hunt_PR, method='brentq',
                     bracket=[PR, PR/PR_fac]).root

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
        else:
            from pdb import set_trace
            set_trace()
    
    def _dP_reverse(self, DS_thermo, w_flux):

        if self.flow_func == 'isentropic':
            return _dP_isenthalpic_reverse(DS_thermo, w_flux)
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