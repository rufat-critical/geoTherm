from geoTherm.thermostate import thermo
import numpy as np
from scipy.optimize import root_scalar


def LMTD(T_hot_in, T_hot_out, T_cold_in, T_cold_out, config='counter'):
    """
    Calculates the Log Mean Temperature Difference (LMTD) for a heat exchanger.

    Parameters:
        T_hot_in (float): Hot fluid inlet temperature (K)
        T_hot_out (float): Hot fluid outlet temperature (K)
        T_cold_in (float): Cold fluid inlet temperature (K)
        T_cold_out (float): Cold fluid outlet temperature (K)
        config (str): 'counter' for counterflow, 'parallel' for parallel flow

    Returns:
        float: LMTD value
    """
    if config == 'counter':
        dT1 = T_hot_in - T_cold_out
        dT2 = T_hot_out - T_cold_in
    elif config == 'parallel':
        dT1 = T_hot_in - T_cold_in
        dT2 = T_hot_out - T_cold_out
    else:
        raise ValueError("Invalid configuration. Use 'counter' or 'parallel'.")

    # Avoid log(1) error if dT1 == dT2
    if abs(dT1 - dT2) < 1e-6:
        return dT1  # LMTD simplifies to dT in this case

    return (dT1 - dT2) / np.log(dT1 / dT2)


def Q_LMTD(hot_fluid,
           cold_fluid,
           mdot_hot,
           mdot_cold,
           UA, config='counter'):
    """
    Solves for heat transfer (Q) using LMTD and energy balance.

    Parameters:
        hot_fluid (thermo): Hot fluid object (thermo object)
        cold_fluid (thermo): Cold fluid object (thermo object)
        mdot_hot (float): Mass flow rate of hot fluid (kg/s)
        mdot_cold (float): Mass flow rate of cold fluid (kg/s)
        UA (float): Overall heat transfer coefficient times area (W/K)
        config (str): 'counter' for counterflow, 'parallel' for parallel flow

    Returns:
        dict: Heat transfer Q (W) and outlet temperatures
    """
    # Extract initial conditions
    H_hot_in = hot_fluid._H
    H_cold_in = cold_fluid._H

    # Create new instances for outlet conditions
    hot_out = thermo.from_state(hot_fluid.state)
    cold_out = thermo.from_state(cold_fluid.state)


    def _find_heat(Q):
        """
        Residual function to solve for Q using UA and LMTD.
        """
        # Update outlet enthalpies
        H_hot_out = H_hot_in - Q / mdot_hot
        H_cold_out = H_cold_in + Q / mdot_cold

        # Assign new enthalpies to hot and cold outlet fluids
        hot_out._HP = H_hot_out, hot_fluid._P
        cold_out._HP = H_cold_out, cold_fluid._P

        # If the cold fluid is hotter than hot fluid then Q is too high
        if cold_out._T > hot_fluid._T:
            return (hot_fluid._T - cold_out._T - 10)*1e5

        # Calculate new LMTD
        LMTD_val = LMTD(hot_fluid._T, hot_out._T,
                        cold_fluid._T, cold_out._T,
                        config)

        return UA * LMTD_val - Q  # Solve for Q numerically

    # Solve for Q using a root finder
    sol = root_scalar(_find_heat, bracket=[0, 1.4e6], method='brentq')

    return sol.root, hot_out._T, cold_out._T


def UA_LMTD(hot_fluid_in, hot_fluid_out, cold_fluid_in, cold_fluid_out,
            mdot_hot, mdot_cold, Q=None, config='counter'):
    """
    Solves for UA using known heat transfer and temperature conditions.

    Parameters:
        hot_fluid_in (thermo): Hot fluid inlet object
        hot_fluid_out (thermo): Hot fluid outlet object
        cold_fluid_in (thermo): Cold fluid inlet object
        cold_fluid_out (thermo): Cold fluid outlet object
        mdot_hot (float): Mass flow rate of hot fluid (kg/s)
        mdot_cold (float): Mass flow rate of cold fluid (kg/s)
        config (str): 'counter' for counterflow, 'parallel' for parallel flow

    Returns:
        float: Overall heat transfer coefficient times area (UA) in W/K
    """

    if Q is None:
        # Compute actual heat transfer based on enthalpy difference
        Q_hot = mdot_hot * (hot_fluid_in._H - hot_fluid_out._H)
        Q_cold = mdot_cold * (cold_fluid_out._H - cold_fluid_in._H)

        # Ensure heat balance consistency
        Q = min(Q_hot, Q_cold)

    # Update outlet states
    hot_fluid_out._HP = hot_fluid_in._H - Q/mdot_hot, hot_fluid_out._P
    cold_fluid_out._HP = cold_fluid_in._H + Q/mdot_cold, cold_fluid_out._P

    # Calculate LMTD
    LMTD_val = LMTD(hot_fluid_in._T, hot_fluid_out._T,
                    cold_fluid_in._T, cold_fluid_out._T,
                    config)

    # Compute UA
    UA = Q / LMTD_val

    return UA, Q
