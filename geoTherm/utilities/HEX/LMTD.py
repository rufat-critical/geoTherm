import numpy as np
from scipy.optimize import fsolve, root_scalar
from geoTherm.common import logger

# Crossflow correction coefficients from Roetzel & Nicole
# a_crossflow[(N_rows, N_passes)]
a_crossflow = {
    (1, 1): np.array([
        [-4.62E-1, -3.13E-2, -1.74E-1, -4.2E-2],
        [5.08E0, 5.29E-1, 1.32E0, 3.47E-1],
        [-1.57E1, -2.37E0, -2.93E0, -8.53E-1],
        [1.72E1, 3.18E0, 1.99E0, 6.49E-1]
    ]),
    (2, 1): np.array([
        [-3.34E-1, -1.54E-1, -8.65E-2, 5.53E-2],
        [3.3E0, 1.28E0, 5.46E-1, -4.05E-1],
        [-8.7E0, -3.35E0, -9.29E-1, 9.53E-1],
        [8.7E0, 2.83E0, 4.71E-1, -7.17E-1]
    ]),
    (3, 1): np.array([
        [-8.74E-2, -3.18E-2, -1.83E-2, 7.1E-3],
        [1.05E0, 2.74E-1, 1.23E-1, -4.99E-2],
        [-2.45E0, -7.46E-1, -1.56E-1, 1.09E-1],
        [3.21E0, 6.68E-1, 6.17E-2, -7.46E-2]
    ]),
    (4, 1): np.array([
        [-4.14E-2, -1.39E-2, -7.23E-3, 6.1E-3],
        [6.15E-1, 1.23E-1, 5.66E-2, -4.68E-2],
        [-1.2E0, -3.45E-1, -4.37E-2, 1.07E-1],
        [2.06E0, 3.18E-1, 1.11E-2, -7.57E-2]
    ]),
    (4, 2): np.array([
        [-6.05E-1, 2.31E-2, 2.94E-1, 1.98E-2],
        [4.34E0, 5.9E-3, -1.99E0, -3.05E-1],
        [-9.72E0, -2.48E-1, 4.32, 8.97E-1],
        [7.54E0, 2.87E-1, -3E0, -7.31E-1]
    ]),
    (4, 4): np.array([
        [-3.39E-1, 2.77E-2, 1.79E-1, -1.99E-2],
        [2.38E0, -9.99E-2, -1.21E0, 4E-2],
        [-5.26E0, 9.04E-2, 2.62E0, 4.94E-2],
        [3.9E0, -8.45E-4, -1.81E0, -9.81E-2]
    ])
}

def LMTD(Thi, Tho, Tci, Tco, config='counter', **kwargs):
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
        dT1 = Thi - Tco
        dT2 = Tho - Tci
    elif config == 'parallel':
        dT1 = Thi - Tci
        dT2 = Tho - Tco
    elif config == 'crossflow':
        return LMTD_crossflow(Thi, Tho, Tci, Tco, **kwargs)
    else:
        logger.critical("Invalid configuration. Use 'counter' or 'parallel'.")

    # Avoid log(1) error if dT1 == dT2
    if abs(dT1 - dT2) < 1e-6:
        return dT1  # LMTD simplifies to dT in this case

    return (dT1 - dT2) / np.log(dT1 / dT2)


def LMTD_crossflow(Thi, Tho, Tci, Tco, N_rows, N_passes):
    """
    Calculates the Log Mean Temperature Difference (LMTD) for a crossflow heat exchanger.

    Parameters:
        T_hot_in (float): Hot fluid inlet temperature (K)
        T_hot_out (float): Hot fluid outlet temperature (K)
        T_cold_in (float): Cold fluid inlet temperature (K)
        T_cold_out (float): Cold fluid outlet temperature (K)
        N_rows (int): Number of rows
        N_passes (int): Number of passes

    Returns:
        float: LMTD value
    """
    LMTD_counter = LMTD(Thi, Tho, Tci, Tco, config='counter')
    r_lm = LMTD_counter/(Thi-Tci)
    R = (Thi - Tho) / (Tco - Tci)

    # Get the correct a_crossflow matrix based on N_rows and N_passes
    a_crossflow_matrix = a_crossflow[(N_rows, N_passes)]

    i, k = np.shape(a_crossflow_matrix)
    i = np.arange(1, i+1)
    k = np.arange(1, k+1)
    sine_terms = np.sin(2*i*np.arctan(R))
    one_m_rlm = (1.0 - r_lm)**(k)

    inner_sum = a_crossflow_matrix @ sine_terms

    return LMTD_counter*(1. - np.sum(inner_sum*one_m_rlm))


def LMTD_F(Thi, Tho, Tci, Tco, N_rows, N_passes):
    """
    Calculates the LMTD correction factor (FT) for a crossflow heat exchanger
    using the Roetzel & Nicole approximation.

    Parameters:
        Thi (float): Hot fluid inlet temperature (K)
        Tho (float): Hot fluid outlet temperature (K)
        Tci (float): Cold fluid inlet temperature (K)
        Tco (float): Cold fluid outlet temperature (K)
        N_rows (int): Number of tube rows
        N_passes (int): Number of passes on the tube side

    Returns:
        float: LMTD correction factor FT (unitless)
    """
    # Compute dimensionless counterflow LMTD
    r_lm = LMTD(Thi, Tho, Tci, Tco, config='counter') / (Thi - Tci)

    # Compute R = effectiveness ratio
    R = (Thi - Tho) / (Tco - Tci)

    # Get coefficients for current configuration
    a_crossflow_matrix = a_crossflow[(N_rows, N_passes)]

    # Index ranges for i and k (1-based)
    i_len, k_len = np.shape(a_crossflow_matrix)
    i = np.arange(1, i_len + 1)
    k = np.arange(1, k_len + 1)

    # Sine terms: sin(2 * i * arctan(R))
    sine_terms = np.sin(2 * i * np.arctan(R))

    # Powers of (1 - r_lm)
    one_m_rlm = (1.0 - r_lm) ** k

    # Matrix-vector product: sum over i
    inner_sum = a_crossflow_matrix @ sine_terms

    # Total correction factor
    return 1.0 - np.sum(inner_sum * one_m_rlm)


def UA(hot_fluid_in, hot_fluid_out, cold_fluid_in, cold_fluid_out,
       mdot_hot, mdot_cold, config='counter'):

    if config == 'counter':
        return UA_counterflow(hot_fluid_in, hot_fluid_out, cold_fluid_in, cold_fluid_out,
                              mdot_hot, mdot_cold)
    else:
        from pdb import set_trace
        set_trace()


def UA_counterflow(hot_fluid_in, hot_fluid_out, cold_fluid_in, cold_fluid_out,
                  mdot_hot, mdot_cold):
    """
    Calculate the overall heat transfer coefficient (UA) for a counterflow heat exchanger.
    
    This function handles different scenarios:
    1. Single phase flow for both fluids
    2. Phase change in cold fluid while hot fluid remains single phase
    
    Parameters:
        hot_fluid_in: Fluid object representing hot fluid inlet state
        hot_fluid_out: Fluid object representing hot fluid outlet state
        cold_fluid_in: Fluid object representing cold fluid inlet state
        cold_fluid_out: Fluid object representing cold fluid outlet state
        mdot_hot: Mass flow rate of hot fluid (kg/s)
        mdot_cold: Mass flow rate of cold fluid (kg/s)
    
    Returns:
        float: Overall heat transfer coefficient (UA)
    """
    # Calculate total heat transfer rate
    Q_hot = mdot_hot * (hot_fluid_in._H - hot_fluid_out._H)
    Q_cold = mdot_cold * (cold_fluid_in._H - cold_fluid_out._H)

    
    # Case 1: Single phase flow (no phase change in either fluid)
    if (hot_fluid_out.phase == hot_fluid_in.phase and 
        cold_fluid_out.phase == cold_fluid_in.phase):

        UA = UA_Q(Q_hot, hot_fluid_in.T, hot_fluid_out.T,
                   cold_fluid_in.T, cold_fluid_out.T, 'counter')
        return UA, [UA]

    # Case 2: Hot fluid remains single phase
    elif hot_fluid_out.phase == hot_fluid_in.phase:
        # Case 2a: Cold fluid remains liquid or supercritical
        if (cold_fluid_in.phase in ['liquid', 'supercritical', 'supercritical_liquid'] and 
            cold_fluid_out.phase in ['liquid', 'supercritical', 'supercritical_liquid']):
            UA = UA_Q(Q_hot, hot_fluid_in.T, hot_fluid_out.T,
                       cold_fluid_in.T, cold_fluid_out.T, 'counter')
            return UA, [UA]

        if (cold_fluid_in._P >= cold_fluid_in._P_crit):
            #and cold_fluid_out._P >= cold_fluid_out._P_crit):
            UA = UA_Q(Q_hot, hot_fluid_in.T, hot_fluid_out.T,
                       cold_fluid_in.T, cold_fluid_out.T, 'counter')
            return UA, [UA]

        # Case 2b: Cold fluid undergoes phase change
        if cold_fluid_in.phase == 'liquid':
            # Create working copies of the fluids
            saturated = cold_fluid_in.from_state(cold_fluid_in.state)
            hot_fluid = hot_fluid_in.from_state(hot_fluid_in.state)

            # Calculate heat transfer for three regions:
            # 1. Liquid heating to saturation
            saturated._PQ = cold_fluid_in._P, 0  # Saturated liquid point
            H_sat_liq = saturated._H
            Q1 = mdot_cold * (H_sat_liq - cold_fluid_in._H)

            # 2. Phase change (latent heat)
            saturated._PQ = cold_fluid_in._P, 1  # Saturated vapor point
            H_sat_vap = saturated._H
            Q2 = mdot_cold * (H_sat_vap - H_sat_liq)

            # 3. Vapor heating
            Q3 = mdot_cold * (cold_fluid_out._H - H_sat_vap)

            # Calculate corresponding hot fluid temperatures at transition points
            hot_fluid._HP = hot_fluid_in._H - (Q_hot - (Q1 + Q2)) / mdot_hot, hot_fluid_in._P
            Thot_1 = hot_fluid._T
            hot_fluid._HP = hot_fluid_in._H - (Q_hot - Q1) / mdot_hot, hot_fluid_in._P
            Thot_2 = hot_fluid._T

            # Calculate UA for each region and sum
            UA1 = UA_Q(Q1, Thot_1, hot_fluid_out._T, cold_fluid_in._T, saturated._T, 'counter')
            UA2 = UA_Q(Q2, Thot_2, Thot_1, saturated._T, saturated._T, 'counter')
            UA3 = UA_Q(Q3, hot_fluid_in._T, Thot_2, saturated._T, cold_fluid_out._T, 'counter')

            return UA1 + UA2 + UA3, [UA1, UA2, UA3]
        else:
            from pdb import set_trace
            set_trace()

    elif cold_fluid_out.phase == cold_fluid_in.phase or (cold_fluid_out.phase == 'supercritical' and cold_fluid_in.phase == 'supercritical_liquid'):
        if (hot_fluid_out.phase in ['liquid', 'supercritical', 'supercritical_liquid'] and 
            hot_fluid_in.phase in ['liquid', 'supercritical', 'supercritical_liquid']):
            UA = UA_Q(Q_hot, hot_fluid_in.T, hot_fluid_out.T,
                      cold_fluid_in.T, cold_fluid_out.T, 'counter') 
            return UA, [UA]
        elif (hot_fluid_in.phase in ['gas', 'supercritical_gas'] and 
              hot_fluid_out.phase in ['liquid']):
            saturated = hot_fluid_in.from_state(hot_fluid_in.state)
            cold_fluid = cold_fluid_in.from_state(cold_fluid_in.state)
            saturated._PQ = hot_fluid_in._P, 1
            H_sat_vap = saturated._H
            Q1 = mdot_hot * (hot_fluid_in._H - H_sat_vap)

            saturated._PQ = hot_fluid_in._P, 0
            H_sat_liq = saturated._H
            Q2 = mdot_hot * (H_sat_vap - H_sat_liq)

            Q3 = mdot_hot * (H_sat_liq - hot_fluid_out._H)

            cold_fluid._HP = cold_fluid_out._H - (Q_hot - (Q1 + Q2))/mdot_cold, cold_fluid_in._P
            Tcold_1 = cold_fluid._T
            cold_fluid._HP = cold_fluid_out._H - (Q_hot - (Q1))/mdot_cold, cold_fluid_in._P
            Tcold_2 = cold_fluid._T


            UA1 = UA_Q(Q1, hot_fluid_in._T, saturated._T, Tcold_1, cold_fluid_out._T, 'counter')
            UA2 = UA_Q(Q2, saturated._T, saturated._T, Tcold_2, Tcold_1, 'counter')
            UA3 = UA_Q(Q3, saturated._T, hot_fluid_out._T, cold_fluid_in._T, Tcold_2, 'counter')

            return UA1 + UA2 + UA3, [UA1, UA2, UA3]
        elif (hot_fluid_in.phase in ['gas', 'supercritical_gas'] and 
              hot_fluid_out.phase in ['supercritical, supercritical_liquid']):
            UA = UA_Q(Q_hot, hot_fluid_in.T, hot_fluid_out.T,
                        cold_fluid_in.T, cold_fluid_out.T, 'counter') 
            return UA, [UA]
        else:
            from pdb import set_trace
            set_trace()

    # Case 3: Other scenarios not yet implemented
    raise NotImplementedError("This heat exchanger configuration is not yet supported")


def UA_Q(Q, T_hot_in, T_hot_out, T_cold_in, T_cold_out, config='counter'):

    UA = Q / LMTD(T_hot_in, T_hot_out, T_cold_in, T_cold_out, config)

    return UA


def LMTD_solve_cold(Q, UA, hot_in, hot_out, cold_in, cold_out_thermo=None, config='counter', **kwargs):
    """
    Solve for the cold fluid outlet temperature in a heat exchanger.

    Parameters:
        Q: Heat transfer rate (W)
        hot_in: Hot fluid inlet temperature (K)
        hot_out: Hot fluid outlet temperature (K)
        cold_in: Cold fluid inlet temperature (K)
        cold_out_thermo: Cold fluid outlet thermodynamics (Fluid object)
        config: 'counter' for counterflow, 'parallel' for parallel flow

    Returns:
        cold_out: Cold fluid outlet temperature (K)
        LMTD: Log mean temperature difference (K)
    """

    if cold_out_thermo is None:
        cold_out_thermo = cold_in.from_state(cold_in.state)

    cold_out_thermo._TP = hot_in._T, cold_in._P

    w_cold_min = Q/(cold_out_thermo._H - cold_in._H)

    def f(w_cold, **kwargs):

        if w_cold < w_cold_min:
            return (w_cold_min - w_cold + 1)*1e5

        cold_out_thermo._HP = cold_in._H + Q/w_cold, cold_in._P
        
        LMdT = LMTD(hot_in._T, hot_out._T, cold_in._T, cold_out_thermo._T, config, **kwargs)

        return Q - UA*LMdT
    
    sol = fsolve(f, w_cold_min, full_output=True)

    if sol[2] != 1:
        from pdb import set_trace
        set_trace()

    w_cold = sol[0]

    cold_out_thermo._HP = cold_in._H + Q/w_cold, cold_in._P

    return w_cold, cold_out_thermo._T, LMTD(hot_in._T, hot_out._T, cold_in._T, cold_out_thermo._T, config, **kwargs)


def LMTD_solve_outlet(UA, hot_in, cold_in, w_hot, w_cold, config='counter', **kwargs):


    hot_out = hot_in.from_state(hot_in.state)
    cold_out = cold_in.from_state(cold_in.state)

    hot_out._TP = cold_in._T, hot_in._P
    cold_out._TP = hot_in._T, cold_in._P

    Q_hot_max = w_hot*(hot_in._H-hot_out._H)
    Q_cold_max = w_cold*(cold_out._H - cold_in._H)

    Q_max = np.min([Q_hot_max, Q_cold_max])


    def f(Q):

        if Q > Q_max:
            return (Q_max - Q - 1)*1e5

        hot_out._HP = hot_in._H - Q/w_hot, hot_in._P
        cold_out._HP = cold_in._H + Q/w_cold, cold_in._P

        LMdT = LMTD(hot_in._T, hot_out._T, cold_in._T, cold_out._T, config, **kwargs)

        return UA*LMdT - Q

    sol = root_scalar(f, bracket=[0, Q_max], method='brentq')

    if sol.converged == False:
        from pdb import set_trace
        set_trace()

    Q = sol.root

    if np.abs(f(Q)) > 1:
        from pdb import set_trace
        set_trace()


    hot_out._HP = hot_in._H - Q/w_hot, hot_in._P
    cold_out._HP = cold_in._H + Q/w_cold, cold_in._P 
    LMdT = LMTD(hot_in._T, hot_out._T, cold_in._T, cold_out._T, config, **kwargs)
    
    return Q, LMdT, hot_out._T, cold_out._T