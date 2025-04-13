from geoTherm.thermostate import thermo
import numpy as np
from scipy.optimize import root_scalar
from geoTherm.units import inputParser, fromSI, units
from geoTherm.utilities.heat_transfer import T_dQ
from geoTherm.logger import logger
from scipy.optimize import root_scalar, minimize_scalar
from .profile import HEXProfile


def pinch_T_discretized(cold_inlet, cold_outlet, hot_inlet, mdot_hot, mdot_cold, dP_hot, n_points=100):

    Q = mdot_cold * (cold_outlet._H - cold_inlet._H)
    hot_outlet = hot_inlet.from_state(hot_inlet.state)
    hot_outlet._HP = hot_inlet._H - Q/mdot_hot, hot_inlet._P - dP_hot

    results = T_dQ(Q, mdot_hot, mdot_cold, n_points=n_points, config='counter',
                   hot_in=hot_inlet, hot_out=hot_outlet,
                   cold_in=cold_inlet, cold_out=cold_outlet,
                   dP_hot=dP_hot, dP_cold=0)

    #HEXProfile(cold_inlet, cold_outlet, hot_inlet, hot_outlet, mdot_hot, mdot_cold, dP_hot, n_points=n_points)
    from pdb import set_trace
    set_trace()
    HEX = HEXProfile(w_hot= mdot_hot, w_cold=mdot_cold, hot_inlet=hot_inlet, hot_outlet=hot_outlet,
                     cold_inlet=cold_inlet, cold_outlet=cold_outlet)
    HEX.res = results
    HEX.evaluate(Q, n_points=100)
    from pdb import set_trace
    set_trace()


def find_pinch_Q(cold_inlet, cold_outlet, hot_inlet, w_hot, dP_hot, T_pinch):
    """
    Find the heat transfer rate that results in a pinch point temperature difference of T_pinch
    """

    # Check that cold_inlet and cold_outlet are not below T_pinch
    if hot_inlet._T - cold_inlet._T < T_pinch:
        logger.warn(f'dT at cold inlet is below T_pinch with no heat transfer: {hot_inlet._T - cold_inlet._T} K')
        return 0
    elif hot_inlet._T - cold_outlet._T < T_pinch:
        logger.warn(f'dT at cold outlet is below T_pinch with no heat transfer: {hot_inlet._T - cold_outlet._T} K')
        return 0

    hot_outlet = hot_inlet.from_state(hot_inlet.state)
    # Max possible heat transfer is when outlet temperature is at T_pinch for counter flow
    hot_outlet._TP = cold_inlet._T + T_pinch, hot_inlet._P - dP_hot
    Q_max = w_hot * (hot_inlet._H - hot_outlet._H)
    Q_min = 0.001

    # Get max and min cold_inlet mdots
    w_cold_max = Q_max/(cold_outlet._H - cold_inlet._H)
    w_cold_min = Q_min/(cold_outlet._H - cold_inlet._H)

    # Generate thermo objects
    hot_thermo = hot_inlet.from_state(hot_inlet.state)
    cold_thermo = cold_inlet.from_state(cold_inlet.state)

    if cold_inlet._P >= cold_inlet._P_crit or cold_outlet._P >= cold_outlet._P_crit:
        mdot_cold, Q = find_pinch_Q_general(cold_inlet, cold_outlet, hot_inlet,
                                   hot_outlet, w_hot, dP_hot, T_pinch, mdot_cold_bounds = [w_cold_min, w_cold_max],
                                   hot_thermo=hot_thermo, cold_thermo=cold_thermo)
        
        return mdot_cold, Q


    def find_Q(w_cold):
        Q = w_cold * (cold_outlet._H - cold_inlet._H)
        try:
            hot_outlet._HP = hot_inlet._H - Q/w_hot, hot_inlet._P - dP_hot
        except:
            from pdb import set_trace
            set_trace()
        
        dT = dT_subcritical_counter_flow(cold_inlet, cold_outlet, hot_inlet, hot_outlet, w_hot, w_cold, Q=Q, dP_hot=dP_hot, hot_thermo=hot_thermo, cold_thermo=cold_thermo)

        return np.min(dT['dT'][:2]) - T_pinch


    # For debugging
    if np.sign(find_Q(w_cold_min)) == np.sign(find_Q(w_cold_max)):
        HEX = HEXProfile(w_hot= w_hot, w_cold=w_cold_min, hot_inlet=hot_inlet,
                         cold_inlet=cold_inlet, cold_outlet=cold_outlet, dP_hot=dP_hot)
        Q = w_cold_min * (cold_outlet._H - cold_inlet._H)
        HEX.evaluate(Q, n_points=100)
        from pdb import set_trace
        set_trace()

    sol = root_scalar(find_Q, bracket=[w_cold_min, w_cold_max], method='brentq')
    mdot_cold = sol.root

    Q = mdot_cold * (cold_outlet._H - cold_inlet._H)

    return mdot_cold, Q

def find_pinch_Q_general(cold_inlet, cold_outlet, hot_inlet, hot_outlet, mdot_hot, dP_hot, T_pinch, 
                                mdot_cold_bounds,
                                hot_thermo=None, cold_thermo=None):


    dP_cold = (cold_outlet._P - cold_inlet._P)

    def dT_Q(Q, Q_tot, mdot_cold):
        hot_thermo._HP = hot_inlet._H - (Q_tot-Q)/mdot_hot, hot_inlet._P - dP_hot*(Q_tot-Q)/Q_tot
        cold_thermo._HP = cold_inlet._H + Q/mdot_cold, cold_inlet._P + dP_cold*Q/Q_tot

        dT = hot_thermo._T - cold_thermo._T
        return dT

    def deltaT(mdot_cold):
        Q = mdot_cold * (cold_outlet._H - cold_inlet._H)
        hot_thermo._HP = hot_inlet._H - Q/mdot_hot, hot_inlet._P - dP_hot

        sol = minimize_scalar(dT_Q, args=(Q, mdot_cold), bounds=[0, Q*.99], method='bounded')

        return sol.fun - T_pinch

    mdot_cold = mdot_cold_bounds[0]

    Q = mdot_cold * (cold_outlet._H - cold_inlet._H)


    sol = root_scalar(deltaT, bracket=[mdot_cold_bounds[0], mdot_cold_bounds[1]], method='brentq')
    mdot_cold = sol.root
    Q = mdot_cold * (cold_outlet._H - cold_inlet._H)
    sol = minimize_scalar(dT_Q, args=(Q, mdot_cold), bounds=[0, Q*.99], method='bounded')

    Q = mdot_cold * (cold_outlet._H - cold_inlet._H)


    return mdot_cold, Q




def dT_subcritical_counter_flow(cold_inlet, cold_outlet, hot_inlet, hot_outlet, w_hot, w_cold, 
                                Q=None, dP_hot=None, hot_thermo=None, cold_thermo=None):
    """
    Calculate temperature differences and heat at key points in a
    subcritical counter-flow heat exchanger.

    Args:
        cold_inlet (thermo): Cold stream inlet state
        cold_outlet (thermo): Cold stream outlet state
        hot_inlet (thermo): Hot stream inlet state
        hot_outlet (thermo): Hot stream outlet state
        w_hot (float): Mass flow rate of hot stream [kg/s]
        w_cold (float): Mass flow rate of cold stream [kg/s]
        Q (float, optional): Heat duty. If None, will be calculated from hot stream states
        dP_hot (float, optional): Total pressure drop in hot stream [Pa]. If None, will be calculated from hot stream states
        hot_thermo (thermo, optional): Hot stream thermo object for reuse. If None, a new one will be created
        cold_thermo (thermo, optional): Cold stream thermo object for reuse. If None, a new one will be created

    Returns:
        dict: Contains arrays of temperature differences (dT), corresponding Q
                values, and temperatures of both streams at each point
    """

    # Total heat and hot pressure drop if not provided
    if Q is None:
        Q = w_hot * (hot_inlet._H - hot_outlet._H)
    if dP_hot is None:
        dP_hot = hot_outlet._P - hot_inlet._P

    # Reuse existing thermo objects if provided, create new ones if needed
    if hot_thermo is None:
        hot_thermo = hot_inlet.from_state(hot_inlet.state)
    if cold_thermo is None:
        cold_thermo = cold_inlet.from_state(cold_inlet.state)

    # Verify energy balance (only in debug/development)
    # Q2 = mdot_cold * (cold_outlet._H - cold_inlet._H)
    # if np.abs(Q - Q2) > 1:
    #     from pdb import set_trace
    #     set_trace()

    # Calculate saturation point properties, assume no pressure drop
    cold_thermo._PQ = cold_inlet._P, 0
    Q_sat = w_cold * (cold_thermo._H - cold_inlet._H)

    hot_thermo._HP = (
        hot_inlet._H - (Q-Q_sat)/w_hot,
        hot_inlet._P + dP_hot * (Q-Q_sat)/Q
    )

    # Return dictionary of data
    return {
        'dT': np.array([
            hot_outlet._T - cold_inlet._T,    # Cold inlet
            hot_thermo._T - cold_thermo._T,   # Saturation point
            hot_inlet._T - cold_outlet._T     # Cold outlet
        ]),
        'Qs': np.array([0, Q_sat, Q])
    }

