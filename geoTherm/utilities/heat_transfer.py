from geoTherm.thermostate import thermo
import numpy as np
from scipy.optimize import root_scalar
from geoTherm.units import inputParser, fromSI, units



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


def T_dQ(Q, mdot_hot, mdot_cold, n_points=100, config='counter',
         hot_in=None, hot_out=None,
         cold_in=None, cold_out=None,
         dP_hot=0, dP_cold=0):

    # Validate inputs
    if (hot_in is None and hot_out is None) or (cold_in is None and cold_out is None):
        raise ValueError("Must specify either inlet or outlet state for both hot and cold fluids")


    if hot_in is not None and hot_out is not None:
        Q_hot = mdot_hot * (hot_in._H - hot_out._H)
        if np.abs(Q_hot - Q) > 1:
            from pdb import set_trace
            set_trace()
    else:
        if hot_in is not None:
            if hot_out is None:
                hot_out = thermo.from_state(hot_in.state)
                hot_out._HP = hot_in._H - Q/mdot_hot, hot_in._P - dP_hot
        else:
            hot_in = thermo.from_state(hot_out.state)
            hot_in._HP = hot_out._H + Q/mdot_hot, hot_out._P + dP_hot        

    if cold_in is not None and cold_out is not None:
        Q_cold = -mdot_cold * (cold_in._H - cold_out._H)
        if np.abs(Q_cold - Q) > 1:
            from pdb import set_trace
            set_trace()
    else:
        if cold_in is not None:
            if cold_out is None:
                cold_out = thermo.from_state(cold_in.state)
                cold_out._HP = cold_in._H + Q/mdot_cold, cold_in._P - dP_cold
        else:
            cold_in = thermo.from_state(cold_out.state)
            cold_in._HP = cold_out._H - Q/mdot_cold, cold_out._P + dP_cold


    dP_hot = hot_out._P - hot_in._P
    dP_cold = cold_out._P - cold_in._P

    if config == 'counter':
        return T_dQ_counter_flow(Q, mdot_hot, mdot_cold,
                               hot_in, cold_in, dP_hot, dP_cold, n_points)
    elif config == 'parallel':
        return T_dQ_parallel_flow(Q, mdot_hot, mdot_cold,
                                hot_in, cold_in,
                                dP_hot, dP_cold, n_points)
    else:
        from pdb import set_trace
        set_trace()


def T_dQ_counter_flow(Q, mdot_hot, mdot_cold,
                      hot_inlet, cold_inlet, dP_hot, dP_cold, n_points=100):
    """
    Calculates temperature profiles by sweeping through heat transfer values,
    with linear pressure drops along the heat exchanger.

    Parameters:
        Q (float): Total heat transfer rate (W)
        mdot_hot (float): Mass flow rate of hot fluid (kg/s)
        mdot_cold (float): Mass flow rate of cold fluid (kg/s)
        n_points (int): Number of points for calculation
        config (str): 'counter' or 'parallel' flow configuration
        hot_in (thermo, optional): Hot fluid inlet state
        hot_out (thermo, optional): Hot fluid outlet state
        cold_in (thermo, optional): Cold fluid inlet state
        cold_out (thermo, optional): Cold fluid outlet state
        dP_hot (float): Pressure drop in hot fluid side (Pa)
        dP_cold (float): Pressure drop in cold fluid side (Pa)

    Returns:
        dict: Dictionary containing temperature and pressure profiles and heat transfer values
    """
   
    dQ = Q / n_points

    # Create arrays for heat transfer steps and pressure profiles
    Q_array = np.zeros(n_points-1)
    T_hot = np.zeros(n_points-1)
    T_cold = np.zeros(n_points-1)
    P_hot = np.zeros(n_points-1)
    P_cold = np.zeros(n_points-1)

    # Create fluid state objects for calculations
    hot_state = thermo.from_state(hot_inlet.state)
    cold_state = thermo.from_state(cold_inlet.state)

    i = 0
    while i < n_points-1:
        # Calculate pressure at current position (linear drop)

        current_Q = dQ * i
        current_P_hot = hot_inlet._P + dP_hot * (Q-current_Q)/Q
        current_P_cold = cold_inlet._P + dP_cold * dQ/Q*i

        # Update hot fluid state
        hot_state._HP = hot_inlet._H - (Q-current_Q)/mdot_hot, current_P_hot

        # Update cold fluid state
        prev_phase = cold_state.phase
        cold_state._HP = cold_inlet._H + current_Q/mdot_cold, current_P_cold

        Q_array[i] = current_Q
        T_hot[i] = hot_state._T
        P_hot[i] = current_P_hot
        T_cold[i] = cold_state._T
        P_cold[i] = current_P_cold

        # If phase change, insert saturation point at i-1
        if cold_state.phase != prev_phase: 
            
            # Create saturation state
            sat_state = thermo.from_state(cold_state.state)
            
            # Phase change detected - insert saturation point at i-1
            if prev_phase == 'liquid' and cold_state.phase == 'two-phase':
                # Calculate state at saturation point
                sat_state._PQ = current_P_cold, 0  # saturated liquid
                i_pos = i

            elif prev_phase == 'two-phase' and cold_state.phase in ['gas', 'supercritical_gas']:
                # Found transition to vapor, calculate Q at saturation point
                sat_state._PQ = current_P_cold, 1  # saturated vapor
                i_pos = i+1

            elif prev_phase == 'supercritical_liquid' and cold_state.phase == 'supercritical':
                sat_state._TP = sat_state._T, current_P_cold
                i_pos = i
            else:
                from pdb import set_trace
                set_trace()

            q_sat = mdot_cold * (cold_state._H - sat_state._H)
            hot_state._HP = hot_inlet._H - (Q-current_Q-q_sat)/mdot_hot, current_P_hot
            # Insert saturation point at i-1
            Q_array = np.insert(Q_array, i_pos, current_Q-q_sat)
            T_hot = np.insert(T_hot, i_pos, hot_state._T)
            P_hot = np.insert(P_hot, i_pos, current_P_hot)
            T_cold = np.insert(T_cold, i_pos, sat_state._T)
            P_cold = np.insert(P_cold, i_pos, current_P_cold)
            i += 1
            n_points += 1
            continue

        # If no phase change, continue
        i += 1

    return {
        'Q': Q_array,
        'T_hot': T_hot,
        'T_cold': T_cold,
        'P_hot': P_hot,
        'P_cold': P_cold,
    }



def T_pinch_counter(cold_inlet, cold_outlet, hot_inlet, hot_outlet, mdot_cold, mdot_hot):
    """
    Calculate heat duty, pinch point, and hot outlet state for a heat exchanger
    using enthalpy changes and state updates
    
    Args:
        mdot_cold (float): Mass flow rate of cold stream [kg/s]
        mdot_hot (float): Mass flow rate of hot stream [kg/s]
        cold_inlet (gt.thermo): Cold stream inlet state
        cold_outlet (gt.thermo): Cold stream outlet state
        hot_inlet (gt.thermo): Hot stream inlet state
        dP_hot (float): Total pressure drop in hot stream [Pa]

    Returns:
        dict: Dictionary containing Q, Tpinch, and hot_outlet
    """

    # Verify Q is equal
    Q1 = mdot_cold * (cold_outlet._H - cold_inlet._H)
    Q2 = mdot_hot * (hot_inlet._H - hot_outlet._H)


    if cold_inlet._P > cold_outlet._P_crit or cold_outlet._P > cold_outlet._P_crit:
        print('Supercritical')
        try:
            dT_cold_inlet = hot_outlet._T - cold_inlet._T
            dT_cold_outlet = hot_inlet._T - cold_outlet._T
            hot_intermediate = hot_inlet.from_state(hot_inlet.state)
            cold_intermediate = cold_inlet.from_state(cold_inlet.state)
            Qmid = Q1/2
            dP_hot = hot_outlet._P - hot_inlet._P
            dP_cold = cold_outlet._P - cold_inlet._P

            hot_intermediate._HP = hot_inlet._H - Qmid/mdot_hot, hot_outlet._P# - dP_hot/10
            cold_intermediate._HP = cold_inlet._H + Qmid/mdot_cold, cold_inlet._P# - dP_cold/2


            mdot_cold_mid = Qmid/(cold_outlet._H - cold_inlet._H)

            dT_cold_mid = hot_intermediate._T - cold_intermediate._T
        except:
            from pdb import set_trace
            set_trace()


        return (dT_cold_inlet, dT_cold_outlet, np.min([dT_cold_inlet, dT_cold_outlet, dT_cold_mid]))
        
        from pdb import set_trace
        set_trace()

    if np.abs(Q1- Q2)>1:
        from pdb import set_trace
        set_trace()


    # Check Phases
    if cold_inlet.phase != 'liquid':
        from pdb import set_trace
        #set_trace()

    if cold_outlet.phase not in ['gas', 'supercritical_gas']:
        from pdb import set_trace
        #set_trace()

    if hot_inlet.phase != hot_outlet.phase:
        from pdb import set_trace
        #set_trace()

    # Calculate intermediate states for cold stream at saturation points
    intermediate_cold = cold_inlet.from_state(cold_inlet.state)
    intermediate_cold._PQ = intermediate_cold._P, 0  # Saturated liquid
    intermediate_hot = hot_inlet.from_state(hot_inlet.state)

    dP_hot = hot_outlet._P - hot_inlet._P
    dQ = mdot_cold * (intermediate_cold._H - cold_inlet._H)

    intermediate_hot._HP = hot_inlet._H - (Q1-dQ)/mdot_hot, hot_inlet._P + dP_hot*(Q1-dQ)/Q1


    dT_cold_inlet = hot_outlet._T - cold_inlet._T
    dT_cold_intermediate = intermediate_hot._T - intermediate_cold._T
    dT_cold_outlet = hot_inlet._T - cold_outlet._T

    return (dT_cold_inlet, dT_cold_outlet, dT_cold_intermediate)


def T_pinch_Q2(cold_inlet, cold_outlet, hot_inlet, mdot_hot, T_pinch, T_pinch_out, dP_cold, dP_hot):
    """
    Calculate heat duty, pinch point, and hot outlet state for a heat exchanger
    using enthalpy changes and state updates
    
    Args:
        mdot_cold (float): Mass flow rate of cold stream [kg/s]
        mdot_hot (float): Mass flow rate of hot stream [kg/s]
        cold_inlet (gt.thermo): Cold stream inlet state
        hot_inlet (gt.thermo): Hot stream inlet state
        T_pinch (float): Minimum temperature difference allowed (K)
        T_pinch_out (float): Minimum temperature difference allowed at outlet (K)

    """


    # Cold Inlet, Cold Outlet
    # 

    # Get Q_max
    hot_out = hot_inlet.from_state(hot_inlet.state)
    hot_out._TP = cold_inlet._T + T_pinch, hot_inlet._P - dP_hot
    Q_max = mdot_hot * (hot_inlet._H - hot_out._H)
    Q_min = 0.001

    def find_Q(Q):
        try:
            hot_out._HP = hot_inlet._H - Q/mdot_hot, hot_inlet._P - dP_hot
        except:
            from pdb import set_trace
            set_trace()
        

        mdot_cold = Q/(cold_outlet._H - cold_inlet._H)

        results = T_dQ(Q, hot_in=hot_inlet,cold_in=cold_inlet,mdot_cold=mdot_cold, mdot_hot=mdot_hot, dP_hot=38e5, dP_cold=1e5,n_points=5)
        
        Tmin = np.min(results['T_hot'] - results['T_cold'])
        
        return Tmin - T_pinch

    from scipy.optimize import root_scalar
    if cold_inlet._P >= cold_inlet._P_crit:
        try:
            sol = root_scalar(find_Q, bracket=[Q_min, Q_max], method='brentq')
            Q = sol.root
        except Exception as e:
            print(e)
            mdot_cold = Q_min/(cold_outlet._H - cold_inlet._H)
            #results = T_dQ(Q_min,hot_in=hot_inlet,cold_in=cold_inlet,mdot_cold=mdot_cold, mdot_hot=mdot_hot,dP_hot=dP_hot*.9,dP_cold=1e5)

            #find_Q(Q_max)
            #mdot_cold = 0.001/(cold_outlet._H - cold_inlet._H)
            #results = T_dQ(1,hot_in=hot_inlet,cold_in=cold_inlet,mdot_cold=mdot_cold, mdot_hot=mdot_hot,dP_hot=dP_hot*.9,dP_cold=1e5)
            from pdb import set_trace
            set_trace()

        return Q, Q/(cold_outlet._H - cold_inlet._H)

    # Check if 
    intermediate_cold = cold_inlet.from_state(cold_inlet.state)
    try:
        intermediate_cold._PQ = intermediate_cold._P, 0  # Saturated liquid
    except:
        from pdb import set_trace
        set_trace()

    if intermediate_cold._T > hot_inlet._T-5+2:
        print('Cold Inlet P is too hot')
        from pdb import set_trace
        set_trace()

    


    
    try:
        sol = root_scalar(find_Q, bracket=[Q_min, Q_max], method='brentq')
        Q = sol.root
    except:
        find_Q(Q_max)
        mdot_cold = 0.001/(cold_outlet._H - cold_inlet._H)
        #results = T_dQ(1,hot_in=hot_inlet,cold_in=cold_inlet,mdot_cold=mdot_cold, mdot_hot=mdot_hot,dP_hot=dP_hot*.9,dP_cold=1e5)
        from pdb import set_trace
        set_trace()

    return Q, Q/(cold_outlet._H - cold_inlet._H)




#def T_pinch_counter_flow(cold_inlet, cold_outlet, hot_inlet, hot_outlet, mdot_cold, mdot_hot)



def find_Pinch_Q(cold_inlet, cold_outlet, hot_inlet, mdot_hot, dP_hot, T_pinch):
    """
    Find the heat transfer rate that results in a pinch point temperature difference of T_pinch
    """

    # Get Q_max
    hot_out = hot_inlet.from_state(hot_inlet.state)
    hot_out._TP = cold_inlet._T + T_pinch, hot_inlet._P - dP_hot
    Q_max = mdot_hot * (hot_inlet._H - hot_out._H)
    Q_min = 0.001

    # Get max and min cold_inlet mdots
    mdot_cold_max = Q_max/(cold_outlet._H - cold_inlet._H)
    mdot_cold_min = Q_min/(cold_outlet._H - cold_inlet._H)


    if cold_inlet._P >= cold_inlet._P_crit:
        from pdb import set_trace
        set_trace()


    from pdb import set_trace
    set_trace()


@inputParser
def T_dQ_plotter(Q, mdot_hot:'MASSFLOW', mdot_cold:'MASSFLOW', n_points, config='counter',
                 hot_in=None, hot_out=None,
                 cold_in=None, cold_out=None,
                 dP_hot:'PRESSURE'=0, dP_cold:'PRESSURE'=0,
                 figsize=(12, 12),
                 show_plot=True):
    """
    Plot temperature and pressure profiles for a heat exchanger with pinch point analysis.
    
    Parameters:
        Q (float): Heat transfer rate (W)
        mdot_hot (float): Mass flow rate of hot fluid (kg/s)
        mdot_cold (float): Mass flow rate of cold fluid (kg/s)
        n_points (int): Number of points for calculation
        config (str): 'counter' or 'parallel' flow configuration
        hot_in (thermo): Hot fluid inlet state
        hot_out (thermo): Hot fluid outlet state
        cold_in (thermo): Cold fluid inlet state
        cold_out (thermo): Cold fluid outlet state
        dP_hot (float): Pressure drop in hot fluid side (Pa)
        dP_cold (float): Pressure drop in cold fluid side (Pa)
        T_unit (str): Temperature unit for plotting ('K' or '°C')
        figsize (tuple): Figure size (width, height)
        show_plot (bool): Whether to display the plot
    
    Returns:
        tuple: (fig, (ax1, ax2)) matplotlib figure and axes objects
        dict: Results dictionary containing temperature and pressure profiles
    """
    import matplotlib.pyplot as plt
    import numpy as np

    # Get temperature and pressure profiles
    results = T_dQ(Q, mdot_hot, mdot_cold, n_points, config,
                  hot_in=hot_in, cold_in=cold_in,
                  dP_hot=dP_hot, dP_cold=dP_cold)

    # Converts units from SI
    results['T_hot'] = fromSI(results['T_hot'], 'TEMPERATURE')
    results['T_cold'] = fromSI(results['T_cold'], 'TEMPERATURE')
    results['P_hot'] = fromSI(results['P_hot'], 'PRESSURE')
    results['P_cold'] = fromSI(results['P_cold'], 'PRESSURE')
    results['Q'] = fromSI(results['Q'], 'POWER')

    # Get display units
    T_unit = units._output_units_for_display['TEMPERATURE']
    P_unit = units._output_units_for_display['PRESSURE']
    Q_unit = units._output_units_for_display['POWER']

    # Create figure with 2 subplots vertically stacked with extra space on right
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=figsize, height_ratios=[3, 2])

    position = np.linspace(0, 100, len(results['T_hot']))
    T_hot = results['T_hot']
    T_cold = results['T_cold']

    # Calculate pinch point
    temp_diff = T_hot - T_cold
    pinch_point = min(temp_diff)
    pinch_idx = np.argmin(temp_diff)
    pinch_position = position[pinch_idx]

    # Plot temperature profiles
    ax1.plot(position, T_hot, 'ro-', label='Hot fluid', linewidth=2)
    ax1.plot(position, T_cold, 'bo-', label='Cold fluid', linewidth=2)

    # Plot pressure profiles
    P_hot = results['P_hot']
    P_cold = results['P_cold']
    ax2.plot(position, P_hot, 'r-', label='Hot fluid', linewidth=2)
    ax2.plot(position, P_cold, 'b-', label='Cold fluid', linewidth=2)

    # Plot pinch point
    ax1.plot(pinch_position, T_hot[pinch_idx], 'go', markersize=8, label='Pinch point')
    ax1.plot(pinch_position, T_cold[pinch_idx], 'go', markersize=8)

    # Move legend and pinch text to the right side
    ax1.legend(bbox_to_anchor=(1.02, 1), loc='upper left')
    
    # Add pinch point text box to the right of the plot, below the legend
    pinch_text = (f"Pinch Point Analysis:\n"
                 f"Position: {pinch_position:.1f}% Q\n"
                 f"Hot T: {T_hot[pinch_idx]:.1f}{T_unit}\n"
                 f"Cold T: {T_cold[pinch_idx]:.1f}{T_unit}\n"
                 f"ΔT: {pinch_point:.1f}{T_unit}\n"
                 f"Q at pinch: {(Q * pinch_position/100):.1f} {Q_unit}")

    ax1.text(1.02, 0.7, pinch_text,
             transform=ax1.transAxes,
             bbox=dict(facecolor='white', edgecolor='black', alpha=0.7),
             fontsize=9,
             verticalalignment='top')
    
    # Add inlet annotations - positioned directly above/below points with offset based on data range
    y_range = max(max(T_hot), max(T_cold)) - min(min(T_hot), min(T_cold))
    y_offset = y_range * 0.1  # Use 10% of the y-range for offset

    # Add inlet annotations
    ax1.annotate(f'{T_hot[0]:.1f}{T_unit}', 
                xy=(0, T_hot[0]),
                xytext=(0, y_offset),
                textcoords='offset points',
                ha='center', 
                va='bottom',
                arrowprops=dict(arrowstyle='->', color='red'))

    ax1.annotate(f'{T_cold[0]:.1f}{T_unit}', 
                xy=(0, T_cold[0]),
                xytext=(0, -y_offset),
                textcoords='offset points',
                ha='center', 
                va='top',
                arrowprops=dict(arrowstyle='->', color='blue'))

    # Add outlet annotations
    ax1.annotate(f'{T_hot[-1]:.1f}{T_unit}', 
                xy=(100, T_hot[-1]),
                xytext=(0, y_offset),
                textcoords='offset points',
                ha='center', 
                va='bottom',
                arrowprops=dict(arrowstyle='->', color='red'))

    ax1.annotate(f'{T_cold[-1]:.1f}{T_unit}', 
                xy=(100, T_cold[-1]),
                xytext=(0, -y_offset),
                textcoords='offset points',
                ha='center', 
                va='top',
                arrowprops=dict(arrowstyle='->', color='blue'))

    # Add padding to y-axis limits to accommodate annotations
    y_min = min(min(T_cold), min(T_hot))
    y_max = max(max(T_cold), max(T_hot))
    padding = y_range * 0.15  # 15% padding
    ax1.set_ylim(y_min - padding, y_max + padding)

    # Add pressure annotations
    P_hot_in = P_hot[0]
    P_hot_out = P_hot[-1]
    P_hot_pinch = P_hot[pinch_idx]
    P_cold_pinch = P_cold[pinch_idx]
    # Customize temperature subplot
    ax1.set_title(f'Heat Exchanger Temperature Profile\nQ = {Q:.1f} {Q_unit}',
                 fontsize=14, pad=20)
    ax1.set_xlabel('Q (%)', fontsize=12)
    ax1.set_ylabel(f'Temperature ({T_unit})', fontsize=12)
    ax1.grid(True)

    # Customize pressure subplot
    ax2.set_title('Pressure Profile', fontsize=14)
    ax2.set_xlabel('Q (%)', fontsize=12)
    ax2.set_ylabel(f'Pressure ({P_unit})', fontsize=12)
    ax2.grid(True)

    # Update the layout to make room for the legend and text box
    plt.subplots_adjust(right=0.85)
    
    if show_plot:
        plt.show()
    
    return fig, (ax1, ax2), {
        **results,
        'pinch_point': pinch_point,
        'pinch_position': pinch_position,
        'pinch_idx': pinch_idx
    }


def plot_HX_profile(T_hot_in, T_hot_out, T_cold_out, Q, P_hot, P_cold, 
                    mdot_hot, mdot_cold,
                    fluid_hot='water', fluid_cold='water', n_points=100,
                    config='counter', plot=True):
    """
    Plots temperature profile in a heat exchanger with possible phase change.
    
    Parameters:
        T_hot_in (float): Hot fluid inlet temperature (K)
        T_hot_out (float): Hot fluid outlet temperature (K)
        T_cold_out (float): Cold fluid outlet temperature (K)
        Q (float): Heat transfer rate (W)
        P_hot (float): Hot fluid pressure (Pa)
        P_cold (float): Cold fluid pressure (Pa)
        mdot_hot (float): Mass flow rate of hot fluid (kg/s)
        mdot_cold (float): Mass flow rate of cold fluid (kg/s)
        fluid_hot (str): Hot fluid type, default 'water'
        fluid_cold (str): Cold fluid type, default 'water'
        n_points (int): Number of points for plotting
        config (str): 'counter' or 'parallel' flow configuration
        plot (bool): Whether to show the plot
    
    Returns:
        tuple: (fig, ax) matplotlib objects if plot=True
        dict: Temperature profiles and position data
    """
    import matplotlib.pyplot as plt
    
    # Initialize fluid states
    hot_in = thermo(fluid_hot)
    hot_in._TP = T_hot_in, P_hot
    
    hot_out = thermo(fluid_hot)
    hot_out._TP = T_hot_out, P_hot
    
    cold_out = thermo(fluid_cold)
    cold_out._TP = T_cold_out, P_cold
    
    # Calculate total enthalpy change
    dH_total = hot_in._H - hot_out._H
    
    # Create position array (0 to 1)
    x = np.linspace(0, 1, n_points)
    
    # Initialize temperature arrays
    T_hot = np.zeros(n_points)
    T_cold = np.zeros(n_points)
    
    # Calculate cold inlet temperature based on energy balance
    cold_in = thermo(fluid_cold)
    H_cold_in = cold_out._H - Q/mdot_cold
    cold_in._HP = H_cold_in, P_cold
    
    # Calculate temperature profiles
    for i, pos in enumerate(x):
        # Hot fluid profile
        hot_state = thermo(fluid_hot)
        H_hot = hot_in._H - pos * Q/mdot_hot
        hot_state._HP = H_hot, P_hot
        T_hot[i] = hot_state._T
        
        # Cold fluid profile (counter or parallel flow)
        cold_state = thermo(fluid_cold)
        if config == 'counter':
            H_cold = cold_in._H + (1 - pos) * Q/mdot_cold  # Use (1-pos) for counter-flow
        else:  # parallel flow
            H_cold = cold_in._H + pos * Q/mdot_cold
        cold_state._HP = H_cold, P_cold
        T_cold[i] = cold_state._T
    
    if plot:
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(x, T_hot - 273.15, 'r-', label=f'{fluid_hot} (hot) - {mdot_hot:.2f} kg/s')
        ax.plot(x, T_cold - 273.15, 'b-', label=f'{fluid_cold} (cold) - {mdot_cold:.2f} kg/s')
        
        ax.set_xlabel('Relative position in heat exchanger')
        ax.set_ylabel('Temperature (°C)')
        ax.set_title('Temperature Profile in Heat Exchanger')
        ax.grid(True)
        ax.legend()
        
        return fig, ax
    
    return {
        'position': x,
        'T_hot': T_hot,
        'T_cold': T_cold,
        'T_cold_in': cold_in._T  # Added cold inlet temperature to output
    }


def T_dQ_pinch(mdot_hot, mdot_cold, dT_pinch, config='counter',
               hot_in=None, cold_in=None, hot_out=None, cold_out=None,
               dP_hot=0, dP_cold=0, n_points=100):
    """
    Calculates heat exchanger profile and total heat transfer based on a specified
    minimum temperature difference (pinch point).

    Parameters:
        mdot_hot (float): Mass flow rate of hot fluid (kg/s)
        mdot_cold (float): Mass flow rate of cold fluid (kg/s)
        dT_pinch (float): Minimum temperature difference allowed (K)
        config (str): 'counter' or 'parallel' flow configuration
        hot_in (thermo, optional): Hot fluid inlet state
        cold_in (thermo, optional): Cold fluid inlet state
        hot_out (thermo, optional): Hot fluid outlet state
        cold_out (thermo, optional): Cold fluid outlet state
        dP_hot (float): Pressure drop in hot fluid side (Pa)
        dP_cold (float): Pressure drop in cold fluid side (Pa)
        n_points (int): Number of points for calculation

    Returns:
        dict: Dictionary containing temperature and pressure profiles, heat transfer values,
              and fluid states at inlet/outlet
    """
    # Validate inputs
    if (hot_in is None and hot_out is None) or (cold_in is None and cold_out is None):
        raise ValueError("Must specify either inlet or outlet state for both hot and cold fluids")

    # Calculate missing states if needed
    if hot_in is not None:
        if hot_out is None:
            hot_out = thermo.from_state(hot_in.state)
    else:
        hot_in = thermo.from_state(hot_out.state)    

    if cold_in is not None:
        if cold_out is None:
            cold_out = thermo.from_state(cold_in.state)
    else:
        cold_in = thermo.from_state(cold_out.state)

    def _check_pinch(Q_test):
        # Use T_dQ to get the temperature profiles for a given Q
        profiles = T_dQ(Q_test, mdot_hot, mdot_cold, n_points, config,
                       hot_in=hot_in, cold_in=cold_in,
                       dP_hot=dP_hot, dP_cold=dP_cold)

        # Calculate temperature difference at all points
        dT = profiles['T_hot'] - profiles['T_cold']

        if any(dT < 0):
            # Heat Flux is too high
            return np.min(dT)*1e5

        # Return difference between minimum temperature difference and target pinch
        return np.min(dT) - dT_pinch

    # Initial guess for maximum possible heat transfer
    # Based on hot fluid cooling to cold inlet temperature plus pinch
    hot_out._TP = cold_in._T + dT_pinch, hot_in._P - dP_hot

    Q_max = mdot_hot * (hot_in._H - hot_out._H)

    # Solve for Q that gives exactly the pinch point temperature difference
    sol = root_scalar(_check_pinch, bracket=[0, Q_max], method='brentq')
    Q = sol.root
    
    # Get final profiles using the solved Q
    results = T_dQ(Q, mdot_hot, mdot_cold, n_points, config,
                   hot_in=hot_in, cold_in=cold_in,
                   dP_hot=dP_hot, dP_cold=dP_cold)
    
    # Add pinch point information to results
    dT = results['T_hot'] - results['T_cold']
    pinch_idx = np.argmin(dT)
    results['pinch_location'] = pinch_idx / (n_points - 1)
    results['dT_pinch_actual'] = dT[pinch_idx]
    
    return results
