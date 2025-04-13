from geoTherm.thermostate import thermo
import numpy as np
from scipy.optimize import root_scalar
from geoTherm.units import inputParser, fromSI, units
import matplotlib.pyplot as plt


class HEXProfile:
    def __init__(self, w_hot=None,
                 w_cold=None,
                 hot_inlet=None,
                 hot_outlet=None,
                 cold_inlet=None,
                 cold_outlet=None,
                 dP_hot=0,
                 dP_cold=0,
                 config='counterflow',):

        self.hot_inlet = hot_inlet
        self.hot_outlet = hot_outlet
        self.cold_inlet = cold_inlet
        self.cold_outlet = cold_outlet
        self.dP_hot = dP_hot
        self.dP_cold = dP_cold
        self.w_hot = w_hot
        self.w_cold = w_cold
        self.config = config


    def evaluate(self, Q, n_points=100):

        if ((self.hot_inlet is None and self.hot_outlet is None)
           or (self.cold_inlet is None and self.cold_outlet is None)):
            raise ValueError("Must specify either inlet or outlet state for "
                             "both hot and cold fluids")


        # Get Inlets and Outlet
        hot_inlet = self.hot_inlet
        hot_outlet = self.hot_outlet
        cold_inlet = self.cold_inlet
        cold_outlet = self.cold_outlet
        dP_hot = self.dP_hot
        dP_cold = self.dP_cold
        w_hot = self.w_hot
        w_cold = self.w_cold

        if hot_inlet is not None and hot_outlet is not None:

            if w_hot is None:
                w_hot = Q/(hot_inlet._H - hot_outlet._H)
            else:
                Q_hot = (hot_inlet._H - hot_outlet._H) * w_hot
                if np.abs(Q - Q_hot) > 1:
                    from pdb import set_trace
                    set_trace()

            # Get dP from the thermo states
            dP_hot = (hot_outlet._P - hot_inlet._P)

        if cold_inlet is not None and cold_outlet is not None:

            if w_cold is None:
                w_cold = Q/(cold_outlet._H - cold_inlet._H)
            else:
                Q_cold = (cold_outlet._H - cold_inlet._H) * w_cold
                if np.abs(Q - Q_cold) > 1:
                    from pdb import set_trace
                    set_trace()

            # Get dP from the thermo states
            dP_cold = (cold_outlet._P - cold_inlet._P)

        if hot_inlet is None:
            hot_inlet = self.hot_outlet.from_state(
                self.hot_outlet.state
                )
            hot_inlet._HP = (
                hot_outlet._H + Q/w_hot,
                hot_outlet._P + dP_hot/w_hot
                )

        if cold_inlet is None:
            cold_inlet = self.cold_outlet.from_state(
                self.cold_outlet.state
                )
            cold_inlet._HP = (
                cold_outlet._H - Q/w_cold,
                cold_outlet._P - dP_cold/w_cold
                )

        if self.config == 'counterflow':
            # Run counterflow analysis
            self.results = T_dQ_counter_flow(Q=Q,
                                             w_hot=w_hot,
                                             w_cold=w_cold,
                                             hot_inlet=hot_inlet,
                                             cold_inlet=cold_inlet,
                                             dP_hot=dP_hot,
                                             dP_cold=dP_cold,
                                             n_points=n_points)

            return self.results
        else:
            from pdb import set_trace
            set_trace()

    @staticmethod
    def plot_results(results, figsize=(12, 8)):
        """Static method to plot HEX results from any results dictionary"""
        if results is None:
            from pdb import set_trace
            set_trace()

        # Converts units from SI
        results['T_hot'] = fromSI(results['T_hot'], 'TEMPERATURE')
        results['T_cold'] = fromSI(results['T_cold'], 'TEMPERATURE')
        results['P_hot'] = fromSI(results['P_hot'], 'PRESSURE')
        results['P_cold'] = fromSI(results['P_cold'], 'PRESSURE')
        results['Q'] = fromSI(results['Q'], 'POWER')

        Q_tot = results['Q'][-1]

        # Get display units
        T_unit = units._output_units_for_display['TEMPERATURE']
        P_unit = units._output_units_for_display['PRESSURE']
        Q_unit = units._output_units_for_display['POWER']

        # Create figure with 2 subplots vertically stacked with extra space on right
        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=figsize, height_ratios=[3, 2, 2])

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

        Quality = results['Quality']
        ax3.plot(position, Quality, 'k.', label='Quality', linewidth=2)

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
                    f"Q at pinch: {(Q_tot * pinch_position/100):.1f} {Q_unit}")

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
        ax1.set_title(f'Heat Exchanger Temperature Profile\nQ = {Q_tot:.1f} {Q_unit}',
                    fontsize=14, pad=20)
        #ax1.set_xlabel('Q (%)', fontsize=12)
        ax1.set_ylabel(f'Temperature ({T_unit})', fontsize=12)
        ax1.grid(True)

        # Customize pressure subplot
       #ax2.set_title('Pressure', fontsize=12)
        #ax2.set_xlabel('Q (%)', fontsize=12)
        ax2.set_ylabel(f'Pressure ({P_unit})', fontsize=12)
        ax2.grid(True)

        #ax3.set_title('Quality', fontsize=12)
        ax3.set_xlabel('Q (%)', fontsize=12)
        ax3.set_ylabel('Quality', fontsize=12)
        ax3.grid(True)

        # Update the layout to make room for the legend and text box
        plt.subplots_adjust(right=0.85)

        plt.show()

    def plot(self, figsize=(10, 8)):
        """Instance method to plot results from the current HEX instance"""
        if not hasattr(self, 'results'):
            raise ValueError("No results available. Run evaluate() first.")
        return self.plot_results(self.results, figsize=figsize)
    


def T_dQ_counter_flow(Q, w_hot, w_cold,
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
    Q_array = np.zeros(n_points+1)
    T_hot = np.zeros(n_points+1)
    T_cold = np.zeros(n_points+1)
    P_hot = np.zeros(n_points+1)
    P_cold = np.zeros(n_points+1)
    Quality = np.zeros(n_points+1)

    # Create fluid state objects for calculations
    hot_state = thermo.from_state(hot_inlet.state)
    cold_state = thermo.from_state(cold_inlet.state)

    i = 0
    j = i
    while i < n_points+1:
        # Calculate pressure at current position (linear drop)

        current_Q = dQ * i
        current_P_hot = hot_inlet._P + dP_hot * (Q-current_Q)/Q
        current_P_cold = cold_inlet._P + dP_cold * dQ/Q*i

        # Update hot fluid state
        hot_state._HP = hot_inlet._H - (Q-current_Q)/w_hot, current_P_hot


        # Update cold fluid state
        prev_phase = cold_state.phase
        cold_state._HP = cold_inlet._H + current_Q/w_cold, current_P_cold

        Q_array[j] = current_Q
        T_hot[j] = hot_state._T
        P_hot[j] = current_P_hot    
        T_cold[j] = cold_state._T
        P_cold[j] = current_P_cold
        Quality[j] = cold_state.Q

        # If phase change, insert saturation point at i-1
        if cold_state.phase != prev_phase: 
            
            # Create saturation state
            sat_state = thermo.from_state(cold_state.state)
            
            # Phase change detected - insert saturation point at i-1
            if prev_phase == 'liquid' and cold_state.phase == 'two-phase':
                # Calculate state at saturation point
                sat_state._PQ = current_P_cold, 0  # saturated liquid
                i_pos = j

            elif prev_phase == 'two-phase' and cold_state.phase in ['gas', 'supercritical_gas']:
                # Found transition to vapor, calculate Q at saturation point
                sat_state._PQ = current_P_cold, 1  # saturated vapor
                i_pos = j+1

            elif prev_phase == 'supercritical_liquid' and cold_state.phase == 'supercritical':
                sat_state._TP = sat_state._T, current_P_cold
                i_pos = j
            else:
                from pdb import set_trace
                set_trace()

            q_sat = w_cold * (cold_state._H - sat_state._H)
            hot_state._HP = hot_inlet._H - (Q-current_Q-q_sat)/w_hot, current_P_hot
            # Insert saturation point at i-1
            Q_array = np.insert(Q_array, i_pos, current_Q-q_sat)
            T_hot = np.insert(T_hot, i_pos, hot_state._T)
            P_hot = np.insert(P_hot, i_pos, current_P_hot)
            T_cold = np.insert(T_cold, i_pos, sat_state._T)
            P_cold = np.insert(P_cold, i_pos, current_P_cold)
            Quality = np.insert(Quality, i_pos, sat_state.Q)
            j += 1
            #n_points += 1
            continue

        # If no phase change, continue
        i += 1
        j += 1


        # Update quality
        Quality[Quality < 0] = -1
        Quality[Quality > 1] = -1

    if Q_array[-1] != Q:
        from pdb import set_trace
        set_trace()

    return {
        'Q': Q_array,
        'T_hot': T_hot,
        'T_cold': T_cold,
        'P_hot': P_hot,
        'P_cold': P_cold,
        'Quality': Quality,
    }
