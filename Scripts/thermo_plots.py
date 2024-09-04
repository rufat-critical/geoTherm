import matplotlib.pyplot as plt
from geoTherm import thermo, units
import numpy as np


def thermo_plot(fluid, properties, 
                state_points=None,
                xscale='linear',
                yscale='linear',
                xlim=None,
                ylim=None,
                isolines=None):
                      
    # Generate thermodynamic plot
    
    # Create a geoTherm thermo object
    thermostate = thermo(fluid=fluid)
    
    # Critical point values
    T_critical = thermostate.pObj.cpObj.T_critical()
    P_critical = thermostate.pObj.cpObj.p_critical()
    Tmin = thermostate.pObj.cpObj.Tmin()
    Tmax = thermostate.pObj.cpObj.Tmax()
    # Temperature sweep
    T_sweep = np.linspace(Tmin, T_critical, 500)
    
    # Initialize arrays with np.nan
    T_liq = np.full(len(T_sweep), np.nan)
    T_vap = np.full(len(T_sweep), np.nan)
    P_liq = np.full(len(T_sweep), np.nan)
    P_vap = np.full(len(T_sweep), np.nan)
    S_liq = np.full(len(T_sweep), np.nan)
    S_vap = np.full(len(T_sweep), np.nan)
    H_liq = np.full(len(T_sweep), np.nan)
    H_vap = np.full(len(T_sweep), np.nan)
    D_liq = np.full(len(T_sweep), np.nan)
    D_vap = np.full(len(T_sweep), np.nan)
    
   
    # Sweep for vapor dome
    for i, T in enumerate(T_sweep):
        
        try:
            # Set thermostate to saturated liquid
            thermostate._TQ = T, 0
            # Store satured liquid properties
            T_liq[i] = thermostate.T
            P_liq[i] = thermostate.P
            S_liq[i] = thermostate.S
            H_liq[i] = thermostate.H
            D_liq[i] = thermostate.density
        except ValueError:
            pass

        try:
            # Set thermostate to saturated vapor
            thermostate._TQ = T, 1
            # Store satured vapor
            T_vap[i] = thermostate.T
            P_vap[i] = thermostate.P
            S_vap[i] = thermostate.S
            H_vap[i] = thermostate.H
            D_vap[i] = thermostate.density
        except ValueError:
            pass         
    
    Units = {'T': units.outputUnits['TEMPERATURE'],
             'P': units.outputUnits['PRESSURE'],
             'H': units.outputUnits['SPECIFICENERGY'],
             'S': units.outputUnits['SPECIFICENTROPY'],
             'D': units.outputUnits['DENSITY']}
    
    VaporProps = {'T': [T_liq, T_vap],
             'P': [P_liq, P_vap],
             'H': [H_liq, H_vap],
             'S': [S_liq, S_vap],
             'D': [D_liq, D_vap],
             'v': [1/D_liq, 1/D_vap]}
    
    PropNames = {'T': f"Temperature [{units.outputUnits['TEMPERATURE']}]",
                 'P': f"Pressure [{units.outputUnits['PRESSURE']}]",
                 'H': f"Enthalpy [{units.outputUnits['SPECIFICENERGY']}]",
                 'S': f"Entropy [{units.outputUnits['SPECIFICENTROPY']}]",
                 'D': f"Density [{units.outputUnits['DENSITY']}]",
                 'v': f"Specific Volume [1/{(units.outputUnits['DENSITY'])}]"}
    
    # Properties we are plotting
    xprop = properties[1]
    yprop = properties[0]
    
    # Create Figure
    fig = plt.figure()

    # Plot Vapor Dome
    # Liq
    plt.plot(VaporProps[xprop][0], VaporProps[yprop][0], color='k')
    # Vap
    plt.plot(VaporProps[xprop][1], VaporProps[yprop][1], color='k')
    
    # Colors to plot the statepoints and isolines with
    # Thanks to chatgpt for the list
    color = ['blue', 'orange', 'green', 'red', 'purple', 'brown',
             'pink', 'gray', 'olive', 'cyan', '#e6194b', '#3cb44b',
             '#ffe119', '#4363d8', '#f58231']
    
    
    # Add state points
    for i, (name, state) in enumerate(state_points.items()):
        # Update thermostate
        thermostate.update_state(state)
        
        # Get the state point properties
        x = getattr(thermostate, xprop)
        y = getattr(thermostate, yprop)

        plt.plot(x, y, 'o', color=color[i])
        # Add a label with the name
        plt.text(x, y, name)
  
    
    # Calculate Isolines and store in iso_dat dictionary
    iso_dat = {}
    if isolines is not None:
        if isinstance(isolines, str):
            for i, (name, state) in enumerate(state_points.items()):
                thermostate.update_state(state)
                iso_val = thermostate._get_property(isolines)
                # Store isolines data in a dictionary
                
                state = {'T':Tmin+5, isolines:iso_val}
                thermostate._update_state(state)
                xmin = thermostate._get_property(xprop)
                state = {'T':Tmax, isolines:iso_val}
                thermostate._update_state(state)
                xmax = thermostate._get_property(xprop)
                x_sweep = np.linspace(xmin, xmax, 500)

                iso_dat[iso_val] ={xprop: np.full(len(x_sweep), np.nan),
                                   yprop: np.full(len(x_sweep), np.nan)}

                for j, x in enumerate(x_sweep):
                    state = {xprop: x, isolines:iso_val}
                    
                    try:
                        thermostate._update_state(state)
                        iso_dat[iso_val][xprop][j] = thermostate.get_property(xprop)
                        iso_dat[iso_val][yprop][j] = thermostate.get_property(yprop)
                    except:
                        pass
                
    # Plot the isolines
    for i, (isoval, x) in enumerate(iso_dat.items()):
        plt.plot(x[xprop], x[yprop], ':', alpha=0.5, color=color[i])
        
    # Make Label with proper units
    plt.xlabel(PropNames[xprop])
    plt.ylabel(PropNames[yprop])
    
    #Set Scaling
    plt.xscale(xscale)
    plt.yscale(yscale)
    
    # Set limits
    if xlim is not None:
        plt.xlim(xlim)
    if ylim is not None:
        plt.ylim(ylim)
  
    return fig


if __name__ == "__main__":
    
    state_points = {'1': {'T': (30, 'degC'), 'P': (1.1, 'atm')},
                    '2': {'T': (33, 'degC'), 'P': (26, 'bar')},
                    '3': {'T': (180 ,'degC'), 'P': (18, 'bar')},
                    '4': {'T': (370, 'degK'), 'P': (2, 'bar')}}
                    
    units.output = 'mixed'    
    fig = thermo_plot('acetone', 'TS', state_points, isolines='P')
    
    plt.show()


