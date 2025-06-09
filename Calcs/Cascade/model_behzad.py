from matplotlib import pyplot as plt
from scipy.optimize import fsolve
import numpy as np
import pandas as pd
import geoTherm as gt
from geoTherm import units
from geoTherm import Claudio_Turbine
from geoTherm.flow_funcs import IsentropicFlow, FlowModel, _dH_isentropic
from geoTherm.utilities.HEX.pinch import find_pinch_Q, find_pinch_Q_hot, find_pinch_T
from scipy.optimize import root_scalar, fsolve
from geoTherm.utilities.heat_transfer import T_dQ_plotter
from geoTherm.utilities.HEX.profile import HEXProfile
from scipy.optimize import minimize_scalar
import CoolProp.CoolProp as CP
from geoTherm.utilities.HEX import LMTD
import os
from matplotlib import pyplot as plt
import matplotlib.pyplot as plt
import seaborn as sns
from utils import post_process


# Load Behzad Axial Turbine Map
Behzad_Map = gt.maps.concepts_map.ConceptsMap('Prelim Axial Turbine Summary_Sweeps.xlsx')

# Solve assuming 1 bar drop
def make_model(fluid, T_hot, T_cold, W_Turb,
               eta_pump,
               dP_hot=(-1, 'bar'),
               dP_cold=(-1, 'bar')):

    # Get Pump Inlet State:
    # T = T_cold + 5deg Pinch
    # P = Pvap at T_cold + 10 degree
    # Create a pump inlet thermo object
    pump_inlet = gt.thermo()
    # Set the inlet temp to T_cold + 10, Pressure is arbitrary
    pump_inlet.TPY = (T_cold+10, 'degC'), 101325, fluid
    # Get the vapor pressure at this temperature
    Pvap = pump_inlet.Pvap
    # Set the pump inlet state to T_cold+5 and calculated Pvap
    pump_inlet.TP = (T_cold+5, 'degC'), Pvap

    # Now Create the model for Turbine Inlet => Pump Inlet
    Turbo = gt.Model([gt.Boundary(name='TurbIn', fluid=fluid, T=(T_hot-5, 'degC'), P=(55,'bar')),
                    gt.TurbineMap(name='Turb', US='TurbIn', DS='TurbOut', map=Behzad_Map),
                    gt.Station(name='TurbOut',fluid=fluid),
                    gt.Pipe(name='ColdHEX',US='TurbOut',DS='PumpIn',w=1,dP=0),
                    gt.Boundary(name='PumpIn', fluid=pump_inlet)])

    Turbo['ColdHEX'].dP = dP_cold

    def find_P(P):
        Turbo['TurbIn'].thermo.TP =Turbo['TurbIn'].T, (P, 'bar')
        Turbo.solve_steady()
        return W_Turb - Turbo['Turb'].W

    def find_P_fsolve(x):
        Turbo['TurbIn'].thermo.TP =Turbo['TurbIn'].T, (x[0], 'bar')
        Turbo.solve_steady()
        return W_Turb - Turbo['Turb'].W
 
    
    # Solve at 70 to get IC
    Turbo['TurbIn'].thermo.TP =Turbo['TurbIn'].T, (85, 'bar')
    Turbo.solve_steady()

    if Turbo['Turb'].W > 3.5e6:    
        try:
            sol = root_scalar(find_P, bracket=[20, 85], method='brentq')
            P_opt = sol.root
        except:
            from pdb import set_trace
            set_trace()
    else:
        print(f"MAX POWER for Turbo at {T_hot} is {Turbo['Turb'].W*1e-6} MW")

    if Turbo['TurbOut'].phase != 'gas':
        print(f"Turbo Outlet is 2-phase at {Turbo['TurbIn'].P*1e-5} bar")
        # Drop Turbo Pressure by 1 bar until we are not 2-phase
        for i in range(0,30):
            Turbo['TurbIn'].thermo._TP =Turbo['TurbIn'].T, Turbo['TurbIn'].thermo._P-1e5
            Turbo.solve_steady()
            if Turbo['TurbOut'].phase == 'gas':
                print(f"Turbo Outlet is gas at {Turbo['TurbIn'].P*1e-5} bar")
                print(f"New Power is {Turbo['Turb'].W*1e-6} MW")
                break
        
        if i >25:
            from pdb import set_trace
            set_trace()
        P_opt = Turbo['TurbIn'].P
            
    

    def Q_hot(hot,cool):
        
        # Get US and DS nodes for cool node
        # US is Pump Outlet
        pumpOut = cool.US_node.thermo
        # DS is Turbine Inlet
        TurbineIn = cool.DS_node.thermo
        
        # Heat needed to bring state to TurbineIn
        return cool._w*(TurbineIn._H - pumpOut._H)

    # Generate the rest of the model
    ORC = gt.Model([gt.fixedFlowPump(name='Pump',
                                     eta=eta_pump,
                                     w=Turbo['Turb'].w,
                                     US='PumpIn',
                                     DS='PumpOut'),
                    gt.Station(name='PumpOut', fluid=fluid),
                    gt.Qdot(name='Heat', cool='HotHEX', Q=Q_hot),
                    gt.Pipe(name='HotHEX', US='PumpOut', DS='TurbIn',w=Turbo['Turb'].w,
                            dP = 0)])
                            
                            
    ORC['HotHEX'].dP = dP_hot
    # Add turbo To this model
    ORC += Turbo
    # Solve the combined Model
    ORC.solve_steady()
    
    if ORC['Turb'].W > 3.501e6:
        print("Trying fsolve")
        # Use fsolve to find the correct pressure
        P_initial = ORC['TurbIn'].P
        sol = fsolve(find_P_fsolve, [P_initial/1e5])  # Convert to bar for fsolve
        P_opt = sol[0] * 1e5  # Convert back to Pa
        ORC['TurbIn'].thermo.TP = ORC['TurbIn'].T, P_opt
        ORC.solve_steady()

    if not ORC.converged:
        from pdb import set_trace
        set_trace()

    # Now Calculate Hot Water mass flow
    water_in = gt.thermo(model='incompressible', cp=4184)
    # Set the object to T_hot temperature, pressure is arbitrary since
    # object is incompressible
    water_in.TP = (T_hot, 'degC'), 101325
    # Copy water_in object
    water_out = water_in.copy()
    # Set state to 80 degC
    water_out.TP = (80, 'degC'), 101325
    # Energy balance to calculate mdot
    w_water = ORC['Heat'].Q/(water_in._H-water_out._H)
    
    # Create water model and add it to overall ORC model
    ORC += gt.Model([gt.Boundary(name='Well', fluid=water_in),
                     gt.fixedFlow(name='WaterHEX', US='Well', DS='WaterOut',w=w_water),
                     gt.Outlet(name='WaterOut', fluid=water_out)])


    
    return ORC


def post_process(ORC):

    gt.units.output='mixed'
    data_point = {
        'T_hot': ORC['Well'].thermo.T,
        'T_cold': ORC['PumpIn'].T-5,
        'P_pump': ORC['PumpIn'].P,
        'P_turbine': ORC['TurbIn'].P,
        'dp_hot': ORC['HotHEX'].dP,
        'Turb_power': ORC['Turb'].W,
        'Pump_power': ORC['Pump'].W,
        'Turb_efficiency': ORC['Turb'].eta,
        'Turb_PR': 1/ORC['Turb'].PR,
        'Pump_PR': ORC['Pump'].PR,
        'Turb_inlet_P': ORC['TurbIn'].P,
        'Turb_inlet_T': ORC['TurbIn'].T,
        'Turb_outlet_P': ORC['TurbOut'].P,
        'Turb_outlet_T': ORC['TurbOut'].T,
        'Turb_Q_in': ORC['Turb'].Q_in,
        'Turb_Q_out': ORC['Turb'].Q_out,
        'Turb_mass_flow': ORC['Turb'].w,
        'Wnet': ORC.performance[0],
        'System_efficiency': ORC.performance[2],
        'Heat_input': ORC['Heat'].Q,
        'water_out_T': ORC['WaterOut'].thermo.T,
        'water_mass_flow': ORC['WaterHEX'].w
    }

    gt.units.output='SI'
    return data_point


# dP scaled to 1 bar at 200C and 35degree outlet
ORC = make_model('isobutane', 170, 35, 3.5e6, 0.7)

data_point = post_process(ORC)


def dP_hot(w, US, DS, w_ref=ORC['Turb']._w, rho_ref=ORC['PumpOut'].thermo._density):
    return -1e5*(w/w_ref)**2*(rho_ref/US._density)
def dP_cold(w, US, DS, w_ref=ORC['Turb']._w, rho_ref=ORC['TurbOut'].thermo._density):
    return -1e5*(w/w_ref)**2#*(rho_ref/US._density)


fervo_data = []
T_hot_range = np.arange(170, 216, 5)  # 170 to 215 in steps of 5
T_cold_values = [0, 12, 35]

# Create results directory if it doesn't exist
results_dir = 'results'
if not os.path.exists(results_dir):
    os.makedirs(results_dir)

# Sweep through temperature combinations
for T_cold in T_cold_values:
    for T_hot in T_hot_range:
        print(f'Working on: T_hot: {T_hot}, T_cold: {T_cold}')
        
        # Create model for current temperature combination
        ORC = make_model('isobutane', T_hot, T_cold, 3.5e6, 0.7, dP_hot=dP_hot,
                         dP_cold=dP_cold)
        
        # Process and store data
        data_point = post_process(ORC)
        fervo_data.append(data_point)


# Create results directory if it doesn't exist
results_dir = 'results'
if not os.path.exists(results_dir):
    os.makedirs(results_dir)

# Save Fervo case data
df_fervo = pd.DataFrame(fervo_data)
df_fervo.to_csv(os.path.join(results_dir, f"isobutane_behzad_map.csv"), index=False)
