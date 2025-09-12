import numpy as np
import geoTherm as gt
from geoTherm.utilities.HEX.pinch import PinchSolver
from geoTherm.utilities.HEX.profile import HEXAnalyzer
from matplotlib import pyplot as plt
from pdb import set_trace
import pandas as pd


# Create Thermo Objects
water = gt.thermo(model='incompressible', cp=4184)
isobutane = gt.thermo()
isobutane.TPY = 300, 101325, 'isobutane:1'
air = gt.thermo()
air.TPY = 300, 101325, 'air'

# Initialize Pinch Solver
WaterPinch = PinchSolver(cold_fluid=isobutane, hot_fluid=water)
RecupPinch = PinchSolver(cold_fluid=isobutane, hot_fluid=isobutane)
AirPinch = PinchSolver(cold_fluid=air, hot_fluid=isobutane)

water_out_fixed = water.copy()
water_out_fixed.TP = (85, 'degC'), 101325


# Load Steves Maps
map1 = 'maps\\20250904-Cascade_Fervo_4stg_Prelim-Maps 1.xlsx'
map2 = 'maps\\20250904-Cascade_Fervo_4stg_Prelim-Maps 2.xlsx'
# Create pandas data frame with the data
map_data = gt.utilities.concepts.excel_reader([map1, map2])
# We can save the reduced data frame as csv
map_data.to_csv('turbine_map.csv')

# Initialize Turbine Map
SteveMap = gt.maps.TurbineMap(map_data)

# Solve assuming 1 bar drop
def make_model(fluid, T_hot, T_cold, W_Turb,
               eta_pump,
               dP_hot=(-1, 'bar'),
               dP_cold=(-1, 'bar'),):

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


    water_in = gt.thermo(model='incompressible', cp=4184)
    # Set the object to T_hot temperature, pressure is arbitrary since
    # object is incompressible
    water_in.TP = (T_hot, 'degC'), 101325
    water_out = water_in.copy()


    def Q_hot(hot, cool, model):
        
        # Get US and DS nodes for cool node
        # US is Pump Outlet
        pumpOut = cool.US_node.thermo
        # DS is Turbine Inlet
        TurbineIn = cool.DS_node.thermo
        
        # Heat needed to bring state to TurbineIn
        return cool._w*(TurbineIn._H - pumpOut._H)

    def Q_cold(hot, cool, model):
        ColdHEX = model['Cold_HEX_In'].thermo
        PumpIn = model['PumpIn'].thermo
        w = model['Turb']._w
        return w*(ColdHEX._H-PumpIn._H)

    def water_flow(US, DS, model):
        return US.thermo._T

    ORC = gt.Model([gt.Boundary(name='PumpIn', fluid=pump_inlet),
              gt.FixedFlowPump(name='Pump', rotor='PumpRotor', eta=eta_pump, w=1, controller='Turb', US='PumpIn', DS='PumpOut'),
              # This is just a dummy rotor and not used in the calculations
              gt.Rotor(name='PumpRotor', N=100),
              gt.Station(name='PumpOut', fluid=fluid),
              gt.FixedDP(name='Recup_Cold', US='PumpOut', DS='Hot_HEX_In', w=1, dP=0),
              gt.Station(name='Hot_HEX_In', fluid=fluid),
              gt.Qdot(name='Heat', cool='Hot-HEX', hot='WaterHEX', Q=Q_hot),
              gt.FixedDP(name='Hot-HEX', US='Hot_HEX_In', DS='TurbIn', w=1, dP=dP_hot),
              gt.Boundary(name='TurbIn', fluid=fluid, T=(T_hot-5, 'degC'), P=(55,'bar')),
              #gt.FixedFlowTurbine(name='Turb', US='TurbIn', DS='TurbOut', rotor='TurbRotor', eta=0.88, w=10),
              gt.TurbineMap(name='Turb', US='TurbIn', DS='TurbOut', rotor='TurbRotor', map=SteveMap),
              gt.Generator(name='Generator', rotor='TurbRotor',
                           drag_loss=.045,
                           gearbox_loss=0.02,
                           generator_loss=0.035),
              gt.Rotor(name='TurbRotor', N=18000),
              gt.Station(name='TurbOut',fluid=fluid),
              gt.FixedDP(name='Recup_Hot', US='TurbOut', DS='Cold_HEX_In', w=1, dP=0),
              gt.Station(name='Cold_HEX_In', fluid=fluid),
              gt.Qdot(name='Recup_Heat', cool='Recup_Cold', hot='Recup_Hot', Q=0),
              gt.FixedDP(name='ColdHEX',US='Cold_HEX_In',DS='PumpIn', w=1, dP=dP_cold),
              gt.Boundary(name='Well', fluid=water_in),
              gt.FixedFlow(name='WaterHEX', US='Well', DS='WaterOut', w=5),
              gt.POutlet(name='WaterOut', fluid=water_out),
              gt.Boundary(name='Ambient', fluid='air', T=(T_cold, 'degC'), P=101325),
              gt.FixedDP(name='AirHEX', US='Ambient', DS='FanInlet', dP=-200),
              gt.Station(name='FanInlet', fluid='air'),
              gt.Qdot(name='Heat_Reject', hot='ColdHEX', cool='AirHEX', Q=Q_cold),
              gt.FixedFlowFan(name='Fan', US='FanInlet', DS='HotAir', rotor='FanRotor', eta=.6, w=1000),
              gt.Rotor(name='FanRotor', N=1000),
              gt.POutlet(name='HotAir', fluid='air')]) 

    return ORC


def ORC_solve(P, ORC, recup=True):

    if P < ORC['PumpIn'].P*1e-5:
        return False
    
    # Set and solve 
    ORC['TurbIn'].thermo.TP = ORC['TurbIn'].T, (P,'bar')
    ORC.solve_steady()

    if recup:
        recup_pinch = RecupPinch.get_pinch_Q(T_pinch=5,
                                        cold_inlet=ORC['PumpOut'].thermo,
                                        hot_inlet=ORC['TurbOut'].thermo,
                                        w_cold=ORC['Turb']._w,
                                        w_hot=ORC['Turb']._w)

        ORC['Recup_Heat']._Q = recup_pinch['Q']
    else:
        ORC['Recup_Heat']._Q = 0

    ORC.solve_steady()
    water_pinch = WaterPinch.get_pinch_Q(T_pinch=5,
                               cold_inlet=ORC['Hot_HEX_In'].thermo,
                               cold_outlet=ORC['TurbIn'].thermo,
                               w_cold = ORC['Turb']._w,
                               hot_inlet=ORC['Well'].thermo)    

    # Get Required Water Flow Rate for Pinch
    ORC['WaterHEX']._w = water_pinch['w_hot']
    ORC.solve_steady()

    
    if ORC['WaterOut'].T < 85+273.15:
        # This is to ensure water is above 85 degC
        water_out = ORC['WaterOut'].thermo.copy()
        water_out.TP = (85, 'degC'), 101325
        w_water = ORC['Heat']._Q/(ORC['Well'].thermo._H-water_out._H)
        ORC['WaterHEX']._w = w_water
        ORC.solve_steady()
    
    air_pinch = AirPinch.get_pinch_Q(T_pinch=5,
                                    cold_inlet=ORC['Ambient'].thermo,
                                    hot_outlet=ORC['PumpIn'].thermo,
                                    hot_inlet=ORC['Cold_HEX_In'].thermo,
                                    w_hot=ORC['Turb']._w)
    
    # Get Required Air Flow Rate for Pinch
    ORC['Fan']._w = air_pinch['w_cold']
    
    # This was scaling to 2.7 MW before using Steve's maps
    #scale = 2.7e6/ORC['Turb'].W
    #ORC['Turb']._w *= scale
    #ORC['WaterHEX']._w *= scale
    #ORC['Recup_Heat']._Q *= scale
    #ORC['Fan']._w *= scale

    ORC.solve_steady()

    return True


def sweep(ORC, recup_flag):

    solution = gt.Solution(ORC, extras=['T_pinch_water', 'T_pinch_recup', 'T_pinch_air'])
    
    # Create Pinch Analysis Objects
    WaterHEX = HEXAnalyzer(ORC['Heat'])
    RecupHEX = HEXAnalyzer(ORC['Recup_Heat'])
    AirHEX = HEXAnalyzer(ORC['Heat_Reject'])
    for i, P in enumerate(pressures):
        print(f"Solving for P = {P:.1f} bar ({i+1}/{len(pressures)})")

        if P < ORC['PumpIn'].P*2*1e-5:
            continue

        if recup_flag == 0:
            ORC_solve(P, ORC, recup=False)
        elif recup_flag == 1:
            ORC_solve(P, ORC, recup=True)

        if ORC['TurbIn'].phase in ['liquid','supercritical_liquid']:
            break
        if ORC['TurbOut'].phase in ['liquid','supercritical_liquid','two-phase']:
            break
        
        
        solution.append(ORC, extras={'T_pinch_water': WaterHEX.T_pinch,
                                     'T_pinch_recup': RecupHEX.T_pinch,
                                     'T_pinch_air': AirHEX.T_pinch})
    

    return solution


# Make logger silent to not output intermediate warnings
gt.logger.set_level('silent')

# Sweep thru parameters
T_hot_sweep = np.array([165, 180, 200, 220])
T_cold_sweep = np.array([0, 12, 21, 35])

fluid='Isobutane'
dP_HEX_sweep = np.array([0.3])
pressures = np.linspace(6, 60, 40)


for T_hot in T_hot_sweep:
    for T_cold in T_cold_sweep:
        for dP_HEX in dP_HEX_sweep:
            print(f"T_hot: {T_hot}, T_cold: {T_cold}, dP_HEX: {dP_HEX}")
            
            # Create Models
            ORC_recup = make_model(fluid,
                                   T_hot=T_hot,
                                   T_cold=T_cold,
                                   W_Turb=2.7e6,
                                   eta_pump=.8,
                                   dP_hot=(-dP_HEX,'bar'),
                                   dP_cold=(-dP_HEX,'bar'))
            
            ORC_no_recup = make_model(fluid,
                 T_hot=T_hot,
                 T_cold=T_cold,
                 W_Turb=2.7e6,
                 eta_pump=.8,
                 dP_hot=(-dP_HEX,'bar'),
                 dP_cold=(-dP_HEX,'bar'))


            print('Solving Recuperated Case')
            solution_recup = sweep(ORC_recup, 1)
            print('Solving Simplified Case')
            solution_no_recup = sweep(ORC_no_recup, 0)

            gt.units.output = 'mixed'
            solution_no_recup.save_csv(f'{fluid}_no_recup_{T_hot}_{T_cold}_{dP_HEX}_steve.csv')
            solution_recup.save_csv(f'{fluid}_recup_{T_hot}_{T_cold}_{dP_HEX}_steve.csv')
            gt.units.output = 'SI'