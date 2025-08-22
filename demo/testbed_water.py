import geoTherm as gt
from matplotlib import pyplot as plt
from geoTherm.utilities.display import print_model_tables
from geoTherm.utilities.HEX.pinch import PinchSolver
from geoTherm.utilities.HEX.profile import HEXProfile

# Working Fluid
fluid = 'acetone'
# Water

acetone = gt.thermo()
acetone.TPY = 303, 101325, fluid
PR_turb = 5
Pin = 3.8
w = 1.4
w_H2O = 6

def heat_rejection_acetone(hot, cool, model):
    # How much heat we need to reject to close cycle
    w = model['Pump']._w
    PumpIn_H = model['PumpIn'].thermo._H
    TurbOut_H = model['TurbOut'].thermo._H
    
    return w*(TurbOut_H - PumpIn_H)
    
def heat_rejection_water(hot, cool, model):
    # How much heat we need to reject to close cycle
    w = model['Water_Pump']._w
    WaterTank_H = model['WaterTank'].thermo._H
    COLD_HEX_OUT_H = model['COLD_HEX_Out'].thermo._H
    return w*(COLD_HEX_OUT_H - WaterTank_H)


# Well Inlet Fluid Object
water_in = gt.thermo(model='incompressible', cp=4184)
water_in.TP = (200, 'degC'), 101325
# Well Outlet
water_out = water_in.copy()

HOT = gt.Model([gt.Boundary(name='Well', fluid=water_in),
                gt.FixedFlow(name='HOT_HEX-HOT', US='Well', DS='HotOut', w=50),
                gt.POutlet(name='HotOut', fluid=water_out)])

ORC = gt.Model([gt.Boundary(name='PumpIn', fluid=fluid),
                gt.FixedFlowPump(name='Pump', rotor='DummyRotor', eta=0.7, w=w, US='PumpIn', DS='PumpOut'),
                gt.Rotor(name='DummyRotor', N=100),
                gt.Station(name='PumpOut', fluid=fluid),
                gt.FixedDP(name='HOT_HEX-COOL', US = 'PumpOut', DS = 'TurbIn', w=w, dP=(-1,'bar')),
                gt.Qdot(name='HOT', cool='HOT_HEX-COOL', Q=(3.2e6, 'BTU/hr')),
                gt.Station(name='TurbIn', fluid=fluid),
                gt.FixedPRTurbine(name='Turb', US='TurbIn', DS='TurbOut', rotor='DummyRotor', PR=5, w=w, eta=0.85),
                gt.Station(name='TurbOut', fluid=fluid),
                gt.FixedDP(name='COLD_HEX-HOT', US = 'TurbOut', DS = 'PumpIn', w=w, dP=(-1,'bar')),
                gt.Qdot(name='Chill', hot='COLD_HEX-HOT', Q=heat_rejection_acetone)])

Cool = gt.Model([gt.Boundary(name='WaterTank', fluid='Water', P=(1, 'bar'), T=(20, 'degC')),
                gt.FixedFlowPump(name='Water_Pump', eta=.7,w=w_H2O,
                                 US='WaterTank',DS='WaterPumpOut', rotor='DummyRotor3'),
                gt.Rotor(name='DummyRotor3', N=100),
                gt.Station(name='WaterPumpOut', fluid='Water'),
                gt.FixedDP(name='COLD_HEX-COLD', US='WaterPumpOut', DS='COLD_HEX_Out', dP=(-1, 'bar')),
                gt.Station(name='COLD_HEX_Out', fluid='Water'),
                gt.FixedDP(name='AirChiller', US='COLD_HEX_Out', DS='WaterTank', dP=(-50, 'psi')),
                gt.Qdot(name='ToAir', hot='AirChiller', Q=heat_rejection_water)])

Air = gt.Model([gt.Boundary(name='AirInlet', fluid='air', P=(1,'bar'), T=(20, 'degC')),
                gt.FixedDP(name='TubeBank', US='AirInlet', DS='FanInlet', dP=(-100, 'Pa')),
                gt.Station(name='FanInlet', fluid='air'),
                gt.FixedFlowPump(name='AirFlow', US='FanInlet', DS='AirOutlet',eta=0.5, w=500, rotor='DummyRotor4'),
                gt.Rotor(name='DummyRotor4', N=100),
                gt.PBoundary(name='AirOutlet', fluid='air', T=300, P=(1,'bar'))])


# Add Models
ORC += Cool
ORC += HOT
ORC += Air
# Update Chill Qdot
ORC['Chill'].cool = 'COLD_HEX-COLD'
ORC['HOT'].hot = 'HOT_HEX-HOT'
ORC['ToAir'].cool= 'FanInlet'

gt.logger.set_level('silent')
ORC.solve_steady()


# Use Pinch Solver to find water_flow rate
Pinch = PinchSolver(cold_fluid=ORC['PumpOut'].thermo, hot_fluid=ORC['Well'].thermo)
result = Pinch.get_pinch_Q(T_pinch=5, cold_inlet=ORC['PumpOut'].thermo, cold_outlet=ORC['TurbIn'].thermo,
                           hot_inlet=ORC['Well'].thermo, w_cold=ORC['Turb'].w)

# Update Well flow rate and re-run
ORC['HOT_HEX-HOT']._w = result['w_hot']

ORC.solve_steady()


gt.units.output='mixed'
print_model_tables(ORC)

Solution = gt.Solution(ORC)
Solution.append(ORC)
Solution.save_csv('testbed_water_example.csv')

# Plot Network
ORC.draw()

# Plot Hex
HEX = HEXProfile(w_hot=ORC['HOT_HEX-HOT']._w,
                 w_cold=ORC['Turb']._w,
                 hot_inlet=ORC['Well'].thermo,
                 cold_inlet=ORC['PumpOut'].thermo,
                 cold_outlet=ORC['TurbIn'].thermo)

HEX.evaluate(ORC['HOT']._Q)
HEX.plot()
