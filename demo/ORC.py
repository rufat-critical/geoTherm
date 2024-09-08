import geoTherm as gt
from matplotlib import pyplot as plt
from scipy.optimize import fsolve

## Water Circuit
HOT_fluid = 'H2O'
## ORC Circuit
ORC_fluid = 'n-Pentane'
## Coolant Circuit
Cool_fluid = 'H2O'

## Hot Well P, T
HOT_P = (40, 'bar')
HOT_T = (473, 'degK')

## ORC Pump Inlet P T
ORC_Pin = (1.4, 'bar')
ORC_Tin = (306, 'degK')

# Turb Pressure Ratio
ORC_Turb_PR = 14.28571429

# Mass Flow
mdot_ORC = (20, 'kg/s')
mdot_H2O = (45, 'kg/s')

ORC = gt.Model([gt.Boundary(name='LowT', fluid=ORC_fluid, P=ORC_Pin, T=ORC_Tin),
                gt.Rotor('ORC_Rotor', N =12007.76906),
                gt.fixedFlowPump(name='Pump', rotor= 'ORC_Rotor', eta=0.7, PR=20/1.4, w=mdot_ORC, US='LowT', DS='PumpOut', D=.1),
                gt.Station(name='PumpOut', fluid=ORC_fluid),
                gt.flow(name='ORC_HEX', US='PumpOut', DS='TurbIn', dP =(0, 'bar'),w=50.232),#, Q=(20, 'MW')),
                #gt.Qdot(name='ORC_Heat', cool='ORC_HEX', Q=(20 , 'MW')),
                gt.Heatsistor('ORC_Heat', hot='WaterHEX', cool='ORC_HEX', Q=(31374557.96)),
                gt.Station(name='TurbIn', fluid=ORC_fluid),
                gt.Turbine(name='Turb', rotor = 'ORC_Rotor', eta=.9, PR=ORC_Turb_PR, w=mdot_ORC, US='TurbIn', DS='TurbOut',
                           D=(0.510569758, 'm')),
                gt.Station(name='TurbOut', fluid=ORC_fluid),
                gt.simpleHEX(name='CoolHex', US = 'TurbOut', DS = 'LowT', w=mdot_ORC, dP=(0,'bar'))])


HOT = gt.Model([gt.Boundary(name='Well', fluid=HOT_fluid, P=HOT_P, T=HOT_T),
                gt.Rotor('DummyRotor', N =15000),
              #gt.staticHEX(name='WaterHEX', US='Well', DS='WaterHEXOut',w=mdot_H2O, dP-=(38, 'bar')),#, Q= -ORC['ORC_HEX']._Q, dP =(38, 'bar'),w=50.232),
              #gt.Qdot(name='ORC_Heat2', cool='WaterHEX', Q=(-21 , 'MW')),
              gt.fixedFlow(name='WaterHEX', US='Well', DS='WaterHEXOut', dP =(-38, 'bar'),w=mdot_H2O),
              gt.Station(name='WaterHEXOut', fluid=HOT_fluid),#, P=(2, 'bar'), T=473+5),
              gt.Outlet(name='Outlet', fluid=HOT_fluid, P=(140, 'bar'), T=500),
              gt.Pump(name='WaterPump',rotor='DummyRotor', eta=.7,PR=140/2,w=mdot_H2O*20,US='WaterHEXOut',DS='Outlet')])


combined = gt.Model()
combined += ORC
combined += gt.Balance('Turbin_Temp', 'Pump.w', 'TurbIn.T', 470, knob_min=0.5, knob_max=100)
combined += HOT
combined.solve_steady()