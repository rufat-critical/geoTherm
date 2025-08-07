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


# Target for Turbine Inlet
turbine_inlet = gt.thermo()
turbine_inlet.TPY = (200, 'degC'), ((1.4+2)*14.2857, 'bar'), ORC_fluid


def Q_transfer(hot, cool, model):
    # Define a function to heat ORC fluid to a pinch of 5 degrees
    return (turbine_inlet._H - model['PumpOut'].thermo._H)*model['Pump']._w
    

def Q_reject(hot, cool, model):
    # Define a function to calculate Q necessary to close cycle
    return (model['TurbOut'].thermo._H - model['LowT'].thermo._H)*model['Pump']._w
    

ORC = gt.Model([gt.Boundary(name='LowT', fluid=ORC_fluid, P=ORC_Pin, T=ORC_Tin),
                gt.FixedFlowPump(name='Pump', eta=0.7, w=mdot_ORC, US='LowT', DS='PumpOut'),
                gt.Station(name='PumpOut', fluid=ORC_fluid),
                gt.FixedDP(name='ORC_HEX-Cold', US='PumpOut', DS='TurbIn', dP =(-2, 'bar'),w=50.232),
                gt.Qdot(name='ORC_Heat', hot='ORC_HEX-Hot', cool='ORC_HEX-Cold', Q=Q_transfer),
                gt.Station(name='TurbIn', fluid=ORC_fluid),
                gt.FixedPRTurbine(name='Turb', eta=.9, PR=ORC_Turb_PR, w=mdot_ORC, US='TurbIn', DS='TurbOut'),
                gt.Station(name='TurbOut', fluid=ORC_fluid),
                gt.Qdot(name='Reject_Heat', hot='CoolHex-Hot', cool='CoolHex-Cold', Q=Q_reject),
                gt.FixedDP(name='CoolHex-Hot', US = 'TurbOut', DS = 'LowT', dP=(-2, 'bar'), w=mdot_ORC)])

HOT = gt.Model([gt.Boundary(name='Well', fluid=HOT_fluid, P=HOT_P, T=HOT_T),
              gt.FixedDP(name='ORC_HEX-Hot', US='Well', DS='WaterHEXOut', dP=(-38, 'bar'),w=mdot_H2O),
              gt.Station(name='WaterHEXOut', fluid=HOT_fluid),
              gt.FixedFlowPump(name='WaterPump', eta=.7, w=mdot_H2O, US='WaterHEXOut',DS='Outlet'),
              gt.POutlet(name='Outlet', fluid=HOT_fluid, P=(140, 'bar'), T=300)])


Cool = gt.Model([gt.Boundary(name='CoolPumpInlet', fluid='Water', P=(1, 'bar'), T=(20, 'degC')),
                 gt.FixedFlowPump(name='ChillerPump', eta=0.7, w=mdot_H2O, US='CoolPumpInlet', DS='CoolPumpOutlet'),
                 gt.Station(name='CoolPumpOutlet', fluid='water'),
                 gt.FixedDP(name='CoolHex-Cold', US='CoolPumpOutlet', DS = 'CoolHexOutlet', dP = (-2, 'bar')),
                 gt.Station(name='CoolHexOutlet', fluid='water'),
                 gt.FixedDP(name='AirCooler', US='CoolHexOutlet', DS='CoolPumpInlet', dP=(-2,'bar'))])
                 
# Combine the models
ORC += HOT
ORC += Cool

# Solve the models
ORC.solve_steady()

# Change output units to mixed
gt.units.output='mixed'

# Draw the model network
ORC.draw()

# Show report
gt.print_model_tables(ORC)