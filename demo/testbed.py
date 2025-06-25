import geoTherm as gt
from matplotlib import pyplot as plt
from gt.utilities.display import print_model_tables

# Working Fluid
fluid = 'acetone'
# Oil
oil = gt.thermo(model='custom', property_file='PG-1.xlsx')


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

def combustion(hot, cool, model):
    # How much burner needs to burn
    w = model['OilPump']._w
    H_hot = model['HOT_HEX_IN'].thermo._H
    H_cold = model['BurnerInlet'].thermo._H
    
    return w*(H_hot-H_cold)
    
ORC = gt.Model([gt.Boundary(name='PumpIn', fluid=fluid, P=(3.8, 'bar'), T=(45, 'degC')),
                gt.fixedFlowPump(name='Pump', eta=0.7, w=w, US='PumpIn', DS='PumpOut'),
                gt.Station(name='PumpOut', fluid=fluid),
                gt.FixedDP(name='HOT_HEX-COOL', US = 'PumpOut', DS = 'TurbIn', w=w, dP=(-1,'bar')),
                gt.Qdot(name='HOT', cool='HOT_HEX-COOL', Q=(3.2e6, 'BTU/hr')),
                gt.Station(name='TurbIn', fluid=fluid),
                gt.FixedPRTurbine(name='Turb', US='TurbIn', DS='TurbOut', PR=5, w=w, eta=0.85),
                gt.Station(name='TurbOut', fluid=fluid),
                gt.FixedDP(name='COLD_HEX-HOT', US = 'TurbOut', DS = 'PumpIn', w=w, dP=(-1,'bar')),
                gt.Qdot(name='Chill', hot='COLD_HEX-HOT', Q=heat_rejection_acetone)])

HOT = gt.Model([gt.Boundary(name='HOT_HEX_IN', fluid=oil, T=(220, 'degC'), P=(10, 'bar')),
                gt.FixedDP(name='HOT_HEX-HOT', US='HOT_HEX_IN', DS='HotOut', dP=(-2, 'bar')),
                gt.Station(name='HotOut', fluid=oil.copy()),
                gt.fixedFlowPump(name='OilPump', eta=0.7, w=4, US='HotOut', DS='BurnerInlet'),
                gt.Station(name='BurnerInlet', fluid = oil.copy()),
                gt.FixedDP(name='Burner', US='BurnerInlet', DS='BurnerOutlet', dP=(-2,'bar')),
                gt.Station(name='BurnerOutlet', fluid=oil.copy()),
                gt.FixedDP(name='HOTPiping', US='BurnerOutlet', DS='HOT_HEX_IN', dP=(-2, 'bar')),
                gt.Qdot(name='COMBUSTION!', cool='Burner', Q=combustion)])

Cool = gt.Model([gt.Boundary(name='WaterTank', fluid='Water', P=(1, 'bar'), T=(20, 'degC')),
                gt.fixedFlowPump(name='Water_Pump', eta=.7,w=w_H2O,
                                 US='WaterTank',DS='WaterPumpOut'),
                gt.Station(name='WaterPumpOut', fluid='Water'),
                gt.FixedDP(name='COLD_HEX-COLD', US='WaterPumpOut', DS='COLD_HEX_Out', dP=(-1, 'bar')),
                gt.Station(name='COLD_HEX_Out', fluid='Water'),
                gt.FixedDP(name='AirChiller', US='COLD_HEX_Out', DS='WaterTank', dP=(-50, 'psi')),
                gt.Qdot(name='ToAir', hot='AirChiller', Q=heat_rejection_water)])
                
Air = gt.Model([gt.Boundary(name='AirInlet', fluid='air', P=(1,'bar'), T=(20, 'degC')),
                gt.FixedDP(name='TubeBank', US='AirInlet', DS='FanInlet', dP=(-100, 'Pa')),
                gt.Station(name='FanInlet', fluid='air'),
                gt.fixedFlowPump(name='AirFlow', US='FanInlet', DS='AirOutlet',eta=0.5, w=500),
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

gt.units.output='mixed'
print_model_tables(ORC)

Solution = gt.Solution(ORC)
Solution.append(ORC)

ORC.draw()
