import geoTherm as gt
from matplotlib import pyplot as plt
from scipy.optimize import fsolve
from pdb import set_trace
import numpy as np
import pandas as pd

## Water Circuit
water = 'H2O'


# Set the default input/output units 
gt.units.input = 'SI'
gt.units.output = 'SI'

def decoupled_ORC(fluid, Tcool, Thot, Cda, PR_turb):
    # Make ORC => Solve initial w => Solve for Heat Loop => Solve for mass flow => Solve using fsolve/bisection
    # Fsolve is being annoying - I really need to implement matrix conditioner
    
    w = 1
    thermo = gt.thermo()
    thermo.TPY = Tcool, 101325, fluid
    
    Pin = thermo.Pvap*1.1
    

    ORC = gt.Model([gt.Boundary(name='LowT', fluid=fluid, P=Pin, T=Tcool),
                    gt.Rotor('ORC_Rotor', N =14009.97841),
                  gt.fixedFlowPump(name='Pump', rotor= 'ORC_Rotor', psi=0.45, eta=0.75, PR=6, w=w, US='LowT', DS='PumpOut'),
                  gt.Station(name='PumpOut', fluid=fluid),
                  #gt.Qdot(name='ORC_Qdot', hot='WaterHEX'),
                  #gt.staticHEX(name='ORC_HEX', US = 'PumpOut', DS = 'TurbIn', w=ww, Q=-HOT['WaterHEX']._Q, dP=(1,'bar'), D=(2, 'in'), L=3),
                  gt.staticHEX(name='ORC_HEX', US='PumpOut', DS='TurbIn',w=50, D=Cda, L=30),
                  gt.TBoundary(name='TurbIn', fluid=fluid, T=Thot-5, P =101325),
                  gt.Turbine(name='Turb', rotor = 'ORC_Rotor', Ns= 7.4, Ds=2.4, eta=1, PR=PR_turb, w=w, US='TurbIn', DS='TurbOut'),
                  gt.Station(name='TurbOut', fluid=fluid),
                  gt.staticHEX(name='CoolHex', US = 'TurbOut', DS = 'LowT', w=50, D=Cda, L=30)])
   
    try:
        ORC.solve()
    except:
        # PR is probably too high
        if ORC['Pump'].PR > 80:
            return ORC
        else:
            from pdb import set_trace
            set_trace()

    HOT = gt.Model([gt.Boundary(name='Well', fluid=water, P=(40, 'bar'), T=Thot),
                gt.Rotor('DummyRotor', N =15000),
              gt.staticHEX(name='WaterHEX', US='Well', DS='WaterHEXOut',w=90, dP=(38, 'bar'), Q= -ORC['ORC_HEX']._Q),
             gt.TBoundary(name='WaterHEXOut', fluid=water, P=(2, 'bar'), T=ORC['PumpOut'].thermo._T+5),
              #gt.Station(name='WaterHEXOut',fluid=water),
              gt.POutlet(name='Outlet', fluid=water, P=(140, 'bar'), T=500),
              gt.fixedFlowPump(name='WaterPump',rotor='DummyRotor', eta=.7,PR=2,w=(90,'kg/s'),US='WaterHEXOut',DS='Outlet')])

    # Make sure Q values are equal
    
    # Update Hex Outlet Condition (Pump out + 5K)
    for i in range(0, 5):
        HOT['WaterHEXOut'].thermo._TP = ORC['PumpOut'].thermo._T+5, HOT['WaterHEXOut'].thermo._P
        HOT.solve()
        w_ORC = HOT['WaterHEX']._Q/(ORC['PumpOut'].thermo._H-ORC['TurbIn'].thermo._H)
        ORC['Pump']._w = w_ORC
    
   
        ORC.solve()
        Qhot = HOT['WaterHEX']._Q
        Qcool = ORC['ORC_HEX']._Q 
        
        if abs(Qhot+Qcool)/abs(Qhot) <.005:
            break



    CombinedModel = gt.Model()
    CombinedModel += ORC
    CombinedModel += HOT
    
    CombinedModel.ORC = ORC
    CombinedModel.HOT = HOT
    
    return CombinedModel


def optimal_PR(fluid, Tcool, Thot, Cda):
    # Sweep Model thru pressure ratios until:
        # Pressure is too high and Turbine inlet is liquid
        # Turbine exhaust is not gassy
        # Net Work starts dropping
        
    PR_sweep = np.arange(2,100,.5)
    Wnet = -1e9
    for i, PR in enumerate(PR_sweep):
        CombinedModel = decoupled_ORC(fluid, Tcool, Thot, Cda, PR)
        
        if CombinedModel['TurbIn'].phase == 'liquid':
            # If inlet is liquid then break out of loop
            # all the other cases are higher pressure and will be liquid
            CombinedModel = decoupled_ORC(fluid, Tcool, Thot, Cda, PR_sweep[i-1])
            break
        
        if (CombinedModel['TurbOut'].phase != 'gas'
           and CombinedModel['TurbOut'].phase !='supercritical_gas'):
            # This is too much expansion and fluid is 2 phase,
            # All other cases will be 2 phase or liq so break out of loop
            CombinedModel = decoupled_ORC(fluid, Tcool, Thot, Cda, PR_sweep[i-1])
            break
        
        if CombinedModel.performance[0] < Wnet:
            CombinedModel = decoupled_ORC(fluid, Tcool, Thot, Cda, PR_sweep[i-1])
            print('LOW POWER')
        
        Wnet = CombinedModel.performance[0]
    
    # Return the optimal state point
    return CombinedModel


fluid_sweep = ['isobutane', 'butane', 'cyclopentane']
cdA_sweep = [1, 1.2, 1.25]
Thot_sweep = [373, 423, 473]
Tpump_sweep = [308, 300, 273]

# Create pandas dataframe
df = pd.DataFrame(columns=['fluid', 'cdA', 'Thot','Tcool', 'eta','Wnet', 'Qin','dT_cool','Qcool',
                            'dT_hot','dP_hot','dP_cool'])


# Sweep, store data
for fluid in fluid_sweep:
    for cdA in cdA_sweep:
        for Thot in Thot_sweep:
            for Tcool in Tpump_sweep:
            
                # Optimal state pt
                Model = optimal_PR('isobutane', Tcool, Thot, cdA)
                rows.append({'fluid': fluid,
                                     'cdA': cdA,
                                     'Thot': Thot,
                                     'Tcool': Tcool,
                                     'eta': Model.performance[2],
                                     'Wnet': Model.performance[0],
                                     'Qin': Model.performance[1],
                                     'dT_hot': Model['TurbIn'].thermo._T - Model['PumpOut'].thermo._T,
                                     'dT_cool': Model['TurbOut'].thermo._T - Model['LowT'].thermo._T,
                                     'dP_hot': Model['TurbIn'].thermo._P - Model['PumpOut'].thermo._P,
                                     'dP_cool': Model['TurbOut'].thermo._P - Model['LowT'].thermo._P})
                                     
# Make Panadas dataframe
df = pd.DataFrame(rows)
# Output to csv
df.to_csv('blah.csv')