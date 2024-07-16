import geoTherm as gt
from matplotlib import pyplot as plt


## Water Circuit
water = 'H2O'


# Set the default input/output units 
gt.units.input = 'SI'
gt.units.output = 'mixed'


# Hot Circuit
HOT = gt.Model([gt.Boundary(name='Well', fluid=water, P=(40, 'bar'), T=473),
              gt.HEX(name='WaterHEX', US='Well', DS='WaterHEXOut', dP =(38, 'bar'),w=50.232),
              gt.TBoundary(name='WaterHEXOut', fluid=water, P=(2, 'bar'), T=314.31038),
              gt.POutlet(name='Outlet', fluid=water, P=(140, 'bar'), T=500),
              gt.fixedWPump(name='WaterPump',eta=.7,PR=1,w=(90,'kg/s'),US='WaterHEXOut',DS='Outlet')])

# Solve Hot
HOT.solve()

## ORC Circuit
fluid = 'IsoButene'

thermo = gt.thermo()
thermo.TPY = 308, 101325, 'isobutane'

ww = 60


ORC = gt.Model([gt.Boundary(name='LowT', fluid=fluid, P=(thermo.Pvap*1.1, 'bar'), T=308),
              gt.Pump(name='Pump', eta=.7, PR=2.1, w=ww, US='LowT', DS='PumpOut'),
              gt.Station(name='PumpOut', fluid=fluid),
              #gt.Qdot(name='ORC_Qdot', hot='WaterHEX'),
              gt.HEX(name='ORC_HEX', US = 'PumpOut', DS = 'TurbIn', w=ww, Q=-HOT['WaterHEX']._Q, dP=(1,'bar'), D=(2, 'in'), L=3),
              gt.Station(name='TurbIn', fluid=fluid),
              gt.fixedWTurbine(name='Turb', eta=.75, PR=1.3, w=ww, US='TurbIn', DS='TurbOut'),
              gt.Station(name='TurbOut', fluid=fluid),
              gt.HEX(name='CoolHex', US = 'TurbOut', DS = 'LowT', w=ww, dP=(1,'bar'))])


# Create empty model
CombinedModel = gt.Model()
# Add Hot
CombinedModel += HOT
# Add ORC
CombinedModel += ORC

CombinedModel += gt.wBalance('TurbInT','Turb','Well.T','TurbIn.T',5)
CombinedModel += gt.TBalance('Blah','WaterHEXOut','WaterHEXOut.T','PumpOut.T',5)
CombinedModel.solve()


ORC['Pump'].PR = 3
CombinedModel.solve()


import numpy as np
PR_sweep = np.arange(2,13,.1)

W_ORC = []
W_Net = []
Mdot = []



for PR in PR_sweep:
    ORC['Pump'].PR = PR
    CombinedModel.solve()
    W_ORC.append(ORC.performance[0])
    W_Net.append(CombinedModel.performance[0])
    Mdot.append(ORC['Pump']._w)
    


print('HOT WATER Circut')
print(HOT)
print('ORC Circuit')
print(ORC)
print('Combined Thermo')
print(CombinedModel)

plt.plot(PR_sweep, W_ORC, label='ORC')
plt.plot(PR_sweep, W_Net, label='W_Net')
plt.xlabel('PR')
plt.ylabel('Work [MW]')
plt.figure()
plt.plot(PR_sweep, Mdot)
plt.xlabel('PR')
plt.ylabel('Mass Flow [kg/s]')
plt.show()
