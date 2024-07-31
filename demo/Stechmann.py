import geoTherm as gt
from matplotlib import pyplot as plt
from scipy.optimize import fsolve



# Inputs
Tcool = 308
Thot = 473
PR_turb = 21.10542
water = 'H2O'


# Set the default input/output units 
gt.units.input = 'SI'
gt.units.output = 'mixed'
#gt.units.input = 'mixed'


# Hot Circuit

# Solve Hot

## ORC Circuit
fluid = 'isobutane'

thermo = gt.thermo()
thermo.TPY = Tcool, 101325, fluid

water = 'H2O'

ww = 135.4649
Pin = 5.58598532
#Pin = thermo.Pvap*1.1
#NSS_target

ORC = gt.Model([gt.Boundary(name='LowT', fluid=fluid, P=(Pin, 'bar'), T=Tcool),
                gt.BoundaryRotor('ORC_Rotor', N =14009.97841),
              gt.fixedFlowPump(name='Pump', rotor= 'ORC_Rotor', eta= 0.7, psi=0.45, PR=6, w=ww, US='LowT', DS='PumpOut'),
              gt.Station(name='PumpOut', fluid=fluid),
              #gt.Qdot(name='ORC_Qdot', hot='WaterHEX'),
              #gt.staticHEX(name='ORC_HEX', US = 'PumpOut', DS = 'TurbIn', w=ww, Q=-HOT['WaterHEX']._Q, dP=(1,'bar'), D=(2, 'in'), L=3),
              gt.staticHEX(name='ORC_HEX', US='PumpOut', DS='TurbIn', dP =(1, 'bar'),w=50.232),
              gt.TBoundary(name='TurbIn', fluid=fluid, T=Thot-5, P =101325),
              gt.Turbine(name='Turb', rotor = 'ORC_Rotor', Ns= 7.4, Ds=2.4, eta=.75, PR=PR_turb, w=ww, US='TurbIn', DS='TurbOut'),
              gt.Station(name='TurbOut', fluid=fluid),
              gt.staticHEX(name='CoolHex', US = 'TurbOut', DS = 'LowT', w=ww, dP=(1,'bar'))])

#ORC.solve()
ORC+= gt.Balance('NSS',knob='LowT.P', feedback='Pump._NSS', setpoint=1200, knob_min=10000, knob_max=1e7,gain=1)
#ORC+= gt.Balance('N',knob='ORC_Rotor.N', feedback='Turb.Ns', knob_min=10000, knob_max=1e7,gain=1)
ORC.solve()

HOT = gt.Model([gt.Boundary(name='Well', fluid=water, P=(40, 'bar'), T=Thot),
                gt.BoundaryRotor('DummyRotor', N =15000),
              gt.staticHEX(name='WaterHEX', US='Well', DS='WaterHEXOut',w=90, dP=(38, 'bar')),#, Q= -ORC['ORC_HEX']._Q, dP =(38, 'bar'),w=50.232),
              gt.TBoundary(name='WaterHEXOut', fluid=water, P=(2, 'bar'), T=ORC['PumpOut'].thermo._T+5),
              gt.POutlet(name='Outlet', fluid=water, P=(140, 'bar'), T=500),
              gt.fixedFlowPump(name='WaterPump',rotor='DummyRotor', eta=.7,PR=2,w=(90,'kg/s'),US='WaterHEXOut',DS='Outlet')])

HOT.solve()
    


ORC.solve()

from pdb import set_trace
set_trace()

    

# Create empty model
CombinedModel = gt.Model()
# Add Hot
CombinedModel += HOT
# Add ORC
CombinedModel += ORC
CombinedModel.Hot = HOT
CombinedModel.ORC = ORC
#CombinedModel += gt.wBalance('TurbInT','Pump','Well.T','TurbIn.T',5)
#CombinedModel += gt.TBalance('Blah','WaterHEXOut','WaterHEXOut.T','PumpOut.T',5)
CombinedModel.solve()

from pdb import set_trace
set_trace()


ORC['Turb'].PR = 3
CombinedModel.solve()


from pdb import set_trace
ORC['Turb'].PR = 7
#ORC.debug =True
ORC.solve()
HOT.solve()
set_trace()

import numpy as np
PR_sweep = np.arange(2,20,.2)

W_ORC = []
W_Net = []
Mdot = []
etaT = []
etaC = []
for PR in PR_sweep:
    ORC['Turb'].PR = PR
    #CombinedModel.solve()
    print(PR)

    ORC.solve()
    HOT.solve()
    W_ORC.append(ORC.performance[0])
    W_Net.append(CombinedModel.performance[0])
    etaT.append(ORC.performance[2])
    etaC.append(CombinedModel.performance[2])

    Mdot.append(ORC['Pump']._w)


print('HOT WATER Circut')
#print(HOT)
print('ORC Circuit')
#print(ORC)
print('Combined Thermo')
#print(CombinedModel)

plt.plot(PR_sweep, W_ORC, label='ORC')
plt.plot(PR_sweep, W_Net, label='W_Net')
plt.xlabel('PR')
plt.ylabel('Work [MW]')
plt.figure()
plt.plot(PR_sweep, Mdot)
plt.xlabel('PR')
plt.ylabel('Mass Flow [kg/s]')

plt.figure()
plt.plot(PR_sweep, etaT, label='ORC')
plt.plot(PR_sweep, etaC, label='W_Net')
plt.show()