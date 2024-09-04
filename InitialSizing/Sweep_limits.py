import geoTherm as gt
from matplotlib import pyplot as plt
from scipy.optimize import fsolve
from pdb import set_trace
import numpy as np

## Water Circuit
water = 'H2O'


# Set the default input/output units 
gt.units.input = 'SI'
gt.units.output = 'mixed'


def make_ORC(fluid, Pin, Tcool, Thot, w, PR_turb):
    ORC = gt.Model([gt.Boundary(name='LowT', fluid=fluid, P=Pin, T=Tcool),
                    gt.Rotor('ORC_Rotor', N =14009.97841),
                  gt.fixedFlowPump(name='Pump', rotor= 'ORC_Rotor', psi=0.45, eta=1, PR=6, w=w, US='LowT', DS='PumpOut'),
                  gt.Station(name='PumpOut', fluid=fluid),
                  #gt.Qdot(name='ORC_Qdot', hot='WaterHEX'),
                  #gt.staticHEX(name='ORC_HEX', US = 'PumpOut', DS = 'TurbIn', w=ww, Q=-HOT['WaterHEX']._Q, dP=(1,'bar'), D=(2, 'in'), L=3),
                  gt.staticHEX(name='ORC_HEX', US='PumpOut', DS='TurbIn',w=50, dP=0),
                  gt.TBoundary(name='TurbIn', fluid=fluid, T=Thot, P =101325),
                  gt.Turbine(name='Turb', rotor = 'ORC_Rotor', Ns= 7.4, Ds=2.4, eta=1, PR=PR_turb, w=w, US='TurbIn', DS='TurbOut'),
                  gt.Station(name='TurbOut', fluid=fluid),
                  gt.staticHEX(name='CoolHex', US = 'TurbOut', DS = 'LowT', w=50, dP=0)])
   
    ORC.solve()

    HOT = gt.Model([gt.Boundary(name='Well', fluid=water, P=(40, 'bar'), T=Thot),
                gt.Rotor('DummyRotor', N =15000),
              gt.staticHEX(name='WaterHEX', US='Well', DS='WaterHEXOut',w=90, dP=(38, 'bar'), Q= -ORC['ORC_HEX']._Q),
             #gt.TBoundary(name='WaterHEXOut', fluid=water, P=(2, 'bar'), T=ORC['PumpOut'].thermo._T+5),
              gt.Station(name='WaterHEXOut',fluid=water),
              gt.POutlet(name='Outlet', fluid=water, P=(140, 'bar'), T=500),
              gt.fixedFlowPump(name='WaterPump',rotor='DummyRotor', eta=.7,PR=2,w=(90,'kg/s'),US='WaterHEXOut',DS='Outlet')])

    CombinedModel = gt.Model()
    CombinedModel += ORC
    CombinedModel += HOT
    return ORC, HOT, CombinedModel


 
fluids = ['isobutane','benzene','ammonia', 'cyclopentane','acetone','toluene','n-Hexane','propane','Propylene', 'butane','CycloHexane','CycloPropane','propyne']
#fluids = ['isobutane', 'ammonia', 'propyne', 'butane','cis-2-Butene','Isohexane']


for fluid in fluids:
    thermo = gt.thermo()
    thermo.TPY = 300, 101325, fluid


from pdb import set_trace
set_trace()

#fluids = ['benzene']
#fluid = 'isobutane'
Tcool = 308
T_Turb = 473-5
w = 50
PR_turb = 20

Max_P = 6.8e6



fluid_data = {}
for fluid in fluids:
    thermo = gt.thermo()
    thermo.TPY = Tcool, 101325, fluid
       
    
    Pin = np.max([thermo._Pvap*1.1, 101325*.9])
       
    ORC, HOT, CombinedModel = make_ORC(fluid, Pin, Tcool, T_Turb, w, 2)


    data = {'PR': [], 'performance': [], 'WTurb': [], 'WPump': [], 'WPumpWater':[], 'netperformance': [],
            'pumpOutletT':[], 'waterOutletT':[], 'Q':[], 'Hwater':[], 'TurbOutletT':[], 'Phi':[]}


    PR_sweep = np.arange(2,min([Max_P/Pin,60]),1)

    for PR in PR_sweep:
        ORC['Turb'].PR = PR
        ORC.solve()
        
        if ORC['TurbIn'].phase == 'liquid':
            # If inlet is liquid then break out of loop
            # all the other cases are higher pressure and will be liquid
            break
        
        if (ORC['TurbOut'].phase != 'gas'
           and ORC['TurbOut'].phase !='supercritical_gas'):
           # This is too much expansion and fluid is 2 phase,
           # All other cases will be 2 phase or liq so break out of loop
            break
        #HOT['WaterHEX']._Q = -ORC['ORC_HEX']._Q
        #HOT.solve()   

        data['PR'].append(PR)
        data['Phi'].append(PR*Pin/1e5)
        data['performance'].append(ORC.performance)
        data['netperformance'].append(CombinedModel.performance)
        data['WTurb'].append(ORC['Turb'].W)
        data['WPump'].append(ORC['Pump'].W)
        data['WPumpWater'].append(HOT['WaterPump'].W)
        data['waterOutletT'].append(HOT['WaterHEXOut'].thermo.T)
        data['pumpOutletT'].append(ORC['PumpOut'].thermo.T)
        data['TurbOutletT'].append(ORC['TurbOut'].thermo.T)
        data['Q'].append(ORC['ORC_HEX']._Q)

            
        #max_eta.append(np.max(data['performance'][:,2]))
        #print(np.max(data['performance'][:,2]))
     
    data['performance'] = np.array(data['performance'])
    data['netperformance'] = np.array(data['netperformance'])
    
    fluid_data[fluid] = data


for fluid in fluids:
    plt.plot(fluid_data[fluid]['PR'], fluid_data[fluid]['performance'][:,2],label=fluid)
    
plt.xlabel('PR')
plt.ylabel(r'$\eta$')
plt.legend()

plt.figure()
for fluid in fluids:
    plt.plot(fluid_data[fluid]['Phi'], fluid_data[fluid]['performance'][:,2],label=fluid)
plt.xlabel('P Pump Outlet [bar]')
plt.ylabel(r'$\eta$')
plt.legend()


np.save('optimal_limit',fluid_data)

from pdb import set_trace
set_trace()





# plt.plot(data['PR'], data['netperformance'][:,0], label='Net')
# plt.plot(data['PR'], data['WTurb'], label ='Turbine')
# plt.plot(data['PR'], data['WPump'], label = 'ORC Pump')
# plt.plot(data['PR'], data['WPumpWater'], label = 'Water Pump')
# plt.xlabel('Turbine Pressure Ratio')
# plt.ylabel('Net Work [MW]')
# plt.legend()
# plt.figure()
# plt.plot(data['PR'], data['performance'][:,2], label='ORC')
# plt.plot(data['PR'], data['netperformance'][:,2], label='Net')
# plt.xlabel('Turbine Pressure Ratio')
# plt.ylabel(r'$\eta$')
# plt.figure()
# plt.plot(data['PR'], data['waterOutletT'],label='H2O')
# plt.plot(data['PR'], data['pumpOutletT'],label='ORC Pump Outlet')
# plt.plot(data['PR'], data['TurbOutletT'],label='ORC Turbine Outlet')
# plt.xlabel('Turbine Pressure Ratio')
# plt.ylabel(r'Temperature [K]')
# plt.legend()
# plt.figure()
# plt.plot(data['PR'], data['Q'])
# plt.xlabel('Turbine Pressure Ratio')
# plt.ylabel(r'Q [MW]')
# plt.show()
