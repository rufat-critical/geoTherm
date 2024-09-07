import geoTherm as gt
from matplotlib import pyplot as plt
from scipy.optimize import fsolve
from pdb import set_trace
import numpy as np
import pandas as pd
from geoTherm import units


fluid = 'acetone'

acetone = gt.thermo()
acetone.TPY = 303, 101325, fluid
PR_turb = 5
Pin = 3.8
w = 1.4

            
ORC = gt.Model([gt.Boundary(name='PumpIn', fluid=fluid, P=(Pin, 'bar'), T=319.8),
                #gt.Station(name='PumpIn', fluid=fluid),
                gt.Rotor('ORC_Rotor', N =40000),
                gt.Rotor('Pump_Rotor', N =14009.97841),
              gt.fixedFlowPump(name='Pump', rotor= 'Pump_Rotor', eta=0.7, PR=5, w=w, US='PumpIn', DS='PumpOut'),
              gt.Station(name='PumpOut', fluid=fluid),
              #gt.Qdot(name='ORC_Qdot', hot='WaterHEX'),
              gt.Pipe(name='ORC_HEX', US = 'PumpOut', DS = 'TurbIn', w=w, dP=(1,'bar'), D=(2, 'in'), L=3),
              gt.Qdot(name='ORC_Heat', cool='ORC_HEX', Q=(3.2e6, 'BTU/hr')),
              #gt.staticHEX(name='ORC_HEX', US = 'PumpOut', DS = 'TurbIn', w=w, Q=(1000, 'kW'), dP=(1,'bar'), D=(2, 'in'), L=3),
              #gt.staticHEX(name='ORC_HEX', US='PumpOut', DS='TurbIn',w=50, dP=(1,'bar')),
              #gt.TBoundary(name='TurbIn', fluid=fluid, T=(160, 'degC'), P =101325),
              #gt.Boundary(name='TurbIn', fluid=fluid, P=(24 ,'bar'), T=470),
              gt.Station(name='TurbIn', fluid=fluid),#, T=Thot-5, P =101325),
              #gt.Turbine_sizer(name='Turb', rotor = 'ORC_Rotor', phi= .1, psi=1.2, PR=PR_turb, w=w, US='TurbIn', DS='TurbOut'),
              gt.Turbine(name='Turb', rotor='ORC_Rotor',US='TurbIn', DS='TurbOut', D= .057225646*2, eta=0.8, PR=5),
              gt.Station(name='TurbOut', fluid=fluid),
              #gt.resistor(name='out', US='TurbIn', DS='TurbOut', area=.1),
              gt.simpleHEX(name='CoolHex', US = 'TurbOut', DS = 'PumpIn', w=w, dP=(1,'bar'))])


#ORC += gt.Balance('mass_balance', knob='Pump.w', feedback='TurbIn.T', setpoint=473, knob_min=0.5, knob_max=1.5)
#ORC += gt.ThermoBalance('mass_balance', knob='PumpIn.T', constant_var='P', feedback='TurbIn.T', setpoint=473, knob_min=300, knob_max=350, gain=-1)

ORC.solve_steady()
#ORC.draw()           
gt.units.output='mixed'           
print(ORC)
#ORC.thermo_plot(plot_type='PT', isolines='P')
#ORC.thermo_plot(plot_type='TS', isolines='P')
set_trace()


