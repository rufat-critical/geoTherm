import geoTherm as gt
from matplotlib import pyplot as plt
from pdb import set_trace
import numpy as np
from geoTherm import units


# Create a model with Acetone
foo = gt.Model([gt.Boundary(name='Inlet', fluid='acetone', P=(500, 'psi'), T=300),
                  gt.resistor(name='Flow1', US='Inlet', DS='Tank1', area= (.1 ,'in**2'), flow_func='incomp'),
                  gt.resistor(name='Flow2', US='Tank1', DS='Tank2', area= (.01 ,'in**2'), flow_func='incomp'),
                  gt.resistor(name='Flow3', US='Tank1', DS='Outlet', area= (.1 ,'in**2'), flow_func='incomp'),
                  gt.Volume(name='Tank1', fluid='acetone', P=(14.7, 'psi'), T=300, volume=(100, 'in**3')),
                  gt.Volume(name='Tank2', fluid='acetone', P=(14.7, 'psi'), T=300, volume=(100, 'in**3')),
                  gt.Boundary(name='Outlet', fluid='acetone', P=(100, 'psi'), T=300)])

foo+=gt.Schedule(name='Flow', knob='Flow3.area', t_points=[0,1e-2, 1.1e-2], y_points=[0, 0, 3e-05])

              
units.output='english'

# Simulate for 1e-3
sol = foo.sim(t_span=[0, 2e-2])

plt.plot(sol['t'], sol['Tank1.P'], label='Tank1')
plt.plot(sol['t'], sol['Tank2.P'], label='Tank2')

plt.xlabel('Time [s]')
plt.ylabel('Pressure [psi]')
plt.legend()

plt.figure()
plt.plot(sol['t'], sol['Flow1.w'], label='1')
plt.plot(sol['t'], sol['Flow2.w'], label='2')
plt.plot(sol['t'], sol['Flow3.w'], label='3')
plt.xlabel('Time [s]')
plt.ylabel('Mass Flow [lbm/s]')
plt.legend()
plt.show()