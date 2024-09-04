import geoTherm as gt
from matplotlib import pyplot as plt
from pdb import set_trace
import numpy as np
from geoTherm import units


# Create a model with Acetone
foo = gt.Model([gt.Boundary(name='Inlet', fluid='water', P=(500, 'psi'), T=300),
                  gt.Pipe(name='InletFlow', US='Inlet', DS='Volume', D= (1 ,'in'), L=(5,'in'), w=50),
                  gt.Volume(name='Volume', fluid='water', P=(200, 'psi'), T=300, volume=(50, 'in**3')),
                  gt.resistor(name='OutletFlow', US='Volume', DS='Outlet', area= (.1 ,'in**2'), flow_func='incomp'),
                  gt.Boundary(name='Outlet', fluid='water', P=(200, 'psi'), T=300)])

foo.solve_steady()

foo+=gt.Schedule(name='Flow', knob='OutletFlow.area', t_points=[0,1e-4], y_points=[6.4516e-05, 0])
# Create a model with Water
bar = gt.Model([gt.Boundary(name='Inlet', fluid='acetone', P=(500, 'psi'), T=300),
                  gt.Pipe(name='InletFlow', US='Inlet', DS='Volume', D= (1 ,'in'), L=(5,'in'), w=50),
                  gt.Volume(name='Volume', fluid='acetone', P=(200, 'psi'), T=300, volume=(50, 'in**3')),
                  gt.resistor(name='OutletFlow', US='Volume', DS='Outlet', area= (.1 ,'in**2'), flow_func='incomp'),
                  gt.Boundary(name='Outlet', fluid='acetone', P=(200, 'psi'), T=300)])
             

bar.solve_steady()


bar+=gt.Schedule(name='Flow', knob='OutletFlow.area', t_points=[0,1e-5], y_points=[6.4516e-05, 0])   
units.output='english'


# Simulate for 1e-3
sol = foo.sim(t_span=[0, 5e-3])
sol2 = bar.sim(t_span=[0, 5e-3])


plt.plot(sol['t'], sol['Volume.P'], label='water')
plt.plot(sol2['t'], sol2['Volume.P'], label='acetone')
plt.xlabel('Time [s]')
plt.ylabel('Pressure [psi]')
plt.legend()

plt.show()