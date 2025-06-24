import geoTherm as gt
import math


def test_tanks():
    foo = gt.Model([gt.Boundary(name='Inlet', fluid='acetone', P=(500, 'psi'), T=300),
                      gt.resistor(name='Flow1', US='Inlet', DS='Tank1', area= (.1 ,'in**2'), flow_func='incomp'),
                      gt.resistor(name='Flow2', US='Tank1', DS='Tank2', area= (.01 ,'in**2'), flow_func='incomp'),
                      gt.resistor(name='Flow3', US='Tank1', DS='Outlet', area= (.1 ,'in**2'), flow_func='incomp'),
                      gt.Volume(name='Tank1', fluid='acetone', P=(14.7, 'psi'), T=300, volume=(100, 'in**3')),
                      gt.Volume(name='Tank2', fluid='acetone', P=(14.7, 'psi'), T=300, volume=(100, 'in**3')),
                      gt.Boundary(name='Outlet', fluid='acetone', P=(100, 'psi'), T=300)])

    foo+=gt.Schedule(name='Flow', knob='Flow3.area', t_points=[0,1e-2, 1.1e-2], y_points=[0, 0, 3e-05])
    
    sol = foo.sim(t_span=[0, 2e-2])
    
    assert(math.isclose(sol.dataframe['Flow3.area'].tail(1), 3e-5, abs_tol=1e-3))
