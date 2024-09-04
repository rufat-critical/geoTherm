import geoTherm as gt
import math


def test_water_flow():
    foo = gt.Model([gt.Boundary(name='Inlet', fluid='water', P=(500, 'psi'), T=300),
                      gt.resistor(name='InletFlow', US='Inlet', DS='Volume', area= (.1 ,'in**2'), flow_func='incomp'),
                      gt.Volume(name='Volume', fluid='water', P=(200, 'psi'), T=300, volume=(50, 'in**3')),
                      gt.resistor(name='OutletFlow', US='Volume', DS='Outlet', area= (.1 ,'in**2'), flow_func='incomp'),
                      gt.Boundary(name='Outlet', fluid='water', P=(200, 'psi'), T=300)])

                      

    sol = foo.sim(t_span=[0, 1e-3])
    
    assert(math.isclose(foo.nodes['Volume'].thermo._P, 2413636.3, abs_tol=1e-3) and
            math.isclose(foo.nodes['Volume'].thermo._H, 114871.893, abs_tol=1e-3))

def test_acetone_flow():
    foo = gt.Model([gt.Boundary(name='Inlet', fluid='acetone', P=(500, 'psi'), T=300),
                      gt.resistor(name='InletFlow', US='Inlet', DS='Volume', area= (.1 ,'in**2'), flow_func='incomp'),
                      gt.Volume(name='Volume', fluid='acetone', P=(200, 'psi'), T=300, volume=(50, 'in**3')),
                      gt.resistor(name='OutletFlow', US='Volume', DS='Outlet', area= (.1 ,'in**2'), flow_func='incomp'),
                      gt.Boundary(name='Outlet', fluid='acetone', P=(200, 'psi'), T=300)])

                      

    sol = foo.sim(t_span=[0, 1e-3])

    assert(math.isclose(foo.nodes['Volume'].thermo._P, 2400032.964, abs_tol=1e-3) and
            math.isclose(foo.nodes['Volume'].thermo._H, -61674.264, abs_tol=1e-3))
    
