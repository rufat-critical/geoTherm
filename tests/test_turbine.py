import geoTherm as gt
import math


def test_turb():

    fluid = 'Water'
    foo = gt.Model([gt.Boundary(name='Inlet', fluid=fluid, P=(1000, 'psi'), T=(600, 'degK')),
                    gt.Turbine(name='Turb', eta=1, PR=6, w=50, US='Inlet', DS='Outlet'),
                    gt.Boundary(name='Outlet', fluid=fluid, P=(14.7, 'psi'), T=(300, 'degK'))])
    
    foo.solve()
    
    PR = foo.nodes['Turb'].PR
    Pin = foo.nodes['Inlet'].thermo._P
    Pout = foo.nodes['Outlet'].thermo._P
    
    assert(math.isclose(PR, Pin/Pout, abs_tol = 1e-5))