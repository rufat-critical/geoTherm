import geoTherm as gt
import math


def test_fixed_flow():

    fluid = 'Water'
    foo = gt.Model([gt.Boundary(name='Inlet', fluid=fluid, P=(1000, 'psi'), T=(600, 'degK')),
                    gt.fixedFlowTurbine(name='Turb', eta=1, w=50, US='Inlet', DS='Outlet'),
                    gt.POutlet(name='Outlet', fluid=fluid, P=(14.7, 'psi'), T=(300, 'degK'))])
                    
    PR = foo.nodes['Turb'].PR
    Pin = foo.nodes['Inlet'].thermo._P
    Pout = foo.nodes['Outlet'].thermo._P

    assert(math.isclose(PR, Pout/Pin, abs_tol = 1e-5))
    
    
def test_fixed_PR():

    fluid = 'Water'
    foo = gt.Model([gt.Boundary(name='Inlet', fluid=fluid, P=(1000, 'psi'), T=(600, 'degK')),
                    gt.fixedPRTurbine(name='Turb', eta=1, PR=20, US='Inlet', DS='Vol', w=5),
                    gt.Station(name='Vol', fluid=fluid),
                    gt.Pipe(name='P1', US='Vol', DS='Outlet', L=1, D=.01),
                    gt.POutlet(name='Outlet', fluid=fluid, P=(14.7, 'psi'), T=(300, 'degK'))])

    foo.solve_steady()

    PR = foo.nodes['Turb'].PR
    Pin = foo.nodes['Inlet'].thermo._P
    Pout = foo.nodes['Vol'].thermo._P
    Tout = foo.nodes['Outlet'].thermo._T

    assert(math.isclose(PR, Pout/Pin, abs_tol = 1e-5))
    assert(math.isclose(Tout, 373.132, abs_tol =1e-3))
        


    

