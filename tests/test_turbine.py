import geoTherm as gt
import math


def test_fixed_flow():

    fluid = 'Water'
    foo = gt.Model([gt.Boundary(name='Inlet', fluid=fluid, P=(1000, 'psi'), T=(600, 'degK')),
                    gt.FixedFlowTurbine(name='Turb', eta=1, rotor='DummyRotor', w=50, US='Inlet', DS='Outlet'),
                    gt.Rotor(name='DummyRotor', N=1000),
                    gt.POutlet(name='Outlet', fluid=fluid, P=(14.7, 'psi'), T=(300, 'degK'))])
                    
    PR = foo.nodes['Turb'].PR
    Pin = foo.nodes['Inlet'].thermo._P
    Pout = foo.nodes['Outlet'].thermo._P

    assert(math.isclose(PR, Pout/Pin, abs_tol = 1e-5))

def test_fixed_flow_variable_eta():
    
    def eta_func(US_thermo, Pe, model):
        # Eta function that is proportional to pressure ratio
        return 1-(Pe/US_thermo._P)*20
    
    fluid = 'air'
    foo = gt.Model([gt.Boundary(name='Inlet', fluid=fluid, P=(1000, 'psi'), T=(600, 'degK')),
                    gt.FixedFlowTurbine(name='Turb', eta=eta_func, rotor='DummyRotor', w=50, US='Inlet', DS='Outlet'),
                    gt.Rotor(name='DummyRotor', N=1000),
                    gt.POutlet(name='Outlet', fluid=fluid, P=(14.7, 'psi'), T=(300, 'degK'))])
                    
    PR = foo.nodes['Turb'].PR
    Pin = foo.nodes['Inlet'].thermo._P
    Pout = foo.nodes['Outlet'].thermo._P
    
    assert math.isclose(PR, Pout/Pin, abs_tol = 1e-5)
    assert math.isclose(foo['Outlet'].T, 305.21, abs_tol=1e-2)
    
def test_fixed_PR():

    fluid = 'Water'
    foo = gt.Model([gt.Boundary(name='Inlet', fluid=fluid, P=(1000, 'psi'), T=(600, 'degK')),
                    gt.FixedPressureRatioTurbine(name='Turb', eta=1, rotor='DummyRotor', PR=20, US='Inlet', DS='Vol', w=5),
                    gt.Rotor(name='DummyRotor', N=1000),
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
    
    
def test_fixed_flow_variable_eta():
    
    def eta_func(US_thermo, Pe, N, model):
        # Eta function that is proportional to pressure ratio
        return 1-(Pe/US_thermo._P)*20
    
    fluid = 'air'
    foo = gt.Model([gt.Boundary(name='Inlet', fluid=fluid, P=(1000, 'psi'), T=(600, 'degK')),
                    gt.FixedFlowTurbine(name='Turb', rotor='DummyRotor', eta=eta_func, w=50, US='Inlet', DS='Outlet'),
                    gt.Rotor(name='DummyRotor', N=1000),
                    gt.POutlet(name='Outlet', fluid=fluid, P=(14.7, 'psi'), T=(300, 'degK'))])
                    
    PR = foo.nodes['Turb'].PR
    Pin = foo.nodes['Inlet'].thermo._P
    Pout = foo.nodes['Outlet'].thermo._P
    
    assert math.isclose(PR, Pout/Pin, abs_tol = 1e-5)
    assert math.isclose(foo['Outlet'].T, 305.21, abs_tol=1e-2)
