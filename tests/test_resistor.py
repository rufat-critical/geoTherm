import geoTherm as gt
import math


def test_incomp_resistor():

    foo = gt.Model([gt.Boundary(name='Inlet', fluid='water', P=(300, 'psi'), T=300),
                      gt.resistor(name='orifice', US='Inlet', DS='Outlet', area= (1 ,'in**2'), flow_func='incomp'),
                      gt.Boundary(name='Outlet', fluid='water', P=(200, 'psi'), T=300)])
                      

    foo.solve_steady()

    assert(math.isclose(foo.nodes['orifice']._w, 23.9268, abs_tol=1e-3))
   

def test_comp_resistor():

    foo = gt.Model([gt.Boundary(name='Inlet', fluid='O2', P=(300, 'psi'), T=300),
                      gt.resistor(name='orifice', US='Inlet', DS='Outlet', area= (.1 ,'in**2'), flow_func='comp'),
                      gt.Boundary(name='Outlet', fluid='O2', P=(200, 'psi'), T=300)])
                      

    foo.solve_steady()
    assert(math.isclose(foo.nodes['orifice']._w, 0.3166, abs_tol=1e-3))
    
def test_isen_resistor():

    foo = gt.Model([gt.Boundary(name='Inlet', fluid='O2', P=(300, 'psi'), T=300),
                      gt.resistor(name='orifice', US='Inlet', DS='Outlet', area= (.1 ,'in**2'), flow_func='isentropic'),
                      gt.Boundary(name='Outlet', fluid='O2', P=(200, 'psi'), T=300)])
                      

    foo.solve_steady()
    assert(math.isclose(foo.nodes['orifice']._w, 0.3166, abs_tol=1e-3))


def test_series_resistor():
        foo = gt.Model([gt.Boundary(name='Inlet', fluid='water', P=(300, 'psi'), T=300),
                        gt.Volume(name='Vol', fluid='water'),
                      gt.resistor(name='orifice', US='Inlet', DS='Vol', area= (.1 ,'in**2'), flow_func='isentropic'),
                      gt.resistor(name='orifice2', US='Vol', DS='Outlet', area= (.1 ,'in**2'), flow_func='isentropic'),
                      gt.Boundary(name='Outlet', fluid='water', P=(200, 'psi'), T=300)])
        
        foo.solve_steady()
        assert(math.isclose(foo.nodes['Vol'].thermo._P, 1723719.623, abs_tol=1e-3)
               and math.isclose(foo.nodes['orifice']._w, 1.6916, abs_tol=1e-3))
