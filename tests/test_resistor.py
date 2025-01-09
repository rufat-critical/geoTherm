import geoTherm as gt
import math
import pytest


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


def series_network(flow_func):
    model = gt.Model([])
    model += gt.Boundary('Inlet', fluid='water', T=(200, 'degC'), P=(10, 'bar'))
    model += gt.resistor('R1', US='Inlet', DS='V1', area=.1, flow_func=flow_func)
    model += gt.Volume('V1', fluid='water')
    model += gt.resistor('R2', US='V1', DS='V2', area=.1, flow_func=flow_func)
    model += gt.Volume('V2', fluid='water')
    model += gt.resistor('R3', US='V2', DS='V3', area=.1, flow_func=flow_func)
    model += gt.Volume('V3', fluid='water')
    model += gt.resistor('R4', US='V3', DS='Outlet', area=.1, flow_func=flow_func)
    model += gt.Boundary('Outlet', fluid='water', T=(200, 'degC'), P=(8, 'bar'))
    
    return model
    
def parallel_network(flow_func):
    model = gt.Model([])
    
    # Define boundaries
    model += gt.Boundary('Inlet', fluid='water', T=(200, 'degC'), P=(10, 'bar'))
    model += gt.Station('V1', fluid='water')
    model += gt.Station('V2', fluid='water')
    
    model += gt.resistor('R1', US='Inlet', DS='V1', area=.01, flow_func=flow_func)
    model += gt.resistor('R2', US='Inlet', DS='V1', area=.01, flow_func=flow_func)
    model += gt.resistor('R3', US='Inlet', DS='V1', area=.001, flow_func=flow_func)
    
    
    model += gt.resistor('R4', US='V1', DS='V2', area=.01,flow_func=flow_func)
    model += gt.resistor('R5', US='V1', DS='V2', area=.01, flow_func=flow_func)

    
    model += gt.resistor('R6', US='V2', DS='Outlet', area=.02, flow_func=flow_func)
    model += gt.resistor('R7', US='V2', DS='Outlet', area=.01, flow_func=flow_func)
    
    
    model += gt.Boundary('Outlet', fluid='water', T=(20, 'degC'), P=(5, 'bar'))
    return model
 
# Function to build the series-parallel network
def series_parallel_network(flow_func):
    model = gt.Model([])

    # Define boundaries
    model += gt.Boundary('Inlet', fluid='water', T=(200, 'degC'), P=(10, 'bar'))
    model += gt.Station('V1', fluid='water')
    model += gt.Station('V2', fluid='water')
    model += gt.Station('V3', fluid='water')

    # Define resistors in series and parallel
    model += gt.resistor('R1', US='Inlet', DS='V1', area=0.1, flow_func=flow_func)
    model += gt.resistor('R2', US='V1', DS='V2', area=0.1,flow_func=flow_func)
    model += gt.resistor('R3', US='V1', DS='V3', area=0.1, flow_func=flow_func)
    model += gt.resistor('R4', US='V2', DS='Outlet', area=0.1, flow_func=flow_func)
    model += gt.resistor('R5', US='V3', DS='Outlet', area=0.1, flow_func=flow_func)

    # Define outlet
    model += gt.Boundary('Outlet', fluid='water', T=(200, 'degC'), P=(8, 'bar'))

    return model

# Test to check if the model converges for each network and flow_func
@pytest.mark.parametrize("flow_func", ["incomp", "isentropic", "comp"])
@pytest.mark.parametrize("network_func", ["series_network", "parallel_network", "series_parallel_network"])
def test_network_convergence(flow_func, network_func):
    """
    Test if each network converges for various flow_func options.
    """
    # Dynamically call the network function with the flow_func argument
    model = globals()[network_func](flow_func)

    # Solve the model
    try:
        model.solve_steady()
    except Exception as e:
        pytest.fail(f"Model {network_func} failed to converge for flow_func={flow_func}. Error: {e}")

    # Check steady-state convergence
    assert model.converged, f"Model {network_func} did not converge for flow_func={flow_func}"
    
