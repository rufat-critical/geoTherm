import geoTherm as gt
import math


def test_flow():

    foo = gt.Model([gt.Boundary('Inlet', fluid='Water', T=300, P=101325*2),
                      gt.Boundary('Outlet', fluid='Water', T=300, P=101325*1.9),
                      gt.flow('flow',w=10,US='Inlet', DS='Outlet', L=1, D=.2, roughness=.5)])
    foo.solve()

    assert(math.isclose(foo.nodes['flow']._w, 21.49994, abs_tol=1e-3))
    

def test_flow2():

    foo = gt.Model([gt.Boundary('Inlet', fluid='Water', T=300, P=101325*2),
                      gt.Boundary('Outlet', fluid='Water', T=300, P=101325*1.9),
                      gt.Station('1',fluid='Water'),
                      gt.flow('F1',w=10,US='Inlet', DS='1', L=1, D=.2, roughness=.5),
                      gt.flow('F2',w=10,US='1',DS='Outlet', L=1, D=.2, roughness=.5)])
    
    foo.solve()
    assert(math.isclose(foo.nodes['F1']._w, foo.nodes['F2']._w, abs_tol=1e-6)
           and math.isclose(foo.nodes['F1']._w, 15.2026, abs_tol=1e-3))

def test_flow_heat():

    foo = gt.Model([gt.Boundary('Inlet', fluid='Water', T=300, P=101325*2),
                      gt.Boundary('Outlet', fluid='Water', T=300, P=101325*1.9),
                      gt.Station('1',fluid='Water'),
                      gt.Qdot('HeatIn',cool='1',Q=5e6),
                      gt.flow('F1',w=10,US='Inlet', DS='1', L=1, D=.2, roughness=.5),
                      gt.flow('F2',w=10,US='1',DS='Outlet', L=1, D=.2, roughness=.5)])
    
    foo.solve()
    assert(math.isclose(foo.nodes['F1']._w, foo.nodes['F2']._w, abs_tol=1e-6)
           and math.isclose(foo.nodes['F1']._w, 15.0353, abs_tol=1e-3)
           and math.isclose(foo.nodes['1'].thermo._T, 379.3266, abs_tol=1e-3))