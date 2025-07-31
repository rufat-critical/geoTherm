import geoTherm as gt
import math


def test_fixed_flow_fan():

    foo = gt.Model([gt.Boundary(name='Ambient', fluid='air', T=(20, 'degC'), P=101325*.8),
                      gt.FixedDP(name='HEX', US='Ambient', DS='Intake', dP=-100),
                      gt.Qdot(name='Hot',cool='HEX',Q=800e3),
                      gt.Station(name='Intake',fluid='air'),
                      gt.FixedFlowFan(name='Fan', US='Intake', DS='Exhaust', eta=0.6,w=50),
                      gt.POutlet(name='Exhaust',fluid='air', T=300, P=101325*.8)])
                  

    foo.solve_steady()
    
    
    assert(math.isclose(foo['Fan']._W, -9125.63886, abs_tol = 1e-5))
