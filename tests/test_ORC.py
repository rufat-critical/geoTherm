import geoTherm as gt
import math


def test_ORC():
    fluid = 'acetone'

    acetone = gt.thermo()
    acetone.TPY = 303, 101325, fluid
    PR_turb = 5
    Pin = 3.8
    w = 1.4

            
    ORC = gt.Model([gt.Boundary(name='PumpIn', fluid=fluid, P=(3.8, 'bar'), T=319.8),
                    gt.Rotor('ORC_Rotor', N =40000),
                    gt.Rotor('Pump_Rotor', N =14009.97841),
                    gt.fixedFlowPump(name='Pump', rotor= 'Pump_Rotor', eta=0.7, PR=5, w=w, US='PumpIn', DS='PumpOut'),
                    gt.Station(name='PumpOut', fluid=fluid),
                    gt.simpleHEX(name='ORC_HEX', US = 'PumpOut', DS = 'TurbIn', w=w, Q=(3.2e6, 'BTU/hr'), dP=(1,'bar'), D=(2, 'in'), L=3),
                    gt.Station(name='TurbIn', fluid=fluid),#, T=Thot-5, P =101325),
                    gt.Turbine(name='Turb', rotor='ORC_Rotor',US='TurbIn', DS='TurbOut', D= .057225646*2, eta=0.8, PR=5),
                    gt.Station(name='TurbOut', fluid=fluid),
                    gt.simpleHEX(name='CoolHex', US = 'TurbOut', DS = 'PumpIn', w=w, dP=(1,'bar'))])

    ORC.solve_steady()
    
    assert(math.isclose(ORC.performance[0], 81670.363, abs_tol=1e-3) 
           and math.isclose(ORC.performance[2], 8.708, abs_tol=1e-3))