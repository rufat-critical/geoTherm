import geoTherm as gt
import math


def test_simple_ORC():
    fluid = 'acetone'
    gt.DEFAULTS.EoS = 'HEOS'

    acetone = gt.thermo()
    acetone.TPY = 303, 101325, fluid
    PR_turb = 5
    Pin = 3.8
    w = 1.4

            
    ORC = gt.Model([gt.Boundary(name='PumpIn', fluid=fluid, P=(3.8, 'bar'), T=319.8),
                    gt.FixedFlowPump(name='Pump', rotor='Rotor', eta=0.7, w=w, US='PumpIn', DS='PumpOut'),
                    gt.Rotor(name='Rotor', N=1000),
                    gt.Station(name='PumpOut', fluid=fluid),
                    gt.FixedDP(name='ORC_HEX', US = 'PumpOut', DS = 'TurbIn', w=w, dP=(-1,'bar')),
                    gt.Qdot(name='Heat', cool='TurbIn', Q=(3.2e6, 'BTU/hr')),
                    gt.Station(name='TurbIn', fluid=fluid),#, T=Thot-5, P =101325),
                    gt.FixedPRTurbine(name='Turb', US='TurbIn', DS='TurbOut', rotor='Rotor', eta=0.8, PR=5, w=w),
                    gt.Station(name='TurbOut', fluid=fluid),
                    gt.FixedDP(name='CoolHex', US = 'TurbOut', DS = 'PumpIn', w=w, dP=(-1,'bar'))])

    ORC.solve_steady()

    assert(math.isclose(ORC.performance[0], 81670.364, abs_tol=1e-3) 
           and math.isclose(ORC.performance[2], 8.708, abs_tol=1e-3))


def test_stechmann():
    # Comparing with Stechmann's Spreadsheets
    gt.DEFAULTS.EoS = 'HEOS'
    ## Water Circuit
    HOT_fluid = 'H2O'
    ## ORC Circuit
    ORC_fluid = 'n-Pentane'
    ## Coolant Circuit
    Cool_fluid = 'H2O'

    ## Hot Well P, T
    HOT_P = (40, 'bar')
    HOT_T = (473, 'degK')

    ## ORC Pump Inlet P T
    ORC_Pin = (1.4, 'bar')
    ORC_Tin = (306, 'degK')

    # Turb Pressure Ratio
    ORC_Turb_PR = 14.28571429

    # Mass Flow
    mdot_ORC = (20, 'kg/s')
    mdot_H2O = (45, 'kg/s')

    ORC = gt.Model([gt.Boundary(name='LowT', fluid=ORC_fluid, P=ORC_Pin, T=ORC_Tin),
                    gt.FixedFlowPump(name='Pump', eta=0.7, rotor='Rotor', w=mdot_ORC, US='LowT', DS='PumpOut'),
                    gt.Rotor(name='Rotor', N=10000),
                    gt.Station(name='PumpOut', fluid=ORC_fluid),
                    gt.FixedDP(name='ORC_HEX', US='PumpOut', DS='TurbIn', dP =(0, 'bar'),w=50.232),#, Q=(20, 'MW')),
                    #gt.Qdot(name='ORC_Heat', cool='ORC_HEX', Q=(20 , 'MW')),
                    gt.Qdot('ORC_Heat', hot='WaterHEXOut', cool='TurbIn', Q=(31374557.96)),
                    gt.Station(name='TurbIn', fluid=ORC_fluid),
                    gt.FixedPRTurbine(name='Turb', eta=.9, PR=ORC_Turb_PR, rotor='Rotor', w=mdot_ORC, US='TurbIn', DS='TurbOut'),
                    gt.Station(name='TurbOut', fluid=ORC_fluid),
                    gt.FixedDP(name='CoolHex', US = 'TurbOut', DS = 'LowT', w=mdot_ORC, dP=(0,'bar'))])


    HOT = gt.Model([gt.Boundary(name='Well', fluid=HOT_fluid, P=HOT_P, T=HOT_T),
                    gt.FixedDP(name='WaterHEX', US='Well', DS='WaterHEXOut', dP=(-38, 'bar'), w=mdot_H2O),
                    gt.Volume(name='WaterHEXOut', fluid=HOT_fluid),
                    gt.FixedFlowPump(name='WaterPump', eta=.7, rotor='DummyRotor', w=mdot_H2O, US='WaterHEXOut',DS='Outlet'),     
                    gt.Rotor('DummyRotor', N =15000),
                    #gt.Qdot('ORC_Heat', hot='WaterHEXOut', Q=(31374557.96)),
                    gt.POutlet(name='Outlet', fluid=HOT_fluid, P=(140, 'bar'), T=HOT_T)])
                

    combined = gt.Model()
    combined += ORC
    combined += gt.Balance('Turbin_Temp', 'Pump.w', 'TurbIn.T', 470, knob_min=0.5, knob_max=100)
    combined += HOT
    combined.solve_steady()
    assert math.isclose(combined.performance[0], 4069798.232, abs_tol=1e-3)
    assert math.isclose(combined.performance[2], 12.972, abs_tol=1e-3)
    assert math.isclose(combined.nodes['TurbIn'].T, 470, abs_tol=1e-3)
    assert math.isclose(combined.nodes['Pump'].w, 48.194, abs_tol=1e-3)