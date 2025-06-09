from geoTherm.utilities.HEX import LMTD
import geoTherm as gt
from geoTherm.utilities.HEX.pinch import find_pinch_Q, find_pinch_Q_hot
from geoTherm.resistance_models.heat import DittusBoelter, Cavallini_Smith_Zecchin, Bringer_Smith

air = gt.thermo(model='incompressible', cp=1006)


def post_process(ORC, Tc):

    # Calculate the LMTD
    UA_hot, UA_hoti = LMTD.UA(ORC['Well'].thermo , ORC['HotOut'].thermo, ORC['PumpOut'].thermo, ORC['TurbIn'].thermo, ORC['HotHEX-Hot']._w, ORC['Pump']._w)

    air.TP = Tc, air.P

    Q_reject = (ORC['TurbOut'].thermo._H-ORC['PumpIn'].thermo._H)*ORC['Pump']._w
    #w_air, air_hot = find_pinch_Q_hot(ORC['TurbOut'].thermo, ORC['PumpIn'].thermo, air, ORC['Pump']._w, Q_reject, 0, 4.5)

    #UA_cold, UA_coldi = LMTD.UA(ORC['TurbOut'].thermo, ORC['PumpIn'].thermo,
    #                            air, air_hot, ORC['Pump']._w, w_air)


    #get_avg_h(ORC['Heat'].Q, ORC['PumpOut'].thermo, ORC['Pump']._w)
    #from pdb import set_trace
    #set_trace()


    data_point = {
        'T_hot': ORC['Well'].thermo._T - 273.15,
        'T_cold': Tc - 273.15,
        'P_pump': ORC['PumpIn'].P,
        'P_turbine': ORC['TurbIn'].P,
        'dp_hot': ORC['HotHEX-Cool']._dP,
        'Turb_power': ORC['Turb'].W,
        'Pump_power': ORC['Pump'].W,
        'Turb_efficiency': ORC['Turb'].eta,
        'Turb_PR': 1/ORC['Turb'].PR,
        'Pump_PR': ORC['Pump'].PR,
        'Turb_inlet_P': ORC['TurbIn'].P,
        'Turb_inlet_T': ORC['TurbIn'].T - 273.15,
        'Turb_outlet_P': ORC['TurbOut'].P,
        'Turb_outlet_T': ORC['TurbOut'].T - 273.15,
        'Turb_Q_in': ORC['Turb'].Q_in,
        'Turb_Q_out': ORC['Turb'].Q_out,
        'Turb_mass_flow': ORC['Turb'].w,
        'Wnet': ORC.performance[0],
        'System_efficiency': ORC.performance[2],
        'Heat_input': ORC['Heat'].Q,
        'water_out_T': ORC['HotOut'].thermo._T - 273.15,
        'UA_hot': UA_hot,
        #'UA_cold': UA_cold,
        'Q_reject': Q_reject,
        #'w_air': w_air,
    }

    return data_point


def get_avg_h(Q, inlet, w, n_tubes=50):


    inlet = inlet.copy()

    h_vals = []

    if inlet.phase in ['liquid', 'gas']:
        DB = DittusBoelter(Dh=(1*.0254))
        h = DB.evaluate(inlet,w/n_tubes)
        from pdb import set_trace
        set_trace()

    from pdb import set_trace
    set_trace()


    # Discretize 100 points between inlet and outlet
    # cALCULATE h
