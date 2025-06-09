from .baseNodes.baseFlow import baseFlowResistor, FixedFlow
from geoTherm.common import inputParser, addQuantityProperty
from geoTherm.utils import cv_2in
from scipy.interpolate import interp1d
from geoTherm.flow_funcs import CvFlow
import numpy as np
from geoTherm.units import toSI, fromSI
import geoTherm as gt
from geoTherm.maps.valve.ball import CvMapSI

@addQuantityProperty
class BallValve(baseFlowResistor):

    _units = baseFlowResistor._units | {'Cv': 'FLOWCOEFFICIENT'}

    def __init__(self, name, US, DS, position, CvMap=None):
        super().__init__(name, US, DS)
        self.position = position

        self.CvMap = CvMap

        if self.CvMap is None:
            self.CvMap = CvMapSI()

        self.flow = CvFlow(self.CvMap.get_cv(self.position))

    def evaluate(self):

        US, DS, _ = self.thermostates()

        #gt.units.output = 'english'
        self.flow._Cv = self._Cv
        #gt.units.output = 'SI'
        #self._Cv = self.CvMap._get_cv(self.position)

        #self._w2 = self.flow._w(US, DS, self.Cv)

        #self.flow2._Cv = self.CvMap2.get_cv(self.position)
        if US.phase in ['two-phase']:
            self._w = self.get_US_w()
            return
        
        
        self._w = self.flow._w(US, DS)


        #self.flow2._Cv = self._Cv
        #self._w2 = self.flow2._w(US, DS)

        #if np.abs(self._w2 - self._w) > .1:
       #     from pdb import set_trace
            #set_trace()

    @property
    def _Cv(self):
        return self.CvMap.get_cv(self.position)

    def _w_max(self, US):
        if US.phase in ['two-phase']:
            return self.get_US_w()
        else:
            return self.flow._w_max(US)

    def get_outlet_state(self, US, w):

        self.flow._Cv = self._Cv
        
        #dP2 = self.flow._dP(US, w)

        #self.flow._Cv = self.CvMap2.get_cv(self.position)
        dP = self.flow._dP(US, w)

        #if np.abs(dP2 - dP) > 100:
        #    from pdb import set_trace
        #    set_trace()

        if US.phase in ['two-phase']:
            return {'P': US._P, 'H': US._H}
            #_US, _DS, _ = self.thermostates()
            #if US._P > DS._P:
            #    return {'P': _DS._P, 'H': US._H}
            #else:
            #    return {'P': _US._P, 'H': US._H}

        return {'P': US._P + dP, 'H': US._H}


    def get_cv(self):
        from pdb import set_trace
        set_trace()
        #interpolate cv_curve

    def get_US_w(self):
        return self.US_node.US_nodes[0]._w


class fixedFlowValve(FixedFlow):

    _units = FixedFlow._units | {'Cv': 'FLOWCOEFFICIENT'}

    def __init__(self, name, US, DS, w:'MASSFLOW', CvMap=None):
        super().__init__(name, US, DS, w)
        self.CvMap = CvMap
        if self.CvMap is None:
            self.CvMap = CvMapSI()

        self.flow = CvFlow(1)

    @property
    def _Cv(self):
        US, DS, _ = self.thermostates()
        self.flow.update_Cv(US, DS, self._w)
        return self.flow._Cv


    @property
    def position(self):
        Cv = self._Cv
        return self.CvMap.get_angle(Cv)


class FlowModel:

    def __init__(self, cv):
        self.cv = cv

    def _w(self, US, DS, Cv):
        return CvFlow._w2(US, DS, Cv)

    def choked_state(self, US):
        DS = US.copy()
        DS._HP = US._H, US._Pvap
        return DS._P, self._w(US, DS, self.cv)
    
    def _w_max(self, US):

        if US.phase in ['liquid', 'supercritical_liquid']:
            DS = US.copy()
            DS._HP = US._H, US._Pvap
            return self._w(US, DS, self.cv)

        else:
            gamma = US.gamma
            P_crit_ratio = (2 / (gamma + 1)) ** (gamma / (gamma - 1))
            DS_P = US._P * P_crit_ratio
            
            Pup_psi = US._P/6894.76
            Pdown_psi = DS_P/6894.76
            Tup_R = US._T*9/5
            deltaP_psi = (Pup_psi - Pdown_psi)
            STD = US.from_state(US.state)
            STD._TP = 288.71, 101325
            SG = STD._density / 1.225
            SCFM_max = self.cv * (0.471 * 22.67 * Pup_psi * np.sqrt(1/(SG * Tup_R)))
            ACFM = (101325/US._P) * (US._T/288.71) * SCFM_max
            w_max = ACFM * US._density/2118.881993105158

            return w_max

    def _dP(self, US, w):

        if US.phase in ['liquid', 'supercritical_liquid']:

            Q_GPM = w / US._density * 15850.3
            deltaP_psi = (Q_GPM/self.cv)**2 * self.SG(US)
            deltaP = deltaP_psi * 6894.76
            return deltaP

        else:
            from pdb import set_trace
            set_trace()

        from pdb import set_trace
        set_trace()

    @classmethod
    def SG(cls, US):

        if US.phase in ['liquid', 'supercritical_liquid']:
            return US._density / 999.0

        else:
            STD = US.from_state(US.state)
            STD._TP = 288.71, 101325
            return STD._density / 1.225