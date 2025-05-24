from .baseNodes.baseFlow import baseInertantFlow
from geoTherm.geometry import external
from geoTherm import pressure_drop, HTC


class ExternalTubeBank(baseInertantFlow):

    def __init__(self, name, US, DS, geometry, w, dP_model=None, HTC_model=None):
        super().__init__(name, US, DS, w)

        self.geometry = geometry

        if dP_model is None:
            if isinstance(geometry, external.tube_bank.ExternalCircularFinnedTubeBank):
                dP_model = pressure_drop.external.tube_bank.ESDU
                self.dP_model = dP_model(self.geometry)
            else:
                from pdb import set_trace
                set_trace()
        else:
            from pdb import set_trace
            set_trace()

        if HTC_model is None:
            if isinstance(geometry, external.tube_bank.ExternalCircularFinnedTubeBank):
                HTC_model = HTC.external.tube_bank.Briggs_Young
                self.HTC_model = HTC_model(self.geometry)
            else:
                from pdb import set_trace
                set_trace()
        else:
            from pdb import set_trace
            set_trace()

    @property
    def _dP(self):

        US, _, _ = self.thermostates()

        return self.dP_model.evaluate(US, self._w)

    def get_outlet_state(self, US, w):

        dP = self.dP_model.evaluate(US, w)
        return {'P': US._P + dP, 'H': US._H}
