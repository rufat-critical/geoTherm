from .baseNodes.baseFlow import baseFlow, baseInertantFlow
from ..units import inputParser

class fixedFlow(baseFlow):

    @inputParser
    def __init__(self, name, US, DS,
                 w:"MASSFLOW"):

        super().__init__(name, US, DS)

        self._w = w

    @property
    def _dP(self):

        return self.DS_node.thermo._P - self.US_node.thermo._P 

    def get_outlet_state(self, US, PR):
        """
        Calculate the thermodynamic state at the outlet (downstream).

        Returns:
            dict: A dictionary containing the enthalpy ('H') and pressure ('P')
                  at the downstream node.
        """

        # Get US, DS Thermo
        #US = self.model.nodes[self.US].thermo
        return {'H': US._H, 'P': US._P*PR}

    def get_inlet_state(self, DS, PR):

        return {'H': DS._H,
                'P': DS._P/PR}


class PressureController(baseFlow):
    
    @inputParser
    def __init__(self, name, US, DS,
                 w:'MASSFLOW',
                 dP:'PRESSURE'=0):
        
        super().__init__(name, US, DS)

        self.w_setpoint = w
        self.dP_setpoint = dP

    
    @property
    def _w(self):
        US, DS, _ = self.thermostates()

        from pdb import set_trace
        set_trace()
        return (self._w_setpoint + (DS._P - US._P)
                - self.dP_setpoint)
                            
                             
    def get_outlet_state(self, US, w):
        from pdb import set_trace
        set_trace()
