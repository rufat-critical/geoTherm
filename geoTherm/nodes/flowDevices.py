from .baseNodes.baseFlow import baseFlow, baseInertantFlow
from ..units import inputParser, addQuantityProperty
from .baseNodes.baseTurbo import baseTurbo
from ..flow_funcs import _dH_isentropic

@addQuantityProperty
class PressureController(baseInertantFlow):
    """Controls pressure by adjusting mass flow rate."""

    _displayVars = ['w', 'P', 'dP']
    _units = baseInertantFlow._units | {
        'P_setpoint': 'PRESSURE'
    }

    _bounds = [0, 1e5]
    @inputParser
    def __init__(self, name, US, DS, w:'MASSFLOW'=0, 
                 P_setpoint:'PRESSURE'=None, Z=(1, 'm**-3')):
        """Initialize pressure controller.

        Args:
            name: Component identifier
            US: Upstream node reference
            DS: Downstream node reference
            w: Initial mass flow rate
            P_setpoint: Target pressure to maintain
            Z: Flow inertance
        """
        super().__init__(name, US, DS, w, Z)
        self._P_setpoint = P_setpoint

    def get_outlet_state(self, US, w):
        """Calculate outlet state based on current conditions."""
        # Pass through enthalpy, pressure will be controlled by evaluate()
        return {'H': US._H, 'P': self._P_setpoint}

    def evaluate(self):
        """Adjust flow to achieve target pressure."""
        DS = self.DS_node.thermo

        # Flow acceleration based on pressure error
        pressure_error = self._P_setpoint - DS._P
        self._wdot = pressure_error / self._Z

@addQuantityProperty
class fixedFlow(baseFlow):

    @inputParser
    def __init__(self, name, US, DS,
                 w:"MASSFLOW"):

        super().__init__(name, US, DS)

        self._w = w

    def thermostates(self):
        if self._w > 0:
            return self.US_node.thermo, self.DS_node.thermo, 1
        else:
            return self.DS_node.thermo, self.US_node.thermo, -1

    @property
    def _dP(self):

        US, DS, _ = self.thermostates()

        return DS._P - US._P

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
                'P': DS._P}#/PR}

    def _set_flow(self, w):
        self._w = w

@addQuantityProperty
class fixedFlowPressure(baseFlow):
    # This adds flow and pressure constraint
    # if dP is too high then it updates upstream pressure
    # to achieve the desired flow and dP
    pass



@addQuantityProperty
class fixedFlowTurbo(baseTurbo, fixedFlow):
    _bounds = [1/100, 100]

    _units = baseFlow._units | {
         'dH': 'SPECIFICENERGY',
         'W': 'POWER'
     }

    _displayVars = ['w', 'PR', 'dP', 'dH', 'W']

    def __init__(self, name, US, DS, w, eta):

        super().__init__(name, US, DS, w)
        self.eta = eta

    def get_outlet_state(self, US, PR):
        """Calculate pump outlet thermodynamic state."""

        dH_is = _dH_isentropic(US, US._P*PR)

        # Check if expansion or compression
        if PR == 1:
            dH = 0
        elif PR > 1:
            # Compression
            dH = dH_is/self.eta
        else:
            # Expansion
            dH = dH_is*self.eta

        return {
            'P': US._P * PR,
            'H': US._H + dH
            }

    @property
    def _dH_is(self):

        US = self.US_node.thermo
        DS = self.DS_node.thermo

        return _dH_isentropic(US, DS._P)

    @property
    def _dH(self):
        US = self.US_node.thermo
        DS = self.DS_node.thermo

        if US._P/DS._P > 1:
            return self._dH_is*self.eta
        else:
            return self._dH_is/self.eta


class PressureController2(baseInertantFlow):
    
    @inputParser
    def __init__(self, name, US, DS,
                 w:'MASSFLOW',
                 dP:'PRESSURE'=0,
                 Z=(1e-5, 'm**-3')):
        
        super().__init__(name, US=US, DS=DS, w=w, Z=Z)

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


@addQuantityProperty
class PressureRatioTurbo(baseInertantFlow, baseTurbo):
    """Turbomachine that maintains a specified pressure ratio by adjusting flow."""

    _bounds = [0, 1e5]

    _units = baseInertantFlow._units | {
        'dH': 'SPECIFICENERGY',
        'W': 'POWER',
    }

    _displayVars = ['w', 'PR', 'dP', 'dH', 'W']

    def __init__(self, name, US, DS, PR,
                 w,
                 eta=1.0, Z=(1, 'm**-3')):
        """Initialize pressure ratio controlled turbomachine.

        Args:
            name: Component identifier
            US: Upstream node reference
            DS: Downstream node reference
            PR: Pressure ratio to maintain (P_out/P_in)
            w: Initial mass flow rate
            PR_bounds: Pressure ratio bounds
            eta: Isentropic efficiency
            Z: Flow inertance
        """
        super().__init__(name=name, US=US, DS=DS, w=w, Z=Z)
        self.PR_setpoint = PR
        self.eta = eta

    def get_outlet_state(self, US, w):
        """Calculate outlet state based on pressure ratio."""
        dH_is = _dH_isentropic(US, US._P * self.PR)

        # Determine work based on compression/expansion
        if self.PR_setpoint == 1:
            dH = 0
        elif self.PR_setpoint > 1:
            # Compression
            dH = dH_is / self.eta
        else:
            # Expansion
            dH = -dH_is * self.eta

        return {
            'P': US._P * self.PR_setpoint,
            'H': US._H + dH
        }

    def evaluate(self):
        """Adjust flow to achieve target pressure ratio."""
        US, DS, _ = self.thermostates()
        DS_target = self.get_outlet_state(US, self._w)

        # Flow acceleration based on pressure error
        pressure_error = DS_target['P'] - DS._P

        self._wdot = pressure_error / self._Z

    @property
    def _dH_is(self):
        """Calculate isentropic enthalpy change."""
        US = self.US_node.thermo
        DS = self.DS_node.thermo
        return _dH_isentropic(US, DS._P)

    @property
    def _dH(self):
        """Calculate actual enthalpy change."""
        US = self.US_node.thermo
        DS = self.DS_node.thermo

        if US._P/DS._P > 1:
            return self._dH_is * self.eta
        else:
            return self._dH_is / self.eta
