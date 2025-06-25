import numpy as np
#from .baseTurbo import Turbo, fixedFlowTurbo, TurboSizer, turbineParameters
from .baseNodes.baseTurbo import Turbo, fixedFlowTurbo, TurboSizer, turbineParameters, baseTurbo
from ..units import addQuantityProperty, inputParser
from ..utils import turb_axial_eta, turb_radial_eta, eps
from ..flow_funcs import _dH_isentropic
from scipy.optimize import fsolve
from .baseNodes.baseFlow import baseInertantFlow
from ..flow_funcs import FlowModel
from scipy.optimize import root_scalar
from ..logger import logger
from .flowDevices import fixedFlow
from ..decorators import state_dict

@addQuantityProperty
class baseTurbine(baseTurbo, turbineParameters):
    """Base turbine class implementing isentropic expansion calculations.

    Extends baseTurbo with specific turbine functionality for:
    - Isentropic enthalpy change calculation
    - Outlet state determination
    """

    _units = baseTurbo._units | turbineParameters._units

    @property
    def _dH_is(self):
        """Calculate isentropic enthalpy change across turbine.

        Returns:
            float: Isentropic specific enthalpy change [J/kg]
        """
        US, DS, _ = self.thermostates()
        return _dH_isentropic(US, DS._P)

    @property
    def _dH(self):
        return self._dH_is*self.eta


@addQuantityProperty
class chokedTurbine(baseTurbo):

    _displayVars = ['w', 'eta_ts', 'dH_is', 'dH', 'W', 'PR']

    def __init__(self, name, US, DS, cdA, total_eta_map, rotor, Ae, flow_func='isen'):

        super().__init__(name=name, US=US, DS=DS)

        self._cdA = cdA
        self._Ae = Ae
        self.total_eta_map = total_eta_map
        self.flow = FlowModel(flow_func, self._cdA)
        self.rotor = rotor

    def initialize(self, model):
        #self.rotor.initialize()
        super().initialize(model)

        # Attach the rotor node
        self.rotor_node = self.model.nodes[self.rotor]
        # Create thermo for total node
        self.total = self.US_node.thermo.from_state(self.US_node.thermo.state)
        # Create thermo for outlet static state
        self.static = self.US_node.thermo.from_state(self.US_node.thermo.state)

    @property
    def _dH(self):
        return self._dH_is*self.eta
    

    def evaluate(self):

        US, DS, flow_sign = self.thermostates()
        
        # Calculate mass flow rate using flow model
        self._w = self.flow._w(US, DS)*flow_sign

        def find_Pt(Pt):
            # Assuming US static state is total
            dH_is_tot = self.flow._dH(US, Pt)

            # Get total efficiency
            self.eta_total = self.total_eta_map.get_efficiency(self.rotor_node.N, US._P/Pt)

            if np.isnan(self.eta_total):
                self.eta_total = 0

            dH_tot = dH_is_tot*self.eta_total
            Hout_tot = US._H + dH_tot

            self.total._HP = Hout_tot, Pt

            self.static._SP = self.total._S, DS._P


            Hs = Hout_tot - .5*(self._w/self.static._density/self._Ae)**2

            # Check if static state is converged
            return Hs - self.static._H

        try:
            root = root_scalar(find_Pt, bracket=[DS._P, DS._P*10], method='brentq')
        except:
            logger.warn(f'{self.name} choked turbine failed to converge')
            return

        # Update total state
        find_Pt(root.root)
        # Update static eta
        self.eta = (US._H - self.static._H)/(-self.flow._dH(US, DS._P))
        self.eta_ts = (US._H - self.total._H)/(-self.flow._dH(US, DS._P))

    @property
    def _dH_is(self):

        US, DS, _ = self.thermostates()
        return self.flow._dH(US, DS._P)
    
    @property
    def _dH_is_tot(self):
        US, DS, _ = self.thermostates()

        self.evaluate

        return self.flow._dH(US, self.total._P)

    def _get_dP(self):
        from pdb import set_trace
        set_trace()

    def get_outlet_state(self, US, w):
        dP, error = self.flow._dP(US, w)
        if error is not None:
            # Too much mass flow, output negative pressure
            # so solver raises pressure
            return {'P': -1e9, 'H': US._H}

        DS_P = US._P + dP
        def find_Pt(Pt, DS_P):
            # Assuming US static state is total
            dH_is_tot = self.flow._dH(US, Pt)

            # Get total efficiency
            eta_total = self.total_eta_map.get_efficiency(self.rotor_node.N, US._P/Pt)

            if np.isnan(eta_total):
                eta_total = 0

            dH_tot = dH_is_tot*eta_total
            Hout_tot = US._H + dH_tot

            self.total._HP = Hout_tot, Pt

            self.static._SP = self.total._S, DS_P


            Hs = Hout_tot - .5*(self._w/self.static._density/self._Ae)**2

            # Check if static state is converged
            return Hs - self.static._H

        try:
            root = root_scalar(find_Pt, args=(DS_P,), bracket=[DS_P, DS_P*2], method='brentq')
        except:
            from pdb import set_trace
            set_trace()


        # Update total state
        find_Pt(root.root, DS_P)

        return {'P': US._P + dP, 'H': self.static._H}

    def _get_outlet_state_PR(self, US, PR):
        # Get outlet state from PR
        # Used in network solver for when flow gets choked

        # Set Static State
        #self.static._HP = US._H, US._P*PR

        DS = US.from_state(US.state)
        DS._HP = US._H, US._P*PR
        # Calculate mass flow
        w = self.flow._w(US, DS)#self.static)

        # Get outlet state wi
        #outlet = self.get_outlet_state(US, w)

        def find_Pt(Pt, DS_P):
            # Assuming US static state is total
            dH_is_tot = self.flow._dH(US, Pt)

            # Get total efficiency
            eta_total = self.total_eta_map.get_efficiency(self.rotor_node.N, US._P/Pt)

            if np.isnan(eta_total):
                eta_total = 0

            dH_tot = dH_is_tot*eta_total
            Hout_tot = US._H + dH_tot

            self.total._HP = Hout_tot, Pt

            self.static._SP = self.total._S, DS_P


            Hs = Hout_tot - .5*(self._w/self.static._density/self._Ae)**2

            # Check if static state is converged
            return Hs - self.static._H

        try:
            root = root_scalar(find_Pt, args=(DS._P,), bracket=[DS._P, DS._P*2], method='brentq')
        except:
            from pdb import set_trace
            set_trace()

        # Update total state
        find_Pt(root.root, DS._P)

        return {'P': US._P*PR, 'H': self.static._H}


class FixedFlowTurbine(baseTurbine, fixedFlow):
    """
    Turbine class where mass flow is fixed to initialization value.
    """

    # State Bounds, defining them here for now
    # In the future can make a control class to check/update bounds
    _displayVars = ['w', 'eta', 'dH', 'W', 'PR']
    _bounds = [0, 1]

    def __init__(self, name, US, DS, w, eta, flow_func='isentropic'):
        """
        Initialize the fixed flow turbine.

        Args:
            name (str): The name of the turbine.
            US (Node): The upstream node.
            DS (Node): The downstream node.
            w (float): The mass flow rate.
            eta (float): The isentropic efficiency.
            flow_func (str): The flow function to use.
        """
        super().__init__(name, US, DS, w, flow_func=flow_func)
        self.eta = eta

    @state_dict
    def _state_dict(self):
        """
        Get the state dictionary containing the node's current state
        information.
        """

        return {'eta': self.eta}

    def get_outlet_state(self, US, PR):

        dH_is = _dH_isentropic(US, US._P*PR)
        dH = dH_is*self.eta

        return {'H': US._H + dH, 'P': US._P*PR}

    def _update_outlet_thermo(self):
        US, _, _ = self.thermostates()
        outlet = self.get_outlet_state(US, self.PR)
        self._ref_thermo._HP = outlet['H'], outlet['P']

from geoTherm.utils import TurbineInterpolator

class TurbineMap(FixedFlowTurbine):

    def __init__(self, name, US, DS, map, flow_func='isentropic'):

        super().__init__(name, US, DS,w=0,eta=1, flow_func=flow_func)

        self.map = map

    def evaluate(self):
        self._w = self.map.get_massflow(self.US_node.thermo._P,
                                        self.US_node.thermo._T,
                                        self.DS_node.thermo._P)
        self.eta = self.map.get_eta_ts(self.US_node.thermo._P,
                                       self.US_node.thermo._T,
                                       self.DS_node.thermo._P)

    def get_outlet_state(self, US, PR):
        self._w = self.map.get_massflow(US._P,
                                        US._T,
                                        US._P*PR)
        self.eta = self.map.get_eta_ts(US._P,
                                       US._T,
                                       US._P*PR)

        return {'P': US._P*PR, 'H': US._H + self._dH}


class FixedFlowTurbineMap(baseTurbine, fixedFlow):

    def __init__(self, name, US, DS, eta_map, w, flow_func='isentropic'):
        super().__init__(name, US, DS, w, flow_func=flow_func)

        if isinstance(eta_map, str):
            self.eta_map = TurbineInterpolator(eta_map)
        else:
            self.eta_map = eta_map

    def evaluate(self):

        US, DS, _ = self.thermostates()
        
        from pdb import set_trace
        set_trace()
        self.eta = self.eta_map.get_optimal_rpm(US._P/DS._P)


    def get_outlet_state(self, US, PR):
        from pdb import set_trace
        set_trace()

    @state_dict
    def _state_dict(self):
        return {'eta_map': self.eta_map.csv_file}

class FixedPRTurbine(baseInertantFlow, baseTurbine):

    _bounds = [0, 1e5]

    _units = baseInertantFlow._units | baseTurbine._units

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
        dH_is = _dH_isentropic(US, US._P / self.PR_setpoint)

        # Determine work based on compression/expansion
        if self.PR_setpoint == 1:
            dH = 0
        else:
            dH = dH_is * self.eta

        return {
            'P': US._P / self.PR_setpoint,
            'H': US._H + dH
        }

    def _get_dP(self):
        return self.US_node.thermo._P/self.PR_setpoint - self.US_node.thermo._P

    def evaluate(self):
        """Adjust flow to achieve target pressure ratio."""
        US, DS = self.US_node.thermo, self.DS_node.thermo
        DS_target = self.get_outlet_state(US, self._w)

        # Flow acceleration based on pressure error
        pressure_error = DS_target['P'] - DS._P

        self._wdot = pressure_error / self._Z

    @property
    def _dH_is(self):
        """Calculate isentropic enthalpy change."""
        US, DS = self.US_node.thermo, self.DS_node.thermo
        return _dH_isentropic(US, DS._P)


@addQuantityProperty
class simpleTurbine(baseTurbine):
    """Simple turbine model with power-law flow-pressure characteristic.

    Implements a turbine model where mass flow varies with pressure ratio
    according to: w = w_nominal * ((PR - 1)/(PR_nominal - 1))^k

    where:
    - w is mass flow rate
    - PR is pressure ratio (P_in/P_out)
    - k is the flow coefficient (default 0.1)
    """
    _displayVars = ['w', 'dH', 'dP', 'W', 'PR_nominal', 'w_nominal', 'eta']
    _units = baseTurbine._units | {
        'W': 'POWER',
        'dH': 'SPECIFICENERGY',
        'dH_is': 'SPECIFICENERGY',
        'w_nominal': 'MASSFLOW'
    }  

    @inputParser
    def __init__(self, name, US, DS, PR_nominal, w_nominal: 'MASSFLOW', eta, k=0.1): 
        """Initialize simple turbine model.

        Args:
            name (str): Component identifier
            US: Upstream node reference
            DS: Downstream node reference
            PR_nominal (float): Design point pressure ratio
            w_nominal (float): Design point mass flow rate
            eta (float): Isentropic efficiency
            k (float, optional): Flow coefficient. Defaults to 0.1
        """
        super().__init__(name, US, DS)
        self._w_nominal = w_nominal
        self.PR_nominal = PR_nominal  # Target pressure ratio
        self.eta = eta  # Isentropic efficiency
        self._k = k  # Flow coefficient

    def evaluate(self):
        """Update turbine state based on pressure ratio.

        Calculates mass flow rate based on current pressure ratio using
        the power-law relationship.
        """
        US, DS, flow_sign = self.thermostates()
        PR = np.max([1, US._P/DS._P])

        # Calculate mass flow using power-law relationship
        self._w = (self._w_nominal *
                   ((PR - 1)/(self.PR_nominal - 1))**self._k *
                   flow_sign)

    def get_outlet_state(self, US, w):
        """Calculate turbine outlet thermodynamic state for given mass flow.

        Uses inverse of the power-law relationship to determine pressure ratio
        from mass flow rate, then calculates outlet state.

        Args:
            US: Upstream thermodynamic state
            w (float): Mass flow rate [kg/s]

        Returns:
            dict: Outlet state with pressure and enthalpy
        """
        # Calculate pressure ratio from mass flow using inverse relationship
        # w/w_nominal = ((PR - 1)/(PR_nominal - 1))^k
        # PR = ((w/w_nominal)^(1/k) * (PR_nominal - 1)) + 1
        flow_ratio = abs(w/self._w_nominal)
        PR = (flow_ratio**(1/self._k) * (self.PR_nominal - 1)) + 1

        # Calculate Isentropic dH
        dH_is = _dH_isentropic(US, US._P/PR)

        return {
            'P': US._P/PR,  # P_out = P_in/PR
            'H': US._H + dH_is* self.eta
        }


@addQuantityProperty
class Turbine(Turbo, turbineParameters):

    def _get_dP(self):
        # Get Upstream Thermo
        US, _ = self._get_thermo()

        # Get delta P across Turbine
        return US._P*(1/self.PR - 1)

    def _get_dH(self):
        """Get enthalpy change across Turbine."""

        return self._dH_is*self.eta

    @property
    def _dH(self):
        return self._dH_is*self.eta

    @property
    def _dH_is(self):

        # Get Upstream Thermo
        US, _,_ = self.thermostates()

        # Isentropic Enthalpy across Turbo Component
        return _dH_isentropic(US, US._P/self.PR)

    def _set_flow(self, w):
        self._w = w
        
    def _update_eta(self):
        # Update Turbine efficiency using efficiency curves

        def hunt_eta(x):
            # Function used to find eta using fsolve

            # Set eta
            self.eta = x[0]

            if self.axial:
                eta_calc = turb_axial_eta(self.phi, self.psi, self.psi_is)
            else:
                eta_calc = turb_radial_eta(self.ns)

            return x[0]-eta_calc

        # Fsolve to find eta
        eta = fsolve(hunt_eta, self.eta)

        # Update Turbine eta and check it's converged
        eta_root = hunt_eta(eta)

        if np.abs(eta_root) > 1e-5:
            # Eta was not updated correctly
            # so go debug and figure out what happened
            from pdb import set_trace
            set_trace()

    def evaluate(self):

        if self.update_eta:
            self._update_eta()

        super().evaluate()

    def get_outlet_state(self, US, w):

        dP = US._P*(1/self.PR - 1)
        dH_is  = _dH_isentropic(US, US._P/self.PR)

        return {'P': US._P + dP,
                'H': US._H + dH_is*self.eta}


class Turbine_sizer(Turbine, TurboSizer):
    """ Turbine Class that sets shaft speed based on input Ns Ds psi or phi"""

    def _update_eta(self):
        # Update Turbine efficiency using efficiency curves

        def hunt_eta(x):
            # Function used to find eta using fsolve

            # Set eta
            self.eta = x[0]
            # Update Rotor
            self._update_rotor()
            # Update Rotor Diameter
            self._D = (self._targets['Ds']*np.sqrt(self._Q_out)
                       / (-self._dH_is)**0.25)

            if self.axial:
                eta_calc = turb_axial_eta(self.phi, self.psi, self.psi_is)
            else:
                eta_calc = turb_radial_eta(self.ns)

            return x[0]-eta_calc

        # Fsolve to find eta
        eta = fsolve(hunt_eta, self.eta)

        # Update Turbine eta and check it's converged
        eta_root = hunt_eta(eta)

        if np.abs(eta_root) > 1e-5:
            # Eta was not updated correctly
            # so go debug and figure out what happened
            from pdb import set_trace
            set_trace()


