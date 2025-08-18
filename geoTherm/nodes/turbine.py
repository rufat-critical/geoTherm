import numpy as np
from .baseNodes.baseTurbo import (Turbo, TurboSizer, turboParameters, baseTurbo,
                                  baseInertantTurbo, FixedFlowTurbo,
                                  FixedPressureRatioTurbo)
from .baseNodes.baseFlow import baseFlowResistor
from ..units import addQuantityProperty, inputParser
from ..utils import turb_axial_eta, turb_radial_eta
from ..flow_funcs import _dH_isentropic
from scipy.optimize import fsolve
from .baseNodes.baseFlow import baseInertantFlow
from ..flow_funcs import FlowModel
from scipy.optimize import root_scalar
from ..logger import logger
from .flowDevices import fixedFlow
from ..decorators import state_dict


class turbineParameters(turboParameters):

    @property
    def phi(self):
        return self._Q_out/(self._D**2*self._u_tip)

    @property
    def psi(self):

        dH = self._get_dH()

        return -dH/self._u_tip**2

    @property
    def _Ns(self):
        """ Turbine Specific Speed Dimensional in SI """
        return self.rotor_node.N*np.sqrt(self._Q_out)/(-self._dH_is)**(0.75)

    @property
    def ns(self):
        """ Turbine Specific Speed Dimensionless """
        return self.rotor_node.Nrad*np.sqrt(self._Q_out)/(-self._dH_is)**(0.75)

    @property
    def _Ds(self):
        """ Turbine Specific Diameter"""
        return self._D/np.sqrt(self._Q_out)*(-self._dH_is)**0.25

    @property
    def AN2(self):
        return self.rotor_node.N**2*self._Q_out


    @property
    def _dH(self):
        return self._dH_is*self.eta


@addQuantityProperty
class baseTurbine(turbineParameters, baseTurbo, baseFlowResistor):
    """Base turbine class implementing isentropic expansion calculations.

    Extends baseTurbo with specific turbine functionality for:
    - Isentropic enthalpy change calculation
    - Outlet state determination
    """

    _units = baseTurbo._units | turbineParameters._units


@addQuantityProperty
class baseInertantTurbine(turbineParameters, baseInertantTurbo):
    """Base inertant turbine class."""

    _units = baseInertantTurbo._units | turbineParameters._units


@addQuantityProperty
class Turbine(baseTurbine):

    _units = baseTurbo._units | baseTurbine._units

    def __init__(self, name, US, DS, rotor, eta=None,
                 flow_func='isen', map=None, cdA=None):


        if flow_func == 'map':
            # Check if map is provided
            if map is None:
                logger.critical('Map is required for flow_func=map')
            self.flow = FlowModel(flow_func, map)
        else:
            # Check if cdA is provided
            if cdA is None:
                logger.critical('cdA is required for flow_func=isen')
            self.flow = FlowModel(flow_func, cdA)

        baseTurbine.__init__(self,
                             name=name,
                             US=US,
                             DS=DS,
                             rotor=rotor,
                             eta=eta)

    def evaluate(self):
        US, DS, flow_sign = self.thermostates()
        self._w = self.flow._w(US, DS, N=self.rotor_node.N)*flow_sign

    def get_outlet_state(self, US, *, w=None, PR=None):
        if w is not None:
            dP, error = self.flow._dP(US, w)
        else:
            dP = -US._P * (1 - PR)

        if dP is None:
            dP = -1e9

        if US._P + dP <= 0:
            return {'P': US._P + dP, 'H': US._H}

        dH_is = self.flow._dH(US, US._P + dP)
        eta = self.eta_func.evaluate(US,
                                     US._P + dP,
                                     self.rotor_node.N,
                                     self.model)
        dH = dH_is*eta

        return {'P': US._P + dP, 'H': US._H + dH}


class FixedFlowTurbine(turbineParameters, FixedFlowTurbo):
    """
    Turbine class where mass flow is fixed to initialization value.
    """

    # State Bounds, defining them here for now
    # In the future can make a control class to check/update bounds
    _displayVars = ['w', 'eta', 'dH', 'W', 'PR']
    _bounds = [0, 1]


    @state_dict
    def _state_dict(self):
        """
        Get the state dictionary containing the node's current state
        information.
        """

        return {'eta': self.eta}

    def get_outlet_state(self, US, *, w=None, PR=None):

        dH_is = _dH_isentropic(US, US._P*PR)

        dH = dH_is*self.eta_func.evaluate(US,
                                          US._P*PR,
                                          self.rotor_node.N,
                                          self.model)

        return {'H': US._H + dH, 'P': US._P*PR}


@addQuantityProperty
class FixedPressureRatioTurbine(turbineParameters, FixedPressureRatioTurbo):
    """Turbine with a fixed pressure ratio.

    A turbine that maintains a constant ratio between inlet and outlet pressure.
    Inherits initialization parameters from FixedPressureRatioTurbo.

    Args:
        name (str): Component identifier
        US: Upstream node reference
        DS: Downstream node reference
        PR (float): Fixed pressure ratio (inlet/outlet pressure)
        w (float): Mass flow rate [kg/s]
        eta (float): Isentropic efficiency
        Z (tuple, optional): Tuple of (value, units) for compressibility factor,
                           defaults to (1, 'm**-3')

    Attributes:
        _bounds (list): Flow rate bounds [min, max] in kg/s, set to [0, 1e5]
    
    Note:
        Unlike the base turbo class, turbine flow bounds are restricted to positive values only.
    """

    _bounds = [0, 1e5]

    _units = baseInertantFlow._units | baseTurbine._units

    def get_outlet_state(self, US, *, w=None, PR=None):
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


class FixedPRTurbine(FixedPressureRatioTurbine):
    """I refer to this in some of the old code. Need refactor and delete"""
    pass


class CdATurbine(baseTurbine, baseFlowResistor):

    def __init__(self, name, US, DS, rotor, eta, cdA, flow_func='isen'):
        baseTurbine.__init__(self, name=name, US=US, DS=DS, rotor=rotor, eta=eta)
        self.flow = FlowModel(flow_func, cdA)

    def get_outlet_state(self, US, *, w=None, PR=None):

        if w is not None:
            dP, error = self.flow._dP(US, w)
        else:
            dP = -US._P * (1 - PR)

        if dP is None:
            dP = -1e9

        dH_is = _dH_isentropic(US, US._P + dP)
        dH = dH_is*self.eta_func.evaluate(US,
                                          US._P + dP,
                                          self.rotor_node.N,
                                          self.model)

        return {'P': US._P + dP, 'H': US._H + dH}


class TurbineMap(baseTurbine):

    def __init__(self, name, US, DS, rotor, map):

        self.map = map
        super().__init__(name, US, DS, rotor, eta=map.eta_func)

    def get_outlet_state(self, US, *, w=None, PR=None):

        from pdb import set_trace
        set_trace()

        self.eta = self.map.get_eta_ts(US._P, US._T, US._P*PR, self.rotor_node.N)

        return super().get_outlet_state(US, w, PR)



@addQuantityProperty
class chokedTurbine(baseTurbine):

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
    def _dH_is_tot(self):
        US, DS, _ = self.thermostates()

        self.evaluate

        return self.flow._dH(US, self.total._P)

    def _get_dP(self):
        from pdb import set_trace
        set_trace()

    def get_outlet_state(self, US, *, w=None, PR=None):
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




from geoTherm.utils import TurbineInterpolator


class TurbineMap(FixedFlowTurbine):

    def __init__(self, name, US, DS, rotor, map, flow_func='isentropic'):

        super().__init__(name, US, DS,w=0,eta=1, flow_func=flow_func)

        self.rotor = rotor
        self.map = map

    def initialize(self, model):
        super().initialize(model)

        self.rotor_node = self.model.nodes[self.rotor]

    def evaluate(self):
        self._w = self.map.get_massflow(self.US_node.thermo._P,
                                        self.US_node.thermo._T,
                                        self.DS_node.thermo._P,
                                        self.rotor_node.N)
        self.eta = self.map.get_eta_ts(self.US_node.thermo._P,
                                       self.US_node.thermo._T,
                                       self.DS_node.thermo._P,
                                       self.rotor_node.N)

    def get_outlet_state(self, US, *, w=None, PR=None):
        self._w = self.map.get_massflow(US._P,
                                        US._T,
                                        US._P*PR,
                                        self.rotor_node.N)
        self.eta = self.map.get_eta_ts(US._P,
                                       US._T,
                                       US._P*PR,
                                       self.rotor_node.N)

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

    def get_outlet_state(self, US, *, w=None, PR=None):
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
class Turbine2(Turbo, turbineParameters):

        
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


