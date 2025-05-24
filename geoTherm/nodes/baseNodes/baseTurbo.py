import numpy as np
from .baseFlow import baseFlow, baseInertantFlow
from ...units import inputParser, addQuantityProperty
from ...logger import logger
from ...thermostate import thermo
from abc import ABC, abstractmethod


@addQuantityProperty
class baseTurbo(baseFlow, ABC):
    """Base class for turbomachinery (pumps, turbines, etc.).
    
    Handles common turbomachinery behavior including:
    - Pressure ratio relationships
    - Efficiency effects
    - Power calculations
    """

    _units = baseFlow._units | {
        'dH_is': 'SPECIFICENERGY',
        'dH': 'SPECIFICENERGY',
        'W': 'POWER',
    }

    _displayVars = ['w', 'eta', 'dH_is', 'dH', 'W', 'PR']

    @property
    @abstractmethod
    def _dH_is(self):
        """Calculate isentropic enthalpy change.
        Must be implemented by derived classes.
        """
        logger.critical("_dH_is must be implemented by derived class")

    @property
    def _W(self):
        """Calculate power transfer to/from fluid.

        Negative power indicates energy addition to fluid (e.g., pump)
        Positive power indicates energy extraction from fluid (e.g., turbine)

        Returns:
            float: Power [W]
        """
        return -self._dH * np.abs(self._w)

    def _set_flow(self, w):
        """Set the mass flow rate"""
        self._w = w
        return False
    
    def initialize(self, model):
        super().initialize(model)

        self._ref_thermo = thermo.from_state(model.nodes[self.US].thermo.state)

    def _update_outlet_thermo(self):

        US, _, _ = self.thermostates()
        outlet = self.get_outlet_state(US, self._w)
        self._ref_thermo._HP = outlet['H'], outlet['P']

    
class fixedFlowTurbo(baseTurbo):

    @inputParser
    def __init__(self, name,
                 US,
                 DS,
                 w:'MASSFLOW',
                 D: 'LENGTH' = 1,       # noqa
                 eta=None,
                 axial=False):
        """
        Initialize the Turbo Node.

        Args:
            name (str): Name of the turbine.
            eta (float): Efficiency of the turbine.
            US (str): Upstream node name.
            DS (str): Downstream node name.
            rotor (str): Rotor Object.
            PR (float): Pressure ratio.
            D (float, optional): Roter Diameter.
            eta(float, optional): Efficiency.
            w (object): Mass flow controller object.
            Ns (float): Specific Speed
            ns (float): Dimensionless Specific Speed
            ds (float): Dimensionless Specific Diameter
            axial (Boolean): Axial or Radial. Default is False
        """

        # Component Name
        self.name = name
        # Component Efficiency
        self.eta = eta
        # Upstream Station
        self.US = US
        # Downstream Station
        self.DS = DS
        # Rotor Diameter
        self._D = D
        # Mass Flow
        self.axial = axial
        self._w_setpoint = w
        self.penalty = False

        if self.eta is None:
            logger.info(f"eta for {self.name} will be calculated using "
                        "Claudio's Curves")
            self.update_eta = True
            # Set to an initial value
            self.eta = 1
        else:
            self.update_eta = False



class fixedPressureRatioTurbo(baseInertantFlow, baseTurbo):
    """Base class for turbomachinery (pumps/compressors) with a fixed pressure ratio.

    Args:
        name (str): Component identifier
        US: Upstream node reference
        DS: Downstream node reference
        PR (float): Fixed pressure ratio (outlet/inlet pressure)
        w (float): Mass flow rate [kg/s]
        eta (float): Isentropic efficiency
        Z (tuple): Tuple of (value, units) for compressibility factor, defaults to (1, 'm**-3')

    Attributes:
        _bounds (list): Flow rate bounds [min, max] in kg/s, set to [-1e5, 1e5]
        _units (dict): Units inherited from baseTurbo class
    """

    _bounds = [-1e5, 1e5]
    _units = baseTurbo._units | baseInertantFlow._units

    _displayVars = ['w', 'PR_setpoint', 'PR', 'eta', 'W', 'Z']

    @inputParser
    def __init__(self, name, US, DS, PR, w:'MASSFLOW', eta, Z=(1, 'm**-3')):

        super().__init__(name=name, US=US, DS=DS, w=w, Z=Z)
        self.eta = eta
        self.PR_setpoint = PR

    def evaluate(self):
        """ Adjust flow to achieve desired pressure ratio"""

        US, DS = self.US_node.thermo, self.DS_node.thermo
        DS_target = self.get_outlet_state(US, self._w)

        pressure_error = DS_target['P'] - DS._P

        self._wdot = pressure_error


class Turbo(baseFlow):
    """Base Turbo Class for turbines and pumps."""

    _displayVars = ['w', 'dP:\u0394P', 'dH:\u0394H', 'W', 'PR', 'Q_in',
                    'Q_out', 'Ns', 'Ds', 'D', 'Mach_in', 'Mach_out',
                    'phi:\u03C6', 'psi:\u03C8', 'psi_is:\u03C8_is', 'U_tip',
                    'eta:\u03B7']

    _units = {'w': 'MASSFLOW', 'W': 'POWER', 'dH': 'SPECIFICENERGY',
              'dP': 'PRESSURE', 'Q_in': 'VOLUMETRICFLOW',
              'Q_out': 'VOLUMETRICFLOW', 'Q': 'POWER', 'Ns': 'SPECIFICSPEED',
              'Ds': 'SPECIFICDIAMETER', 'D': 'LENGTH', 'U_tip': 'VELOCITY'}

    # Bounds on flow variables
    bounds = [-1e5, 1e5]

    @inputParser
    def __init__(self, name,
                 US,
                 DS,
                 rotor,
                 PR=2,
                 D: 'LENGTH' = 1,       # noqa
                 eta=None,
                 w: 'MASSFLOW' = 1,     # noqa
                 axial=False):
        """
        Initialize the Turbo Node.

        Args:
            name (str): Name of the turbine.
            eta (float): Efficiency of the turbine.
            US (str): Upstream node name.
            DS (str): Downstream node name.
            rotor (str): Rotor Object.
            PR (float): Pressure ratio.
            D (float, optional): Roter Diameter.
            eta(float, optional): Efficiency.
            w (float, optional): Mass flow rate.
            Ns (float): Specific Speed
            ns (float): Dimensionless Specific Speed
            ds (float): Dimensionless Specific Diameter
            axial (Boolean): Axial or Radial. Default is False
        """

        # Component Name
        self.name = name
        # Component Efficiency
        self.eta = eta
        # Upstream Station
        self.US = US
        # Downstream Station
        self.DS = DS
        # Rotor Object
        self.rotor = rotor
        # Pressure Ratio
        self.PR = PR
        # Rotor Diameter
        self._D = D
        # Mass Flow
        self._w = w

        self.axial = axial

        self.penalty = False

        if self.eta is None:
            logger.info(f"eta for {self.name} will be calculated using "
                        "Claudio's Curves")
            self.update_eta = True
            # Set to an initial value
            self.eta = 1
        else:
            self.update_eta = False

        if self.PR is None:
            msg = f'No PR input specified for {self.name}, setting it to 2'
            logger.warn(msg)
            self.PR = 1.5

        if self._w is None:
            msg = f'No w input specified for {self.name}, setting it to 1'
            logger.warn(msg)
            self._w = 1

    def initialize(self, model):

        # Create a reference thermo object to evaluate outlet properties
        self._ref_thermo = thermo.from_state(model.nodes[self.US].thermo.state)

        self.rotor_node = model.nodes[self.rotor]

        self.US_node = model.nodes[self.US]
        self.DS_node = model.nodes[self.DS]

        return super().initialize(model)

    def _update_outlet_thermo(self):
        
        US, _, _ = self.thermostates()
        outlet = self.get_outlet_state(US, self._w)
        self._ref_thermo._HP = outlet['H'], outlet['P']

    @property
    def _W(self):
        # Power
        dH = self._get_dH()
        return -dH*np.abs(self._w)


#class InertantTurbo(Turbo, baseInertantFlow):
#    pass

class turboParameters:

    _units = {'Q_in': 'VOLUMETRICFLOW',
              'Q_out': 'VOLUMETRICFLOW',
              'U_tip': 'VELOCITY',
              'c0_is': 'VELOCITY',
              'N': 'RPM'}

    @property
    def _Q_in(self):
        # Volumeetric flow in

        # Get Thermo States
        US, _, _ = self.thermostates()

        # Get Upstream
        return self._w/US._density

    @property
    def _Q_out(self):
        # Volumetric Flow Out

        self._update_outlet_thermo()
        DS_density = self._ref_thermo._density

        # Get Downstream node density
        return self._w/DS_density

    @property
    def _U_tip(self):
        # Rotor Tip Speed
        return self._D*self.rotor_node.Nrad/2

    @property
    def psi_is(self):
        # Isentropic Work Coefficient
        return np.abs(self._dH_is)/self._U_tip**2

    @property
    def _c0_is(self):
        # Isentropic spouting velocity
        return np.sqrt(2*np.abs(self._dH_is))

    @property
    def Mach_in(self):
        # Mach Number using upstream sound speed

        return self._c0_is/self.US_node.thermo.sound

    @property
    def Mach_out(self):
        # Mach Number using downstream sound speed
        self._update_outlet_thermo()

        try:
            DS_sound = self._ref_thermo.sound
        except Exception:
            return 'two-phase'

        return self._c0_is/DS_sound

    @property
    def N(self):
        # Get the rotor shaft speed in RPM
        return self.rotor_node.N


class pumpParameters(turboParameters):

    _units = turboParameters._units | {'NPSP': 'PRESSURE',
                                       'u_tip': 'VELOCITY'}

    @property
    def _NPSP(self):
        try:
            return self.US_node.thermo._P - self._ref_thermo._Pvap
        except:
            return 0

    @property
    def _u_tip(self):
        return np.sqrt(self._dH_is/self.psi)

    @property
    def _Ds(self):
        """ Pump Specific Diameter """
        return self._D/np.sqrt(self._Q_in)*(self._dH_is)**0.25

    @property
    def phi(self):
        return self._Q_in/(self._D**2*self._u_tip)

    @property
    def psi(self):
        dH = self._get_dH()

        return dH/self._U_tip**2

    @property
    def _Ns(self):
        """ Pump Specific Speed Dimensional in SI """
        return self.rotor_node.N*np.sqrt(self._Q_in)/(self._dH_is)**(0.75)

    @property
    def _NSS(self):
        return (self.rotor_node.N*np.sqrt(self._Q_in)
                / (np.abs(self._NPSP)
                   / (9.81*self.US_node.thermo._density))**0.75)


class turbineParameters(turboParameters):

    _units = turboParameters._units | {'NPSP': 'PRESSURE',
                                       'u_tip': 'VELOCITY'}

    @property
    def phi(self):
        return self._Q_out/(self._D**2*self._U_tip)

    @property
    def psi(self):

        dH = self._get_dH()

        return -dH/self._U_tip**2

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


class TurboSizer(Turbo):

    @inputParser
    def __init__(self, name,
                 US,
                 DS,
                 rotor,
                 PR=2,
                 eta=None,
                 w: 'MASSFLOW' = 1,                 # noqa
                 Ns: 'SPECIFICSPEED' = None,        # noqa
                 ns=None,
                 Ds: 'SPECIFICDIAMETER' = None,     # noqa
                 phi=None,
                 psi=None,
                 axial=False):
        """
        Initialize the Turbo Node.

        Args:
            name (str): Name of the turbine.
            eta (float): Efficiency of the turbine.
            US (str): Upstream node name.
            DS (str): Downstream node name.
            rotor (str): Rotor Object.
            PR (float): Pressure ratio.
            D (float, optional): Roter Diameter.
            eta(float, optional): Efficiency.
            w (float, optional): Mass flow rate.
            Ns (float): Specific Speed
            ns (float): Dimensionless Specific Speed
            ds (float): Dimensionless Specific Diameter
            axial (Boolean): Axial or Radial. Default is False
        """

        # Component Name
        self.name = name
        # Component Efficiency
        self.__eta = eta
        # Upstream Station
        self.US = US
        # Downstream Station
        self.DS = DS
        # Rotor Object
        self.rotor = rotor
        # Pressure Ratio
        self.PR = PR

        # Mass Flow
        self._w = w

        self.targets = {'Ns': Ns,
                        'ns': ns,
                        'Ds': Ds,
                        'phi': phi,
                        'psi': psi}

        self._targets = {'Ns': Ns,
                         'ns': ns,
                         'Ds': Ds,
                         'phi': phi,
                         'psi': psi}

        self.axial = axial

        if self.__eta is None:
            logger.info(f"eta for {self.name} will be calculated using "
                        "Claudio's Curves")
            self.update_eta = True
            # Set to an initial value
            self.eta = 1
        else:
            self.update_eta = False

        if self.PR is None:
            msg = f'No PR input specified for {self.name}, setting it to 2'
            logger.warn(msg)
            self.PR = 1.5

        if self._w is None:
            msg = f'No w input specified for {self.name}, setting it to 1'
            logger.warn(msg)
            self._w = 1

    def initialize(self, model):
        super().initialize(model)

        # Sort thru targets list and make sure only 2 targets are specified
        targets = [k for k, v in self.targets.items() if v is not None]

        if len(targets) != 2:
            logger.critical(f"Turbo Sizer '{self.name} can only have 2 "
                            "targets specified. Possible options are:\n"
                            'Ns, ns, Ds, phi, psi')

        self._target_NsDs()

    def _target_NsDs2(self):
        # Get the target Ns and Ds from specified targets

        if self.targets['Ns'] is not None:
            Ns_target = self.targets['Ns']
        elif self.targets['ns'] is not None:
            Ns_target = self.targets['ns']*60/(2*np.pi)
        else:
            if self.targets['Ds'] is not None:
                Ds = self.targets['Ds']
                if self.targets['phi'] is not None:
                    phi = self.targets['phi']
                    psi = (Ds*np.sqrt(phi))**4
                elif self.targets['psi'] is not None:
                    psi = self.targets['psi']
                    phi = (psi**(0.25)/Ds)**2
                else:
                    from pdb import set_trace
                    set_trace()
            elif (self.targets['phi'] is not None
                  and self.targets['psi'] is not None):
                phi = self.targets['phi']
                psi = self.targets['psi']
            else:
                from pdb import set_trace
                set_trace()

            Ns_target = 60*np.sqrt(phi)/(np.pi*psi**0.75)

        if self.targets['Ds'] is not None:
            Ds_target = self.targets['Ds']
        else:

            if self.targets['phi'] is not None:
                phi = self.targets['phi']
                psi = (60*np.sqrt(phi)/(np.pi*Ns_target))**(4/3)
            elif self.targets['psi'] is not None:
                psi = self.targets['psi']
                phi = (Ns_target*np.pi*psi**.75/60)**2
            else:
                from pdb import set_trace
                set_trace()

            Ds_target = psi**(1/4)/np.sqrt(phi)

        return Ns_target, Ds_target

    def _target_NsDs(self):
        # Get the target Ns and Ds from specified targets

        if self.targets['Ns'] is not None:
            Ns_target = self.targets['Ns']
            self._targets['Ns'] = Ns_target
        elif self.targets['ns'] is not None:
            Ns_target = self.targets['ns']*60/(2*np.pi)
        else:
            if self.targets['Ds'] is not None:
                Ds = self.targets['Ds']
                if self.targets['phi'] is not None:
                    phi = self.targets['phi']
                    psi = (Ds*np.sqrt(phi))**4
                elif self.targets['psi'] is not None:
                    psi = self.targets['psi']
                    phi = (psi**(0.25)/Ds)**2
                else:
                    from pdb import set_trace
                    set_trace()
            elif (self.targets['phi'] is not None
                  and self.targets['psi'] is not None):
                phi = self.targets['phi']
                psi = self.targets['psi']
            else:
                from pdb import set_trace
                set_trace()

            Ns_target = 60*np.sqrt(phi)/(np.pi*psi**0.75)

        if self.targets['Ds'] is not None:
            Ds_target = self.targets['Ds']
        else:

            if self.targets['phi'] is not None:
                phi = self.targets['phi']
                psi = (60*np.sqrt(phi)/(np.pi*Ns_target))**(4/3)
            elif self.targets['psi'] is not None:
                psi = self.targets['psi']
                phi = (Ns_target*np.pi*psi**.75/60)**2
            else:
                from pdb import set_trace
                set_trace()

            Ds_target = psi**(1/4)/np.sqrt(phi)

        # Calculate the Turbo Parameters
        self._targets['Ns'] = Ns_target
        self._targets['ns'] = Ns_target*(2*np.pi)/60
        self._targets['Ds'] = Ds_target
        self._targets['psi'] = (60/(np.pi*Ds_target*Ns_target))**2
        self._targets['phi'] = (self._targets['psi']**(1/4)/Ds_target)**2

    def evaluate(self):
        # Calculate Node and rotor parameters

        # Calculate Ns, Ds, phi, psi
        self._target_NsDs()

        # Update Rotor Diameter
        self._D = self._targets['Ds']*np.sqrt(self._Q_out)/(-self._dH_is)**0.25

        # Update the rotor speed
        self._update_rotor()

        # Check if eta is fixed or being updated
        if self.update_eta:
            self._update_eta()

        super().evaluate()

    def _update_rotor(self):
        self.rotor_node.N = (self._targets['Ns']*(-self._dH_is)**0.75
                             / np.sqrt(self._Q_out))

    @property
    def _Ns(self):
        """ Turbine Specific Speed Dimensional in SI """
        return super()._Ns

    @property
    def ns(self):
        """ Turbine Specific Speed Dimensionless """
        return super().ns

    @property
    def Ds(self):
        return super().Ds

    def update_targets(self, Ns=None, ns=None, Ds=None, phi=None, psi=None):

        # Check if more than 2 specified
        self.targets = {'Ns': Ns,
                        'ns': ns,
                        'Ds': Ds,
                        'phi': phi,
                        'psi': psi}

        # Sort thru targets list and make sure only 2 targets are specified
        targets = [k for k, v in self.targets.items() if v is not None]

        if len(targets) != 2:
            logger.critical(f"Turbo Sizer '{self.name} can only have 2 "
                            "targets specified. Possible options are:\n"
                            'Ns, ns, Ds, phi, psi')

        self.evaluate()

