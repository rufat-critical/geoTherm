"""
Base classes for turbomachinery components in geoTherm.

This module provides the foundational classes for modeling turbomachinery
components such as turbines, pumps, and compressors. It includes:

- Efficiency function classes for different turbomachinery types
- Parameter calculation classes for common turbomachinery properties
- Mixin classes for adding turbomachinery functionality to flow components
- Base classes for different turbomachinery configurations
- Specialized classes for fixed pressure ratio and sizing operations

Classes:
    EtaFunction: Efficiency function for basic turbomachinery
    EtaRotorFunction: Efficiency function including rotor speed
    turboParameters: Base class for turbomachinery parameter calculations
    TurboMixin: Mixin class for adding turbo functionality to flow components
    baseTurboRotor: Base class for turbomachinery with rotor dynamics
    baseTurbo: Base class for basic turbomachinery components
    baseInertantTurbo: Base class for turbomachinery with inertial effects
    FixedFlowTurbo: Base class for turbomachinery with fixed mass flow
    FixedPressureRatioTurbo: Base class for fixed pressure ratio turbomachinery
    Turbo: Comprehensive turbomachinery class with sizing capabilities
    TurboSizer: Specialized class for turbomachinery sizing operations
"""

import numpy as np
from .baseFlow import baseFlow, baseInertantFlow, FixedFlow
from ...units import inputParser, addQuantityProperty
from ...logger import logger
from ...thermostate import thermo
from ...utils import UserDefinedFunction
from ...flow_funcs import _dH_isentropic


class EtaFunction(UserDefinedFunction):
    """
    A specialized UserDefinedFunction for efficiency calculations.
    
    This class provides a standardized interface for efficiency calculations
    in turbomachinery components that don't require rotor speed information.
    
    Parameters:
        US_thermo: Upstream thermodynamic state
        Pe: Exit pressure
        model: Reference to the parent model
    """
    parameters = {'US_thermo', 'Pe', 'model'}


class EtaRotorFunction(UserDefinedFunction):
    """
    A specialized UserDefinedFunction for efficiency calculations that
    includes the rotor speed.
    
    This class extends EtaFunction to include rotor speed in efficiency
    calculations, which is necessary for many turbomachinery performance
    correlations.
    
    Parameters:
        US_thermo: Upstream thermodynamic state
        Pe: Exit pressure
        N: Rotor speed [rpm]
        model: Reference to the parent model
    """
    parameters = {'US_thermo', 'Pe', 'N', 'model'}


class turboParameters:
    """
    Base class for turbomachinery parameter calculations.
    
    This class provides common properties and calculations used across
    all turbomachinery components, including volumetric flows, velocities,
    Mach numbers, and power calculations.
    
    Attributes:
        _units (dict): Mapping of property names to their unit types
    """
    
    _units = {
        'Q_in': 'VOLUMETRICFLOW',
        'Q_out': 'VOLUMETRICFLOW',
        'U_tip': 'VELOCITY',
        'c0_is': 'VELOCITY',
        'dH_is': 'SPECIFICENERGY',
        'W': 'POWER',
        'u_tip': 'VELOCITY',
    }

    @property
    def _Q_in(self):
        """
        Calculate inlet volumetric flow rate.
        
        Returns:
            float: Inlet volumetric flow rate [m³/s]
        """
        US, _, _ = self.thermostates()
        return self._w / US._density

    @property
    def _Q_out(self):
        """
        Calculate outlet volumetric flow rate.
        
        Returns:
            float: Outlet volumetric flow rate [m³/s]
        """
        self._update_outlet_thermo()
        DS_density = self._ref_thermo._density
        return self._w / DS_density

    @property
    def _c0_is(self):
        """
        Calculate isentropic spouting velocity.
        
        The spouting velocity is the theoretical velocity that would be
        achieved in an isentropic expansion from inlet to outlet pressure.
        
        Returns:
            float: Isentropic spouting velocity [m/s]
        """
        return np.sqrt(2 * np.abs(self._dH_is))

    @property
    def Mach_in(self):
        """
        Calculate inlet Mach number.
        
        Returns:
            float: Inlet Mach number [-]
        """
        return self._c0_is / self.US_node.thermo.sound

    @property
    def Mach_out(self):
        """
        Calculate outlet Mach number.
        
        Returns:
            float or str: Outlet Mach number [-] or 'two-phase' if two-phase flow
        """
        self._update_outlet_thermo()
        
        try:
            DS_sound = self._ref_thermo.sound
            return self._c0_is / DS_sound
        except Exception:
            return 'two-phase'

    @property
    def _dH_is(self):
        """
        Calculate isentropic enthalpy change across turbomachinery.
        
        Returns:
            float: Isentropic specific enthalpy change [J/kg]
        """
        US, DS, _ = self.thermostates()
        return _dH_isentropic(US, DS._P)

    @property
    def _W(self):
        """
        Calculate power transfer to/from fluid.
        
        Negative power indicates energy addition to fluid (e.g., pump)
        Positive power indicates energy extraction from fluid (e.g., turbine)
        
        Returns:
            float: Power [W]
        """
        return -self._dH * np.abs(self._w)

    @property
    def _u_tip(self):
        return np.sqrt(self._dH_is/self.psi)
    

class TurboMixin(turboParameters):
    """
    Mixin class containing turbomachinery functionality.

    This mixin can be combined with any flow class to add turbomachinery
    capabilities including efficiency handling and power calculations.

    Attributes:
        _units (dict): Additional units for turbomachinery properties
        _displayVars (list): Variables to display for this component
    """

    _units = {
        'dH_is': 'SPECIFICENERGY',
        'dH': 'SPECIFICENERGY',
        'W': 'POWER',
        'N': 'RPM',
        'U_tip': 'VELOCITY',
        'c0_is': 'VELOCITY'}

    _displayVars = ['w', 'eta', 'dH_is', 'dH', 'W', 'PR']

    def __init__(self, rotor, eta):
        """
        Initialize the turbo mixin.

        Args:
            eta: Efficiency value or function
        """
        self.eta = eta
        self.rotor = rotor

    def initialize(self, model):
        """
        Initialize the turbo mixin with model reference.

        Args:
            model: Reference to the parent model
        """
        super().initialize(model)
        self.rotor_node = model.nodes[self.rotor]
        self._ref_thermo = thermo.from_state(model.nodes[self.US].thermo.state)

    @property
    def eta(self):
        """
        Get the current efficiency value including rotor speed.
        
        Returns:
            float: Current efficiency [-]
        """
        return self.eta_func.evaluate(
            self.US_node.thermo,
            self.DS_node.thermo._P,
            self.rotor_node.N,
            self.model
        )

    @eta.setter
    def eta(self, value):
        """
        Set the efficiency value or function including rotor speed.
        
        Args:
            value: Efficiency value or function
        """
        self.eta_func = EtaRotorFunction(value)

    @property
    def N(self):
        """
        Get the current rotor speed.
        
        Returns:
            float: Rotor speed [rpm]
        """
        return self.rotor_node.N

    @property
    def _U_tip(self):
        """
        Calculate rotor tip speed.
        
        Returns:
            float: Rotor tip speed [m/s]
        """
        return self._D * self.rotor_node.Nrad / 2

    @property
    def psi_is(self):
        """
        Calculate isentropic work coefficient.

        The work coefficient is a dimensionless parameter relating the
        isentropic work to the rotor tip speed.

        Returns:
            float: Isentropic work coefficient [-]
        """
        return np.abs(self._dH_is) / self._U_tip**2



@addQuantityProperty
class baseTurbo(TurboMixin, baseFlow):
    """
    Base class for turbomachinery components.

    This class combines flow functionality with turbomachinery capabilities
    to provide a complete base for pumps, turbines, and compressors.

    Attributes:
        _units (dict): Combined units from base classes
        _displayVars (list): Display variables from TurboMixin
    """

    _units = baseFlow._units | TurboMixin._units
    _displayVars = TurboMixin._displayVars

    def __init__(self, name, US, DS, rotor, eta):
        """
        Initialize the base turbo component.

        Args:
            name (str): Component name
            US (str): Upstream node name
            DS (str): Downstream node name
            eta: Efficiency value or function
        """
        baseFlow.__init__(self, name=name, US=US, DS=DS)
        TurboMixin.__init__(self, rotor, eta)



@addQuantityProperty
class baseInertantTurbo(TurboMixin, baseInertantFlow):
    """
    Base class for turbomachinery with inertial effects.
    
    This class extends baseInertantFlow with turbomachinery capabilities,
    allowing for dynamic modeling of turbomachinery components.
    
    Attributes:
        _units (dict): Combined units from base classes
        _displayVars (list): Display variables from TurboMixin
    """
    
    _units = baseInertantFlow._units | TurboMixin._units
    _displayVars = TurboMixin._displayVars

    @inputParser
    def __init__(self, name, US, DS, w: 'MASSFLOW', rotor, eta, Z: 'INERTANCE' = 1):
        """
        Initialize the base inertant turbo component.
        
        Args:
            name (str): Component name
            US (str): Upstream node name
            DS (str): Downstream node name
            w (float): Mass flow rate [kg/s]
            eta: Efficiency value or function
            Z (float): Flow inertance [m⁻³]
        """
        baseInertantFlow.__init__(self, name=name, US=US, DS=DS, w=w, Z=Z)
        TurboMixin.__init__(self, rotor, eta)


@addQuantityProperty
class FixedFlowTurbo(TurboMixin, FixedFlow):
    """
    Base class for turbomachinery with fixed mass flow rate.
    
    This class is useful for modeling turbomachinery where the mass flow
    rate is controlled externally and remains constant.
    
    Attributes:
        _units (dict): Combined units from base classes
        _displayVars (list): Display variables from TurboMixin
    """
    
    _units = FixedFlow._units | TurboMixin._units
    _displayVars = TurboMixin._displayVars

    @inputParser
    def __init__(self, name, US, DS, w: 'MASSFLOW', rotor, eta,
                 controller=None):
        """
        Initialize the fixed flow turbo component.
        
        Args:
            name (str): Component name
            US (str): Upstream node name
            DS (str): Downstream node name
            w (float): Fixed mass flow rate [kg/s]
            eta: Efficiency value or function
        """
        FixedFlow.__init__(self, name=name, US=US, DS=DS, w=w,
                           controller=controller)
        TurboMixin.__init__(self, rotor, eta)


@addQuantityProperty
class FixedPressureRatioTurbo(baseInertantTurbo):
    """
    Base class for turbomachinery with fixed pressure ratio.
    
    This class is designed for pumps and compressors that maintain a
    constant pressure ratio between inlet and outlet. The mass flow
    rate is adjusted dynamically to achieve the target pressure ratio.
    
    Attributes:
        _bounds (list): Flow rate bounds [min, max] in kg/s
        _units (dict): Combined units from base classes
        _displayVars (list): Variables to display for this component
    """
    
    _bounds = [-1e5, 1e5]
    _units = baseTurbo._units | baseInertantFlow._units
    _displayVars = ['w', 'PR_setpoint', 'PR', 'eta', 'W', 'Z']

    @inputParser
    def __init__(self, name, US, DS, PR, w: 'MASSFLOW', rotor, eta, Z=(1, 'm**-3')):
        """
        Initialize the fixed pressure ratio turbo component.
        
        Args:
            name (str): Component identifier
            US (str): Upstream node name
            DS (str): Downstream node name
            PR (float): Fixed pressure ratio (outlet/inlet pressure)
            w (float): Initial mass flow rate [kg/s]
            eta: Isentropic efficiency
            Z (tuple): Tuple of (value, units) for compressibility factor
        """
        super().__init__(name=name, US=US, DS=DS, w=w, rotor=rotor, eta=eta, Z=Z)
        self.PR_setpoint = PR

    def evaluate(self):
        """
        Adjust flow to achieve desired pressure ratio.
        
        This method calculates the pressure error and adjusts the mass
        flow rate to drive the system toward the target pressure ratio.
        """
        US, DS = self.US_node.thermo, self.DS_node.thermo
        DS_target = self.get_outlet_state(US, w=self._w)
        
        pressure_error = DS_target['P'] - DS._P
        self._wdot = pressure_error


class Turbo(baseFlow):
    """
    Comprehensive turbomachinery class for turbines and pumps.
    
    This class provides a complete implementation for turbomachinery
    components with extensive parameter calculations, sizing capabilities,
    and performance analysis tools.
    
    Attributes:
        _displayVars (list): Comprehensive list of display variables
        _units (dict): Units for all turbomachinery properties
        bounds (list): Flow rate bounds for the component
    """
    
    _displayVars = [
        'w', 'dP:ΔP', 'dH:ΔH', 'W', 'PR', 'Q_in', 'Q_out', 'Ns', 'Ds', 'D',
        'Mach_in', 'Mach_out', 'phi:φ', 'psi:ψ', 'psi_is:ψ_is', 'U_tip', 'eta:η'
    ]
    
    _units = {
        'w': 'MASSFLOW', 'W': 'POWER', 'dH': 'SPECIFICENERGY',
        'dP': 'PRESSURE', 'Q_in': 'VOLUMETRICFLOW', 'Q_out': 'VOLUMETRICFLOW',
        'Q': 'POWER', 'Ns': 'SPECIFICSPEED', 'Ds': 'SPECIFICDIAMETER',
        'D': 'LENGTH', 'U_tip': 'VELOCITY'
    }
    
    bounds = [-1e5, 1e5]

    @inputParser
    def __init__(self, name, US, DS, rotor, PR=2, D: 'LENGTH' = 1, eta=None,
                 w: 'MASSFLOW' = 1, axial=False):
        """
        Initialize the Turbo component.
        
        Args:
            name (str): Component name
            US (str): Upstream node name
            DS (str): Downstream node name
            rotor (str): Rotor node name
            PR (float, optional): Pressure ratio. Defaults to 2.
            D (float, optional): Rotor diameter [m]. Defaults to 1.
            eta (float, optional): Efficiency. If None, will be calculated.
            w (float, optional): Mass flow rate [kg/s]. Defaults to 1.
            axial (bool, optional): Whether axial or radial design. Defaults to False.
        """
        self.name = name
        self.eta = eta
        self.US = US
        self.DS = DS
        self.rotor = rotor
        self.PR = PR
        self._D = D
        self._w = w
        self.axial = axial
        self.penalty = False

        # Handle efficiency initialization
        if self.eta is None:
            logger.info(f"eta for {self.name} will be calculated using Claudio's Curves")
            self.update_eta = True
            self.eta = 1  # Initial value
        else:
            self.update_eta = False

        # Set default values if not provided
        if self.PR is None:
            logger.warn(f'No PR input specified for {self.name}, setting it to 1.5')
            self.PR = 1.5

        if self._w is None:
            logger.warn(f'No w input specified for {self.name}, setting it to 1')
            self._w = 1

    def initialize(self, model):
        """
        Initialize the turbo component with model references.
        
        Args:
            model: Reference to the parent model
            
        Returns:
            Result of parent class initialization
        """
        self._ref_thermo = thermo.from_state(model.nodes[self.US].thermo.state)
        self.rotor_node = model.nodes[self.rotor]
        self.US_node = model.nodes[self.US]
        self.DS_node = model.nodes[self.DS]
        
        return super().initialize(model)

    def _update_outlet_thermo(self):
        """
        Update the outlet thermodynamic state reference.
        
        This method calculates the outlet state and updates the reference
        thermodynamic object for downstream property calculations.
        """
        US, _, _ = self.thermostates()
        outlet = self.get_outlet_state(US, self._w)
        self._ref_thermo._HP = outlet['H'], outlet['P']

    @property
    def _W(self):
        """
        Calculate power transfer.
        
        Returns:
            float: Power [W]
        """
        dH = self._get_dH()
        return -dH * np.abs(self._w)


class TurboSizer(Turbo):
    """
    Specialized turbomachinery class for sizing operations.
    
    This class extends Turbo with sizing capabilities, allowing the
    specification of design parameters such as specific speed, specific
    diameter, flow coefficient, and work coefficient to determine
    optimal turbomachinery geometry.
    
    The class requires exactly 2 sizing parameters to be specified from:
    - Ns: Specific speed [rpm]
    - ns: Dimensionless specific speed [-]
    - Ds: Specific diameter [-]
    - phi: Flow coefficient [-]
    - psi: Work coefficient [-]
    """
    
    @inputParser
    def __init__(self, name, US, DS, rotor, PR=2, eta=None, w: 'MASSFLOW' = 1,
                 Ns: 'SPECIFICSPEED' = None, ns=None, Ds: 'SPECIFICDIAMETER' = None,
                 phi=None, psi=None, axial=False):
        """
        Initialize the Turbo Sizer component.
        
        Args:
            name (str): Component name
            US (str): Upstream node name
            DS (str): Downstream node name
            rotor (str): Rotor node name
            PR (float, optional): Pressure ratio. Defaults to 2.
            eta (float, optional): Efficiency. If None, will be calculated.
            w (float, optional): Mass flow rate [kg/s]. Defaults to 1.
            Ns (float, optional): Specific speed [rpm]
            ns (float, optional): Dimensionless specific speed [-]
            Ds (float, optional): Specific diameter [-]
            phi (float, optional): Flow coefficient [-]
            psi (float, optional): Work coefficient [-]
            axial (bool, optional): Whether axial or radial design. Defaults to False.
        """
        self.name = name
        self.__eta = eta
        self.US = US
        self.DS = DS
        self.rotor = rotor
        self.PR = PR
        self._w = w
        
        # Store sizing targets
        self.targets = {'Ns': Ns, 'ns': ns, 'Ds': Ds, 'phi': phi, 'psi': psi}
        self._targets = {'Ns': Ns, 'ns': ns, 'Ds': Ds, 'phi': phi, 'psi': psi}
        
        self.axial = axial

        # Handle efficiency initialization
        if self.__eta is None:
            logger.info(f"eta for {self.name} will be calculated using Claudio's Curves")
            self.update_eta = True
            self.eta = 1  # Initial value
        else:
            self.update_eta = False

        # Set default values if not provided
        if self.PR is None:
            logger.warn(f'No PR input specified for {self.name}, setting it to 1.5')
            self.PR = 1.5

        if self._w is None:
            logger.warn(f'No w input specified for {self.name}, setting it to 1')
            self._w = 1

    def initialize(self, model):
        """
        Initialize the turbo sizer and validate sizing parameters.
        
        Args:
            model: Reference to the parent model
        """
        super().initialize(model)

        # Validate that exactly 2 sizing targets are specified
        targets = [k for k, v in self.targets.items() if v is not None]
        
        if len(targets) != 2:
            logger.critical(
                f"Turbo Sizer '{self.name}' can only have 2 targets specified. "
                "Possible options are: Ns, ns, Ds, phi, psi"
            )

        self._target_NsDs()

    def _target_NsDs(self):
        """
        Calculate target specific speed and specific diameter from sizing parameters.
        
        This method converts various combinations of sizing parameters into
        the fundamental specific speed (Ns) and specific diameter (Ds) values
        needed for turbomachinery sizing.
        """
        # Calculate target specific speed
        if self.targets['Ns'] is not None:
            Ns_target = self.targets['Ns']
        elif self.targets['ns'] is not None:
            Ns_target = self.targets['ns'] * 60 / (2 * np.pi)
        else:
            # Calculate Ns from other parameters
            if self.targets['Ds'] is not None:
                Ds = self.targets['Ds']
                if self.targets['phi'] is not None:
                    phi = self.targets['phi']
                    psi = (Ds * np.sqrt(phi))**4
                elif self.targets['psi'] is not None:
                    psi = self.targets['psi']
                    phi = (psi**(0.25) / Ds)**2
                else:
                    raise ValueError("Insufficient parameters to calculate Ns")
            elif (self.targets['phi'] is not None and self.targets['psi'] is not None):
                phi = self.targets['phi']
                psi = self.targets['psi']
            else:
                raise ValueError("Insufficient parameters to calculate Ns")

            Ns_target = 60 * np.sqrt(phi) / (np.pi * psi**0.75)

        # Calculate target specific diameter
        if self.targets['Ds'] is not None:
            Ds_target = self.targets['Ds']
        else:
            # Calculate Ds from other parameters
            if self.targets['phi'] is not None:
                phi = self.targets['phi']
                psi = (60 * np.sqrt(phi) / (np.pi * Ns_target))**(4/3)
            elif self.targets['psi'] is not None:
                psi = self.targets['psi']
                phi = (Ns_target * np.pi * psi**0.75 / 60)**2
            else:
                raise ValueError("Insufficient parameters to calculate Ds")

            Ds_target = psi**(1/4) / np.sqrt(phi)

        # Store all calculated parameters
        self._targets['Ns'] = Ns_target
        self._targets['ns'] = Ns_target * (2 * np.pi) / 60
        self._targets['Ds'] = Ds_target
        self._targets['psi'] = (60 / (np.pi * Ds_target * Ns_target))**2
        self._targets['phi'] = (self._targets['psi']**(1/4) / Ds_target)**2

    def evaluate(self):
        """
        Evaluate the turbo sizer and update geometry.
        
        This method calculates the required geometry based on sizing parameters
        and updates the rotor diameter and speed accordingly.
        """
        # Calculate sizing parameters
        self._target_NsDs()

        # Update rotor diameter based on specific diameter
        self._D = (self._targets['Ds'] * np.sqrt(self._Q_out) / 
                   (-self._dH_is)**0.25)

        # Update rotor speed
        self._update_rotor()

        # Update efficiency if needed
        if self.update_eta:
            self._update_eta()

        super().evaluate()

    def _update_rotor(self):
        """
        Update rotor speed based on specific speed requirements.
        """
        self.rotor_node.N = (self._targets['Ns'] * (-self._dH_is)**0.75 / 
                             np.sqrt(self._Q_out))

    @property
    def _Ns(self):
        """
        Get turbine specific speed in SI units.
        
        Returns:
            float: Specific speed [rpm]
        """
        return super()._Ns

    @property
    def ns(self):
        """
        Get dimensionless specific speed.
        
        Returns:
            float: Dimensionless specific speed [-]
        """
        return super().ns

    @property
    def Ds(self):
        """
        Get specific diameter.
        
        Returns:
            float: Specific diameter [-]
        """
        return super().Ds

    def update_targets(self, Ns=None, ns=None, Ds=None, phi=None, psi=None):
        """
        Update sizing targets and recalculate geometry.
        
        Args:
            Ns (float, optional): Specific speed [rpm]
            ns (float, optional): Dimensionless specific speed [-]
            Ds (float, optional): Specific diameter [-]
            phi (float, optional): Flow coefficient [-]
            psi (float, optional): Work coefficient [-]
        """
        # Update targets
        self.targets = {'Ns': Ns, 'ns': ns, 'Ds': Ds, 'phi': phi, 'psi': psi}

        # Validate that exactly 2 targets are specified
        targets = [k for k, v in self.targets.items() if v is not None]
        
        if len(targets) != 2:
            logger.critical(
                f"Turbo Sizer '{self.name}' can only have 2 targets specified. "
                "Possible options are: Ns, ns, Ds, phi, psi"
            )

        self.evaluate()

