import numpy as np
from .baseClasses import statefulFlowNode
from ..units import inputParser
from ..logger import logger
from ..thermostate import thermo


class Turbo(statefulFlowNode):
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
        self.__eta = eta
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

        # Create a reference thermo object to evaluate outlet properties
        self._ref_thermo = thermo.from_state(model.nodes[self.US].thermo.state)

        self.rotor_node = model.nodes[self.rotor]

        self.US_node = model.nodes[self.US]
        self.DS_node = model.nodes[self.DS]

        return super().initialize(model)

    def _update_outlet_thermo(self):
        outlet = self.get_outlet_state()
        self._ref_thermo._HP = outlet['H'], outlet['P']

    @property
    def eta(self):
        return self.__eta

    @eta.setter
    def eta(self, eta):
        self.__eta = eta

    @property
    def _W(self):
        # Power
        dH = self._get_dH()
        return -dH*np.abs(self._w)

    @property
    def _Q_in(self):
        # Volumeetric flow in

        # Get Thermo States
        US, _ = self._get_thermo()

        # Get Upstream
        return self._w/US.thermo._density

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
        # Inlet Mach Number
        return self._c0_is/self.US_node.thermo.sound

    @property
    def Mach_out(self):
        # Outlet Mach Number
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

    @property
    def _Ns(self):
        """
        Placeholder method to be overwritten by specific flow nodes.
        """
        logger.critical(f"{self.name} of type {type(self)} is missing a "
                        "_Ns(self) property")

    @property
    def Ds(self):
        """
        Placeholder method to be overwritten by specific flow nodes.
        """
        logger.critical(f"{self.name} of type {type(self)} is missing a "
                        "Ds(self) property")

    @property
    def _dH_is(self):
        """
        Placeholder method to be overwritten by specific flow nodes.
        """
        logger.critical(f"{self.name} of type {type(self)} is missing a "
                        "_dH_is(self) property")

    @property
    def psi(self):
        """
        Placeholder method to be overwritten by specific flow nodes.
        """
        logger.critical(f"{self.name} of type {type(self)} is missing a "
                        "psi(self) property")

    @property
    def Q(self):
        "Heat flow from Turbo Machine"
        return 0


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
