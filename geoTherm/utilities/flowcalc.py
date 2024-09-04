import geoTherm as gt
from ..flow_funcs import (
    _dH_isentropic, _dH_isentropic_perfect, _dH_incompressible,
    _w_isen, _dP_isen, _w_incomp, _dP_incomp, _total_to_static,
    _static_to_total, _dP_comp, dP_incomp, dP_isen, dP_comp,
    w_comp, w_incomp, w_isen, _w_comp, _w_incomp, _w_isen,
    dH_isentropic, dH_isentropic_perfect, dH_incompressible,
    Mach_isentropic, total_to_static_Mach, static_to_total_Mach,
    _Mach_total_perfect, _Mach_static_perfect, perfect_Mach_from_ratio,
    perfect_ratio_from_Mach
)
from rich.console import Console
from rich.table import Table
from ..thermostate import thermo
from ..utils import eps
from ..logger import logger
from units import addQuantityProperty, inputParser
import numpy as np



@addQuantityProperty
class flowCalc:
    """
    Flow Calculator for compressible,
    incompressible, and isentropic flow conditions.
    """
    _units = {'w_flux': 'MASSFLUX'}

    @inputParser
    def __init__(self, total=None,
                 static=None,
                 w_flux: 'MASSFLUX' = None, # noqa
                 PR=None,
                 M=None):

        self._total = total
        self._static = static
        self.__w_flux = w_flux
        self.__PR = PR
        self.__M = M
        self.initialize()

    @property
    def _w_flux(self):
        """Mass flux between total and static states"""
        U = np.sqrt(2 * (self._total._H - self._static._H))
        return self._static._density * U

    @property
    def PR(self):
        """Pressure Ratio (static pressure / total pressure)."""
        return self._static._P / self._total._P

    @property
    def M(self):
        """Mach number."""
        U = np.sqrt(2 * (self._total._H - self._static._H))
        return U / self._static.sound

    def __repr__(self):
        """
        Return a string representation of the flowCalc Object

        This method is called when the object is displayed in the console
        (e.g., by simply typing the object name in an interactive session).
        """
        return self.__makeTable()

    def __str__(self):
        """
        Return a string representation of the flowCalc Object

        This method is called when the object is printed using the print() function.
        """
        return self.__makeTable()


    def __makeTable(self, **kwargs):
        """
        Create a formatted table representation of the Model object.

        Returns:
            str: A string containing the formatted table representation of the Node.
        """

        # Initialize model if it's not yet initialized
        # self.initialize()

        # Parameters to plot:
        params = ['P', 'PR', 'dP', 'T', 'TR', 'H', 'HR', 'dH', 'D', 'DR', 'a',
                  'M', 'U', 'w_flux', 'gamma', 'AR']

        table = Table(title='Flow Calcs')
        # Add columns for Node name and its parameters
        table.add_column('Properties')
        table.add_column(self.boundary['label'])
        table.add_column(self.isentropic['label'])
        table.add_column(self.incompressible['label'])
        table.add_column(self.compressible['label'])



        for param in params:
            param_row = [param]
            if param in self.boundary:
                param_row += [f"{self.boundary[param]:.3f}"]
            else:
                param_row += ['']

            vals = [self.isentropic[param], self.incompressible[param],
                    self.compressible[param]]
            for val in vals:
                if isinstance(val, (float, int)):
                    param_row += [f"{val:.3f}"]
                else:
                    param_row += [f"{val}"]

            table.add_row(*param_row)

        console = Console(**kwargs)
        with console.capture() as capture:
            # Print Node and Performance Table
            console.print(table)

        return capture.get()


    @property
    def total(self):
        return self._total
    
    @total.setter
    def total(self, total):
        # Check if thermostate
        self._total = total
        self._static = None
        self.initialize()

    @property
    def static(self):
        return self._static

    @static.setter
    def static(self, static):
        # Check if thermostate
        self._total = None
        self._static = static
        self.initialize()


    def initialize(self):
        """
        Initialize the flow calculation based on the provided parameters.
        """
        if self._total is None and self._static is None:
            logger.critical("Either a total or static state must be defined.")
        elif self._total is not None and self._static is not None:
            logger.critical("Specify either a total or static state, not both")
        elif self.__PR is None and self.__M is None and self.__w_flux is None:
            logger.critical('Either Pressure Ratio, Mach or mass flux '
                            'must be specified')

        if self._total is None:
            self.boundary = {'label': 'Static State',
                             'P': self._static.P,
                             'T': self._static.T,
                             'D': self._static.density,
                             'H': self._static.H,
                             'a': self._static.sound,
                             'gamma': self._static.gamma}
            self._total = thermo.from_state(self._static.state)
            self.update_state(static=False)
        elif self._static is None:
            self.boundary = {'label': 'Total State',
                             'P': self._total.P,
                             'T': self._total.T,
                             'D': self._total.density,
                             'H': self._total.H,
                             'a': self._total.sound,
                             'gamma': self._total.gamma}
            self._static = thermo.from_state(self._total.state)
            self.update_state(static=True)


    def update_state(self, static=True):
        """
        Update isentropic, incompressible, and compressible flow states.
        """
        self._update_isentropic(static=static, w_flux=self.__w_flux, PR=self.__PR, M=self.__M)
        self._update_incompressible(static=static, w_flux=self.__w_flux, PR=self.__PR, M=self.__M)
        self._update_compressible(static=static, w_flux=self.__w_flux, PR=self.__PR, M=self.__M)


    def _update_isentropic(self, static=True, w_flux=None, PR=None, M=None):

        if static:
            label = 'Static State\n(Isentropic)'
            if w_flux is not None:
                self._static = _total_to_static(self._total, w_flux,
                                                supersonic=False,
                                                static=self._static)
            elif PR is not None:
                self._static._SP = self._total._S, self._total._P*PR
            elif M is not None:
                self._static = total_to_static_Mach(self._total, M,
                                                    static=self._static)
            else:
                from pdb import set_trace
                set_trace()

            a = self.static.sound
            gamma = self.static.gamma
            T, P, H, D = (self.static.T, self.static.P, self.static.H,
                          self.static.density)
        else:
            label = 'Total State\n(Isentropic)'
            if w_flux is not None:
                self._total = _static_to_total(self._static, w_flux,
                                               supersonic=False,
                                               total=self._total)
            elif PR is not None:
                self._total._SP = self._static._S, self._static._P/PR
            elif M is not None:
                self._total = static_to_total_Mach(self._static, M,
                                                   total=self._total)
            a = self.total.sound
            gamma = self.total.gamma
            T, P, H, D = (self.total.T, self.total.P, self.total.H,
                          self.total.density)

        # Calculate Mach
        M = Mach_isentropic(self._total, self._static)
        # Calculate mass flux
        w_flux = w_isen(self._total, self._static, 1)

        if M >= 1:
            supersonic = True
        else:
            supersonic=False

        self.isentropic = {'label': label,
                           'P': P,
                           'PR': self._static._P/self._total._P,
                           'T': T,
                           'TR': self._static._T/self._total._T,
                           'H': H,
                           'HR': self._static._H/self._total._H,
                           'DR': self._static.density/self._total._density,
                           'a': a,
                           'M': M,
                           'D': D,
                           'U': M*self._static.sound,
                           'w_flux': w_flux,
                           'dP': dP_isen(self._total, (w_flux, 'kg/m**2/s')),
                           'dH': dH_isentropic(self._total, (self._static._P, 'Pa')),
                           'gamma': gamma,
                           'AR': perfect_ratio_from_Mach(M, self._static.gamma, 'AR')}


    def _update_incompressible(self, w_flux=None, PR=None, M=None, static=False):

        if static:
            label = 'Static State\n(Incompressible)'
            if w_flux is not None:
                dP = _dP_incomp(self._total, w_flux)
            elif PR is not None:
                dP = -self._total._P*(1-PR)
            elif M is not None:
                logger.warn("Can't calculate incompressible flow properties using "
                            "Mach number - will use isentropic outlet pressure")
                PR = self._static._P/self._total._P
                dP = -self._total._P*(1-PR)

            self._static._DP = self._total._D, self._total._P + dP
            w_flux = _w_incomp(self._total, self._static, 1)
            dP = dP_incomp(self._total, w_flux)
            PR = self._static._P/self._total._P
            dH = _dH_incompressible(self._total, self._static._P)
            HR = (self._total._H + dH)/self._total._H
            U = w_flux/(self._total._density)
            q = 1/2*self._total._density*U**2
            
        else:
            label = 'Total State\n(Incompressible)'
            if w_flux is not None:
                dP = -_dP_incomp(self._static, w_flux)
            elif PR is not None:
                dP = -(PR-1)*self._static._P
            elif M is not None:
                logger.warn("Can't calculate incompressible flow properties using "
                            "Mach number - will use isentropic outlet pressure")
                PR = self._static._P/self._total._P
                dP = -(PR-1)*self._static._P

            self._total._DP = self._static._D, self._static._P + dP            

        w_flux = _w_incomp(self._total,self._static, 1)
        dP = dP_incomp(self._total, w_flux)
        PR = self._static._P/self._total._P
        U = w_flux/(self._static.density)
        q = 1/2*self._total._density*U**2
        dH = _dH_incompressible(self._total, self._static._P)

        if static:      
            HR = (self._total._H + dH)/self._total._H
            T, P, H, D = (self.total.T, self.total.P*PR, self.total.H*HR,
                          self.total.density)
        else:
            HR = (self._static._H)/(self._static._H-dH)
            T, P, H, D = (self.static.T, self.static.P/PR, self.static.H/HR,
                          self.total.density)

        self.incompressible = {'label': label,
                               'PR': PR,
                               'TR': 1,
                               'HR': HR,
                               'T': T,
                               'P': P,
                               'H': H,
                               'D': D,
                               'DR': 1,
                               'a': '\u221E',
                               'q': q,
                               'M': 0,
                               'U': U,
                               'w_flux': w_flux,
                               'dP': dP,
                               'dH': dH,
                               'gamma': 1,
                               'AR': np.nan}

    def _update_compressible(self, w_flux=None, PR=None, M=None, static=False):

        if static:
            label = 'Static State\n(Compressible Ideal Gas)'
            gamma = self._total.gamma

            if w_flux is not None:
                M = _Mach_total_perfect(self._total, w_flux)
            elif PR is not None:
                M = perfect_Mach_from_ratio(PR, gamma, 'PR')
        else:
            label = 'Total State\n(Compressible Ideal Gas)'
            gamma = self._static.gamma
            if w_flux is not None:
                M = _Mach_static_perfect(self._static, w_flux)
            elif PR is not None:
                M = perfect_Mach_from_ratio(PR, gamma)


        PR = perfect_ratio_from_Mach(M, gamma, 'PR')
        DR = perfect_ratio_from_Mach(M, gamma, 'HR')
        TR = perfect_ratio_from_Mach(M, gamma, 'TR')
        HR = perfect_ratio_from_Mach(M, gamma, 'HR')

        if static:
            self._static._DP = self._total._density*DR, self._total._P*PR
            T, P, H, D = (TR*self.total.T, PR*self.total.P, HR*self.total.H,
                          DR*self.total.density)
        else:
            self._total._DP = self._static._density/DR, self._static._P/PR
            T, P, H, D = (self.static.T/TR, self.static.P/PR, self.static.H/HR,
                          self.static.density/DR)


        w_flux = _w_comp(self._total, self._static, 1)
        aR = perfect_ratio_from_Mach(M, gamma, 'soundR')

        a = aR*np.sqrt(gamma*self._total._P/self._total._density)
        U = M*a


        dH = _dH_isentropic_perfect(self._total, self._static._P)
        dP = dP_comp(self._total, w_flux)

        self.compressible = {'label': label,
                             'PR': PR,
                             'TR': TR,
                             'HR': HR,
                             'DR': DR,
                             'T': T,
                             'P': P,
                             'H': H,
                             'D': D,
                             'a': a,
                             'M': M,
                             'U': U,
                             'w_flux': w_flux,
                             'dP': dP,
                             'dH': dH,
                             'gamma': gamma,
                             'AR': perfect_ratio_from_Mach(M, gamma, 'AR')}

    # Total State | Static State (Isentropic) | Static State (Incompressible) | Static State (Compressible)
    # P
    # T
    # H
    # q
    # Density
    # Area Ratio
    # U
    # M
    # a
    # w
    # dP
    # dH
    # 

    #def comp_calc(self):
    
    #def isentropic_calc(self):
        