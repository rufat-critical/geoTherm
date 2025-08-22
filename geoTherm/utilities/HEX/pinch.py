from geoTherm.thermostate import thermo
import numpy as np
from scipy.optimize import root_scalar
from geoTherm.units import inputParser, fromSI, units
from geoTherm.utilities.heat_transfer import T_dQ
from geoTherm.logger import logger
from scipy.optimize import root_scalar, minimize_scalar
from .profile import HEXProfile
from geoTherm.utils import eps


def dT_Q_parallel(Q, hot_inlet, cold_inlet, w_hot, w_cold):
    pass


def dT_Q_counter(Q, hot_inlet, cold_outlet, w_hot, w_cold,
                 cold_inlet_thermo=None,
                 hot_outlet_thermo=None,
                 dP_cold=0,
                 dP_hot=0):

    if hot_outlet_thermo is None:
        hot_outlet_thermo = hot_inlet.from_state(hot_inlet.state)

    if cold_inlet_thermo is None:
        cold_inlet_thermo = cold_outlet.from_state(cold_outlet.state)

    if w_hot == 0:
        hot_outlet_thermo._HP = hot_inlet._H, hot_inlet._P
    else:
        hot_outlet_thermo._HP = (hot_inlet._H - Q/w_hot,
                                 hot_inlet._P + dP_hot)

    if w_cold == 0:
        cold_inlet_thermo._HP = cold_outlet._H, cold_outlet._P
    else:
        cold_inlet_thermo._HP = (cold_outlet._H - Q/w_cold,
                                 cold_outlet._P - dP_cold)

    return hot_outlet_thermo._T - cold_inlet_thermo._T


def find_pinch_Q_counter(Q, hot_inlet, cold_outlet,
                         w_hot, w_cold,
                         cold_inlet_thermo=None,
                         hot_outlet_thermo=None,
                         dP_cold=0,
                         dP_hot=0):

    if Q == 0:
        return hot_inlet._T - cold_outlet._T

    if cold_inlet_thermo is None:
        cold_inlet_thermo = cold_outlet.from_state(cold_outlet.state)
    if hot_outlet_thermo is None:
        hot_outlet_thermo = hot_inlet.from_state(hot_inlet.state)

    def dT_Q(Q_fraction):
        Q_i = Q_fraction * Q
        dP_cold_i = Q_fraction * dP_cold
        dP_hot_i = Q_fraction * dP_hot
        return dT_Q_counter(Q_i, hot_inlet, cold_outlet, w_hot, w_cold,
                            cold_inlet_thermo=cold_inlet_thermo,
                            hot_outlet_thermo=hot_outlet_thermo,
                            dP_cold=dP_cold_i,
                            dP_hot=dP_hot_i)

    Q_bnds = np.linspace(0, 1, 5)
    sol1 = minimize_scalar(dT_Q, bounds=[Q_bnds[0], Q_bnds[1]], method='bounded')
    sol2 = minimize_scalar(dT_Q, bounds=[Q_bnds[1], Q_bnds[2]], method='bounded')
    sol3 = minimize_scalar(dT_Q, bounds=[Q_bnds[2], Q_bnds[3]], method='bounded')

    return np.min([sol1.fun, sol2.fun, sol3.fun])


def get_pinch_point(Q, cold_inlet, hot_inlet, w_hot, w_cold, config='counter',
                    cold_outlet_thermo=None,
                    cold_inlet_thermo=None,
                    hot_outlet_thermo=None,
                    hot_inlet_thermo=None,
                    dP_hot=0,
                    dP_cold=0):

    if cold_outlet_thermo is None:
        cold_outlet_thermo = cold_inlet.from_state(cold_inlet.state)

    if config == 'counter':

        cold_outlet_thermo._HP = (cold_inlet._H + Q/w_cold,
                                  cold_inlet._P + dP_cold)

        return find_pinch_Q_counter(Q,
                                    hot_inlet=hot_inlet,
                                    cold_outlet=cold_outlet_thermo,
                                    w_hot=w_hot,
                                    w_cold=w_cold,
                                    cold_inlet_thermo=cold_inlet_thermo,
                                    hot_outlet_thermo=hot_outlet_thermo,
                                    dP_cold=dP_cold,
                                    dP_hot=dP_hot)
    else:
        from pdb import set_trace
        set_trace()

def _find_w_hot_hot_outlet_counter(T_pinch, cold_inlet, cold_outlet, w_cold, hot_inlet,
                                   hot_outlet_thermo=None,
                                   cold_inlet_thermo=None,
                                   cold_outlet_thermo=None):

    # Create a thermo object to output
    hot_outlet_thermo = hot_inlet.from_state(hot_inlet.state)
    
    if cold_inlet_thermo is None:
        cold_inlet_thermo = cold_inlet.from_state(cold_inlet.state)
    if cold_outlet_thermo is None:
        cold_outlet_thermo = cold_inlet.from_state(cold_inlet.state)
    
    cold_outlet_thermo._TP = cold_outlet._T, cold_inlet._P
    dP_cold = cold_outlet._P - cold_inlet._P

    # Calculate Q
    Q = w_cold * (cold_outlet._H - cold_inlet._H)

    def dT(T):
        hot_outlet_thermo._TP = T, hot_inlet._P

        w_hot = Q/(hot_inlet._H - hot_outlet_thermo._H)

        return find_pinch_Q_counter(Q=Q,
                                    hot_inlet=hot_inlet,
                                    cold_outlet=cold_outlet_thermo,
                                    w_hot=w_hot,
                                    w_cold=w_cold,
                                    cold_inlet_thermo=cold_inlet_thermo,
                                    hot_outlet_thermo=hot_outlet_thermo) - T_pinch

    def dT(T):
        hot_outlet_thermo._TP = T, hot_inlet._P
        w_hot = Q/(hot_inlet._H - hot_outlet_thermo._H+eps)

        return get_pinch_point(Q=Q,
                        cold_inlet=cold_inlet,
                        hot_inlet=hot_inlet,
                        w_hot=w_hot,
                        w_cold=w_cold,
                        config='counter',
                        cold_outlet_thermo=cold_outlet_thermo,
                        cold_inlet_thermo=cold_inlet_thermo,
                        hot_outlet_thermo=hot_outlet_thermo,
                        dP_cold=dP_cold) - T_pinch

    T_max = hot_inlet._T
    T_min = cold_outlet._T

    for i in range(10):
        if np.sign(dT(T_min)) != np.sign(dT(T_max)):
            break
        else:
            T_max = T_min
            T_min /= 1.1

    sol = root_scalar(dT, bracket=[T_min, T_max], method='brentq')

    hot_outlet_thermo._TP = sol.root, hot_inlet._P
    w_hot = Q/(hot_inlet._H - hot_outlet_thermo._H)

    return Q, hot_outlet_thermo, w_hot


def _find_w_hot_cold_inlet_counter(T_pinch, cold_outlet, hot_inlet, hot_outlet, w_cold,
                                  cold_inlet_thermo=None,
                                  hot_outlet_thermo=None):

    if cold_inlet_thermo is None:
        cold_inlet_thermo = cold_outlet.from_state(cold_outlet.state)

    if hot_outlet_thermo is None:
        hot_outlet_thermo = hot_inlet.from_state(hot_inlet.state)

    def dT(Q):
        w_hot = Q/(hot_inlet._H - hot_outlet._H)
        return find_pinch_Q_counter(Q=Q, 
                                    hot_inlet=hot_inlet,
                                    cold_outlet=cold_outlet,
                                    w_hot=w_hot,
                                    w_cold=w_cold,
                                    cold_inlet_thermo=cold_inlet_thermo,
                                    hot_outlet_thermo=hot_outlet_thermo) - T_pinch

    cold_inlet_thermo._TP = (hot_outlet._T - T_pinch), cold_inlet_thermo._P
    Q_max = w_cold * (cold_outlet._H - cold_inlet_thermo._H)

    Q_min = 1e-5
    for i in range(10):
        if np.sign(dT(Q_max)) != np.sign(dT(Q_min)):
            break
        else:
            Q_min = Q_max
            Q_max *= 1.1


    sol = root_scalar(dT, bracket=[1e-5, Q_max], method='brentq')

    Q = sol.root
    w_hot = Q/(hot_inlet._H - hot_outlet._H)
    cold_inlet_thermo._HP = cold_outlet._H - Q/w_cold, cold_outlet._P

    return Q, w_hot, cold_inlet_thermo

def _find_cold_outlet_hot_outlet_counter(T_pinch, cold_inlet, hot_inlet, w_cold, w_hot,
                                         cold_outlet_thermo=None,
                                         cold_inlet_thermo=None,
                                         hot_outlet_thermo=None):

    if cold_outlet_thermo is None:
        cold_outlet_thermo = cold_inlet.from_state(cold_inlet.state)

    if hot_outlet_thermo is None:
        hot_outlet_thermo = hot_inlet.from_state(hot_inlet.state)

    if cold_inlet_thermo is None:
        cold_inlet_thermo = cold_inlet.from_state(cold_inlet.state)


    def dT(Q):
        # Update Cold Outlet Thermo
        cold_outlet_thermo._HP = cold_inlet._H + Q/w_cold, cold_inlet._P

        return find_pinch_Q_counter(Q=Q,
                                    hot_inlet=hot_inlet,
                                    cold_outlet=cold_outlet_thermo,
                                    w_hot=w_hot,
                                    w_cold=w_cold,
                                    cold_inlet_thermo=cold_inlet_thermo,
                                    hot_outlet_thermo=hot_outlet_thermo) - T_pinch

    # Get max Q
    cold_outlet_thermo._TP = hot_inlet._T, cold_inlet._P
    Q_max = w_cold * (cold_outlet_thermo._H - cold_inlet._H)

    sol = root_scalar(dT, bracket=[1e-5, Q_max], method='brentq')

    Q = sol.root

    cold_outlet_thermo._HP = cold_inlet._H + Q/w_cold, cold_inlet._P
    hot_outlet_thermo._HP = hot_inlet._H - Q/w_hot, hot_inlet._P

    return Q, cold_outlet_thermo, hot_outlet_thermo


def _find_w_cold_hot_outlet_counter(T_pinch, cold_inlet, cold_outlet,
                                    hot_inlet, w_hot,
                                    cold_inlet_thermo=None,
                                    hot_outlet_thermo=None):

    # Create temporary thermo objects to update state
    if cold_inlet_thermo is None:
        cold_inlet_thermo = cold_inlet.from_state(cold_inlet.state)
    if hot_outlet_thermo is None:
        hot_outlet_thermo = hot_inlet.from_state(hot_inlet.state)

    def dT(Q):
        w_cold = Q/((cold_outlet._H - cold_inlet._H))

        return find_pinch_Q_counter(Q=Q,
                                    hot_inlet=hot_inlet,
                                    cold_outlet=cold_outlet,
                                    w_hot=w_hot,
                                    w_cold=w_cold,
                                    cold_inlet_thermo=cold_inlet_thermo,
                                    hot_outlet_thermo=hot_outlet_thermo) - T_pinch

    # Find Q Max
    hot_outlet_thermo._TP = cold_inlet._T, hot_inlet._P
    Q_max = w_hot * (hot_inlet._H - hot_outlet_thermo._H)

    sol = root_scalar(dT, bracket=[1e-5, Q_max], method='brentq')
    Q = sol.root

    hot_outlet_thermo._HP = hot_inlet._H - Q/w_hot, hot_inlet._P
    w_cold = Q/(cold_outlet._H - cold_inlet._H)

    return Q, hot_outlet_thermo, w_cold


class PinchSolver:
    """
    A solver for heat exchanger pinch point analysis.

    This class provides methods to analyze heat exchangers and determine pinch points,
    which are the locations where the temperature difference between hot and cold streams
    is minimized. The solver supports counter-flow heat exchanger configurations.

    Attributes:
        config (str): Heat exchanger configuration ('counter' for counter-flow)
        _hot_inlet_fluid (thermo): Reference hot inlet fluid state
        _hot_outlet_fluid (thermo): Reference hot outlet fluid state  
        _cold_inlet_fluid (thermo): Reference cold inlet fluid state
        _cold_outlet_fluid (thermo): Reference cold outlet fluid state

    Example:
        >>> cold_fluid = thermo('water', T=300, P=1e6)
        >>> hot_fluid = thermo('steam', T=500, P=1e6)
        >>> solver = PinchSolver(cold_fluid, hot_fluid, config='counter')
        >>> result = solver.get_pinch_Q(T_pinch=10, cold_inlet=cold_in, 
        ...                            hot_inlet=hot_in, w_cold=1.0, w_hot=1.0)
    """

    def __init__(self, cold_fluid, hot_fluid, config='counter'):
        """
        Initialize the PinchSolver with reference fluids.

        Args:
            cold_fluid (thermo): Reference cold fluid state
            hot_fluid (thermo): Reference hot fluid state
            config (str, optional): Heat exchanger configuration. 
                                  Defaults to 'counter' for counter-flow.

        Raises:
            ValueError: If config is not 'counter' (other configurations not yet supported)
        """
        if config not in ['counter']:
            raise ValueError(f"Configuration '{config}' not supported. Only 'counter' is currently supported.")

        self.config = config
        self.update_reference_fluids(cold_fluid, hot_fluid)

    def update_reference_fluids(self, cold_fluid, hot_fluid):
        """
        Update the reference fluid states for both hot and cold streams.

        This method creates new thermo objects from the provided fluid states
        to serve as reference points for calculations.

        Args:
            cold_fluid (thermo): Cold fluid state to use as reference
            hot_fluid (thermo): Hot fluid state to use as reference
        """
        self._hot_inlet_fluid = hot_fluid.from_state(hot_fluid.state)
        self._hot_outlet_fluid = hot_fluid.from_state(hot_fluid.state)
        self._cold_inlet_fluid = cold_fluid.from_state(cold_fluid.state)
        self._cold_outlet_fluid = cold_fluid.from_state(cold_fluid.state)

    def _update_reference_fluid(self, input_fluid, reference_fluid):
        """
        Update a reference fluid if the input fluid has different properties.

        This method compares the thermodynamic and fluid properties of the input
        and reference fluids, and updates the reference if they differ.

        Args:
            input_fluid (thermo): Input fluid state to compare
            reference_fluid (thermo): Reference fluid to potentially update

        Returns:
            thermo: Updated reference fluid (same object if no update needed)
        """
        input_state = input_fluid.state
        reference_state = reference_fluid.state

        # Check if the fluid compositions match
        if (input_state['thermo'] | input_state['fluid']) != (reference_state['thermo'] | reference_state['fluid']):
            reference_fluid = reference_fluid.from_state(input_state)

        return reference_fluid

    def get_pinch_point(self, Q,
                        cold_inlet, hot_inlet,
                        w_hot, w_cold,
                        dP_cold=0, dP_hot=0,
                        config='counter'):
        """
        Calculate the minimum temperature difference (pinch point) for given conditions.

        Args:
            Q (float): Heat transfer rate [W]
            w_hot (float): Mass flow rate of hot stream [kg/s]
            w_cold (float): Mass flow rate of cold stream [kg/s]

        Returns:
            float: Minimum temperature difference at the pinch point [K]

        Note:
            This method uses the reference fluids stored in the solver instance.
            Use get_pinch_Q() for more flexible input specification.
        """
        self._cold_outlet_fluid = self._update_reference_fluid(cold_inlet, self._cold_outlet_fluid)
        self._cold_inlet_fluid = self._update_reference_fluid(cold_inlet, self._cold_inlet_fluid)
        self._hot_inlet_fluid = self._update_reference_fluid(hot_inlet, self._hot_inlet_fluid)
        self._hot_outlet_fluid = self._update_reference_fluid(hot_inlet, self._hot_outlet_fluid)


        return get_pinch_point(Q=Q,
                               cold_inlet=cold_inlet,
                               hot_inlet=hot_inlet,
                               w_hot=w_hot,
                               w_cold=w_cold,
                               config=self.config,
                               cold_outlet_thermo=self._cold_outlet_fluid,
                               cold_inlet_thermo=self._cold_inlet_fluid,
                               hot_outlet_thermo=self._hot_outlet_fluid,
                               hot_inlet_thermo=self._hot_inlet_fluid,
                               dP_cold=dP_cold,
                               dP_hot=dP_hot)

    def get_pinch_Q(self, T_pinch, cold_inlet=None, cold_outlet=None,
                    hot_inlet=None, hot_outlet=None, w_cold=None, w_hot=None):
        """
        Solve for heat exchanger parameters given a target pinch point temperature difference.

        This method determines the missing parameters (exactly 2 must be None) that result
        in the specified pinch point temperature difference. The solver supports three
        different scenarios based on which parameters are unspecified.

        Args:
            T_pinch (float): Target pinch point temperature difference [K]
            cold_inlet (thermo, optional): Cold stream inlet state
            cold_outlet (thermo, optional): Cold stream outlet state  
            hot_inlet (thermo, optional): Hot stream inlet state
            hot_outlet (thermo, optional): Hot stream outlet state
            w_cold (float, optional): Mass flow rate of cold stream [kg/s]
            w_hot (float, optional): Mass flow rate of hot stream [kg/s]

        Returns:
            dict: Dictionary containing all heat exchanger parameters:
                - 'Q': Heat transfer rate [W]
                - 'w_hot': Mass flow rate of hot stream [kg/s]
                - 'w_cold': Mass flow rate of cold stream [kg/s]
                - 'cold_inlet': Cold stream inlet state
                - 'cold_outlet': Cold stream outlet state
                - 'hot_inlet': Hot stream inlet state
                - 'hot_outlet': Hot stream outlet state

        Raises:
            ValueError: If exactly 2 parameters are not None (must specify 4 out of 6)

        Note:
            The solver currently only supports counter-flow configurations.
            Other configurations will raise NotImplementedError.
        """
        # Count how many inputs are None
        none_count = sum(1 for x in [cold_inlet, cold_outlet, hot_inlet, hot_outlet, w_cold, w_hot] if x is None)

        # Check if exactly 2 inputs are missing (4 specified)
        if none_count != 2:
            raise ValueError(f"Exactly 2 parameters must be unspecified. Currently {none_count} parameters are missing. "
                           f"Please specify 4 out of 6 parameters: cold_inlet, cold_outlet, hot_inlet, hot_outlet, w_cold, w_hot")

        # Scenario 1: Solve for cold_inlet and w_hot
        if cold_inlet is None and w_hot is None:
            if self.config == 'counter':

                self._update_reference_fluid(cold_outlet, self._cold_outlet_fluid)
                self._update_reference_fluid(hot_inlet, self._hot_inlet_fluid)

                Q, w_hot, cold_inlet = _find_w_hot_cold_inlet_counter(
                    T_pinch, cold_outlet, hot_inlet, hot_outlet, w_cold)
            else:
                raise NotImplementedError(f"Configuration '{self.config}' not yet implemented")

        # Scenario 2: Solve for cold_outlet and hot_outlet
        elif cold_outlet is None and hot_outlet is None:
            if self.config == 'counter':

                self._update_reference_fluid(cold_inlet, self._cold_outlet_fluid)
                self._update_reference_fluid(cold_inlet, self._cold_inlet_fluid)
                self._update_reference_fluid(hot_inlet, self._hot_outlet_fluid)


                Q, cold_outlet, hot_outlet = _find_cold_outlet_hot_outlet_counter(
                    T_pinch, cold_inlet, hot_inlet, w_cold, w_hot,
                    cold_outlet_thermo=self._cold_outlet_fluid,
                    cold_inlet_thermo=self._cold_inlet_fluid,
                    hot_outlet_thermo=self._hot_outlet_fluid)
            else:
                raise NotImplementedError(f"Configuration '{self.config}' not yet implemented")

        # Scenario 3: Solve for hot_outlet and w_cold
        elif hot_outlet is None and w_cold is None:
            if self.config == 'counter':

                self._update_reference_fluid(cold_inlet, self._cold_inlet_fluid)
                self._update_reference_fluid(hot_inlet, self._hot_outlet_fluid)

                Q, hot_outlet, w_cold = _find_w_cold_hot_outlet_counter(
                    T_pinch, cold_inlet, cold_outlet, hot_inlet, w_hot,
                    cold_inlet_thermo=self._cold_inlet_fluid,
                    hot_outlet_thermo=self._hot_outlet_fluid)
            else:
                raise NotImplementedError(f"Configuration '{self.config}' not yet implemented")

        elif hot_outlet is None and w_hot is None:
            if self.config == 'counter':

                self._update_reference_fluid(cold_inlet, self._cold_inlet_fluid)
                self._update_reference_fluid(cold_outlet, self._cold_outlet_fluid)

                Q, hot_outlet, w_hot = _find_w_hot_hot_outlet_counter(
                    T_pinch, cold_inlet, cold_outlet, w_cold, hot_inlet,
                    hot_outlet_thermo=self._hot_outlet_fluid,
                    cold_inlet_thermo=self._cold_inlet_fluid,
                    cold_outlet_thermo=self._cold_outlet_fluid)
        
        else:
            raise ValueError("Invalid combination of specified/unspecified parameters. "
                           "Please ensure exactly 2 parameters are None.")

        return {
            'Q': Q,
            'w_hot': w_hot,
            'w_cold': w_cold,
            'cold_inlet': cold_inlet,
            'cold_outlet': cold_outlet,
            'hot_inlet': hot_inlet,
            'hot_outlet': hot_outlet
        }


def dT_subcritical_counter_flow(cold_inlet, cold_outlet, hot_inlet, hot_outlet, w_hot, w_cold, 
                                Q=None, dP_hot=None, hot_thermo=None, cold_thermo=None):
    """
    Calculate temperature differences and heat at key points in a
    subcritical counter-flow heat exchanger.

    Args:
        cold_inlet (thermo): Cold stream inlet state
        cold_outlet (thermo): Cold stream outlet state
        hot_inlet (thermo): Hot stream inlet state
        hot_outlet (thermo): Hot stream outlet state
        w_hot (float): Mass flow rate of hot stream [kg/s]
        w_cold (float): Mass flow rate of cold stream [kg/s]
        Q (float, optional): Heat duty. If None, will be calculated from hot stream states
        dP_hot (float, optional): Total pressure drop in hot stream [Pa]. If None, will be calculated from hot stream states
        hot_thermo (thermo, optional): Hot stream thermo object for reuse. If None, a new one will be created
        cold_thermo (thermo, optional): Cold stream thermo object for reuse. If None, a new one will be created

    Returns:
        dict: Contains arrays of temperature differences (dT), corresponding Q
                values, and temperatures of both streams at each point
    """

    # Total heat and hot pressure drop if not provided
    if Q is None:
        Q = w_hot * (hot_inlet._H - hot_outlet._H)
    if dP_hot is None:
        dP_hot = hot_outlet._P - hot_inlet._P

    # Reuse existing thermo objects if provided, create new ones if needed
    if hot_thermo is None:
        hot_thermo = hot_inlet.from_state(hot_inlet.state)
    if cold_thermo is None:
        cold_thermo = cold_inlet.from_state(cold_inlet.state)

    # Verify energy balance (only in debug/development)
    # Q2 = mdot_cold * (cold_outlet._H - cold_inlet._H)
    # if np.abs(Q - Q2) > 1:
    #     from pdb import set_trace
    #     set_trace()

    # Calculate saturation point properties, assume no pressure drop
    cold_thermo._PQ = cold_inlet._P, 0

    Q_sat = w_cold * (cold_thermo._H - cold_inlet._H)

    hot_thermo._HP = (
        hot_inlet._H - (Q-Q_sat)/w_hot,
        hot_inlet._P + dP_hot * (Q-Q_sat)/Q
    )

    # Return dictionary of data
    return {
        'dT': np.array([
            hot_outlet._T - cold_inlet._T,    # Cold inlet
            hot_thermo._T - cold_thermo._T,   # Saturation point
            hot_inlet._T - cold_outlet._T     # Cold outlet
        ]),
        'Qs': np.array([0, Q_sat, Q])
    }
