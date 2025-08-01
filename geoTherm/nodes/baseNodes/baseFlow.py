#from .baseFlowResistor import baseFlowResistor
from .baseNode import Node
from geoTherm.common import addQuantityProperty, inputParser, logger
import numpy as np
from abc import ABC, abstractmethod
#from ...nodes.heatsistor import Qdot
from geoTherm.decorators import state_dict
from ...units import units


@addQuantityProperty
class baseFlow(Node, ABC):
    """Base class for all flow components.

    Defines the basic interface and properties that all flow components must
    implement. Flow components must handle mass flow rates and pressure
    differences.

    Attributes:
        _displayVars (list): Variables to display in output
        _units (dict): Unit definitions for quantities
        _bounds (list): Valid range for mass flow rate [min, max]
        _unidirectional (bool): If True, only allows forward flow
    """
    _displayVars = ['w', 'dP', 'dH']
    _units = {'w': 'MASSFLOW', 'dP': 'PRESSURE', 'dH': 'SPECIFICENERGY', 'Ue': 'VELOCITY'}
    _bounds = [1e-5, 1e5]

    def __init__(self, name, US, DS, unidirectional=False):
        """Initialize flow component.

        Args:
            name (str): Component identifier
            US (str): Upstream node name
            DS (str): Downstream node name
            unidirectional (bool): If True, only allows forward flow
        """
        self.name = name
        self.US = US
        self.DS = DS
        self._w = 0
        self.penalty = False
        self._unidirectional = unidirectional

    @state_dict
    def _state_dict(self):
        """Returns a dictionary containing the component's state information.

        This property extends the base state dictionary from the parent class by adding
        flow-specific configuration details.

        Returns:
            dict: A dictionary containing:
                - All state information from the parent class
                - A 'config' key with flow-specific details:
                    - 'US': Name of the upstream node
                    - 'DS': Name of the downstream node
        """
        return {'US': self.US,
                'DS': self.DS}

    def initialize(self, model):
        """Initialize node with model and set up node connections.

        Validates node connections and sets up references to upstream and 
        downstream nodes. Ensures flow nodes have correct connectivity.

        Args:
            model: Model containing node map and other nodes.

        Raises:
            CriticalError: If node connections are invalid.
        """
        super().initialize(model)
        node_map = self.model.node_map[self.name]

        # Validate upstream and downstream connections
        if len(node_map['US']) != 1 or len(node_map['DS']) != 1:
            logger.critical(
                f"Flow Node {self.name} must have exactly one upstream and "
                f"one downstream node. Current connections: "
                f"US: {node_map['US']}, DS: {node_map['DS']}"
            )

        # Validate node map strings match initialization values
        if node_map['US'][0] != self.US or node_map['DS'][0] != self.DS:
            logger.critical(
                f"Node mapping mismatch for '{self.name}': "
                f"Expected US: {self.US}, DS: {self.DS}, "
                f"got US: {node_map['US'][0]}, DS: {node_map['DS'][0]}"
            )

        # Validate heat transfer connections
        if node_map['hot'] or node_map['cool']:
            for node in node_map['hot'] + node_map['cool']:
                pass
                #if not isinstance(model.nodes[node], Qdot):
                #    logger.critical(
                #        f"Flow Node {self.name} can only have Qdot elements "
                #        f"as heat connections. Invalid connections:\n"
                #        f"hot: {node_map['hot']}\ncool: {node_map['cool']}"
               #     )

        # Set node references
        self.US_node = model.nodes[node_map['US'][0]]
        self.DS_node = model.nodes[node_map['DS'][0]]
        self.hot_node = model.nodes[node_map['hot'][0]] if node_map['hot'] else None
        self.cool_node = model.nodes[node_map['cool'][0]] if node_map['cool'] else None
        self._fixed_flow = False

    def thermostates(self):
        """
        Get the inlet and outlet thermo states based on Pressure
        """

        # Handle Backflow
        if self.US_node.thermo._P >= self.DS_node.thermo._P:
            US, DS = self.US_node.thermo, self.DS_node.thermo
            flow_sign = 1
        else:
            US, DS = self.DS_node.thermo, self.US_node.thermo
            flow_sign = -1

        return US, DS, flow_sign

    @abstractmethod
    def get_outlet_state(self, US, *, w=None, PR=None):
        """Calculate outlet state given inlet conditions and flow rate.

        Must be implemented by derived classes.

        Args:
            US: Upstream thermodynamic state
            w (float): Mass flow rate [kg/s]

        Returns:
            dict: Outlet state with pressure and enthalpy
        """
        pass

    def _get_outlet_state(self, US, *, w=None, PR=None):
        """Calculate outlet state including heat transfer effects.

        This method extends the base outlet state calculation by adding the heat
        transfer contribution (Q) from any connected heat nodes (hot/cool nodes).
        The heat transfer is added to the outlet enthalpy.

        Args:
            US: Upstream thermodynamic state
            w (float): Mass flow rate [kg/s]

        Returns:
            dict: Outlet state dictionary containing pressure and enthalpy,
                  with heat transfer effects included in the enthalpy

        Note:
            The heat transfer Q is divided by mass flow rate to convert
            from total heat transfer [W] to specific enthalpy change [J/kg]
        """
        # Get base outlet state first
        outlet_state = self.get_outlet_state(US, w=w, PR=PR)

        if PR is not None:
            w = self._w

        # Add heat transfer contribution to enthalpy if flow rate is non-zero
        if w != 0:  # Avoid division by zero
            outlet_state['H'] += self._Q/ w

        return outlet_state

    @property
    def _Ue(self):
        # FLow Speed at the exit
        US, DS, _ = self.thermostates()

        outlet_state = self.get_outlet_state(US, w=self._w)

        outlet = US.from_state(US.state)
        outlet.update_state(outlet_state)

        return self._w/(outlet._density*self._area)

    @property
    def Mach_exit(self):

        US, DS, _ = self.thermostates()

        outlet_state = self.get_outlet_state(US, w=self._w)

        outlet = US.from_state(US.state)
        outlet.update_state(outlet_state)

        Ue = self._w/(outlet._density*self._area)

        return Ue/outlet._sound


    @property
    def _Q(self):

        Q = 0
        for hot_node in self.model.node_map[self.name]['hot']:
            Q += self.model.nodes[hot_node]._Q
        for cool_node in self.model.node_map[self.name]['cool']:
            Q -= self.model.nodes[cool_node]._Q

        return Q


    def _get_dH(self, US, w):
        """Calculate outlet state given inlet conditions and flow rate.

        Must be implemented by derived classes.

        Args:
            US: Upstream thermodynamic state
            w (float): Mass flow rate [kg/s]

        Returns:
            float: Specific enthalpy change [J/kg]
        """

        if w == 0:
            return 0
        
        dH = 0   
        if self.hot_node is not None:
            dH += self.hot_node._Q/w
        if self.cool_node is not None:
            dH -= self.cool_node._Q/w
        
        return dH


    def get_US_state(self, DS, w):

        # Get Upstream Node
        # w must be positive

        # Get the Inlet State
        US_state = self.get_inlet_state(DS, w)

        return US_state

    @property
    def _dP(self):
        """Calculate pressure difference across component.

        Returns:
            float: Pressure difference [Pa]
        """
        US, DS, _ = self.thermostates()
        return DS._P - US._P

    @property
    def PR(self):
        """Calculate actual pressure ratio (outlet/inlet).

        For pumps, PR > 1 indicates normal operation (compression),
        while PR < 1 indicates reverse flow.

        Returns:
            float: Pressure ratio (P_downstream / P_upstream)
        """
        US, DS, _ = self.thermostates()
        return DS._P / US._P

    @property
    def _dH(self):
        """Calculate enthalpy change across component.
        Must be implemented by derived classes if non-zero.

        Returns:
            float: Specific enthalpy change [J/kg]
        """

        US, DS, _ = self.thermostates()
        return self._get_dH(US, self._w)


@addQuantityProperty
class baseInertantFlow(baseFlow):
    """Base class for flow components with inertial effects.

    Handles dynamic flow behavior with inertance.
    """
    _units = {**baseFlow._units,
              'wdot': 'MASSFLOWDERIV',
              'Z': 'INERTANCE'}

    @inputParser
    def __init__(self, name, US, DS, w:'MASSFLOW'=0,
                 Z:'INERTANCE'=1):
        """Initialize inertant flow component.

        Args:
            name (str): Component identifier
            US: Upstream node reference
            DS: Downstream node reference
            w (float): Initial mass flow rate [kg/s]
            Z (float): Flow inertance [m**-3]
        """
        super().__init__(name=name, US=US, DS=DS)
        self._w = w
        self._Z = Z
        self.penalty = False

    @state_dict
    def _state_dict(self):
        return {'w': (self.w, units.output_units['MASSFLOW']),
                'Z': (self.Z, units.output_units['INERTANCE'])}

    def update_state(self, x):
        """
        Update the state of the node.

        Args:
            x (float): New state value to set.
        """
        if self._bounds[0] < x[0] < self._bounds[1]:
            self._w = x[0]
            self.penalty = False
        else:
            logger.warn(f"Can't update state for {self.name} to: "
                        f"{x[0]}, it's outside the bounds: {self._bounds}")
            if x < self._bounds[0]:
                self.penalty = (self._bounds[0] - x + 10)*1e8
                self._w = self._bounds[0]
            elif x > self._bounds[1]:
                self.penalty = (x - self._bounds[1] - 10)*1e8
                self._w = self._bounds[1]           

    def evaluate(self):
        """Calculate flow acceleration based on pressure imbalance.

        The flow rate should:
        - Decrease (negative wdot) when downstream pressure is too high
        - Increase (positive wdot) when downstream pressure is too low

        For example:
        - If DS._P = 110 bar and DS_target['P'] = 100 bar:
          Flow sees too much back pressure, so wdot < 0 to reduce flow
        - If DS._P = 90 bar and DS_target['P'] = 100 bar:
          Flow sees less resistance, so wdot > 0 to increase flow
        """
        US, DS, flow_sign = self.thermostates()
        DS_target = self.get_outlet_state(US, w=self._w)

        # Pressure difference driving the flow acceleration
        # Positive when target pressure > actual pressure (flow increases)
        # Negative when target pressure < actual pressure (flow decreases)
        self._wdot = (DS_target['P'] - DS._P) / self._Z * flow_sign


    @property
    def xdot(self):
        """Calculate state derivative."""
        if self.penalty is not False:
            return np.array([self.penalty])
        return np.array([self._wdot])

    @property
    def x(self):
        """Current state (mass flow rate)."""
        return np.array([self._w])

    def thermostates(self):
        """
        Get the inlet and outlet thermo states based on flow
        """

        if self._w >= 0:
            return self.US_node.thermo, self.DS_node.thermo, 1
        else:
            return self.DS_node.thermo, self.US_node.thermo, -1


    def _set_flow(self, w):
        """
        Set the flow rate and get outlet state.

        Args:
            w (float): Flow rate.

        Returns:
            tuple: Downstream node name and downstream state.
        """

        self._w = w

        return False


class baseFlowResistor(baseFlow):
    """Base class for a flow node that calculates flow in between stations."""

    @property
    def _dH(self):
        """Flow resistors are isenthalpic by default.

        Returns:
            float: Always 0 for basic flow resistors
        """
        return 0

    def _set_flow(self, w):
        """
        Set the flow rate and get outlet state.

        Args:
            w (float): Flow rate.

        Returns:
            tuple: Downstream node name and downstream state.
        """

        self._w = w

        return False

    def is_choked(self):
        """Check if the flow is choked.

        Returns:
            bool: True if the flow is choked, False otherwise.
        """

        from pdb import set_trace
        set_trace()
        return False


    def is_outlet_choked(self, US, DS, w):
        """Check if the outlet flow is choked.

        Returns:
            bool: True if the outlet flow is choked, False otherwise.
        """

        w_max = self.flow._w_max(US)


        from pdb import set_trace
        set_trace()


    def _w_max(self, US):
        """Get the maximum mass flow rate for the flow resistor.

        Args:
            US (thermo): Upstream state.

        Returns:
            float: Maximum mass flow rate.
        """

        if US.phase in ['two-phase']:
            from pdb import set_trace
            set_trace()
        return self.flow._w_max(US)

    def evaluate(self):

        # Get US, DS Thermo
        US, DS, flow_sign = self.thermostates()
        self._w = self.flow._w(US, DS)*flow_sign

    def get_outlet_state(self, US, *, w=None, PR=None):
        """Calculate outlet thermodynamic state for a flow resistor.

        This method calculates the downstream pressure based on either the mass
        flow rate or pressure ratio, while maintaining isenthalpic flow (no
        enthalpy change).

        Args:
            US: Upstream thermodynamic state object
            w (float, optional): Mass flow rate [kg/s]. If provided, calculates
                               pressure drop using the flow model's pressure
                               drop correlation.
            PR (float, optional): Pressure ratio (downstream/upstream). If
                                provided, calculates pressure drop directly
                                from the ratio.

        Returns:
            dict: Outlet state dictionary containing:
                - 'H': Enthalpy [J/kg] (same as upstream, isenthalpic flow)
                - 'P': Pressure [Pa] (upstream pressure + pressure drop)

        Note:
            - Either 'w' or 'PR' must be provided, but not both
            - If pressure drop calculation fails (returns None), a large
              negative pressure drop (-1e9 Pa) is used to signal the solver to
              reduce mass flow
            - This method assumes isenthalpic flow (no enthalpy change across
              the resistor)
        """
        # Calculate pressure drop based on input parameters
        if w is not None:
            dP, error = self.flow._dP(US, np.abs(w))
        elif PR is not None:
            dP = -US._P * (1 - PR)
        else:
            logger.critical(
                "Either 'w' (mass flow rate) or 'PR' (pressure ratio) "
                "must be provided"
            )

        # Handle failed pressure drop calculation
        if dP is None:
            # Set large negative pressure drop to signal solver to reduce
            # mass flow
            dP = -1e9

        return {'H': US._H, 'P': US._P + dP}


class FixedFlow(baseFlow):
    """
    A flow component with a fixed flow rate.
    """

    @inputParser
    def __init__(self, name, US, DS, w:'MASSFLOW'):
        super().__init__(name, US, DS)
        self._w = w

    def thermostates(self):
        if self._w > 0:
            return self.US_node.thermo, self.DS_node.thermo, 1
        else:
            return self.DS_node.thermo, self.US_node.thermo, -1

    def get_outlet_state(self, US, *, w=None, PR=None):
        """
        Calculate the thermodynamic state at the outlet (downstream).

        Returns:
            dict: A dictionary containing the enthalpy ('H') and pressure ('P')
                  at the downstream node.
        """

        # Get US, DS Thermo
        #US = self.model.nodes[self.US].thermo
        return {'H': US._H, 'P': US._P*PR}


    @state_dict
    def _state_dict(self):
        return {'w': (self.w, units.output_units['MASSFLOW'])}
