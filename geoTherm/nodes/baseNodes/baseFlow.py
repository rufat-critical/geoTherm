#from .baseFlowResistor import baseFlowResistor
from .baseNode import Node
from ...logger import logger
from ...units import addQuantityProperty, inputParser
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
    _units = {'w': 'MASSFLOW', 'dP': 'PRESSURE', 'dH': 'SPECIFICENERGY'}
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
    def get_outlet_state(self, US, w):
        """Calculate outlet state given inlet conditions and flow rate.

        Must be implemented by derived classes.

        Args:
            US: Upstream thermodynamic state
            w (float): Mass flow rate [kg/s]

        Returns:
            dict: Outlet state with pressure and enthalpy
        """

        dP = self._get_dP(US, w)
        dH = self._get_dH(US, w)

        return {'H': US._H + dH,
                'P': US._P + dP}

    def _get_outlet_state(self, US, w):
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
        outlet_state = self.get_outlet_state(US, w)

        # Add heat transfer contribution to enthalpy if flow rate is non-zero
        if w != 0:  # Avoid division by zero
            outlet_state['H'] += self._Q/ w

        return outlet_state

    @property
    def _Q(self):

        Q =0
        if self.hot_node is not None:
            Q += self.hot_node._Q
        if self.cool_node is not None:
            Q -= self.cool_node._Q

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
        US, DS, _ = self.thermostates()
        return DS._P / US._P

    @property
    def _dH(self):
        """Calculate enthalpy change across component.
        Must be implemented by derived classes if non-zero.

        Returns:
            float: Specific enthalpy change [J/kg]
        """

        if self.hot_node:
            return self.hot_node._Q/self._w
        elif self.cool_node:
            return -self.cool_node._Q/self._w
        else:
            return 0
    

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
                 Z:'INERTANCE'=None):
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
        US, DS, _ = self.thermostates()
        DS_target = self.get_outlet_state(US, self._w)

        # Pressure difference driving the flow acceleration
        # Positive when target pressure > actual pressure (flow increases)
        # Negative when target pressure < actual pressure (flow decreases)
        self._wdot = (DS_target['P'] - DS._P) / self._Z

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


class FixedFlow(baseFlow):
    """
    A flow component with a fixed flow rate.
    """
    def __init__(self, name, US, DS, w):
        super().__init__(name, US, DS)
        self._w = w

    def thermostates(self):
        if self._w > 0:
            return self.US_node.thermo, self.DS_node.thermo, 1
        else:
            return self.DS_node.thermo, self.US_node.thermo, -1

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

    def _set_flow(self, w):
        self._w = w
