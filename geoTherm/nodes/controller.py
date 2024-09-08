from .node import Node
from ..utils import parse_component_attribute
import numpy as np
from .baseClasses import ThermoNode
from ..logger import logger


class BaseController(Node):
    """
    Base class for controllers that manipulate a specified parameter (knob).
    """

    def initialize(self, model):
        """
        Initialize the controller and parse the knob attribute.

        Args:
            model: The model containing the components to be controlled.
        """
        super().initialize(model)
        # Parse the knob attribute and link it to the corresponding node
        knob_node, self.knob_attr = parse_component_attribute(self.knob)
        self.knob_node = model.nodes[knob_node]

    @property
    def knob_val(self):
        """
        Get the current value of the knob (controlled attribute).

        Returns:
            The current value of the knob.
        """
        return getattr(self.knob_node, self.knob_attr)

    def set_knob(self, value):
        """
        Set the value of the knob (controlled attribute).

        Args:
            value: The new value to set for the knob.
        """
        setattr(self.knob_node, self.knob_attr, value)


class Schedule(BaseController):
    """
    A controller that sets a knob value based on a predefined schedule
    (function of time).

    Args:
        name: The name of the schedule.
        knob: The attribute to control.
        t_points: The time points for the schedule.
        y_points: The corresponding values at each time point.
    """

    _displayVars = ['knob']

    def __init__(self, name, knob, t_points, y_points):
        self.name = name
        self.knob = knob
        self.t_points = np.array(t_points)
        self.y_points = np.array(y_points)

    def evaluate(self):
        """
        Evaluate the schedule and set the knob value based on the current
        model time.
        """
        val = np.interp(self.model.t, self.t_points, self.y_points)
        self.set_knob(val)


class Balance(BaseController):
    """
    A controller that adjusts a knob to balance a system by comparing
    feedback with a setpoint.

    Args:
        name: The name of the balance controller.
        knob: The attribute to control.
        feedback: The feedback attribute to compare with the setpoint.
        setpoint: The desired setpoint value.
        gain: The proportional gain for the controller (default 0.2).
        knob_min: The minimum allowable value for the knob (default -inf).
        knob_max: The maximum allowable value for the knob (default inf).
    """

    _displayVars = ['knob', 'x', 'xdot']

    def __init__(self, name, knob, feedback, setpoint, gain=0.2,
                 knob_min=-np.inf, knob_max=np.inf):

        self.name = name
        self.knob = knob
        self.feedback = feedback
        self.setpoint = setpoint
        self.gain = gain
        self.knob_min = knob_min
        self.knob_max = knob_max
        self.penalty = False

    def initialize(self, model):
        """
        Initialize the balance controller and parse the feedback attribute.

        Args:
            model: The model containing the components to be controlled.
        """
        super().initialize(model)
        feedback_node, self.feedback_attr = parse_component_attribute(
            self.feedback
        )

        self.feedback_node = model.nodes[feedback_node]

        if not isinstance(self, ThermoBalance):
            # If this is not ThermoBalance, ensure knob_node is not a
            # ThermoNode. Check what type of node is specified
            if isinstance(self.knob_node, ThermoNode):
                logger.critical(f"Knob '{self.knob}' in Balance '{self.name}' "
                                "is associated with a thermoNode. You need to "
                                "use a thermoBalance Object!"
                                )

    @property
    def feedback_val(self):
        """
        Get the current value of the feedback attribute.

        Returns:
            The current value of the feedback attribute.
        """
        return getattr(self.feedback_node, self.feedback_attr)

    @property
    def x(self):
        """
        Get the current state (knob value) as a NumPy array.

        Returns:
            The current state (current knob value).
        """
        return np.array([self.knob_val])

    def update_state(self, x):
        """
        Update the state of the controller based on the new knob value.

        Args:
            x: The new state (knob value) to apply.
        """

        if x[0] < self.knob_min:
            self.penalty = (self.knob_min - x[0] + 10)*1e8
            return
        elif x[0] > self.knob_max:
            self.penalty = (self.knob_max - x[0] - 10)*1e8
            return
        else:
            self.penalty = False

        self.set_knob(x[0])

    @property
    def xdot(self):
        if self.penalty is not False:
            return np.array([self.penalty])

        return np.array([(self.setpoint - self.feedback_val)*self.gain])


class ThermoBalance(Balance):
    """
    A specialized Balance controller for ThermoNode objects.

    This controller ensures that one state variable is varied while keeping
    another state variable constant.

    Args:
        name: The name of the balance controller.
        knob: The attribute to control.
        feedback: The feedback attribute to compare with the setpoint.
        setpoint: The desired setpoint value.
        constant_var: The thermodynamic property to keep constant.
        gain: The proportional gain for the controller (default 0.2).
        knob_min: The minimum allowable value for the knob (default -inf).
        knob_max: The maximum allowable value for the knob (default inf).
    """

    def __init__(self, name, knob, feedback, setpoint, constant_var,
                 gain=0.2, knob_min=-np.inf, knob_max=np.inf):

        super().__init__(name=name,
                         knob=knob,
                         feedback=feedback,
                         setpoint=setpoint,
                         gain=gain,
                         knob_min=knob_min,
                         knob_max=knob_max)

        self.constant_var = constant_var

    def initialize(self, model):

        """
        Initialize the ThermoBalance controller, ensuring that the knob node
        is a ThermoNode and the constant_var is a valid thermodynamic property.

        Args:
            model: The model containing the components to be controlled.
        """

        super().initialize(model)

        # Ensure that the knob_node is a ThermoNode
        if not isinstance(self.knob_node, ThermoNode):
            raise TypeError(
                f"Knob '{self.knob}' in ThermoBalance '{self.name}' "
                "must be associated with a thermoNode."
            )

        if (self.constant_var not in ['T', 'P', 'H', 'S', 'U', 'Q']):
            logger.critical(
                f"Invalid constant_var '{self.constant_var}' in ThermoBalance "
                f"'{self.name}'. \n"
                "Possible options are: 'T', 'P', 'H', 'S', 'U', 'Q'."
                )

        # Store the initial state with the constant_var fixed
        self._state = {
            self.constant_var: getattr(self.knob_node, self.constant_var),
            self.knob_attr: getattr(self.knob_node, self.knob_attr)
            }

    def set_knob(self, value):
        """
        Set the knob value and update the state of the ThermoNode while
        keeping the constant state variable fixed.

        Args:
            value: The new value to set for the knob.
        """

        self._state[self.knob_attr] = value

        self.knob_node.thermo.update_state(self._state)
