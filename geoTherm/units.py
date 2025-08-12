import pint
from dataclasses import dataclass
import inspect
import importlib
from geoTherm.logger import logger
from functools import wraps
import os
import sys
import numpy as np
from rich.console import Console
from rich.table import Table
from .unitSystems import _DISPLAY_FORMAT


def get_classes_from_module(module_name):
    """
    Import all classes from a given module and store them in a dictionary.

    Args:
        module_name (str): The name of the module to import classes from.

    Returns:
        dict: A dictionary where keys are class names and values are
              class objects.
    """
    module_path = os.path.abspath(os.path.dirname(__file__))
    if module_path not in sys.path:
        sys.path.append(module_path)

    module = importlib.import_module(module_name)
    classes = {
        name: cls for name, cls in inspect.getmembers(module, inspect.isclass)
        if cls.__module__ == module_name
    }

    return classes


# Load the Unit Systems
unitSystems = get_classes_from_module('unitSystems')


@dataclass
class UnitSystem:
    """
    Units Class for specifying and converting units.

    Handles the input and output unit systems.
    """
    __slots__ = ['_input', '_output']

    def __init__(self, input: str = 'SI', output: str = 'SI'):
        self._input = input.upper()
        self._output = output.upper()

    @property
    def input(self) -> str:
        """Get the current input unit system."""
        return self._input

    @input.setter
    def input(self, value: str):
        """Set the input unit system."""
        self._input = value.upper()

    @property
    def output(self) -> str:
        """Get the current output unit system."""
        return self._output

    @output.setter
    def output(self, value: str):
        """Set the output unit system."""
        self._output = value.upper()

    @property
    def input_units(self):
        """Get the units for the current input unit system."""
        return unitSystems[self._input].units

    @property
    def output_units(self):
        """Get the units for the current output unit system."""
        return unitSystems[self._output].units

    @property
    def _output_units_for_display(self):
        """Get the units for the current display unit system."""

        # Get the units for the current output unit system
        UNITS = self.output_units
        # Create a dictionary of units with display units
        for key, value in UNITS.items():
            if value in _DISPLAY_FORMAT:
                UNITS[key] = _DISPLAY_FORMAT[value]

        return UNITS

    @property
    def SI_units(self):
        """Get SI Units."""
        return unitSystems['SI'].units

    def __str__(self):
        """Return a formatted string representation of the UnitSystem."""
        return self._make_table()

    def __repr__(self):
        """Return a formatted string representation of the UnitSystem."""
        return self._make_table()

    def _make_table(self):
        """
        Create a formatted table representation of the UnitSystem using rich.

        Returns:
            str: The formatted string representation of the UnitSystem.
        """
        table = Table()
        table.add_column("Quantity", style="bold")
        table.add_column(f"Input Unit: {self._input}")
        table.add_column(f"Output Unit: {self._output}")

        for quantity in sorted(self.input_units.keys()):
            table.add_row(quantity, self.input_units[quantity],
                          self.output_units[quantity])

        # Capture the table output using the rich console
        console = Console()
        with console.capture() as capture:
            console.print(table)
        return capture.get()


# Initialize UnitSystem
units = UnitSystem()


class UnitHandler:
    """Class for handling units using the pint library."""

    def __init__(self):
        """Initialize the UnitHandler with a pint UnitRegistry."""
        self.ureg = pint.UnitRegistry()
        self.Q_ = self.ureg.Quantity

    def convert(self, value, input_unit, output_unit):
        """
        Convert quantities between different units.

        Args:
            value: The numeric value to convert.
            input_unit: The unit of the input value.
            output_unit: The desired unit of the output value.

        Returns:
            float: The converted value in the desired units.
        """

        return self.Q_(value, input_unit).to(output_unit).magnitude

    def parse_units(self, input_value, quantity):
        """
        Parse and convert input values to SI units.

        Args:
            input_value: The value to convert.
            quantity: The type of quantity being converted (e.g., 'LENGTH').

        Returns:
            float: The value converted to SI units, or the original value
                  if no conversion is needed.

        Raises:
            ValueError: If the input format is invalid.
        """
        # Handle None case
        if input_value is None:
            return None

        # Handle numeric types
        numeric_types = (float, int, np.integer, np.floating)
        if isinstance(input_value, numeric_types):
            if units.input == 'SI':
                return input_value
            input_unit = unitSystems[units.input].units[quantity]
            output_unit = unitSystems['SI'].units[quantity]
            return self.convert(input_value, input_unit, output_unit)

        # Handle sequences (tuple, list, ndarray)
        if isinstance(input_value, (tuple, list, np.ndarray)):
            if len(input_value) == 1:
                return self.parse_units(input_value[0], quantity)

            elif isinstance(input_value[1], str):
                value, unit = input_value
                is_valid = (isinstance(value, numeric_types) and
                            isinstance(unit, str))
                
                if is_valid:
                    return self.convert(
                        value, unit, unitSystems['SI'].units[quantity])
                elif isinstance(value, (list, tuple, np.ndarray)):
                    return np.array([self.parse_units((v, unit), quantity)
                                     for v in value])
                logger.critical(
                    "Two-element sequence must be (numeric_value, unit_string)"
                    )
            else:
                return np.array([self.parse_units(v, quantity)
                                     for v in input_value])
        
        # If none of the above cases match, return as-is
        return input_value


# Create an instance of UnitHandler
unit_handler = UnitHandler()
unit_converter = unit_handler.convert


# Functions to handle units in/out for node classes
def toSI(value, quantity):
    """
    Convert a given value from input units to SI units.

    Args:
        value (float): The value to convert.
        quantity (str): The type of quantity (e.g., 'TEMPERATURE').

    Returns:
        float: The value converted to SI units.
    """
    return unit_handler.parse_units(value, quantity)


def fromSI(value, quantity):
    """
    Convert a given value from SI units to the output units.

    Args:
        value (float): The value to convert.
        quantity (str): The type of quantity (e.g., 'TEMPERATURE').

    Returns:
        float: The value converted to the output units.
    """
    # Get the SI unit for the quantity
    SIunit = units.SI_units[quantity]
    try:
        outputUnit = units.output_units[quantity]
    except KeyError:
        from pdb import set_trace
        set_trace()

    # Convert the value from SI units to the desired output units
    return unit_handler.convert(value, input_unit=SIunit,
                                output_unit=outputUnit)

# These are copied from thermo._units
# I check these in inputParser state and can't import thermo directly
# because of circular import


THERMO_UNITS = {
    'T': 'TEMPERATURE',          # Temperature
    'P': 'PRESSURE',             # Pressure
    'D': 'DENSITY',              # Density
    'H': 'SPECIFICENERGY',       # Specific Enthalpy
    'S': 'SPECIFICENTROPY',      # Specific Entropy
    'U': 'SPECIFICENERGY',       # Specific Internal Energy
    'cp': 'SPECIFICHEAT',        # Specific Heat at Constant Pressure
    'cv': 'SPECIFICHEAT',        # Specific Heat at Constant Volume
    'density': 'DENSITY',        # Density
    'viscosity': 'VISCOSITY',    # Viscosity
    'sound': 'VELOCITY',         # Velocity
    'Q': None                    # Quality
}


def inputParser(init):
    """
    Decorator to check function annotations and update to SI units.
    Arguments with no annotation are treated as-is.
    """

    @wraps(init)
    def wrapper(self, *args, **kwargs):
        # Retrieve function signature and argument names
        sig = inspect.signature(init)
        arg_names = list(sig.parameters.keys())[1:]

        # Create a dictionary of inputs from args and kwargs
        inputs = dict(zip(arg_names, args))
        inputs.update(kwargs)

        # Validate input arguments against function signature
        invalid_args = [kwarg for kwarg in kwargs if kwarg not in arg_names]
        if invalid_args:
            err_str = (f"Invalid argument names specified for nodetype: "
                       f"{type(self)}")
            if 'name' in inputs:
                err_str = (f"Invalid argument names entered for "
                           f"node: '{inputs['name']}' of type: {type(self)}")
            logger.error(f'{err_str}\nInvalid arguments: {invalid_args}')
            logger.info(f'Valid argument names are: {arg_names}')
            raise RuntimeError('Errors Encountered')

        # Retrieve default values for function arguments
        # defaults = {
        #     var: val.default if val.default != val.empty else None
        #     for var, val in sig.parameters.items() if var != 'self'
        # }

        # Store the original inputs
        # self.__inputs = {**defaults, **inputs}

        # Process each input argument for conversion to SI units
        for name, val in inputs.items():
            if val is None:
                continue

            # Check if the input is a 'state'
            if name == 'state':
                # Convert each state value to SI units
                state = {Q: unit_handler.parse_units(v, THERMO_UNITS[Q])
                         for Q, v in dict(val).items()}
                inputs[name] = state
                continue

            # Check if the parameter has a type annotation
            elif (sig.parameters[name].annotation is not
                  inspect.Parameter.empty):
                quantity = sig.parameters[name].annotation
                inputs[name] = unit_handler.parse_units(val, quantity)

        # Call the original init method with the converted inputs
        return init(self, **inputs)

    return wrapper

def parse_state_dict(state_dict):
    state = {Q: unit_handler.parse_units(v, THERMO_UNITS[Q])
                for Q, v in state_dict.items()}
    return state


def output_converter(quantity):
    """
    A decorator that converts the output of a function from SI units to the
    specified geotherm output unit.

    Args:
        quantity (str): The unit quantity string (must be defined
                        in unitSystems.py).

    Returns:
        function: A wrapper function that applies the conversion factor.
    """
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            # Call the original function to get the output in SI units
            output = func(*args, **kwargs)
            # Convert output from SI to the specified geotherm output unit
            return fromSI(output, quantity)

        return wrapper

    return decorator


def addQuantityProperty(cls):
    """
    Decorator to add properties for quantities in a class.

    This decorator converts SI quantities (e.g., _T) to output units
    when accessed as a regular value (e.g., T).

    Args:
        cls (class): The class to decorate.

    Returns:
        class: The decorated class with added properties for quantities.
    """

    # Check if _units is defined in the class
    if not hasattr(cls, '_units'):
        print(f"The class {cls.__name__} does not have a _units attribute.")
        return cls

    # Add properties for each quantity defined in the class's '_units'
    for quantity in cls._units:
        # This is the SI quantity attribute
        SI_quantity = f"_{quantity}"

        # Define getter and setter for converting between SI and output units
        def getter(self, attr=SI_quantity, quantity=quantity):
            return fromSI(getattr(self, attr), cls._units[quantity])

        def setter(self, value, attr=SI_quantity, quantity=quantity):
            setattr(self, attr, toSI(value, cls._units[quantity]))

        # Add the property to the class
        setattr(cls, quantity, property(getter, setter))

    return cls
