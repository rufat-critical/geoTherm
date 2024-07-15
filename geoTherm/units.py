import pint
from dataclasses import dataclass
import inspect
import importlib
import os
import sys
import numpy as np
from rich.console import Console
from rich.table import Table


def getClassesFromModule(module_name):
    """
    Import all classes from a given module and store them in a dictionary.

    Args:
        module_name (str): The name of the module to import classes from.

    Returns:
        dict: A dictionary where the keys are class names and the values are class objects.
    """
    # Add the directory of the module to the system path
    module_path = os.path.abspath(os.path.dirname(__file__))
    if module_path not in sys.path:
        sys.path.append(module_path)
    
    # Import the module
    module = importlib.import_module(module_name)

    # Get all classes from the module
    classes = {name: cls for name, cls in inspect.getmembers(module, inspect.isclass) if cls.__module__ == module_name}

    return classes

# Load the Unit Systems
unitSystems = getClassesFromModule('unitSystems')


@dataclass
class UnitSystem:
    """
    Units Class for specifying and converting units.
    This class handles the input and output unit systems.
    """

    __slots__ = ['_input', '_output']

    def __init__(self, input: str = 'SI', output: str = 'SI'):
        self._input = input.upper()
        self._output = output.upper()

    @property
    def input(self) -> str:
        """
        Get the current input unit system.
        
        Returns:
            str: The current input unit system.
        """
        return self._input
    
    @input.setter
    def input(self, value: str):
        """
        Set the input unit system.
        
        Args:
            value (str): The unit system to set as input (e.g., 'SI', 'ENGLISH').
        """
        self._input = value.upper()
    
    @property
    def output(self) -> str:
        """
        Get the current output unit system.
        
        Returns:
            str: The current output unit system.
        """
        return self._output
    
    @output.setter
    def output(self, value: str):
        """
        Set the output unit system.
        
        Args:
            value (str): The unit system to set as output (e.g., 'SI', 'ENGLISH').
        """
        self._output = value.upper()
    
    @property
    def inputUnits(self):
        """
        Get the units for the current input unit system.
        
        Returns:
            dict: A dictionary of units for the input unit system.
        """
        return unitSystems[self._input].units
    
    @property
    def outputUnits(self):
        """
        Get the units for the current output unit system.
        
        Returns:
            dict: A dictionary of units for the output unit system.
        """
        return unitSystems[self._output].units
    
    @property
    def SIUnits(self):
        """ Get SI Units"""

        return unitSystems['SI'].units

    def __str__(self):
        """
        Return a string representation of the UnitSystem, formatted nicely using rich.
        
        Returns:
            str: The formatted string representation of the UnitSystem.
        """
        return self._makeTable()

    def __repr__(self):
        """
        Return a string representation of the UnitSystem, formatted nicely using rich.
        
        Returns:
            str: The formatted string representation of the UnitSystem.
        """
        return self._makeTable()      

    def _makeTable(self):
        """
        Create a formatted table representation of the UnitSystem using rich.
        
        Returns:
            str: The formatted string representation of the UnitSystem.
        """
        
        table = Table()
        # Add columns for the quantity, input units, and output units
        table.add_column("Quantity", style="bold")
        table.add_column(f"Input Unit: {self._input}")
        table.add_column(f"Output Unit: {self._output}")
        
        # Loop through the input and output units and add rows to the table
        for quantity, unit in self.inputUnits.items():
            table.add_row(quantity, unit, self.outputUnits[quantity])

        # Capture the table output using the rich console
        console = Console()
        with console.capture() as capture:
            console.print(table)
        return capture.get()

# Initialize UnitSystem
units = UnitSystem()


class unitHandler:
    """
    Class for handling units using the pint library.
    """

    def __init__(self):
        """
        Initialize the UnitHandler with a pint UnitRegistry.
        """
        ureg = pint.UnitRegistry()
        self.Q_ = ureg.Quantity

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
    
    def parseUnits(self, input_value, quantity):
        """
        Parse and convert input values to SI units.

        This method checks the input type and converts it to SI units if necessary.

        Args:
            input_value: The value to convert.
            quantity: The type of quantity being converted (e.g., 'LENGTH').

        Returns:
            float: The value converted to SI units, or the original value if no conversion is needed.
        """
        if isinstance(input_value, (float, int, np.integer, np.floating)):
            # Convert to SI if input is not already in SI
            if units.input == 'SI':
                return input_value
            else:
                return self.convert(input_value, 
                                    input_unit=unitSystems[units.input].units[quantity],
                                    output_unit=unitSystems['SI'].units[quantity])
        elif input_value is None:
            return None
        elif isinstance(input_value, (tuple, list, np.ndarray)):
            # Handle cases where input is a tuple, list, or array with a value and a unit
            if (isinstance(input_value[0], (float, int, np.int64, np.float64)) and isinstance(input_value[1], str)):
                return self.convert(input_value[0],
                                    input_unit=input_value[1],
                                    output_unit=unitSystems['SI'].units[quantity])
        else:
            return input_value
            
            
# Create an instance of UnitHandler
unit_handler = unitHandler()


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
    return unit_handler.parseUnits(value, quantity)

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
    SIunit = units.SIUnits[quantity]
    outputUnit = units.outputUnits[quantity]
    
    # Convert the value from SI units to the desired output units
    return unit_handler.convert(value, input_unit=SIunit,
                                output_unit=outputUnit)

# These are copied from thermo._units
# I check these in inputParser state and can't import thermo directly
# because of circular import 

thermoUnits = {'T': 'TEMPERATURE',         # Temperature
               'P': 'PRESSURE',            # Pressure
               'H': 'SPECIFICENERGY',      # Specific Enthalpy
               'S': 'SPECIFICENTROPY',     # Specific Entropy
               'U': 'SPECIFICENERGY',      # Specific Internal Energy
               'cp': 'SPECIFICHEAT',       # Specific Heat at Constant Pressure
               'cv': 'SPECIFICHEAT',       # Specific Heat at Constant Volume
               'density': 'DENSITY',       # Density
               'viscosity': 'VISCOSITY',   # Viscosity
               'sound': 'VELOCITY'         # Velocity
               }

## UNIT DECORATORS
def inputParser(init):
    """
    Decorator to check function annotations and update to SI units.
    Arguments with no annotation are treated as is.
    """

    def wrapper(self, *args, **kwargs):
        # Get the function signature
        sig = inspect.signature(init)
        arg_names = list(sig.parameters.keys())[1:]
        inputs = dict(zip(arg_names, args))
        inputs.update(kwargs)

        # Get default argument inputs
        defaults = {var: val.default if val.default != val.empty else None
                    for var, val in sig.parameters.items() if var != 'self'}
        
        # Store the original inputs
        #self.__inputs = {**defaults, **inputs}

        # Loop through each argument
        for name, val in inputs.items():
            if val is None:
                continue
            
            # Check if the input is a 'state'
            if name == 'state':
                # This should be a dictionary with 2 thermodynamic properties
                # Loop through the dictionary and convert each value to SI units
                state = dict(val)
                for Q, v in state.items():
                    state[Q] = unit_handler.parseUnits(v, 
                                                       thermoUnits[Q])
                
                inputs[name] = state
                continue
                
            # Check if the parameter has a type annotation
            elif sig.parameters[name].annotation is not inspect.Parameter.empty:
                quantity = sig.parameters[name].annotation
                # Convert the input value to SI units
                inputs[name] = unit_handler.parseUnits(val, quantity)

        # Call the original init method with the converted inputs
        return init(self, **inputs)
    return wrapper 

def inputParser(init):
    """
    Decorator to check function annotations and update to SI units.
    Arguments with no annotation are treated as is.
    """

    def wrapper(self, *args, **kwargs):
        # Get the function signature
        sig = inspect.signature(init)
        arg_names = list(sig.parameters.keys())[1:]
        inputs = dict(zip(arg_names, args))
        inputs.update(kwargs)

        # Get default argument inputs
        defaults = {var: val.default if val.default != val.empty else None
                    for var, val in sig.parameters.items() if var != 'self'}
        
        # Store the original inputs
        #self.__inputs = {**defaults, **inputs}

        # Loop through each argument
        for name, val in inputs.items():
            if val is None:
                continue
            
            # Check if the input is a 'state'
            if name == 'state':
                # This should be a dictionary with 2 thermodynamic properties
                # Loop through the dictionary and convert each value to SI units
                state = dict(val)
                for Q, v in state.items():
                    state[Q] = unit_handler.parseUnits(v, 
                                                       thermoUnits[Q])
                
                inputs[name] = state
                continue
                
            # Check if the parameter has a type annotation
            elif sig.parameters[name].annotation is not inspect.Parameter.empty:
                quantity = sig.parameters[name].annotation
                # Convert the input value to SI units
                inputs[name] = unit_handler.parseUnits(val, quantity)

        # Call the original init method with the converted inputs
        return init(self, **inputs)
    return wrapper 


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

    for quantity in cls._units:
        # Loop through attributes in the class that are listed in _units

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
