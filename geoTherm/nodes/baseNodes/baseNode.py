from rich.console import Console
from rich.table import Table
import re


class Node:
    """
    Base Node Class that provides a structured and formatted node
    representation.

    This class is intended to be inherited by specific node types and should
    not be used standalone.
    """

    _displayVars = []  # Variables to display in the node table
    _units = {}  # Dictionary of quanities with associated units

    @property
    def _state_dict(self):
        """
        Returns a dictionary containing the basic identification information
        of the node.

        Returns:
            dict: A dictionary with the following keys:
                - 'Node_Name': The name of the node
                - 'Node_Type': The class name of the node type
        """
        return {'Node_Type': type(self).__name__,
                'config': {}}

    def initialize(self, model):
        """
        Attach the model to the component.

        Args:
            model: The model instance to link with the component.
        """
        self.model = model

    def __repr__(self):
        """
        Return a string representation of the Node object.

        This method is called when the object is displayed in the console
        (e.g., by simply typing the object name in an interactive session).
        """
        return f"Node: '{self.name}', Type: {type(self)}"

    def __str__(self):
        """
        Return a string representation of the Node object.

        This method is called when the object is printed using the print()
        function.
        """
        return self.__makeTable()

    def __makeTable(self):
        """
        Create a formatted table representation of the Node object.

        Returns:
            str: A string containing the formatted table representation of the
            Node.
        """

        # Generate a table without a title
        print(f'{type(self)}')
        table = Table()

        # Add columns for Node name and its parameters
        table.add_column("Node")
        table.add_column("Parameters")

        # Generate the parameter string using the _generateParamStr method
        paramStr = self._generateParamStr()

        # Add a row to the table with the node name and its parameters
        table.add_row(self.name, paramStr)

        # Create a console object to capture the table output
        console = Console()

        with console.capture() as capture:
            console.print(table)

        # Return the captured table output as a string
        return capture.get()
    def _generateParamStr(self):
        """
        Create a formatted string of display variables for the Node object.

        Returns:
            str: A string containing the formatted display variables.
        """

        # Initialize a list to hold the formatted variables
        vars_str = []

        # Iterate over each display variable in the node
        for dVar in self._displayVars:
            # Parse the display variable format using regex
            pattern = r'^(?:([^:]+):)?([^;]+)(?:;(.+))?$'
            match = re.match(pattern, dVar)
            
            if match:
                var, display_name, fmt = match.groups()
                # If var is None, use display_name as the variable name
                var = var if var is not None else display_name
            else:
                # Fallback for invalid format
                var = display_name = dVar
                fmt = None

            try:
                val = getattr(self, var)
            except AttributeError:
                val = 'error'

            # Format the variable depending on its type and format specification
            if isinstance(val, (int, float)):
                if fmt:
                    # Use custom format if provided
                    vars_str.append(f"{display_name}:{val:{fmt}}")
                else:
                    # Default to 5 significant digits
                    vars_str.append(f"{display_name}:{val:0.5g}")
            elif isinstance(val, dict):
                if fmt:
                    # Apply format to dictionary values if format is specified
                    val = ' '.join([f"{k}:{v:{fmt}}" for k, v in val.items()])
                else:
                    # No formatting if fmt is not specified
                    val = ' '.join([f"{k}:{v}" for k, v in val.items()])
                vars_str.append(f"{display_name}: {val}")
            else:
                vars_str.append(f"{display_name}: {val}")

        # Join the formatted variables into a single string
        return ' |'.join(vars_str)

    def evaluate(self):
        """
        Placeholder for component evaluation logic.

        Subclasses should implement this method.
        """
        pass
