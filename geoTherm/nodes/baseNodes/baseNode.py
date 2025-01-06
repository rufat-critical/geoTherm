from rich.console import Console
from rich.table import Table


class Node:
    """
    Base Node Class that provides a structured and formatted node
    representation.

    This class is intended to be inherited by specific node types and should
    not be used standalone.
    """

    _displayVars = []  # Variables to display in the node table
    _units = {}  # Dictionary of quanities with associated units

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
            # Get the value of the display variable
            if ':' in dVar:
                var, dVar = dVar.split(':')
            else:
                var = dVar

            try:
                val = getattr(self, var)
            except AttributeError:
                val = 'error'

            # Format the variable depending on its type
            if isinstance(val, (int, float)):
                # Format numeric variables with 5 significant digits
                vars_str.append(f"{dVar}:{val:0.5g}")
            else:
                # Format non-numeric variables as is
                vars_str.append(f"{dVar}: {val}")

        # Join the formatted variables into a single string
        return ' |'.join(vars_str)

    def evaluate(self):
        """
        Placeholder for component evaluation logic.

        Subclasses should implement this method.
        """
        pass
