from rich.console import Console
from rich.table import Table

class Node:
    """ The Base Node Class that makes a pretty
    node printout

    THIS should be inherited and NOT Standalone"""

    _displayVars = []

    def initialize(self, model):
        """
        Initialize the component by attaching a reference to the model.

        This method sets the `model` attribute of the component to the
        provided model. It is used to link the component to a model instance,
        allowing the component to interact with the model.

        Args:
            model: The model instance to attach to the component.
        """
        # Initialize component by attaching reference to model
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
        Vars = []

        # Iterate over each display variable in the node
        for dVar in self._displayVars:
            # Get the value of the display variable
            if ':' in dVar:
                var, dVar = dVar.split(':')
            else:
                var = dVar

            try:
                val = getattr(self, var)
            except:
                val = 'error'

            # Format the variable depending on its type
            if isinstance(val, (int, float)):
                # Format numeric variables with 5 significant digits
                Vars.append(f"{dVar}:{val:0.5g}")
            else:
                # Format non-numeric variables as is
                Vars.append(f"{dVar}: {val}")

        # Join the formatted variables into a single string
        return ' |'.join(Vars)

    def evaluate(self):
        pass