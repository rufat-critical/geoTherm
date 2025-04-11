from rich.console import Console
from rich.table import Table


class modelTable:
    """ The Base Model Class that makes a pretty
    model printout. Similar to Base Node Class but
    for entire model

    THIS should be inherited and NOT Standalone"""

    def __repr__(self):
        """
        Return a string representation of the Model object.

        This method is called when the object is displayed in the console
        (e.g., by simply typing the object name in an interactive session).
        """
        return self.__makeTable()

    def __str__(self):
        """
        Return a string representation of the Model object.

        This method is called when the object is printed using the print()
        function.
        """
        return self.__makeTable()

    def __makeTable(self, **kwargs):
        """
        Create a formatted table representation of the Model object.

        Returns:
            str: A string containing the formatted table representation of the
                 Node.
        """

        # Initialize model if it's not yet initialized
        if not self.initialized:
            self.initialize()

        # Evaluate Nodes
        self.evaluate_nodes()

        table = Table(title='Current Model State')
        # Add columns for Node name and its parameters
        table.add_column("Node")
        table.add_column("Parameters")
        table.add_column("US")
        table.add_column("DS")

        # Loop thru node list and print nodes
        for name, node in self.nodes.items():

            table.add_row(f'{name}',
                          node._generateParamStr(),
                          ','.join(self.node_map[name]['US']),
                          ','.join(self.node_map[name]['DS']))

        # Get Performance metrics
        Wnet, Qin, eta = self.performance
        pText = f'Wnet: {Wnet:.05f} | Qin: {Qin:.05f} | \u03B7: {eta:.05f}\n'
        pTable = Table.grid()
        pTable.add_row(pText)
        console = Console(**kwargs)
        with console.capture() as capture:
            # Print Node and Performance Table
            console.print(table)
            console.print(pTable)

            # console.rule(f'[bold white] Wnet: {Wnet:.05f} eta: {eta:.05f}')

        return capture.get()


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



            if ';' in dVar:
                var, fmt = dVar.split(';')
            else:
                fmt = None

            if ':' in dVar:
                var, dVar = dVar.split(':')
            else:
                var = dVar



            try:
                val = getattr(self, var)
            except:
                val = 'error'

            if fmt is not None:
                from pdb import set_trace
                set_trace()

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
