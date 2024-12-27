from .baseFlow import baseFlow
from ...logger import logger


class baseFlowResistor(baseFlow):
    """Base class for a flow node that calculates flow in between stations."""


    @property
    def _dH(self):
        return 0

    def initialize(self, model):
        """
        Initialize the node with the model, setting up connections to upstream
        and downstream nodes.

        Args:
            model: The model containing the node map and other nodes.
        """

        super().initialize(model)

        # Initialize attributes if not already defined and no property
        # is defined
        if not hasattr(self, '_w'):
            self._w = 0
        if not hasattr(self, '_dP'):
            self._dP = 0

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
