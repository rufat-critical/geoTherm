from .baseFlow import baseFlow
from ...logger import logger


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

        return self.flow._w_max(US)
