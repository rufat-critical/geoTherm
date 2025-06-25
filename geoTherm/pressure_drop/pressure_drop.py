class PressureDrop:
    """
    A class to handle pressure drop calculations in a fluid system.
    
    This class can handle both constant pressure drops and dynamic pressure drops
    that are calculated based on upstream conditions and flow rate.
    """
    def __init__(self, dP):
        """
        Initialize the PressureDrop object.

        Args:
            dP: Either a constant pressure drop value (int/float) or a callable function
                that calculates pressure drop based on upstream conditions and flow rate.
        """
        # Initialize attributes first
        self.constant_dP = None
        self._dP_func = None

        # Then set the pressure drop
        self._dP = dP

    @property
    def _dP(self):
        if self.constant_dP is not None:
            return self.constant_dP
        else:
            return self._dP_func

    @_dP.setter
    def _dP(self, dP):
        self.constant_dP = None

        if isinstance(dP, (int, float)):
            # Handle constant pressure drop case
            self.constant_dP = dP
            def dP_func(US, w, dP=dP):
                return dP
        elif callable(dP):
            # Handle dynamic pressure drop case
            dP_func = dP
        else:
            raise ValueError('dP must be a number or a callable')

        self._dP_func = dP_func

    def evaluate(self, US, w):
        """
        Evaluate the pressure drop for given upstream conditions and flow rate.

        Args:
            US: Upstream conditions
            w: Flow rate

        Returns:
            float: The calculated pressure drop
        """
        return self._dP_func(US, w)