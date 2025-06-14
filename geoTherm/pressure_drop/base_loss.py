
class baseLossModel:
    """Base class for pressure loss calculations."""
    def __init__(self, geometry):
        self.geometry = geometry

        # Initialize K and dP
        self.K = None
        self._dP = None
        self.Re = None
        self.f = None
