from .baseClasses import Node
from .heat import HTC
from ..units import addQuantityProperty, inputParser
import numpy as np
from ..utils import Re_calc


@addQuantityProperty
class WallSurface(Node):
    """
    Represents a wall surface with properties and methods to calculate thermal resistance.

    Attributes:
        name (str): Name of the wall surface.
        volume (str): Volume identifier for the associated node.
        _D (float): Hydraulic diameter of the wall surface.
        _L (float): Length of the wall surface.
        hotSide (bool): Flag indicating if the wall is on the hot side.
    """

    _displayVars = ['R', 'U']
    _units = {'R': 'THERMALRESISTANCE', 'D': 'LENGTH', 'L': 'LENGTH', 'A': 'AREA',
              'U': 'CONVECTION'}


    @inputParser
    def __init__(self,name, volume,
                 D: 'LENGTH',
                 L: 'LENGTH',
                 L_ht: 'LENGTH'=None,
                 hotSide = True):
        """
        Initializes the WallSurface object with provided parameters.

        Args:
            name (str): Name of the wall surface.
            volume (str): Volume identifier for the associated node.
            D (float): Hydraulic diameter of the wall surface.
            L (float): Length of the wall surface.
            hotSide (bool): Flag indicating if the wall is on the hot side (default is True).
        """

        self.name = name
        self.volume = volume
        self._D = D
        self._L = L
        self._L_ht = L_ht

        if self._L_ht is None:
            self._L_ht = self._D

        # Flag to see if wall is hotter than fluid
        self.hotSide = hotSide

    def update_geometry(self, D=None, L=None):
        """
        Updates the geometry of the wall surface.

        Args:
            D (float, optional): New hydraulic diameter.
            L (float, optional): New length.
        """

        if D is not None:
            self._D = D
        if L is not None:
            self._L = L

    def initialize(self, model):
        """
        Initializes the wall surface with the provided model.

        Args:
            model: The model containing volume nodes and other components.
        """
        self.model = model
        self.volNode = self.model[self.volume]
        self.htc = HTC(self.volNode.thermo, self._L_ht)

    @property
    def _R(self):
        """
        Calculates and returns the convective thermal resistance.

        Returns:
            float: The thermal resistance.
        """

        Nu = self.htc.Nu_dittus_boelter(self.volNode.Re, heating=self.hotSide)

        return 1/(self.htc._h(Nu)*self._A)

    @property
    def _U(self):
        """
        Calculates and returns the convective coefficient

        Returns:
            float: The convection coefficient
        """

        Nu = self.htc.Nu_dittus_boelter(self.volNode.Re, heating=self.hotSide)

        return self.htc._h(Nu)

    @property
    def _A(self):
        """
        Calculates and returns the area of the wall surface.

        This method should be implemented in subclasses.

        Raises:
            NotImplementedError: If the method is not implemented in a subclass.
        """
        raise NotImplementedError("This method should be implemented in subclasses")


class PlaneWallSurface(WallSurface):
    """
    Represents a plane wall surface with specific area calculation.
    """
    @property
    def _A(self):
        """
        Calculates and returns the area of the plane wall surface.

        Returns:
            float: The area of the plane wall surface.
        """
        return self._D * self._L


class CylindricalWallSurface(WallSurface):
    """
    Represents a cylindrical wall surface with specific area calculation.
    """
    @property
    def _A(self):
        """
        Calculates and returns the area of the cylindrical wall surface.

        Returns:
            float: The area of the cylindrical wall surface.
        """
        return np.pi * self._D * self._L
    

@addQuantityProperty
class Wall(Node):
    """
    Represents a generic wall with properties and methods to calculate thermal resistance.
    """
    _displayVars = ['R']
    _units = {}

    @property
    def _R(self):
        """
        Calculates and returns the thermal resistance.

        This method should be implemented in subclasses.

        Raises:
            NotImplementedError: If the method is not implemented in a subclass.
        """
        raise NotImplementedError("This method should be implemented in subclasses")


class CylindricalWall(Wall):
    """
    Represents a cylindrical wall with specific thermal resistance calculation.
    """
    _units = {'k': 'CONDUCTIVITY', 'L': 'LENGTH', 'D': 'LENGTH', 't': 'LENGTH',
              'D1': 'LENGTH', 'D2': 'LENGTH'}


    @inputParser
    def __init__(self, name, k: 'CONDUCTIVITY', L: 'LENGTH', D: 'LENGTH', t: 'LENGTH',
                 R=None):
        """
        Initializes the CylindricalWall object with provided parameters.

        Args:
            name (str): Name of the cylindrical wall.
            k (float): Thermal conductivity of the wall material.
            L (float): Length of the wall.
            D (float): Inner diameter of the cylindrical wall.
            t (float): Wall Thickness.
        """

        self.name = name
        self._k = k
        self._L = L
        self._D = D
        self._t = t

        self._D1 = self._D
        self._D2 = self._D + 2*self._t

        self.__R = R

    @property
    def _U(self):

        return self._k/self._t

    @property
    def _R(self):
        """
        Calculates and returns the thermal resistance of the cylindrical wall.

        Returns:
            float: The thermal resistance.
        """
        if self.__R is not None:
            return self.__R
        
        return np.log(self._D2 / self._D1) / (2 * np.pi * self._L * self._k)

   

class Heatsistor_surface(Node):

    _displayVars = ['_UA']

    def __init__(self, name, layers=[]):

        self.name = name
        self.layers = layers
        self.__L = 1

    def initialize(self, model):

        self.model = model
        for layer in self.layers:
            layer.initialize(model)
            layer._L = self._L

    @property
    def _L(self):
        return self.__L

    @_L.setter
    def _L(self, L):
        self.__L = L
        for layer in self.layers:
            layer._L = L

    @property
    def _R(self):
        # Calculate Equivalent Resistance

        R = 0
        for layer in self.layers:
            R+=layer._R

        return R

    @property
    def _UA(self):
        return 1/self._R

    @property
    def _R_layers(self):
        # Get R values for all the layers
        return {layer.name: layer._R for layer in self.layers}

    @property
    def _UA_layers(self):
        # Get UA for all the layers
        return {layer.name: 1/layer._R for layer in self.layers}

    @property
    def _U_layers(self):
        # Get UA for all the layers
        return {layer.name: layer._U for layer in self.layers}