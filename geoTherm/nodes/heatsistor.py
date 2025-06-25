from .baseNodes.baseThermal import baseThermal, baseHeatsistor
from .baseNodes.baseThermo import baseThermo
from .baseNodes.baseFlow import baseFlow
from ..units import inputParser, addQuantityProperty, units, toSI
from ..logger import logger
import numpy as np
from ..resistance_models.heat import HTC
from ..decorators import state_dict


@addQuantityProperty
class Heatsistor(baseHeatsistor):

    _displayVars = ['Q', 'R', 'hot', 'cool']
    _units = {'Q': 'POWER', 'R': 'THERMALRESISTANCE'}

    @inputParser
    def __init__(self, name, hot, cool, R:'THERMALRESISTANCE'):
        self.name = name
        self.hot = hot
        self.cool = cool
        self._R = R

    @state_dict
    def _state_dict(self):
        return {'R': (self._R, 'THERMALRESISTANCE')}

    def evaluate(self):
        T_hot = self.model[self.hot].thermo._T
        T_cold = self.model[self.cool].thermo._T
        self._Q = (T_hot-T_cold)/self._R

    def get_outlet_state(self):

        # Check Temp
        if self._Q > 0:
            T_hot = self.model[self.hot].thermo._T
            D = self.model[self.cool].thermo._density
        else:
            T_hot = self.model[self.cool].thermo._T
            D = self.model[self.hot].thermo._density

        return {'T': T_hot - self._Q*self._R,
                'D': D}

    def get_inlet_state(self):

        if self._Q >0:
            T_cold = self.model[self.cool].thermo._T
            D = self.model[self.hot].thermo._density
        else:
            T_cold = self.model[self.hot].thermo._T
            D = self.model[self.cool].thermo._density

        return {'T': T_cold + self._Q*self._R,
                'D': D}

    def get_cool_state(self, hot_thermo, Q):

        return {'T': hot_thermo._T - Q*self._R,
                'D': hot_thermo._density}



class ConvectiveResistor2(Heatsistor):

    @inputParser
    def __init__(self, name, flow, boundary, HTC):

        self.name = name
        self.flow = flow
        self.boundary = boundary

        self.cool = flow
        self.hot = boundary

        self._H = 1e10

    def evaluate(self):

        self._Q = (self.hot_node.thermo._T - self.cool_node.thermo._T)*self._H
        from pdb import set_trace
        #set_trace()

    def get_cool_state(self, hot_thermo, Q):

        return {'T': hot_thermo._T -Q*self._H,
                'D': hot_thermo._density}

        from pdb import set_trace
        set_trace()


    @property
    def Q2(self):
        from pdb import set_trace
        set_trace()
        return 100



class ConvectiveResistor(Heatsistor):

    def __init__(self, name, cool, hot, h_hot=None, h_cool=None, A_cool=None, k_wall=15, A_hot=None, layers=None):

        self.name = name
        self.cool = cool
        self.hot = hot

        self.h_hot = h_hot
        self.h_cool = h_cool
        self.A_cool = A_cool
        self.A_hot = A_hot
        self.k_wall = k_wall
        self.layers = layers
        self._Q = 0


    def initialize(self, model):
        super().initialize(model)

        if isinstance(self.hot_node, baseThermo):
            if self.h_hot is None:
                logger.critical(f"Convective Resistor is connected to a thermo node {self.name} "
                                f"but no h_hot is provided")
            else:
                # Check if h_hot is a correlation name or constant value
                if isinstance(self.h_hot, str):
                    self.htc_hot = HTC(self.h_hot, self.hot_node)
                else:
                    self.htc_hot = HTC('constant', None, h=self.h_hot)

        if isinstance(self.cool_node, baseThermo):
            if self.h_cool is None:
                logger.critical(f"Convective Resistor is connected to a thermo node {self.name} "
                                f"but no h_cool is provided")
            else:
                # Check if h_cool is a correlation name or constant value
                if isinstance(self.h_cool, str):
                    self.htc_cool = HTC(self.h_cool, self.cool_node)
                else:
                    self.htc_cool = HTC('constant', None, h=self.h_cool)

        if self.h_hot is None or self.h_cool is None:
            from pdb import set_trace
            set_trace()


        if isinstance(self.cool_node, baseFlow):
            self.geometry = self.cool_node.geometry
            self._Acool = self.geometry._Ain
            self._Ahot = self.geometry._Aout

        if isinstance(self.hot_node, baseFlow):
            self.geometry = self.hot_node.geometry
            self._Ahot = self.geometry._Ain
            self._Acool = self.geometry._Aout

        

    #def _R(self):

        #R_hot = 1/(self.h_hot*self._Ahot)
        #R_cool = 1/(self.h_cool*self._Acool)

      #  R_wall = np.log(self.geometry._Do/self.geometry._Di)/(2*np.pi*self.geometry._L*self.k_wall)

     #   return R_hot + R_cool + R_wall


    def evaluate(self): #, w, US):
        
       # #if self.htc_cool
        #self.h_hot = self.htc_hot.evaluate(self.hot_node._w, self.hot_node.thermo)
        #self.h_cool = self.htc_cool.evaluate(self.cool_node._w, self.cool_node.thermo)

        if isinstance(self.h_hot, (float, int)):
            R_hot = 1/(self.h_hot*self._Ahot)
        else:
            from pdb import set_trace
            set_trace()

        if isinstance(self.h_cool, (float, int)):
            R_cool = 1/(self.h_cool*self._Acool)
        else:
            if self.h_cool == 'DB':
                self.htc_cool = HTC('Dittus-Boelter', self.cool_node)
                US, _, _ = self.cool_node.thermostates()
                Nu_cool = self.htc_cool.evaluate(US)
                h_cool = Nu_cool*US._conductivity/self.geometry._Di
                R_cool = 1/(h_cool*self._Acool)
                from pdb import set_trace
                #set_trace()
            else:
                from pdb import set_trace
                set_trace()

        R_wall = np.log(self.geometry._Do/self.geometry._Di)/(2*np.pi*self.geometry._L*self.k_wall)

        self._R = R_hot + R_cool + R_wall



        if isinstance(self.hot_node, baseThermo):
            T_hot = self.hot_node.thermo._T
        else:
            from pdb import set_trace
            set_trace()

        if isinstance(self.cool_node, baseThermo):
            T_cool = self.cool_node.thermo._T
        else:
            US, _, _ = self.cool_node.thermostates()
            T_cool = US._T

        self._Q = (T_hot - T_cool)/self._R




@addQuantityProperty
class Qdot(baseThermal):
    """A thermal component that represents a heat transfer between two nodes.
    
    This class handles heat transfer (Q) between a hot and cold node. The heat transfer
    can be specified either as a constant value or as a function that calculates the
    heat transfer based on the states of the connected nodes.

    Attributes:
        _displayVars (list): Variables displayed in the table ['Q', 'hot', 'cool']
        _units (dict): Units for the variables {'Q': 'POWER'}
        name (str): Name of the component
        hot (str): Name of the hot node
        cool (str): Name of the cold node
        _Q (float or callable): Heat transfer rate in watts or a function to
        calculate it
    """

    _displayVars = ['Q', 'hot', 'cool']
    _units = {'Q': 'POWER'}

    def __init__(self, name, hot=None, cool=None, Q=0):
        """Initialize a Qdot component.

        Args:
            name (str): Name of the component
            hot (str, optional): Name of the hot node. Defaults to None.
            cool (str, optional): Name of the cold node. Defaults to None.
            Q (float or callable, optional): Heat transfer rate in watts or a function
                that calculates heat transfer. The function should accept hot_node and
                cool_node as arguments. Defaults to 0.
        """
        self.name = name
        self.hot = hot
        self.cool = cool
        self.hot_node = None
        self.cool_node = None
        self._Q = Q

    @property
    def _Q(self):
        """Get the current heat transfer rate.

        If Q was initialized as a function, it will be called with the current
        hot and cold node states to calculate the heat transfer.

        Returns:
            float: Heat transfer rate in watts
        """
        return self._Q_func(self.hot_node, self.cool_node, self.model)

    @_Q.setter
    def _Q(self, value):
        """Set the heat transfer rate or function.

        Args:
            value (float or callable): Either a constant heat transfer rate in watts
                or a function that calculates heat transfer. The function should accept
                hot_node and cool_node as arguments.

        Raises:
            TypeError: If value is neither a numeric value nor a callable
        """
        if callable(value):
            self._Q_func = value
        elif isinstance(value, (int, float)):
            def Q_func(hot_node, cool_node, model, Q=value):
                return Q
            self._Q_func = Q_func
        elif isinstance(value, (tuple, list)):
            value = toSI(value, 'POWER')
            def Q_func(hot_node, cool_node, model, Q=value):
                return Q
            self._Q_func = Q_func
        else:
            raise TypeError("Q must be either a callable or a numeric value")

    @state_dict
    def _state_dict(self):
        """Get the state dictionary containing the node's current state information.

        Returns:
            dict: A dictionary containing the current state with units:
                {'Q': (heat_transfer_rate, 'POWER')}
        """
        return {'Q': (self.Q, units.output_units['POWER'])}
