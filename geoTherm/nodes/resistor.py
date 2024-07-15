from geoTherm.units import inputParser, addQuantityProperty
from .baseClass import flowNode


# PUT THIS IN A LINE TO FIX MASS FLOW RATE!
# USE THIS TO THROTTLE FLOW AND EVALUATE PROPERTIES

@addQuantityProperty
class fixedFlow(flowNode):
    """ Resistor Object where mass flow is fixed """
    pass

    _units = {'w': 'MASSFLOW'}
    _displayVars = ['w']

    @inputParser
    def __init__(self, name, US, DS,
                w:'MASSFLOW'):

        self.name = name
        self.US = US
        self.DS = DS
        self._w = w

    def getOutletState(self):

        # Get the Downstream thermo state
        US = self.model.nodes[self.US].thermo

        return {'H': US._H, 'P': US._P}