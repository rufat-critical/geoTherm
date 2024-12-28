from .baseNodes.baseFlow import baseFlow


class fixedFlow(baseFlow):

    def __init__(self, name, US, DS,
                 w:"MASSFLOW"):

        super().__init__(name, US, DS)

        self._w = w

    @property
    def _dP(self):

        return self.DS_node.thermo._P - self.US_node.thermo._P 

    def get_outlet_state(self, US, PR):
        """
        Calculate the thermodynamic state at the outlet (downstream).

        Returns:
            dict: A dictionary containing the enthalpy ('H') and pressure ('P')
                  at the downstream node.
        """

        # Get US, DS Thermo
        #US = self.model.nodes[self.US].thermo
        return {'H': US._H, 'P': US._P*PR}

    def get_inlet_state(self, DS, PR):

        return {'H': DS._H,
                'P': DS._P/PR}
