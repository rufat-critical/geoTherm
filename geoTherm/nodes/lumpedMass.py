from .baseNodes.baseNode import Node


class LumpedMass(Node):
    pass

# Material Properties
class SS316(LumpedMass):

    @property
    def _density(self):
        return 8000

    @property
    def _k(self):
        return 16.2

    @property
    def _Cp(self):
        return 480
