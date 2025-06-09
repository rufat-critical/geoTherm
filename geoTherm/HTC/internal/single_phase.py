from geoTherm.common import logger
from ..HTC import BaseHTC


class Dittus_Boelter(BaseHTC):

    def evaluate(self, thermo, w, heating=True):

        Nu = self.Nu(thermo, w, heating=heating)
        return Nu*thermo._conductivity/self.geometry._Dh

    def Nu(self, thermo, w, heating=True):

        Re = self.Re(thermo, w)
        Pr = thermo.prandtl

        # Check Applicability
        if 0.6 <= Pr <= 160:
            pass
        else:
            logger.warn("Dittus-Boelter relation is outside valid range"
                        "of 0.6<=Pr<=160, "
                        f"current {Pr}")

        if Re >= 1e4:
            pass
        else:
            logger.warn("Dittus-Boelter relation is outside valid range"
                        f"Re>1e4, current {Re}")

        # Check what Exponent to use for Nusselt #
        if heating:
            n = 0.4
        else:
            n = 0.3

        return 0.023*Re**(0.8)*Pr**n
