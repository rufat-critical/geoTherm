from ..logger import logger
from .HTC import BaseHTC
from .single_phase import Dittus_Boelter
import numpy as np


class Shah(BaseHTC):

    def __init__(self, geometry):
        super().__init__(geometry)
       
        self.DB = Dittus_Boelter(geometry)

    def evaluate(self, thermo, w):

        if thermo.phase == 'two-phase':
            if thermo.Q >0.999:
                return self.DB.evaluate(thermo, w, heating=False)
            else:
                thermo_liq = thermo.copy()
                thermo_liq._PQ = thermo._P, 0
                h_db_liq = self.DB.evaluate(thermo_liq, w, heating=False)
                Pred = thermo._P/thermo._P_crit

                return h_db_liq*((1-thermo.Q)**0.8 + 3.8*thermo.Q**0.76*(1-thermo.Q)**0.04/(Pred**0.38))

        else:
            logger.warn("Shah correlation is only valid for two-phase flow, returning Dittus-Boelter")
            return self.DB.evaluate(thermo, w)


class Cavallini_Smith_Zecchin(BaseHTC):

    def Nu(self, thermo, w):

        temp_thermo = thermo.copy()
        temp_thermo._PQ = thermo._P, 0
        mul = temp_thermo._viscosity
        Prl = temp_thermo.prandtl
        rhol = temp_thermo._density
        temp_thermo._PQ = thermo._P, 1
        mug = temp_thermo._viscosity
        rhog = temp_thermo._density

        G = w/(np.pi/4*self.params['Dh']**2)


        Re = G*self.params['Dh']/mul
        Rel = Re*(1-thermo.Q)
        Reg = Re*thermo.Q/(mug/mul)
        Reeq = Reg*(mug/mul)*(rhol/rhog)**0.5 + Rel

        return 0.05*Reeq**0.8*Prl**0.33

    def evaluate(self, thermo, w):
        Nu = self.Nu(thermo, w)

        liq_thermo = thermo.copy()
        liq_thermo._PQ = thermo._P, 0
        return Nu*liq_thermo._conductivity/self.params['Dh']
