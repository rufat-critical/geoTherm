import geoTherm as gt
from pdb import set_trace


thermo = gt.thermo(state={'P':101325, 'Q':0.5})

thermo.HP = thermo.H, thermo.P
thermo.TPY = 300, 101325, 'O2:1'

import CoolProp
from CoolProp.Plots import PropertyPlot
plot = PropertyPlot('HEOS::isobutane', 'TS', unit_system='EUR', tp_limits='ORC')
plot.calc_isolines(CoolProp.iQ, num=11)
plot.calc_isolines(CoolProp.iP, iso_range=[1,50], num=10, rounding=True)
plot.draw()
plot.isolines.clear()
plot.props[CoolProp.iP]['color'] = 'green'
plot.props[CoolProp.iP]['lw'] = '0.5'
plot.calc_isolines(CoolProp.iP, iso_range=[1,50], num=10, rounding=False)
plot.show()


set_trace()