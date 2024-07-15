import CoolProp
from CoolProp.Plots import PropertyPlot
from pdb import set_trace
plot = PropertyPlot('HEOS::isobutane', 'TS', unit_system='SI', tp_limits='ORC')
plot.calc_isolines(CoolProp.iQ, num=11)
plot.calc_isolines(CoolProp.iP, iso_range=[101325,101325*10], num=10, rounding=True)
#plot.calc_isolines(CoolProp.iP, iso_range=[1,50], num=10, rounding=True)
plot.draw()
#plot.isolines.clear()
#plot.props[CoolProp.iP]['color'] = 'green'
#plot.props[CoolProp.iP]['lw'] = '0.5'
#plot.calc_isolines(CoolProp.iP, iso_range=[1,50], num=10, rounding=False)

#



plot.show()


set_trace()