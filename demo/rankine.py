import geoTherm as gt
from pdb import set_trace

bar = 1e5

fluid = 'water'


# A = gt.Model([gt.PBoundary(name='P', fluid=fluid, P=3*bar),
          # gt.HEX_T(name='HEX1', US = 'P', DS = 'Tin', Tout=473, dP =.1*bar, w=100),
          # gt.Station(name='Tin', fluid=fluid),
          # gt.Turbine(name='Turb', eta=0.75, PR=3, w=100, US='Tin', DS='Tout'),
          # gt.Station(name='Tout', fluid=fluid),
          # gt.HEX_T(name='HEX2', US = 'Tout', DS = 'Cin', Tout=320, dP = 1*bar, w=100),
          # gt.Station(name='Cin', fluid=fluid),
          # gt.Pump(name='Pump', eta=0.75, PR=20, w=100, US='Cin', DS='P')])
          

# A = gt.Model([gt.Boundary(name='P', fluid=fluid, P=4*bar, T=300),
              # gt.Pump(name='Pump', eta=0.75, PR=5, w=100, US='P', DS='Cout'),
              # gt.Station(name='Cout', fluid=fluid),
              # gt.HEX_T(name='HEX1', US = 'Cout', DS = 'Tin', Tout=473, dP =1*bar, w=100),
              # gt.Station(name='Tin', fluid=fluid),
              # gt.Turbine(name='Turb', eta=0.75, PR=3, w=100, US='Tin', DS='Tout'),
              # gt.Station(name='Tout', fluid=fluid),
              # gt.HEX_T(name='HEX2', US = 'Tout', DS = 'P', Tout=320, dP = 0*bar, w=100)])

fluid = 'Isobutane'
w = 100##3.77*1e5/60/60
A = gt.Model([gt.PBoundary(name='P', fluid=fluid, P=5*bar, T=300),
              gt.Pump(name='Pump', eta=1, PR=6, w=w, US='P', DS='Cout'),
              gt.Station(name='Cout', fluid=fluid),
              gt.HEX_T(name='HEX1', US = 'Cout', DS = 'Tin', Tout=360, D=5*.0254, L=3, w=w),
              gt.Station(name='Tin', fluid=fluid),
              gt.Turbine(name='Turb', eta=1, PR=3, w=w, US='Tin', DS='Tout'),
              gt.Station(name='Tout', fluid=fluid),
              gt.HEX_T(name='HEX2', US = 'Tout', DS = 'P', Tout=300, D=8*.0254, L=3, w=w)])




A.initialize()
A


#A.evaluate(A.x
from scipy.optimize import fsolve
A.nodes['HEX1'].error(A)
fsolve(A.evaluate,A.x)
print(f"Pmax: {A.nodes['Cout'].P*14.7/101325}, Tpump:{A.nodes['Cout'].T}")
print(f"Pump Inlet Phase: {A.nodes['P'].phase}, Outlet Phase: {A.nodes['Cout'].phase}")
print(f"HEX dP: {A.nodes['HEX1'].dP*14.7/101325}")
print(f"Turbine In T: {A.nodes['Tin'].T} P:{A.nodes['Tin'].P*14.7/101325}")
print(f"Turbine Out Phase: {A.nodes['Tout'].phase}")
print(f"Turbine PR: {A.nodes['Turb'].PR}")
print(f"Turb: {A.nodes['Turb'].W*1e-6} Pump: {A.nodes['Pump'].W*1e-6}")
print(f"Condensor: dP: {A.nodes['HEX2'].dP*14.7/101325}")
print((A.nodes['Turb'].W-A.nodes['Pump'].W)*1e-6)

gt.units.output='ENGLISH'

print(A)
set_trace()



#gt.Model([gt.PBoundary(name='1', 'H2O', P=10*bar),
#          gt.HEX(name='HEX1', US = '1', DS = '2', Tout=473),
#          gt.Station(name='2', 'H2O'),
#          gt.Turbine(name='T', US='2', DS='3', eta=0.75, PR=3),
#          gt.Station(name='3', 'H2O'),
#          gt.HEX(name='HEX2', US='3', DS='4', Tout=300),
#          gt.Station(name='4', 'H2O'),
#          gt.Pump(name='P', US='4', DS='1', eta = 0.75, PR=4)])
          
#gt.Model.solve()

set_trace()

