import geoTherm as gt
from pdb import set_trace
from matplotlib import pyplot as plt



mdot_H2O = 700
mdot_iso = 57.1
L_HEX = (10, 'm')

# Outlets are dummy nodes that we need to specify to build the model
Shell_inlet = gt.Boundary('Sinlet', fluid='isobutane', state={'T':378.74, 'P':(6.2, 'bar')})
Shell_outlet = gt.POutlet('Soutlet', fluid='isobutane', state={'T':317, 'P':590857.8792414305})
Tube_inlet = gt.Boundary('Tinlet', fluid='water', state={'T':311.8, 'P':202497.70642367066})
Tube_outlet = gt.POutlet('Toutlet', fluid='water', state={'T':312, 'P':(1.5, 'bar')})

# HEX
shell_vol = gt.hexFlow('Shell', 'Sinlet', 'Soutlet', 'HEX',w=57.1)
tube_vol = gt.hexFlow('Tube', 'Tinlet', 'Toutlet','HEX', w=700)

HEX = gt.HEX('HEX', shell='Shell', tube='Tube', L=L_HEX, D_tube=.0127-2*0.0012446, D_shell=1, n_tubes=4000,
            wall_th=0.0012446, k_wall= 14.4, n_points=200)

model = gt.Model([Shell_inlet, Shell_outlet, Tube_inlet, Tube_outlet, shell_vol, tube_vol, HEX])

# This uses fsolve to determine what the Tube Outlet State should be
#model['HEX'].evaluate()
# This is used to specify tube outlet and get Inlet
model['HEX'].set_tube_outlet([Tube_outlet.thermo._H, Tube_outlet.thermo._P])


# Change output to mixed unit system (mixed has Power as MW)
gt.units.output='mixed'

# Print Output
print(model['HEX'])


# Plot Stuff
plt.plot(model['HEX'].hexCalcs['L'], model['HEX'].hexCalcs['Q']*1e-6)
plt.xlabel("Length [m]")
plt.ylabel("Q [MW]")
plt.figure()
plt.plot(model['HEX'].hexCalcs['L'], model['HEX'].hexCalcs['shell']['T'],label='shell')
plt.plot(model['HEX'].hexCalcs['L'], model['HEX'].hexCalcs['tube']['T'],label='Tube')
plt.xlabel("Length [m]")
plt.ylabel("T [K]")
plt.legend()
plt.figure()
plt.plot(model['HEX'].hexCalcs['L'], model['HEX'].hexCalcs['shell']['P']*1e-5,label='shell')
plt.plot(model['HEX'].hexCalcs['L'], model['HEX'].hexCalcs['tube']['P']*1e-5,label='Tube')
plt.xlabel("Length [m]")
plt.ylabel("P [bar]")
plt.legend()
plt.figure()
plt.plot(model['HEX'].hexCalcs['L'], model['HEX'].hexCalcs['R'])
plt.xlabel("Length [m]")
plt.ylabel("R [K/W]")
plt.show()

#model.solve(netSolver=False)

from pdb import set_trace
set_trace()


