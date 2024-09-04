from geoTherm import thermoPlotter
from matplotlib import pyplot as plt
from geoTherm import units

# Change output units to mixed
units.output = 'mixed'

# Plot acetone
acetone = thermoPlotter('acetone')
acetone.add_state_point(name='1', state={'T': (30, 'degC'), 'P': (1.1, 'atm')})
acetone.add_state_point(name='2', state={'T': (33, 'degC'), 'P': (26, 'bar')})
acetone.add_state_point(name='3', state={'T': (180, 'degC'), 'P': (18, 'bar')})
acetone.add_state_point(name='4', state={'T': (360, 'degK'), 'P': (2, 'bar')})
# Add Pressure isolines
acetone.add_isoline('P', '3')
acetone.add_isoline('P', '4')
# Make TS plot
acetone.plot('TS', show=False)

# 2.5 M Btu

# Remove isolines
acetone.remove_isolines()
# Add Entropy Isolines
acetone.add_isoline('S', '3')
acetone.add_isoline('S', '4')
# Also make Pv plot
acetone.plot('Pv', xscale='log', show=False)

# Plot multiple fluids
fluids = thermoPlotter('acetone')
fluids.add_fluid('n-Pentane')
fluids.add_fluid('isobutane')
fluids.add_fluid('butane')
fluids.add_fluid('water')
fluids.plot('TS', show=False)
fluids.plot('PT', show=False)

# Show all the plots
plt.show()