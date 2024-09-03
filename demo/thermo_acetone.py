from geoTherm import thermoPlotter
from matplotlib import pyplot as plt
from geoTherm import units

# Change output units to mixed
units.output = 'mixed'

# Plot acetone
acetone = thermoPlotter('acetone')
acetone.add_state_point(name='1', state={'T': (30, 'degC'), 'P': (1.1, 'bar')})
acetone.add_state_point(name='2', state={'T': (31.5, 'degC'), 'P': (15.7, 'bar')})
acetone.add_state_point(name='3', state={'T': (180, 'degC'), 'P': (14.7, 'bar')})
acetone.add_state_point(name='4', state={'T': (360, 'degK'), 'P': (2.1, 'bar')})
# Add Pressure isolines
#acetone.add_isoline('P', '2')
#acetone.add_isoline('P', '3')

acetone.add_process_line('compression','1','2','S',line_style=':')
acetone.add_process_line('Heating','2','3','P')
acetone.add_process_line('expansion','3','4','S')
acetone.add_process_line('cooling','4','1','P')


import mplcursors
import plotly.tools as tls
mplcursors.cursor(hover=True)

# Make TS plot
fig = acetone.plot('TS', show=False)

fig.show()
#plotly_fig = tls.mpl_to_plotly(fig)
#plotly_fig.show()

#acetone.plot('Pv', xscale='log', show=False)

# 2.5 M Btu

# Remove isolines
#acetone.remove_isolines()
# Add Entropy Isolines
#acetone.add_isoline('S', '3')
#acetone.add_isoline('S', '4')
# Also make Pv plot
#acetone.plot('TS', xscale='log', show=False)

# Plot multiple fluids
#fluids = thermoPlotter('acetone')
#fluids.add_fluid('n-Pentane')
#fluids.add_fluid('isobutane')
#fluids.add_fluid('butane')
#fluids.add_fluid('water')
#fluids.plot('TS', show=False)
#fluids.plot('PT', show=False)

# Show all the plots
#plt.show()

from pdb import set_trace
set_trace()