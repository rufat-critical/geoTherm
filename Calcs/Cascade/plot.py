import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np

# Read the data
df = pd.read_csv('results/isobutane_behzad_map.csv')

# Create a list of parameters to plot
parameters = [
    ('P_turbine', 'Turbine Inlet Pressure (bar)'),
    ('P_pump', 'Pump Inlet Pressure (bar)'),
    ('Turb_PR', 'Turbine Pressure Ratio'),
    ('Pump_PR', 'Pump Pressure Ratio'),
    ('Turb_power', 'Turbine Power (MW)'),
    ('Pump_power', 'Pump Power (MW)'),
    ('Heat_input', 'Heat Input (MW)'),
    ('Wnet', 'Net Power (MW)'),
    ('System_efficiency', 'System Efficiency'),
    ('Turb_mass_flow', 'Turbine Mass Flow (kg/s)'),
    ('Turb_Q_in', 'Turbine Q_in (m³/s)'),
    ('Turb_Q_out', 'Turbine Q_out (m³/s)'),
    ('water_mass_flow', 'Water Mass Flow (kg/s)'),
    ('dp_hot', 'Hot Side Pressure Drop (bar)')
]

# Convert pressure values to bar
df['P_turbine'] = df['P_turbine']
df['P_pump'] = df['P_pump']
df['dp_hot'] = df['dp_hot']

# Convert power values to MW
df['Turb_power'] = df['Turb_power']
df['Pump_power'] = df['Pump_power']
df['Heat_input'] = df['Heat_input']
df['Wnet'] = df['Wnet']

# Create subplots
fig = make_subplots(
    rows=5, cols=3,
    subplot_titles=[param[1] for param in parameters],
    vertical_spacing=0.1,
    horizontal_spacing=0.1
)

# Define a fixed color palette for T_cold values
colors = ['#1f77b4', '#ff7f0e', '#2ca02c']  # Blue, Orange, Green

# Plot each parameter
for idx, (param, title) in enumerate(parameters):
    row = idx // 3 + 1
    col = idx % 3 + 1
    
    # Plot for each T_cold value
    for i, t_cold in enumerate(sorted(df['T_cold'].unique())):
        mask = df['T_cold'] == t_cold
        fig.add_trace(
            go.Scatter(
                x=df[mask]['T_hot'],
                y=df[mask][param],
                name=f'T_cold = {t_cold}°C' if idx == 0 else None,  # Only show legend for first subplot
                mode='lines+markers',
                line=dict(color=colors[i]),
                marker=dict(color=colors[i]),
                showlegend=True if idx == 0 else False  # Only show legend for first subplot
            ),
            row=row, col=col
        )

# Update layout
fig.update_layout(
    height=1500,
    width=1500,
    title_text="System Parameters forBehzad's Axial Turbine for 3.5MW Target",
    showlegend=True,
    legend=dict(
        yanchor="bottom",
        y=1.02,  # Position just below the title
        xanchor="center",
        x=0.5,   # Center the legend
        orientation="h"  # Make the legend horizontal
    )
)

# Update y-axis labels
for i, (param, title) in enumerate(parameters):
    row = i // 3 + 1
    col = i % 3 + 1
    fig.update_yaxes(title_text=title, row=row, col=col)

# Update x-axis labels
for i in range(1, 6):
    fig.update_xaxes(title_text="Hot Temperature (°C)", row=i, col=1)

# Save the plot as HTML
fig.write_html("results/orc_parameters_plot.html")

# Show the plot
fig.show()