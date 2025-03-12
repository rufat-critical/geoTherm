from geoTherm.nodes.baseNodes.baseThermo import baseThermo
from geoTherm.nodes.baseNodes.baseThermal import baseThermal
from geoTherm.nodes.flowDevices import baseFlow
import pandas as pd
from tabulate import tabulate
from geoTherm.unitSystems import SI, ENGLISH, MIXED
import geoTherm.units


def print_model_tables(model):
    """
    Print formatted tables of thermodynamic, flow, and thermal nodes in the model.
    
    Args:
        model: geoTherm model instance
    """

    # Get unit system from geoTherm units
    units = geoTherm.units.output_units
    
    # Collect thermo node data
    thermo_data = []
    flow_data = []
    thermal_data = []
    
    for name, node in model.nodes.items():
        if isinstance(node, baseThermo):
            thermo_data.append({
                'Node': name,
                f"P [{units['PRESSURE']}]": f"{node.thermo.P:.2e}",
                f"T [{units['TEMPERATURE']}]": f"{node.thermo.T:.2f}",
                f"H [{units['SPECIFICENERGY']}]": f"{node.thermo.H:.2e}"
            })
        elif isinstance(node, baseFlow):
            # Get upstream and downstream connections
            us_name = node.US_node.name if hasattr(node, 'US_node') else 'None'
            ds_name = node.DS_node.name if hasattr(node, 'DS_node') else 'None'
            
            flow_data.append({
                'Node': name,
                'Upstream': us_name,
                'Downstream': ds_name,
                f"ṁ [{units['MASSFLOW']}]": f"{node.w:.3f}",
                f"ΔP [{units['PRESSURE']}]": f"{node.dP:.2e}",
                f"ΔH [{units['SPECIFICENERGY']}]": f"{node.dH:.2e}"
            })
        elif isinstance(node, baseThermal):
            hot_name = node.hot
            cold_name = node.cool
            
            thermal_data.append({
                'Node': name,
                'Hot Side': hot_name,
                'Cold Side': cold_name,
                f"Q [{units['POWER']}]": f"{node.Q:.2e}",
            })
    
    # Create DataFrames
    thermo_df = pd.DataFrame(thermo_data)
    flow_df = pd.DataFrame(flow_data)
    thermal_df = pd.DataFrame(thermal_data)
    
    print("\nThermodynamic Nodes:")
    print(tabulate(thermo_df, headers='keys', tablefmt='psql', showindex=False))
    
    print("\nFlow Nodes:")
    print(tabulate(flow_df, headers='keys', tablefmt='psql', showindex=False))
    
    print("\nThermal Nodes:")
    print(tabulate(thermal_df, headers='keys', tablefmt='psql', showindex=False))