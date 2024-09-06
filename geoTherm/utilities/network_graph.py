from ..nodes.baseClasses import ThermoNode, flowNode
from ..nodes.rotor import Rotor
from ..nodes.turbine import Turbine
from ..nodes.pump import Pump
from ..nodes.heat import staticHEX
from ..units import units
from ..logger import logger
import pyyed
from plantuml import PlantUML
import webbrowser
from pathlib import Path


def generate_dot_code(model):
    """
    Generates DOT code representing the graph structure of the nodes and
    their connections in the given model.

    Args:
        model (object): geoTherm Model Class

    Returns:
        str: The generated DOT code as a string.
    """

    # Get Nodes and Node Map from the model
    nodes = model.nodes
    node_map = model.node_map
    # Get the output units
    u = units.output_units

    # Start building the DOT graph
    dot = 'digraph G {\nnode [width=0, height=0, margin=0];\n'

    # Iterate through all nodes to define their properties in the DOT format
    for name, node in nodes.items():

        # Define properties for different node types
        if isinstance(node, ThermoNode):
            shape = 'circle'
            if node.thermo.phase == 'gas':
                color = 'red'
            else:
                color = 'blue'
            label = (
                f"{name}\nT: {node.T:.1f} {u['TEMPERATURE']}\n"
                f"P: {node.P:.1f} {u['PRESSURE']}"
            )
        elif isinstance(node, Turbine):
            shape = 'trapezium'
            color = 'black'
            label = (
                f"{name}\nw: {node.w:.2f} {u['MASSFLOW']}\n"
                f"W: {node.W:.1f} {u['POWER']}\nPR: {node.PR:.2f}"
            )
        elif isinstance(node, Pump):
            shape = 'invtrapezium'
            color = 'black'
            label = (
                f"{name}\nw: {node.w:.2f} {u['MASSFLOW']}\n"
                f"W: {node.W:.1f} {u['POWER']}\nPR: {node.PR:.2f}"
            )
        elif isinstance(node, flowNode):
            if node.name == 'Turb':
                from pdb import set_trace
                set_trace()
            shape = 'rectangle'
            color = 'black'
            label = f"{name}\nw: {node.w:.2f} {u['MASSFLOW']}\n"
        elif isinstance(node, Rotor):
            color = 'black'
            shape = 'box'
            label = f"{name}\n N:{node.N:.2f}\n {u['ROTATIONSPEED']}"
        else:
            color = 'orange'
            shape = 'circle'
            label = f"{name}"

        # Add the node to the DOT graph
        dot += f'{name} [shape={shape}, label="{label}", color={color}]\n'

    # Iterate through the node map to define connections
    for name, nMap in node_map.items():
        for US in nMap['US']:
            dot += f'{US} -> {name};\n'

        # Connect Rotor Node to turbo objects
        if isinstance(nodes[name], Rotor):
            for load in nodes[name].loads:
                dot += f'{name} -> {load} [dir=both, style=dashed];\n'

    dot += '}'

    return dot


def make_dot_diagram(model, file_path, auto_open=True):
    """
    Generates a DOT plot representing the model's node network and saves it
    as an SVG file.

    Args:
        model (Model): The geoTherm model containing nodes and their
                       connections.
        file_path (str): The path to save the generated SVG file.
        auto_open (bool): Whether to automatically open the SVG file after
                          creation.
    """

    # Generate DOT code for the model
    dot_code = generate_dot_code(model)

    # Initialize the PlantUML server (using the public PlantUML server)
    plantuml = PlantUML(url='http://www.plantuml.com/plantuml/svg/')

    # Render the diagram and save it to the output file
    with open(file_path, 'wb') as f:
        f.write(plantuml.processes(dot_code))

    logger.info(f"DOT diagram saved to: '{file_path}'")

    # Optionally open the file in the web browser
    if auto_open:
        webbrowser.open(Path(file_path))


def estimate_label_size(label):
    """
    Estimates the width and height of a label based on its text content.

    Args:
        label (str): The text label to estimate the size for.

    Returns:
        tuple: Estimated width and height as integers.
    """
    lines = label.split("\n")
    max_line_length = max(len(line) for line in lines)
    width = max_line_length * 10  # Rough width per character
    height = len(lines) * 20      # Rough height per line

    return width, height


def make_graphml_diagram(model, file_path):
    """
    Generates a GraphML file representing the graph structure of the nodes
    and their connections in the given model.

    Args:
        model (Model): The geoTherm model containing nodes and their
                       connections.
        file_path (str): The path to save the generated GraphML file.
    """

    # Initialize the GraphML object
    g = pyyed.Graph()

    # Get Nodes and Node Map from the model
    nodes = model.nodes
    node_map = model.node_map
    # Get the output units
    u = units.output_units

    # Iterate through all nodes to define their properties in the GraphML
    # format
    for name, node in nodes.items():

        label = f'{name}\n'

        # Define properties for different node types
        if isinstance(node, ThermoNode):
            shape = 'ellipse'
            if node.thermo.phase == 'liquid':
                border_color = '#0000FF'  # Blue
            else:
                border_color = '#FF0000'  # Red
            fill_color = '#FFFFFF'  # White
            label += (
                f"T: {node.T:.1f} {u['TEMPERATURE']}\n"
                f"P: {node.P:.1f} {u['PRESSURE']}"
            )
        elif isinstance(node, Turbine):
            shape = 'trapezoid2'
            border_color = '#000000'  # Black
            fill_color = '#FFFFFF'  # White
            label += (
                f"w: {node.w:.2f} {u['MASSFLOW']}\n"
                f"W: {node.W:.1f} {u['POWER']}\nPR: {node.PR:.2f}\n"
                f"\u03B7: {node.eta:.1f}"
            )
        elif isinstance(node, Pump):
            shape = 'trapezoid'
            border_color = '#000000'  # Black
            fill_color = '#FFFFFF'  # White
            label += (
                f"w: {node.w:.2f} {u['MASSFLOW']}\n"
                f"W: {node.W:.1f} {u['POWER']}\nPR: {node.PR:.2f}\n"
                f"\u03B7: {node.eta:.1f}"
                )
        elif isinstance(node, staticHEX):
            shape = 'diamond'
            border_color = '#000000'  # Black
            fill_color = '#FFFFFF'  # White
            label += (
                f"w: {node.w:.2f} {u['MASSFLOW']}\n"
                f"Q: {node.Q:.1f} {u['POWER']}\n"
                f"dP: {node.dP:.1f} {u['PRESSURE']}\n")            
        elif isinstance(node, flowNode):
            shape = 'diamond'
            border_color = '#000000'  # Black
            fill_color = '#FFFFFF'  # White
            label += f"w: {node.w:.2f} {u['MASSFLOW']}\n"
        elif isinstance(node, Rotor):
            shape = 'rectangle'
            border_color = '#000000'  # Black
            fill_color = '#FFFFFF'  # White
            label += f"N:{node.N:.2f}\n {u['ROTATIONSPEED']}"
        else:
            from pdb import set_trace
            set_trace()

        # Estimate size based on the label content
        width, height = estimate_label_size(label)

        # Enforce square aspect ratio for thermoNode instances
        if isinstance(node, ThermoNode):
            width = min([width, height])*1.5
            height = width

        g.add_node(
            name, label=label, shape=shape, border_color=border_color,
            shape_fill=fill_color, width=str(width), height=str(height))

    # Iterate through the node map to define connections
    for name, nMap in node_map.items():
        for US in nMap.get('US', []):
            g.add_edge(US, name)

        # Connect Rotor Node to turbo objects
        if isinstance(nodes[name], Rotor):
            for load in nodes[name].loads:
                g.add_edge(name, load, arrowhead='standard',
                           line_type='dashed')
                g.add_edge(load, name, arrowhead='standard',
                           line_type='dashed')

    g.write_graph(file_path)

    logger.info(f"graphml diagram saved to: '{file_path}'")
