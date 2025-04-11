import pandas as pd
import os.path
from geoTherm.units import toSI


def tube_reader(xls_path):
    """
    Read tube data from an Excel file and return a structured dictionary of tube information.

    Args:
        xls_path (str): Path to the Excel file containing tube data

    Returns:
        dict: A nested dictionary containing tube data organized by fluid node, part number, and component

    Raises:
        FileNotFoundError: If the Excel file doesn't exist
        ValueError: If the file is not an Excel file
        Exception: If there's an error reading the Excel file
    """
    # Check if file exists
    if not os.path.exists(xls_path):
        raise FileNotFoundError(f"Excel file '{xls_path}' not found. Please check the file path and try again.")

    # Check if file has the correct extension
    if not xls_path.endswith(('.xlsx', '.xlsm', '.xls')):
        raise ValueError(f"File '{xls_path}' is not an Excel file. Please provide a file with .xlsx, .xlsm, or .xls extension.")

    try:
        df = pd.read_excel(xls_path, sheet_name="Tube Lengths", header=2, dtype={'Part Number': str})
    except Exception as e:
        raise Exception(f"Error reading Excel file '{xls_path}': {str(e)}")


    UNITS = {
        'Z': 'mm',
        'Angle': 'deg',
        'Length': 'mm',
        'Diameter': 'in'
    }

    # Forward fill the categorical columns
    df[['Fluid Node', 'Part Number', 'Flow Component']] = df[['Fluid Node', 'Part Number', 'Flow Component']].ffill()

    # Select relevant columns
    df_relevant = df[[
        'Fluid Node',
        'Part Number',
        'Flow Component',
        'Vertical Height Change (Delta Z)',
        'Diameter (in)',
        'Bend Angle',
        'Length (m)'
    ]].copy()

    # Drop rows where all measurement columns are NA
    df_relevant = df_relevant.dropna(subset=[
        'Vertical Height Change (Delta Z)',
        'Diameter (in)',
        'Bend Angle',
        'Length (m)'
    ], how='all')

    # Build the tube data dictionary
    tube_data = {}

    for _, row in df_relevant.iterrows():
        fluid_node = row['Fluid Node']
        part = row['Part Number']
        component = row['Flow Component']

        # Initialize nested structure if it doesn't exist
        if fluid_node not in tube_data:
            tube_data[fluid_node] = {}
        if part not in tube_data[fluid_node]:
            tube_data[fluid_node][part] = {}
        if component not in tube_data[fluid_node][part]:
            tube_data[fluid_node][part][component] = []

        # Create a dimension dictionary for this row
        dimension = {
            'Z': 0,
            'D': 0,
            'Angle': 0,
            'L': 0
        }

        # Add data if it exists, otherwise keep the zero default
        if pd.notna(row['Vertical Height Change (Delta Z)']):
            dimension['Z'] = toSI((row['Vertical Height Change (Delta Z)'], UNITS['Z']), 'LENGTH')
        if pd.notna(row['Diameter (in)']):
            dimension['D'] = toSI((row['Diameter (in)'], UNITS['Diameter']), 'LENGTH')
        if pd.notna(row['Bend Angle']):
            dimension['Angle'] = toSI((row['Bend Angle'], UNITS['Angle']), 'ANGLE')
        if pd.notna(row['Length (m)']):
            dimension['L'] = toSI((row['Length (m)'], UNITS['Length']), 'LENGTH')

        tube_data[fluid_node][part][component].append(dimension)

    return tube_data
