import pandas as pd
import os.path
from geoTherm.units import toSI
import numpy as np


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


def fluid_property_reader(xls_path):
    """
    Read fluid property data from an Excel file and return a pandas DataFrame with SI-converted values.
    The Excel file should have a structure where:
    - A row contains "DATA" followed by property names
    - A row contains "UNITS" followed by corresponding units
    - The actual data follows below these rows

    Args:
        xls_path (str): Path to the Excel file containing fluid property data

    Returns:
        pandas.DataFrame: A DataFrame containing fluid properties with values converted to SI units.
                         Original units are stored in the column names.

    Raises:
        FileNotFoundError: If the Excel file doesn't exist
        ValueError: If the file is not an Excel file, required keywords not found, or unrecognized property header
        Exception: If there's an error reading the Excel file
    """
    # Check if file exists
    if not os.path.exists(xls_path):
        raise FileNotFoundError(f"Excel file '{xls_path}' not found. Please check the file path and try again.")

    # Check if file has the correct extension
    if not xls_path.endswith(('.xlsx', '.xlsm', '.xls')):
        raise ValueError(f"File '{xls_path}' is not an Excel file. Please provide a file with .xlsx, .xlsm, or .xls extension.")

    # Define recognized property headers and their corresponding SI quantity types
    PROPERTY_QUANTITIES = {
        'Temperature': 'TEMPERATURE',
        'Pressure': 'PRESSURE',
        'Density': 'DENSITY',
        'Specific Heat': 'SPECIFICHEAT',
        'Thermal Conductivity': 'CONDUCTIVITY',
        'Vapor Pressure': 'PRESSURE',
        'Kinematic Viscosity': 'KINEMATICVISCOSITY',
    }

    try:
        # Read the entire Excel file without specifying header
        df = pd.read_excel(xls_path, header=None)
        
        # Find the row containing "DATA"
        data_row = df[df.apply(lambda x: x.astype(str).str.contains('DATA', case=False, na=False)).any(axis=1)].index
        if len(data_row) == 0:
            raise ValueError("Could not find 'DATA' keyword in the Excel file")
        data_row = data_row[0]
        
        # Find the row containing "UNITS"
        units_row = df[df.apply(lambda x: x.astype(str).str.contains('UNITS', case=False, na=False)).any(axis=1)].index
        if len(units_row) == 0:
            raise ValueError("Could not find 'UNITS' keyword in the Excel file")
        units_row = units_row[0]
        
        # Find the column containing "DATA"
        data_col = df.iloc[data_row].astype(str).str.contains('DATA', case=False, na=False)
        data_col = data_col[data_col].index[0]
        
        # Get property names and units
        properties = df.iloc[data_row, (data_col + 1):].dropna().tolist()
        
        # Validate property headers
        for prop in properties:
            if prop not in PROPERTY_QUANTITIES:
                raise ValueError(f"Unrecognized property header: '{prop}'. Valid headers are: {', '.join(PROPERTY_QUANTITIES.keys())}")
        
        units = df.iloc[units_row, (data_col + 1):len(properties) + data_col + 1].tolist()
        
        # Read the actual data, starting from the row after units
        data_df = df.iloc[(units_row + 1):, (data_col + 1):len(properties) + data_col + 1]
        data_df.columns = properties
        
        # Create a new DataFrame for SI-converted values
        si_df = pd.DataFrame()
        
        # Convert each column to SI units
        for prop, unit in zip(properties, units):
            if unit.strip():  # Only convert if unit is specified
                try:
                    # Convert non-NaN values to SI units
                    si_df[f"{prop}"] = data_df[prop].apply(
                        lambda x: toSI((x, unit), PROPERTY_QUANTITIES[prop]) if pd.notna(x) else x
                    )
                except Exception as e:
                    print(f"Warning: Could not convert {prop} to SI units. Error: {str(e)}")
                    si_df[f"{prop}"] = data_df[prop]
            else:
                # If no unit specified, keep original values
                si_df[prop] = data_df[prop]
        
        return si_df

    except Exception as e:
        raise Exception(f"Error reading Excel file '{xls_path}': {str(e)}")


def concepts_excel_reader(xlsx_path, output_csv_path=None):
    """
    Load and process data from Excel file.
    
    Args:
        xlsx_path (str): Path to Excel file
        output_csv_path (str, optional): Path to save processed data as CSV. If None, no CSV is saved.
        
    Returns:
        pd.DataFrame: Processed DataFrame with all required parameters
    """
    # Load all sheets
    xls = pd.read_excel(xlsx_path, sheet_name=None, header=None)
    
    # Parameters to extract and their actual labels
    desired_params = {
        'p0.in': 'p0.in',
        'T0.in': 'T0.in',
        'p.out': 'p.out',
        'Power(Shaft)': 'Power(Shaft)',
        'ETA_ts_ad': 'ETA_ts_ad',
        'm.out': 'm.out'
    }
    
    # List to hold all data rows
    all_data = []
    
    # Process each sheet
    for sheet_name, df in xls.items():
        if not sheet_name.endswith("C Sweeps"):
            continue
        
        try:
            # Transpose and clean
            df_t = df.transpose()
            df_t.columns = df_t.iloc[1]  # Second row has actual headers
            df_t_clean = df_t.drop([0, 1]).reset_index(drop=True)
            
            # Find matched columns
            matched_params = {
                k: v for k, v in desired_params.items() if v in df_t_clean.columns
            }
            
            # Skip if nothing found
            if not matched_params:
                print(f"[{sheet_name}] Skipping — no matched parameters.")
                continue
            
            # Process each row of data
            for idx in range(len(df_t_clean)):
                row_data = {}
                valid_row = True
                
                # Check if all required parameters are present and valid
                for param, label in matched_params.items():
                    value = df_t_clean[label].iloc[idx]
                    if pd.isna(value):
                        valid_row = False
                        break
                    try:
                        value = float(value)
                        # Additional validation for efficiency and pressure
                        if param == 'ETA_ts_ad' and (value < 0 or value > 1):
                            valid_row = False
                            break
                        if param in ['p0.in', 'p.out'] and value <= 0:
                            valid_row = False
                            break
                        # Convert pressure from bar to Pa
                        if param in ['p0.in', 'p.out']:
                            value = value * 1e5
                        row_data[param] = value
                    except (ValueError, TypeError):
                        valid_row = False
                        break
                
                # Only add complete rows
                if valid_row and len(row_data) == len(matched_params):
                    all_data.append(row_data)
        
        except Exception as e:
            print(f"[{sheet_name}] Failed: {e}")

    # Create final DataFrame preserving the order
    final_df = pd.DataFrame(all_data)

    # Calculate pressure ratio
    final_df['PR_ts'] = final_df['p0.in'] / final_df['p.out']

    # Calculate corrected mass flow
    T_ref = 288.15  # K
    p_ref = 101325  # Pa

    # Calculate corrected mass flow from actual mass flow
    final_df['m_c'] = final_df['m.out'] * np.sqrt(final_df['T0.in']/T_ref) * (p_ref/final_df['p0.in'])

    # Save to CSV if output path is provided
    if output_csv_path:
        final_df.to_csv(output_csv_path, index=False)
        print(f"Processed data saved to: {output_csv_path}")

    return final_df