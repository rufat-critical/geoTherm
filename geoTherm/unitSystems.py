from dataclasses import dataclass

### UNIT Systems for geoTherm

@dataclass
class SI:
    """
    SI Units Class

    This class defines the standard units used in the International System of Units (SI)
    for various physical quantities.
    """
    units = {
        'AREA': 'm**2',                  # Square meter for area
        'DENSITY': 'kg/m**3',            # Kilogram per cubic meter for density
        'ENERGY': 'J',                   # Joule for energy
        'LENGTH': 'm',                   # Meter for length
        'MASS': 'kg',                    # Kilogram for mass
        'MASSFLOW': 'kg/s',              # Kilogram per second for mass flow rate
        'POWER': 'W',                    # Watt for power
        'PRESSURE': 'Pa',                # Pascal for pressure
        'SPECIFICENERGY': 'J/kg',        # Joule per kilogram for specific energy
        'SPECIFICENTROPY': 'J/kg/degK',  # Joule per kilogram per Kelvin for specific entropy
        'SPECIFICHEAT': 'J/kg/degK',     # Joule per kilogram per Kelvin for specific heat
        'SPECIFICPOWER': 'W/kg',         # Watt per kilogram for specific power
        'TEMPERATURE': 'degK',           # Kelvin for temperature
        'VELOCITY': 'm/s',               # Meter per second for velocity
        'VISCOSITY': 'Pa*s',             # Pascal-second for viscosity
        'VOLUME': 'm**3'                 # Cubic meter for volume
    }

@dataclass
class ENGLISH:
    """
    English Units Class

    This class defines the standard units used in the English unit system
    for various physical quantities.
    """
    units = {
        'AREA': 'in**2',                   # Square inch for area
        'DENSITY': 'lb/in**3',             # Pound per cubic inch for density
        'ENERGY': 'Btu',                   # British thermal unit for energy
        'LENGTH': 'in',                    # Inch for length
        'MASS': 'lb',                      # Pound for mass
        'MASSFLOW': 'lb/s',                # Pound per second for mass flow rate
        'POWER': 'Btu/s',                  # British thermal unit per second for power
        'PRESSURE': 'psi',                 # Pounds per square inch for pressure
        'SPECIFICENERGY': 'Btu/lb',        # British thermal unit per pound for specific energy
        'SPECIFICENTROPY': 'Btu/lb/degR',  # British thermal unit per pound per Rankine for specific entropy
        'SPECIFICHEAT': 'Btu/lb/degR',     # British thermal unit per pound per Rankine for specific heat
        'SPECIFICPOWER': 'Btu/s/lb',       # British thermal unit per second per pound for specific power
        'TEMPERATURE': 'degR',             # Rankine for temperature
        'VELOCITY': 'ft/s',                # Feet per second for velocity
        'VISCOSITY': 'lb/in/s',            # Pound per inch per second for viscosity
        'VOLUME': 'in**3'                  # Cubic inch for volume
    }

@dataclass
class MIXED:
    """
    Mixed Units Class

    This class defines a mix of SI and English units used for various physical quantities.
    """
    units = {
        'AREA': 'in**2',                   # Square inch for area
        'DENSITY': 'kg/m**3',              # Kilogram per cubic meter for density
        'ENERGY': 'MJ',                    # Megajoule for energy
        'LENGTH': 'in',                    # Inch for length
        'MASS': 'lb',                      # Pound for mass
        'MASSFLOW': 'kg/s',                # Kilogram per second for mass flow rate
        'POWER': 'MW',                     # Megawatt for power
        'PRESSURE': 'bar',                 # Pounds per square inch for pressure
        'SPECIFICENERGY': 'kJ/kg',         # Kilojoule per kilogram for specific energy
        'SPECIFICENTROPY': 'kJ/kg/degK',   # Kilojoule per kilogram per Kelvin for specific entropy
        'SPECIFICHEAT': 'kJ/kg/K',         # Kilojoule per kilogram per Kelvin for specific heat
        'SPECIFICPOWER': 'MW/kg',          # Megawatt per kilogram for specific power
        'TEMPERATURE': 'degK',             # Kelvin for temperature
        'VELOCITY': 'm/s',                 # Meter per second for velocity
        'VISCOSITY': 'Pa*s',               # Pascal-second for viscosity
        'VOLUME': 'in**3'                  # Cubic inch for volume
    }
