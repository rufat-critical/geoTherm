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
        'DENSITY': 'kg/m**3',            # Kilogram per cubic
        'ENERGY': 'J',                   # Joule
        'LENGTH': 'm',                   # Meter 
        'MASS': 'kg',                    # Kilogram 
        'MASSFLOW': 'kg/s',              # Kilogram per second
        'POWER': 'W',                    # Watt
        'PRESSURE': 'Pa',                # Pascal
        'SPECIFICENERGY': 'J/kg',        # Joule per kilogram
        'SPECIFICENTROPY': 'J/kg/degK',  # Joule per kilogram per Kelvin
        'SPECIFICHEAT': 'J/kg/degK',     # Joule per kilogram per Kelvin
        'SPECIFICPOWER': 'W/kg',         # Watt per kilogram
        'TEMPERATURE': 'degK',           # Kelvin
        'VELOCITY': 'm/s',               # Meter per second
        'VISCOSITY': 'Pa*s',             # Pascal-second
        'CONDUCTIVITY': 'W/m/K',         # Watts per meter-Kelvin
        'VOLUME': 'm**3',                # Cubic meter
        'THERMALRESISTANCE': 'degK/W',   # Thermal Resistance
        'CONVECTION': 'W/m**2/degK',        # Convection coeff
        'VOLUMETRICFLOW': 'm**3/s'
    }

@dataclass
class ENGLISH:
    """
    English Units Class

    This class defines the standard units used in the English unit system
    for various physical quantities.
    """
    units = {
        'AREA': 'in**2',                   # Square inch
        'DENSITY': 'lb/in**3',             # Pound per cubic inch
        'ENERGY': 'Btu',                   # British thermal unit
        'LENGTH': 'in',                    # Inch
        'MASS': 'lb',                      # Pound
        'MASSFLOW': 'lb/s',                # Pound per second
        'POWER': 'Btu/s',                  # British thermal unit per second
        'PRESSURE': 'psi',                 # Pounds per square inch
        'SPECIFICENERGY': 'Btu/lb',        # British thermal unit per pound
        'SPECIFICENTROPY': 'Btu/lb/degR',  # British thermal unit per pound per Rankine
        'SPECIFICHEAT': 'Btu/lb/degR',     # British thermal unit per pound per Rankine
        'SPECIFICPOWER': 'Btu/s/lb',       # British thermal unit per second per pound
        'TEMPERATURE': 'degR',             # Rankine
        'VELOCITY': 'ft/s',                # Feet per second
        'VISCOSITY': 'lb/in/s',            # Pound per inch per second
        'CONDUCTIVITY': 'Btu/hr/ft/F',     # BTU per hour per foot per Fahrenheit
        'VOLUME': 'in**3',                 # Cubic inch
        'THERMALRESISTANCE': 'degR/Btu/s',  # Thermal Resistance
        'CONVECTION': 'Btu/s/ft**2/degR',   # Convection coeff
        'VOLUMETRICFLOW': 'ft**3/s'
    }

@dataclass
class MIXED:
    """
    Mixed Units Class

    This class defines a mix of SI and English units used for various physical quantities.
    """
    units = {
        'AREA': 'in**2',                   # Square inch
        'DENSITY': 'kg/m**3',              # Kilogram per cubic meter
        'ENERGY': 'MJ',                    # Megajoule
        'LENGTH': 'in',                    # Inch
        'MASS': 'lb',                      # Pound
        'MASSFLOW': 'kg/s',                # Kilogram per second
        'POWER': 'MW',                     # Megawatt
        'PRESSURE': 'bar',                 # Pounds per square inch
        'SPECIFICENERGY': 'kJ/kg',         # Kilojoule per kilogram
        'SPECIFICENTROPY': 'kJ/kg/degK',   # Kilojoule per kilogram per Kelvin
        'SPECIFICHEAT': 'kJ/kg/K',         # Kilojoule per kilogram per Kelvin
        'SPECIFICPOWER': 'MW/kg',          # Megawatt per kilogram
        'TEMPERATURE': 'degK',             # Kelvin
        'VELOCITY': 'm/s',                 # Meter per second
        'VISCOSITY': 'Pa*s',               # Pascal-second
        'CONDUCTIVITY': 'W/m/K',           # Watts per meter-Kelvin
        'VOLUME': 'in**3',                 # Cubic inch
        'THERMALRESISTANCE': 'degK/W',      # Thermal Resistance
        'CONVECTION': 'W/m**2/degK',        # Convection coeff
        'VOLUMETRICFLOW': 'm**3/s'
    }
