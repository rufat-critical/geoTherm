from dataclasses import dataclass


_DISPLAY_FORMAT = {
    # Temperature units
    'degC': '°C',                      # Celsius
    'degF': '°F',                      # Fahrenheit 
    'degK': 'K',                       # Kelvin
    'degR': '°R',                      # Rankine

    # Basic units
    'm**2': 'm²',                      # Square meter
    'm**3': 'm³',                      # Cubic meter
    'in**2': 'in²',                    # Square inch
    'in**3': 'in³',                    # Cubic inch
    'ft**2': 'ft²',                    # Square feet
    'ft**3': 'ft³',                    # Cubic feet

    # Common compound units
    'kg/m**3': 'kg/m³',                # Density
    'm**3/kg': 'm³/kg',                # Specific volume
    'kg/s/m**2': 'kg/(s·m²)',          # Mass flux
    'W/m**2/degK': 'W/(m²·K)',         # Heat transfer coefficient
    'W/m/K': 'W/(m·K)',                # Thermal conductivity
    'J/kg/degK': 'J/(kg·K)',           # Specific heat
    'kJ/kg/K': 'kJ/(kg·K)',            # Specific heat (kilo)
    'Btu/lb/degR': 'Btu/(lb·°R)',      # Specific heat (English)
    'Pa*s': 'Pa·s',                    # Pascal-second

    # Basic units without changes
    'J': 'J',                          # Joule
    'm': 'm',                          # Meter
    'kg': 'kg',                        # Kilogram
    'Pa': 'Pa',                        # Pascal
    'W': 'W',                          # Watt
    'N': 'N',                          # Newton
    'bar': 'bar',                      # Bar
    'psi': 'psi',                      # Pounds per square inch
    'rpm': 'rpm',                      # Revolutions per minute
    }


@dataclass
class SI:
    """
    SI Units Class

    This class defines the standard units used in the International System
    of Units (SI) for various physical quantities.
    """
    units = {
        'AREA': 'm**2',                    # Square meter
        'DENSITY': 'kg/m**3',              # Kilogram per cubic meter
        'SPECIFICVOLUME': 'm**3/kg',       # Cubic meter per kilogram
        'ENERGY': 'J',                     # Joule
        'LENGTH': 'm',                     # Meter
        '1/LENGTH': 'm**-1',               # Reciprocal meter
        'MASS': 'kg',                      # Kilogram
        'MASSFLOW': 'kg/s',                # Kilogram per second
        'MASSFLOWDERIV': 'kg/s**2',        # Kilogram per second squared
        'MOLARMASS': 'kg/kmol',            # Kilogram per kilomole
        'MASSFLUX': 'kg/s/m**2',           # Kilogram per second per square
                                           # meter
        'POWER': 'W',                      # Watt
        'PRESSURE': 'Pa',                  # Pascal
        'SPECIFICENERGY': 'J/kg',          # Joule per kilogram
        'SPECIFICENTROPY': 'J/kg/degK',    # Joule per kilogram per Kelvin
        'SPECIFICHEAT': 'J/kg/degK',       # Joule per kilogram per Kelvin
        'SPECIFICPOWER': 'W/kg',           # Watt per kilogram
        'TEMPERATURE': 'degK',             # Kelvin
        'VELOCITY': 'm/s',                 # Meter per second
        'VISCOSITY': 'Pa*s',               # Pascal-second
        'KINEMATICVISCOSITY': 'm**2/s',    # Square meter per second
        'CONDUCTIVITY': 'W/m/K',           # Watt per meter-Kelvin
        'VOLUME': 'm**3',                  # Cubic meter
        'THERMALRESISTANCE': 'degK/W',     # Kelvin per Watt
        'CONVECTION': 'W/m**2/degK',       # Watt per square meter per Kelvin
        'VOLUMETRICFLOW': 'm**3/s',        # Cubic meter per second
        'SPECIFICSPEED': 'rpm*(m**3/s)**(0.5)/(J/kg)**0.75',
        'SPECIFICDIAMETER': 'm*(J/kg)**(1/4)/(m**3/s)**(0.5)',
        'ROTATIONSPEED': 'rpm',            # Revolutions per minute
        'SURFACETENSION': 'N/m',
        'INERTANCE': 'm**-3',
        'GASCONSTANT': 'J/kmol/K',
        'RPM': 'rpm',
        'ANGLE': 'deg',
        'FLOWCOEFFICIENT': 'm**3/s/Pa**(0.5)',
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
        'SPECIFICVOLUME': 'in**3/lb',      # Cubic inch per pound
        'ENERGY': 'Btu',                   # British thermal unit
        'LENGTH': 'in',                    # Inch
        '1/LENGTH': 'in**-1',               # Reciprocal inch
        'MASS': 'lb',                      # Pound
        'MOLARMASS': 'lb/mol',           # Pound per lbmol
        'MASSFLOW': 'lb/s',                # Pound per second
        'MASSFLOWDERIV': 'lb/s**2',        # Pound per second squared
        'MASSFLUX': 'lbs/s/in**2',         # Pounds per second per square inch
        'POWER': 'Btu/s',                  # British thermal unit per second
        'PRESSURE': 'psi',                 # Pounds per square inch
        'SPECIFICENERGY': 'Btu/lb',        # British thermal unit per pound
        'SPECIFICENTROPY': 'Btu/lb/degR',  # BTU per pound per Rankine
        'SPECIFICHEAT': 'Btu/lb/degR',     # BTU per pound per Rankine
        'SPECIFICPOWER': 'Btu/s/lb',       # BTU per second per pound
        'TEMPERATURE': 'degR',             # Rankine
        'VELOCITY': 'ft/s',                # Feet per second
        'VISCOSITY': 'lb/in/s',            # Pound per inch per second
        'KINEMATICVISCOSITY': 'ft**2/s',    # Square feet per second
        'CONDUCTIVITY': 'Btu/hr/ft/F',     # BTU per hour per foot per
                                           # Fahrenheit
        'VOLUME': 'in**3',                 # Cubic inch
        'THERMALRESISTANCE': 'degR/Btu/s',  # Rankine per BTU per second
        'CONVECTION': 'Btu/s/ft**2/degR',  # BTU per second per square foot
                                           # per Rankine
        'VOLUMETRICFLOW': 'gallons/min',       # Cubic foot per second
        'SPECIFICSPEED': 'rpm*(ft**3/s)**(0.5)/(ft*lbf/lb)**(3/4)',
        'SPECIFICDIAMETER': 'ft*(ft*lbf/lb)**(1/4)/(ft**3/s)**(0.5)',
        'ROTATIONSPEED': 'rpm',            # Revolutions per minute
        'SURFACETENSION': 'lbf/in',
        'INERTANCE': 'in**-3',
        'GASCONSTANT': 'ft*lbf/lb/degR',
        'RPM': 'rpm',
        'ANGLE': 'deg',
        'FLOWCOEFFICIENT': 'gal/min/psi**(0.5)',
    }

@dataclass
class ENGLISHFT:
    """
    English Units Class

    This class defines the standard units used in the English unit system
    for various physical quantities.
    """
    units = {
        'AREA': 'ft**2',                   # Square inch
        'DENSITY': 'lb/ft**3',             # Pound per cubic inch
        'SPECIFICVOLUME': 'ft**3/lb',      # Cubic inch per pound
        'ENERGY': 'Btu',                   # British thermal unit
        'LENGTH': 'ft',                    # Inch   
        '1/LENGTH': 'ft**-1',               # Reciprocal inch
        'MASS': 'lb',                      # Pound
        'MOLARMASS': 'lb/mol',           # Pound per lbmol
        'MASSFLOW': 'lb/s',                # Pound per second
        'MASSFLOWDERIV': 'lb/s**2',        # Pound per second squared
        'MASSFLUX': 'lbs/s/ft**2',         # Pounds per second per square inch
        'POWER': 'Btu/s',                  # British thermal unit per second
        'PRESSURE': 'psi',                 # Pounds per square inch
        'SPECIFICENERGY': 'Btu/lb',        # British thermal unit per pound
        'SPECIFICENTROPY': 'Btu/lb/degR',  # BTU per pound per Rankine
        'SPECIFICHEAT': 'Btu/lb/degR',     # BTU per pound per Rankine
        'SPECIFICPOWER': 'Btu/s/lb',       # BTU per second per pound
        'TEMPERATURE': 'degR',             # Rankine
        'VELOCITY': 'ft/s',                # Feet per second
        'VISCOSITY': 'lb/in/s',            # Pound per inch per second
        'KINEMATICVISCOSITY': 'ft**2/s',    # Square feet per second
        'CONDUCTIVITY': 'Btu/hr/ft/F',     # BTU per hour per foot per
                                           # Fahrenheit
        'VOLUME': 'ft**3',                 # Cubic inch
        'THERMALRESISTANCE': 'degR/Btu/s',  # Rankine per BTU per second
        'CONVECTION': 'Btu/s/ft**2/degR',  # BTU per second per square foot
                                           # per Rankine
        'VOLUMETRICFLOW': 'gallons/min',       # Cubic foot per second
        'SPECIFICSPEED': 'rpm*(ft**3/s)**(0.5)/(ft*lbf/lb)**(3/4)',
        'SPECIFICDIAMETER': 'ft*(ft*lbf/lb)**(1/4)/(ft**3/s)**(0.5)',
        'ROTATIONSPEED': 'rpm',            # Revolutions per minute
        'SURFACETENSION': 'lbf/in',
        'INERTANCE': 'ft**-3',
        'GASCONSTANT': 'ft*lbf/lb/degR',
        'RPM': 'rpm',
        'ANGLE': 'deg',
        'FLOWCOEFFICIENT': 'gal/min/psi**(0.5)',
    }



@dataclass
class MIXED:
    """
    Mixed Units Class

    This class defines a mix of SI and English units used for various
    physical quantities.
    """
    units = {
        'AREA': 'in**2',                   # Square inch
        'DENSITY': 'kg/m**3',              # Kilogram per cubic meter
        'SPECIFICVOLUME': 'm**3/kg',       # Cubic meter per kilogram
        'ENERGY': 'MJ',                    # Megajoule
        'LENGTH': 'in',                    # Inch
        '1/LENGTH': 'in**-1',               # Reciprocal inch
        'MASS': 'lb',                      # Pound
        'MOLARMASS': 'kg/kmol',
        'MASSFLOW': 'kg/s',                # Kilogram per second
        'MASSFLOWDERIV': 'kg/s**2',        # Kilogram per second squared
        'MASSFLUX': 'kg/s/m**2',           # Kilogram per second per square
                                           # meter
        'POWER': 'MW',                     # Kilowatt
        'PRESSURE': 'bar',                 # Bar
        'SPECIFICENERGY': 'kJ/kg',         # Kilojoule per kilogram
        'SPECIFICENTROPY': 'kJ/kg/degK',   # Kilojoule per kilogram per Kelvin
        'SPECIFICHEAT': 'kJ/kg/K',         # Kilojoule per kilogram per Kelvin
        'SPECIFICPOWER': 'MW/kg',          # Megawatt per kilogram
        'TEMPERATURE': 'degC',             # Celsius
        'VELOCITY': 'm/s',                 # Meter per second
        'VISCOSITY': 'Pa*s',               # Pascal-second
        'KINEMATICVISCOSITY': 'm**2/s',    # Square meter per second
        'CONDUCTIVITY': 'W/m/K',           # Watt per meter-Kelvin
        'VOLUME': 'in**3',                 # Cubic inch
        'THERMALRESISTANCE': 'degK/W',     # Kelvin per Watt
        'CONVECTION': 'W/m**2/degK',       # Watt per square meter per Kelvin
        'VOLUMETRICFLOW': 'gallons/min',        # Cubic meter per second
        'SPECIFICSPEED': 'rpm*(m**3/s)**(0.5)/(J/kg)**0.75',
        'SPECIFICDIAMETER': 'm*(J/kg)**(1/4)/(m**3/s)**(0.5)',
        'ROTATIONSPEED': 'rpm',            # Revolutions per minute
        'SURFACETENSION': 'N/m',
        'INERTANCE': 'm**-3',
        'GASCONSTANT': 'J/kmol/K',
        'RPM': 'rpm',
        'ANGLE': 'deg',
        'FLOWCOEFFICIENT': 'gal/min/psi**(0.5)',
    }
