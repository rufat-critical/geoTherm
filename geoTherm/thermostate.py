import numpy as np
import CoolProp as cp
from pdb import set_trace
from .utils import parseState, parseComposition
from .units import inputParser, addQuantityProperty, fromSI, toSI, units
from rich.console import Console
from rich.table import Table
from .logger import logger

class coolprop_wrapper:
    """ Wrapper for CoolProp, makes it easy to interface with thermo state """

    def __init__(self, cDict=None, state=None, stateVars=None, cType='Y'):
        """
        Initialize coolprop object.
        
        Args:
            cDict (dict): Fluid composition dictionary.
            state (dict): Dictionary containing thermodynamic state variables.
            stateVars (str): State variables to use for initialization.
            cType (str): Composition type specified as mass or mole fraction ('X', 'Y').
        """
        
        self.cDict = cDict

        # Update composition based on input or use default
        if cDict is not None:
            self.updateComposition(cDict, cType=cType)
        else:
            self.updateComposition({'H2O': 1}, cType='Y')

        # Update state based on input or use default 
        if state is not None:
            self.updateState(state, stateVars)
        else:
            self.updateState(state={'T': 300, 'P': 101325}, stateVars='TP')

        # Generate dictionary with phase index
        self.phaseIndx = {
            cp.iphase_liquid: 'liquid',
            cp.iphase_supercritical: 'supercritical',
            cp.iphase_supercritical_gas: 'supercritical_gas',
            cp.iphase_supercritical_liquid: 'supercritical_liquid',
            cp.iphase_critical_point: 'critical',
            cp.iphase_gas: 'gas',
            cp.iphase_twophase: 'two-phase',
            cp.iphase_unknown: 'unknown'
        }

    def updateComposition(self, cDict, cType):
        """
        Update the composition of the CoolProp object.
        
        Args:
            cDict (dict): Fluid composition dictionary.
            cType (str): Composition type specified as mass or mole fraction ('X', 'Y').
        """
        # Update CoolProp object using Helmholtz Equation of State
        self.cpObj = cp.AbstractState("HEOS", '&'.join(cDict.keys()))

        # Set the mass or mole fractions via list comprehension
        if cType == 'Y':
            self.cpObj.set_mass_fractions([cDict[species] for species in cDict])
        elif cType == 'X':
            self.cpObj.set_mole_fractions([cDict[species] for species in cDict])
        else:
            raise ValueError(f'Invalid cType specified: {cType}, specify only X or Y')


    def updateState(self, state, stateVars):
        """
        Update the state of the CoolProp object.
        
        Args:
            state (dict): Dictionary containing thermodynamic state variables.
            stateVars (str): State variables to use for updating the state.
        """
        # Map state variable pairs to CoolProp input pairs
        input_pairs = {
            'TP': (cp.PT_INPUTS, ('P', 'T')),
            'TS': (cp.SmassT_INPUTS, ('S', 'T')),
            'HP': (cp.HmassP_INPUTS, ('H', 'P')),
            'SP': (cp.PSmass_INPUTS, ('P', 'S')),
            'DU': (cp.DmassUmass_INPUTS, ('D', 'U')),
            'PU': (cp.PUmass_INPUTS, ('P', 'U')),
            'DP': (cp.DmassP_INPUTS, ('D', 'P')),
            'HS': (cp.HmassSmass_INPUTS, ('H', 'S')),
            'DH': (cp.DmassHmass_INPUTS, ('D', 'H')),
            'TQ': (cp.QT_INPUTS, ('Q', 'T')),
            'PQ': (cp.PQ_INPUTS, ('P', 'Q')),
            'TD': (cp.DmassT_INPUTS, ('D', 'T'))
        }

        # Retrieve the appropriate CoolProp input pair and variable names
        if stateVars in input_pairs:
            inputs, variables = input_pairs[stateVars]
            # Update the CoolProp object with the correct input pair and values
            self.cpObj.update(inputs, state[variables[0]], state[variables[1]])
        else:
            raise ValueError(f"No valid input configuration for state variables: {stateVars}")


    def getProperty(self, property):
        """
        Get the thermodynamic property of a CoolProp object.
        
        Args:
            property (str): Property name.
        
        Returns:
            float: Property value.
        """
        if property == 'Pvap':
            # Get the vapor pressure by setting Q to 0
            vapor = cp.AbstractState("HEOS", self.cpObj.name())
            vapor.update(cp.QT_INPUTS, 0.0, self.cpObj.T())
            return float(vapor.p())
        elif property == 'phase':
            return self.phaseIndx[self.cpObj.phase()]

        # CoolProp name
        coolprop_name = self.pDict(property)

        # Return the CoolProp property value
        return getattr(self.cpObj, coolprop_name)()


    def pDict(self, property):
        """
        Return the CoolProp name of a specific property.
        
        Args:
            property (str): Property name.
        
        Returns:
            str: CoolProp property name.
        """
        prop_dict = {
            'T': 'T',
            'P': 'p',
            'H': 'hmass',
            'S': 'smass',
            'U': 'umass',
            'cp': 'cpmass',
            'cv': 'cvmass',
            'density': 'rhomass',
            'viscosity': 'viscosity',
            'Y': 'get_mass_fractions',
            'X': 'get_mole_fractions',
            'species_names': 'fluid_names',
            'sound': 'speed_sound',
            'Q': 'Q',
            'conductivity': 'conductivity'
        }

        return prop_dict[property]  


def addThermoAttributes(cls):
    """
    Decorator to add thermodynamic attributes to a class.
    
    Args:
        cls (class): Class to which attributes are added.
    
    Returns:
        class: Class with added attributes.
    """
    propertyList = ['T', 'P', 'Q', 'H', 'phase']
    for name in propertyList:
        def getter(self, name=name):  # Default argument to bind the current name
            return getattr(self.thermo, name)
        setattr(cls, name, property(getter))
    return cls


#@addQuantityProperty
def addThermoGetters(getterList):
    """
    Decorator to add thermodynamic properties to a class.
    
    Args:
        propertyList (list): List of property names to add.
    
    Returns:
        function: Decorator function.
    """
    def decorator(cls):
        for name in getterList:
            if '_' in name:
                prop = name[1:]
            else:
                prop = name

            def getter(self, prop=prop): 
                return self.pObj.getProperty(prop)
            setattr(cls, name, property(getter))                
        return cls
    return decorator


def addThermoSetters(setterList):
    """
    Decorator to add thermodynamic setters to a class.
    
    Args:
        setterList (list): List of setter names to add.
    
    Returns:
        function: Decorator function.
    """
    def decorator(cls):
        for name in setterList:
            properties = [prop for prop in name]
            
            # Getter method that returns a tuple for the corresponding properties in SI units
            def getterSI(self, name=name, properties=properties):
                return tuple(self.pObj.getProperty(p) for p in properties)
            
            # Getter method that returns a tuple for the corresponding properties in specified units
            def getterUnit(self, name=name, properties=properties):
                return tuple(
                    fromSI(self.pObj.getProperty(p), cls._units[p]) if p in cls._units 
                    else self.pObj.getProperty(p) for p in properties
                )

            if len(properties) == 1:
                # Setter for updating the composition
                def setter(self, value, name=name, properties=properties):
                    cDict = parseComposition(value)
                    self.pObj.updateComposition(cDict, cType=properties[0])    

                # Apply the getter and setter for single property (composition)
                setattr(cls, name, property(getterSI, setter))                   

            elif len(properties) == 2:
                
                # Setter for updating the state in SI units
                def setterSI(self, value, name=name, properties=properties):
                    self.pObj.updateState({properties[0]: value[0], properties[1]: value[1]}, stateVars=name)

                # Setter for updating the state with unit conversion
                def setterUnit(self, value, name=name, properties=properties):
                    # Create array with converted values in SI units
                    val = np.array(value)
                    for i, v in enumerate(val):
                        if properties[i] in cls._units:
                            val[i] = toSI(v, cls._units[properties[i]])
                    
                    self.pObj.updateState({properties[0]: val[0], properties[1]: val[1]},
                                          stateVars=name)

                # Apply the getters and setters for state variables (2 properties)
                setattr(cls, f'_{name}', property(getterSI, setterSI))
                setattr(cls, name, property(getterUnit, setterUnit))

            else:
                # Setter for updating both state and composition in SI units
                def setterSI(self, value, name=name, properties=properties):
                    cDict = parseComposition(value[2])
                    self.pObj.updateComposition(cDict, cType=properties[2])    
                    self.pObj.updateState({properties[0]: value[0], properties[1]: value[1]}, stateVars=name[:2])

                # Setter for updating both state and composition with unit conversion
                def setterUnit(self, value, name=name, properties=properties):
                    cDict = parseComposition(value[2])
                    self.pObj.updateComposition(cDict, cType=properties[2])

                    # Create array with converted values in SI units
                    val = np.array(value[:2],dtype='object')
                    
                    for i, v in enumerate(val):
                        if properties[i] in cls._units:
                            val[i] = toSI(v, cls._units[properties[i]])
                    
                    self.pObj.updateState({properties[0]: val[0], properties[1]: val[1]},
                                          stateVars=name[:2])
                                    
                # Apply the getters and setters for state and composition (3 properties)
                setattr(cls, f'_{name}', property(getterSI, setterSI))
                setattr(cls, name, property(getterUnit, setterUnit))          
            
        return cls
    return decorator

# Add thermo properties, the _ means they have a unit that's described in units
# class
thermoGetters = ['_T', '_P', 'Q', '_H', '_S', '_U', '_cp', '_cv', '_density',
                 '_viscosity', 'species_names', 'phase', '_sound', '_Pvap',
                 '_conductivity']

thermoSetters = ['TP', 'TS', 'HP', 'SP', 'DU', 'PU', 'DP', 'HS',
                 'DH', 'TQ', 'PQ', 'TD','TPY', 'X', 'Y']


@addThermoGetters(thermoGetters)
@addThermoSetters(thermoSetters)
@addQuantityProperty
class thermo:
    """ Class for tracking thermodynamic state """

    # This variable prevents other attributes from being defined for thermo
    __slots__ = ['pObj', 'T', '_T', 'P', '_P', 'Q', 'H', '_H', 'S', '_S',
                 'U', '_U', 'cp', '_cp', 'cv', '_cv', 'density',
                 '_density', 'viscosity', '_viscosity', 'Y', 'X', 
                 'species_names', 'TP', '_TP', 'TD', '_TD', 'TQ', '_TQ',
                 'TH', '_TH', 'TS', '_TS', 'HP', '_HP', 'HS', '_HS',
                 'DU', '_DU', 'TPY', '_TPY', 'thermoModel', 'sound', '_sound',
                 '_Pvap', '_conductivity']

    _units = {'T': 'TEMPERATURE',               # Temperature
              'P': 'PRESSURE',                  # Pressure
              'H': 'SPECIFICENERGY',            # Specific Enthalpy
              'S': 'SPECIFICENTROPY',           # Specific Entropy
              'U': 'SPECIFICENERGY',            # Specific Internal Energy
              'cp': 'SPECIFICHEAT',             # Specific Heat at Constant P
              'cv': 'SPECIFICHEAT',             # Specific Heat at Constant V
              'density': 'DENSITY',             # Density
              'viscosity': 'VISCOSITY',         # Viscosity
              'conductivity': 'CONDUCTIVITY',   # Conductivity
              'sound': 'VELOCITY',              # Velocity
              'Pvap': 'PRESSURE'}               # Vapor Pressure

    @inputParser
    def __init__(self, fluid='H2O', state=None, model='coolprop'):
        """
        Initialize the thermo object.
        
        Args:
            fluid (str): Fluid composition.
            state (dict): Initial state.
            model (str): Model to use ('coolprop').
        """

        self.thermoModel = model

        # Parse fluid composition
        cDict = parseComposition(fluid)
        # Check if state is specified, if not then use default state
        if state is None:
            state = {'T': 300, 'P': 101325}
            stateVars = 'TP'
        else:
            # Parse the state and composition
            stateVars = parseState(state)


        if self.thermoModel == 'coolprop':
            self.pObj = coolprop_wrapper(cDict=cDict, state=state, stateVars=stateVars)
        else:
            set_trace()


    def updateState(self, state):
        """
        Update the thermodynamic state of the object.

        Args:
            state (dict): Dictionary containing thermodynamic state variables.
        """
        # Parse the state variables
        stateVars = parseState(state)
        self.pObj.updateState(state=state, stateVars=stateVars)

    def _updateState(self, state):
        """
        Update the thermodynamic state of the object where inputs are in SI

        Args:
            state (dict): Dictionary containing thermodynamic state variables 
            in SI units
        """

        # Parse the state variables
        stateVars = parseState(state)
        self.pObj.updateState(state=state, stateVars=stateVars)

    def getProperty(self, property):
        """
        Get the thermodynamic property.
        
        Args:
            property (str): Property name.
        
        Returns:
            float: Property value.
        """   

        if property == 'prandtl':
            return self.prandtl

        return self.pObj.getProperty(property)     

    @property
    def state(self):
        """
        Output a dictionary defining the thermodynamic state.
        
        Returns:
            dict: Dictionary containing the thermodynamic state.
        """
        return {
            'fluid': self.Ydict,
            'state': {'H': (self._H, 'J/kg'),
                      'P': (self._P, 'Pa')},
            'model': self.thermoModel
        }

    @staticmethod
    def from_state(state):
        """
        Generate a thermo class from the state dictionary.
        
        Args:
            state (dict): State dictionary.
        
        Returns:
            thermo: Thermo object.
        """
        return thermo(**state)

    def __makeTable(self):
        """
        Create a formatted table representation of the thermostate using rich.
        
        Returns:
            str: The formatted string representation of the UnitSystem.
        """
        
        table = Table(title='Thermo Object')
        # Add columns for the quantity, input units, and output units
        table.add_column("Property", style="bold")
        table.add_column("Value")
        table.add_column("Units")

        table.add_row('TEMPERATURE', f'{self.T:0.5g}',
                      units.outputUnits['TEMPERATURE'])
        table.add_row('PRESSURE', f'{self.P:0.5g}',
                      units.outputUnits['PRESSURE'])
        table.add_row('VAPOR P', f'{self.Pvap:0.5g}',
                      units.outputUnits['PRESSURE'])
        table.add_row('DENSITY', f'{self.density:0.5g}',
                      units.outputUnits['DENSITY'])
        table.add_row('Cp', f'{self.cp:0.5g}',
                      units.outputUnits['SPECIFICHEAT'])
        table.add_row('Viscosity', f'{self.viscosity:0.5g}',
                      units.outputUnits['VISCOSITY'])        
        table.add_row('Fluid', str(self.Ydict), '')
        table.add_row('PHASE', self.phase, '')

        # Capture the table output using the rich console
        console = Console()
        with console.capture() as capture:
            console.print(table)
        return capture.get()
    
    def __repr__(self):
        return self.__makeTable()
    
    def __str__(self):
        return self.__makeTable()

    # THESE ARE ADDITIONAL PROPERTIES THAT ARE NOT DEFINED IN THERMOGETTERS
    @property
    def Ydict(self):
        """
        Return the dictionary containing species mass fractions.

        Returns:
            dict: Dictionary of species mass fractions.
        """
        species_names = self.pObj.getProperty('species_names')
        Y = self.pObj.getProperty('Y')
        return {name: Y[i] for i, name in enumerate(species_names)}

    @property 
    def gamma(self):
        return self._cp/self._cv

    @property
    def prandtl(self):
        
        # Return gas Prandtl for 2 phase
        if 0 < self.Q < 1:
            # Get OG State
            HP0 = self._HP
            # Set state to saturated vapor
            self._PQ = self._P,1
            # Get Prandtl #
            prandtl = self.prandtl
            # Revert state back to OG state
            self._HP = HP0
            return prandtl

        if self._cp < 0:
            # There is some issue in coolprop tables where cp for some
            # 2 phase mixture is evaluated as a negative value => this 
            # affects heat transfer calcs and produces erronous predictions
            # This is a fix if cp is negative by using linear approx of
            # liq and vapor cp values  
            logger.warn("Negative Cp detected in Prandl Calc, "
                        "Will approximate properties as: liq*(1-Q) + Q*vap")

            cp = self._two_phase_average(self.Q, 'cp')
            viscosity = self._two_phase_average(self.Q, 'viscosity')
            conductivity = self._two_phase_average(self.Q, 'conductivity')

            # Return prandtl with the interpolated calcs
            return cp*viscosity/conductivity

        return self._cp*self._viscosity/self._conductivity

    def _two_phase_average(self, Q, prop):
        # Average properties in 2 phase at constant Pressure

        # Get the initial HP values and Q
        HP0 = self._HP

        # Set the quality to 0 at the same pressure and evaluate properties
        self._PQ = self._P, 0
        prop0 = self.getProperty(prop)
        # Set the quality to 1 at the same pressure and evaluate properties
        self._PQ = self._P, 1
        prop1 = self.getProperty(prop)

        # Revert state back to initial state
        self._HP = HP0
        # Linearly interpolate properties
        return prop0*(1-Q) + prop1*Q
