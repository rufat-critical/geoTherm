import numpy as np
import CoolProp as cp
from .units import inputParser, addQuantityProperty, fromSI, toSI, units, parse_state_dict
from rich.console import Console
from rich.table import Table
from .logger import logger
import re
from .utils import R_ideal


# Utility functions for thermostate
def parseState(stateDict):
    """Check the stateDictionary for quantity code.

    Args:
        stateDict (dict): Dictionary containing 2 state variables.

    Returns:
        str: State Variable code.
    """

    if not stateDict or len(stateDict) != 2:
        logger.critical("The thermodynamic state must be defined with exactly "
                        "2 state variables. the current stateDict is : "
                        f"'{stateDict}'")

    # Creates a set using dict keys from stateDict, we'll use this to
    # compare with quanities defined below
    stateVars = set(stateDict)

    # These are all the property inputs that have been programmed so far
    # These are refered to in update_state method in thermostate.py
    quantities = {'TP': {'T', 'P'}, 
                  'TS': {'T', 'S'},
                  'HP': {'H', 'P'},
                  'SP': {'S', 'P'},
                  'DU': {'D', 'U'},
                  'PU': {'P', 'U'},
                  'DP': {'D', 'P'},
                  'HS': {'H', 'S'},
                  'DH': {'D', 'H'},
                  'TQ': {'T', 'Q'},
                  'PQ': {'P', 'Q'},
                  'TD': {'T', 'D'},
                  'DS': {'D', 'S'},
                  'HQ': {'H', 'Q'}}

    # Check if the set of stateVars matches any of the sets in quantities
    for code, vars in quantities.items():
        if stateVars == vars:
            return code

    # If we reached the end of the loop without returning then the quantity
    # code hasn't been coded yet
    logger.critical(f'Invalid Thermostate Variables specified: {stateDict}')


def parseComposition(composition):
    """Parse the composition into a dictionary containing species and quantities.

    Args:
        composition (str, list, dict): Fluid composition input.

    Returns:
        dict: Composition dictionary.
    """
    if composition is None:
        return None

    if isinstance(composition, str):
        # Use Regular Expression to parse composition string

        # Search for composition strings in the form of:
        # species:quantity, species: quantity
        cRe = re.compile(r'([A-Z0-9]*):([\.0-9]*)', re.IGNORECASE)

        comp = cRe.findall(composition)

        # Check if length of this is 0
        # If it is then maybe single name was specified for fluid
        if len(comp) == 0:
            composition = composition.split()
            if len(composition) == 1:
                comp = [(composition[0], 1.0)]
            else:
                from pdb import set_trace
                # Something is wrong with the composition string
                set_trace()

        # Make Dictionary containing composition
        composition = {species[0]: float(species[1])
                       for species in comp}

    elif isinstance(composition, (np.ndarray, list)):
        # If composition is specified as array of values then loop thru species
        # and generate composition
        composition = {name: composition[i] for i, name
                        in enumerate(self.species_names)}

    elif isinstance(composition, dict):
        # If input is a dictionary then we guchi
        pass

    # Normalize all the quantities
    tot = sum(composition.values())
    # Normalize by total sum
    composition = {species:q/tot for species, q in composition.items()}


    return composition


class IncompressibleState:
    """Incompressible fluid state model"""
    pass


coolprop_property_names = {
            'T': 'T',
            'P': 'p',
            'H': 'hmass',
            'S': 'smass',
            'U': 'umass',
            'cp': 'cpmass',
            'cv': 'cvmass',
            'D': 'rhomass',
            'density': 'rhomass',
            'viscosity': 'viscosity',
            'X': 'get_mole_fractions',
            'sound': 'speed_sound',
            'Q': 'Q',
            'conductivity': 'conductivity',
            "P_crit": "p_critical",
            "T_crit": "T_critical",
            "surface_tension": "surface_tension"
        }


coolprop_phase_index = {
    cp.iphase_liquid: 'liquid',
    cp.iphase_supercritical: 'supercritical',
    cp.iphase_supercritical_gas: 'supercritical_gas',
    cp.iphase_supercritical_liquid: 'supercritical_liquid',
    cp.iphase_critical_point: 'critical',
    cp.iphase_gas: 'gas',
    cp.iphase_twophase: 'two-phase',
    cp.iphase_unknown: 'unknown'
}



class coolprop_wrapper:
    """ Wrapper for CoolProp, makes it easy to interface with thermo state """

    def __init__(self, cDict=None, state=None, stateVars=None, cType='Y', model='coolprop',
                 EoS='HEOS'):
        """
        Initialize coolprop object.

        Args:
            cDict (dict): Fluid composition dictionary.
            state (dict): Dictionary containing thermodynamic state variables.
            stateVars (str): State variables to use for initialization.
            cType (str): Composition type specified as mass or mole fraction ('X', 'Y').
        """

        self.cDict = cDict

        # The Equation of State
        self.EoS = EoS

        # Update composition based on input or use default
        if cDict is not None:
            self.updateComposition(cDict, cType=cType)
        else:
            # Default State
            self.updateComposition({'Water': 1}, cType='Y')

        # Update state based on input or use default 
        if state is not None:
            self.update_state(state, stateVars)
        else:
            self.update_state(state={'T': 300, 'P': 101325}, stateVars='TP')

    def updateComposition(self, cDict, cType):
        """
        Update the composition of the CoolProp object.

        Args:
            cDict (dict): Fluid composition dictionary.
            cType (str): Composition type specified as mass or mole fraction ('X', 'Y').
        """
        # Update CoolProp object using EoS
        self.cpObj = cp.AbstractState(self.EoS, '&'.join(cDict.keys()))

        #    self.cpObj = cp.AbstractState("HEOS", '&'.join(cDict.keys()))
        #elif self.model == 'refprop':
        #    self.cpObj = cp.AbstractState("REFPROP", '&'.join(cDict.keys()))
        #else:
        #    from pdb import set_trace
        #    set_trace()

        # Set the mass or mole fractions via list comprehension
        if cType == 'Y':
            self.cpObj.set_mass_fractions([cDict[species] for species in cDict])
        elif cType == 'X':
            self.cpObj.set_mole_fractions([cDict[species] for species in cDict])
        else:
            raise ValueError(f'Invalid cType specified: {cType}, specify only X or Y')


    def update_state(self, state, stateVars):
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
            'TD': (cp.DmassT_INPUTS, ('D', 'T')),
            'DS': (cp.DmassSmass_INPUTS, ('D', 'S')),
            'HQ': (cp.HmassQ_INPUTS, ('H', 'Q'))
        }

        # Retrieve the appropriate CoolProp input pair and variable names
        if stateVars in input_pairs:
            inputs, variables = input_pairs[stateVars]
            # Update the CoolProp object with the correct input pair and values
            self.cpObj.update(inputs, state[variables[0]], state[variables[1]])
        else:
            raise ValueError(f"No valid input configuration for state variables: {stateVars}")

    def getProperty(self, property):
        return self.get_property(property)

    def get_property(self, property):
        """
        Get the thermodynamic property of a CoolProp object.

        Args:
            property (str): Property name.

        Returns:
            float: Property value.
        """
        #if property == 'Pvap':
            # Get the vapor pressure by setting Q to 0
        #    vapor = cp.AbstractState("HEOS", self.cpObj.name())
        #    vapor.update(cp.QT_INPUTS, 0.0, self.cpObj.T())
        #    return float(vapor.p())
        if property == 'Tvap':
            vapor = cp.AbstractState("HEOS", self.cpObj.name())
            vapor.update(cp.PQ_INPUTS, self.cpObj.p(), 0)
            return float(vapor.T())
        #elif property == 'phase':
        #    return self.phaseIndx[self.cpObj.phase()]

        #if property == 'viscosity' and self.model=='coolprop':
        #    if self.cpObj.fluid_names() == ['Acetone']:
                # Coolprop doesn't have an Acetone Viscosity model so using the
                # model from: 
                # http://ddbonline.ddbst.de/VogelCalculation/VogelCalculationCGI.exe?component=Acetone
         #       A = -3.37955
         #       B = 553.403
         #       C = -46.9657
         #       return np.exp(A+B/(C+self.cpObj.T()))*1e-3
        if property == 'molecular_weight':
            # Convert from kg/mol to kg/kmol
            return self.cpObj.molar_mass()*1e3
        if property == 'sound':
            # Determine if the fluid is in the two-phase region
            Q = self.cpObj.Q()

            if 0 < Q < 1:  # Check if in the two-phase region
                # Create a CoolProp state object for vapor/liquid properties
                vapor = cp.AbstractState("HEOS", self.cpObj.name())

                # Calculate speed of sound in the liquid phase (Q = 0)
                vapor.update(cp.QT_INPUTS, 0.0, self.cpObj.T())
                a_0 = vapor.speed_sound()

                # Calculate speed of sound in the gas phase (Q = 1)
                vapor.update(cp.QT_INPUTS, 1.0, self.cpObj.T())
                a_1 = vapor.speed_sound()
                # Perform linear interpolation based on the quality Q
                return a_1 * Q + a_0 * (1 - Q)

        if property == 'viscosity':
            return self.viscosity


        # CoolProp name
        if property in coolprop_property_names:
            coolprop_property = coolprop_property_names[property]
            return getattr(self.cpObj, coolprop_property)()
        else:
            return getattr(self, property)


    @property
    def Pvap(self):

        if self.EoS in ['HEOS', 'REFPROP']:
            # Return vapor pressure if Helmholtz or Refprop EoS
            vapor = cp.AbstractState(self.EoS, self.cpObj.name())
            vapor.update(cp.QT_INPUTS, 0.0, self.cpObj.T())
            return float(vapor.p())
        else:
            logger.info(f'Pvap is not defined for EoS: {self.EoS}')
            return 0

    @property
    def phase(self):
        if self.EoS == 'INCOMP':
            return "INCOMPRESSIBLE"
        else:
            return coolprop_phase_index[self.cpObj.phase()]

    @property
    def viscosity(self):
        # Viscosity Calc if using coolprop for acetone
        if self.EoS == 'HEOS' and self.cpObj.fluid_names() == ['Acetone']:
            # Coolprop doesn't have an Acetone Viscosity model so using the
            # model from: 
            # http://ddbonline.ddbst.de/VogelCalculation/VogelCalculationCGI.exe?component=Acetone
            A = -3.37955
            B = 553.403
            C = -46.9657
            return np.exp(A+B/(C+self.cpObj.T()))*1e-3
        else:
            return getattr(self.cpObj, 'viscosity')()

    @property
    def species_names(self):
        if self.EoS == 'INCOMP':
            return np.array([f'{self.cpObj.name()}'])
        else:
            return self.cpObj.fluid_names()

    @property
    def Y(self):
        if self.EoS == 'INCOMP':
            return np.array([1])
        else:
            return self.cpObj.get_mass_fractions()


def addThermoAttributes(cls):
    """Decorator to add thermodynamic attributes to a class."""
    propertyList = ['T', 'P', 'Q', 'H', 'phase', 'density']
    for name in propertyList:
        def getter(self, name=name):
            return getattr(self.thermo, name)
        setattr(cls, name, property(getter))
    return cls


def addThermoGetters(getterList):
    """Decorator to add thermodynamic properties to a class."""
    def decorator(cls):
        for name in getterList:
            prop = name[1:] if name.startswith('_') else name
            def getter(self, prop=prop):
                return self.pObj.getProperty(prop)
            setattr(cls, name, property(getter))
        return cls
    return decorator


def addThermoSetters(setterList):
    """Decorator to add thermodynamic setters to a class."""
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
                    self.pObj.update_state({properties[0]: value[0], properties[1]: value[1]}, stateVars=name)

                # Setter for updating the state with unit conversion
                def setterUnit(self, value, name=name, properties=properties):
                    # Create array with converted values in SI units
                    val = list(value)
                    for i, v in enumerate(val):
                        if properties[i] in cls._units:
                            val[i] = toSI(v, cls._units[properties[i]])
                    self.pObj.update_state({properties[0]: val[0], properties[1]: val[1]},
                                          stateVars=name)

                # Apply the getters and setters for state variables (2 properties)
                setattr(cls, f'_{name}', property(getterSI, setterSI))
                setattr(cls, name, property(getterUnit, setterUnit))

            else:
                # Setter for updating both state and composition in SI units
                def setterSI(self, value, name=name, properties=properties):
                    cDict = parseComposition(value[2])
                    self.pObj.updateComposition(cDict, cType=properties[2])    
                    self.pObj.update_state({properties[0]: value[0], properties[1]: value[1]}, stateVars=name[:2])

                # Setter for updating both state and composition with unit conversion
                def setterUnit(self, value, name=name, properties=properties):
                    cDict = parseComposition(value[2])
                    self.pObj.updateComposition(cDict, cType=properties[2])

                    # Create array with converted values in SI units
                    val = list(value[:2])
                    for i, v in enumerate(val):
                        if properties[i] in cls._units:
                            val[i] = toSI(v, cls._units[properties[i]])

                    self.pObj.update_state({properties[0]: val[0], properties[1]: val[1]},
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
                 '_Tvap', '_conductivity', '_D', '_T_crit', '_P_crit',
                 '_molecular_weight', '_surface_tension']

thermoSetters = ['TP', 'TS', 'HP', 'SP', 'DU', 'PU', 'DP', 'HS',
                 'DH', 'HQ', 'TQ', 'PQ', 'TD', 'TPY', 'HPY', 'X', 'Y']


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
                 'DU', '_DU', 'TPY', '_TPY', 'model', 'sound', '_sound',
                 '_Pvap', '_Tvap', '_conductivity', 'HQ', '_HQ', '_P_crit',
                 'P_crit', '_T_crit', 'T_crit', 'molecular_weight',
                 '_molecular_weight', '_surface_tension', 'HPY', '_HPY']

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
              'Pvap': 'PRESSURE',               # Vapor Pressure
              'Tvap': 'TEMPERATURE',            # Vapor Temperature
              'P_crit': 'PRESSURE',             # Critical Pressure
              'T_crit': 'TEMPERATURE',          # Critical Temperature
              'molecular_weight': 'MOLARMASS',
              'surface_tension': 'SURFACETENSION',
              'gas_constant': 'GASCONSTANT'}

    def __init__(self, fluid=None, state=None, model='coolprop', **kwargs):
        """
        Initialize the thermo object.
        
        Args:
            fluid (str): Fluid composition.
            state (dict): Initial state.
            model (str): Model to use ('coolprop').
        """

        self.model = model

        # Parse fluid composition
        cDict = parseComposition(fluid)

        if state is not None:  
            # Convert units
            state = parse_state_dict(state)
            # Parse the state and composition
            stateVars = parseState(state)
        else:
            stateVars = None

        if self.model == 'coolprop':
            self.pObj = coolprop_wrapper(cDict=cDict, state=state,
                                         stateVars=stateVars,
                                         model=model, **kwargs)
        else:
            logger.critical(f'Invalid thermo model used in input: {model}'
                            "The Valid Models are: 'coolprop'")


    def update_state(self, state, composition=None, cType='Y'):
        """
        Update the thermodynamic state of the object.

        Args:
            state (dict): Dictionary containing thermodynamic state variables.
        """
        # Parse the state variables
        stateVars = parseState(state)

        state = dict(state)
        # Convert input to SI
        state[stateVars[0]] = toSI(state[stateVars[0]],
                                   self._units[stateVars[0]])
        state[stateVars[1]] = toSI(state[stateVars[1]],
                                   self._units[stateVars[1]])

        if composition is not None:
            cDict = parseComposition(composition)
            self.pObj.updateComposition(cDict, cType=cType)

        self.pObj.update_state(state=state, stateVars=stateVars)        

    def _update_state(self, state):
        """
        Update the thermodynamic state of the object where inputs are in SI

        Args:
            state (dict): Dictionary containing thermodynamic state variables 
            in SI units
        """

        # Parse the state variables
        stateVars = parseState(state)
        self.pObj.update_state(state=state, stateVars=stateVars)

    def _update_state(self, state, composition=None, cType='Y'):
        """
        Update the thermodynamic state of the object where inputs are in SI

        Args:
            state (dict): Dictionary containing thermodynamic state variables 
            in SI units
        """
        # Parse the state variables
        stateVars = parseState(state)

        if composition is not None:
            cDict = parseComposition(composition)
            self.pObj.updateComposition(cDict, cType=cType)

        self.pObj.update_state(state=state, stateVars=stateVars)


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



        return self.pObj.get_property(property)     


    def get_property(self, property):
        """
        Get the thermodynamic property.

        Args:
            property (str): Property name.

        Returns:
            float: Property value.
        """

        if property == 'prandtl':
            return self.prandtl

        if property in self._units:
            return fromSI(self.pObj.get_property(property),
                        self._units[property])
        else:
            return self.pObj.get_property(property)

    def _get_property(self, property):
        """
        Get the thermodynamic property in SI units.

        Args:
            property (str): Property name.

        Returns:
            float: Property value.
        """

        if property == 'prandtl':
            return self.prandtl

        return self.pObj.get_property(property)     


    @property
    def state(self):
        """
        Output a dictionary defining the thermodynamic state.
        
        Returns:
            dict: Dictionary containing the thermodynamic state.
        """

        if self.model == 'coolprop':
            return {
                'fluid': self.Ydict,
                'state': {'D': (self._D, 'kg/m^3'),
                        'P': (self._P, 'Pa')},
                'EoS': self.pObj.EoS,
                'model': self.model
            }
        else:
            return {
            'fluid': self.Ydict,
            'state': {'D': (self._D, 'kg/m^3'),
                    'P': (self._P, 'Pa')},
            'model': self.model
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
                      units.output_units['TEMPERATURE'])
        table.add_row('PRESSURE', f'{self.P:0.5g}',
                      units.output_units['PRESSURE'])
        
        if (self.phase == 'supercritical_gas'
                or self.phase =='supercritical_liquid'):
            table.add_row('VAPOR P', 'supercritical')
        else:
            table.add_row('VAPOR P', f'{self.Pvap:0.5g}',
                          units.output_units['PRESSURE'])
                               
        table.add_row('DENSITY', f'{self.density:0.5g}',
                      units.output_units['DENSITY'])
        table.add_row('Cp', f'{self.cp:0.5g}',
                      units.output_units['SPECIFICHEAT'])
        table.add_row('Viscosity', f'{self.viscosity:0.5g}',
                      units.output_units['VISCOSITY'])        
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
    def _gas_constant(self):
        return R_ideal/self._molecular_weight

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
