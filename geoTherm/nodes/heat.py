from .node import Node
from .baseClasses import Heat, statefulHeatNode, statefulFlowNode, flowNode
from .flow import flow
from ..units import inputParser, addQuantityProperty
from ..logger import logger
from ..utils import dP_pipe
import numpy as np
from pdb import set_trace
from scipy.optimize import fsolve
import geoTherm as gt

# The heat stuff needs some serious reorg

@addQuantityProperty
class HTC:
    # This tracks and calculates convection coefficients

    _units = {'L': 'LENGTH', 'h':'CONVECTION'}

    @inputParser
    def __init__(self, thermo, L:'LENGTH'):
        self.thermo = thermo
        # Characteristic length
        self._L = L

    def Nu_dittus_boelter(self, Re, heating=True):
        # Dittus Boelter heat transfer correlation

        # Check Applicability
        if 0.6 <= self.thermo.prandtl <= 160:
            pass
        else:
            logger.warn("Dittus-Boelter relation is outside valid range"
                        "of 0.6<=Pr<=160, "
                        f"current {self.thermo.prandtl}")

        if Re >= 1e4:
            pass
        else:
            logger.warn("Dittus-Boelter relation is outside valid range"
                        f"Re>1e4, current {Re}")

        # Check what Exponent to use for Nusselt #
        if heating:
            n = 0.4
        else:
            n = 0.3

        return 0.023*Re**(0.8)*self.thermo.prandtl**n

    def _h(self, Nu):

        # Calculate heat transfer coefficient
        return Nu*self.thermo._conductivity/self._L

    def getR(self, correlation):
        # Calculate Re, Pr, properties
        # 
    #    if correlation == 'DB':
    #   
        pass

class LumpedMass(Node):
    # This specifies a wall node with constant k at the moment

    def __init__(self, name, k, L, t=None, D1=None, D2=None, coord='cylind'):
        self.name = name
        self._k = k
        self._L = L
        self._t = t
    
    def update_area(self, L):
        self._L = L

    @property
    def _R(self):
        return self._k*self._L/self._t

@addQuantityProperty
class Heatsistor2(Node):
    # This calculates resistance

    _displayVars = ['Q', '_UA']
    _units = {'Q': 'POWER'}

    def __init__(self, name, cool, hot, A_hot=None, A_cool=None, h_cool=None, h_hot=None, wall=None):
        
        self.name = name
        self.cool = cool
        self.hot = hot
        
        self._A_cool = A_cool
        self._A_hot = A_hot
        self.wall = wall

        self._h_cool = h_cool
        self._h_hot = h_hot


    def initialize(self, model):
        self.model = model
        
        # Volume Node
        # Check if convection specified if not error

        # Convection Object for cool Node

        # Convection Object for hotNode
        if self.wall is None:
            self._Rwall = 0
        else:
            from pdb import set_trace
            set_trace()


        self.hotNode = model.nodes[self.hot]
        self.coolNode = model.nodes[self.cool]
        self.evaluate()

        # Check if coolH, hotH specified
        # Check if coolA, coolH specified
        # If not check if the connected component has the info
        # if not then output error
        self._Q = 0.1#self._UA*(self.hotNode.thermo._T - self.coolNode.thermo._T) 

    @property
    def _UA(self):

        # 1/(h*A) + ln(r2/r1)/(2*pi*L*k) + 1/(h*A)
        
        self._Rhot = 1/(self._h_hot*self._A_hot)
        self._Rcool = 1/(self._h_cool*self._A_cool)

        return 1/(self._Rhot + self._Rcool + self._Rwall)


    def evaluate5(self):

        # get the overall UA
        self._Qtarget = self._UA*(self.hotNode.thermo._T - self.coolNode.thermo._T) 

    def updateState(self, Q):

        if Q <0 and False:
            self._Q = 0
            self.penalty = (0-Q[0]+10)*1e5
            return
        else:
            self.penalty = False

        self._Q = Q[0]*1e6

    @property
    def x(self):
        return np.array([self._Q*1e-6])

    @property
    def error(self):

        if self.penalty is not False:
            return np.array([self.penalty])

        self._Qtarget = self._UA*(self.hotNode.thermo._T - self.coolNode.thermo._T)
        

        #from pdb import set_trace
        #set_trace()

        # PENALTY APPROACH!! TO ADDRESS Q if hot T > cool T then Q needs to be positive!

        return np.array([-(self._Q - self._Qtarget)])
        from pdb import set_trace
        set_trace()




class HeatsistorStateful(Heatsistor2):
    pass



@addQuantityProperty
class HEXConnector5(statefulHeatNode):
    """ Connector that connects outlet T to Qin HEX """
    """ Qdot is calculated from outlet T Hex"""

    _displayVars = ['Q']
    _units = {'Q': 'POWER'}

    def __init__(self, name, hot, cool):
        self.name = name
        self.hot = hot
        self.cool = cool
        self._Q = 0

    def initialize(self, model):

        # Check the types of hot and cool
        # If Hot and Cool do stuff with signs
        hot = self.model.nodes[self.hot]
        cool = self.model.nodes[self.cool]

        # Check instances
        hot_hexT = isinstance(hot, HEX_T)
        hot_hexQ = isinstance(hot, HEX_Q)
        cool_hexT = isinstance(cool, HEX_T)
        cool_hexQ = isinstance(cool, HEX_Q)

        # Identify proper nodes
        if (hot_hexT and cool_hexQ) or (hot_hexQ and cool_hexT):
            if hot_hexT:
                self._hexT = hot
                self._hexQ = cool
            else:
                self._hexT = cool
                self._hexQ = hot

            # Q is negative of hexT
            self._Q = -self._hexT._Q

        else:
            msg = "Incorrect HEX types specified to HEXConnector " \
                f"'{self.name}', There needs to be 1 HEX_T and 1 HEX_Q Type"
            logger.error(msg)
            raise ValueError(msg)

        # Do rest of initialization 
        return super().initialize(model)

    def updateState(self, x):
        self._Q = x[0]

        self._hexQ._Q = x[0]

    @property
    def error(self):
        return self._Q + self._hexT._Q


@addQuantityProperty
class staticHEX(flow):
    pass


class QController(statefulHeatNode):
    
    @inputParser
    def __init__(self, name, node, T_setpoint):
        self.name = name
        self.node = node
        self.cool = node
        self._T_setpoint = T_setpoint

    @ property
    def error(self):

        return self.model.nodes[self.node].thermo._T - self._T_setpoint

@addQuantityProperty
class Qdot2(statefulHeatNode):

    _displayVars = ['Q']
    _units = {'Q': 'POWER'}

    @inputParser
    def __init__(self, name, hotNode, coolNode=None, Q:'POWER'=0):
        self.name = name
        self.node = hotNode
        self.cool = coolNode
        self._Q = Q

    def initialize(self, model):
        from pdb import set_trace
        set_trace()
        return super().initialize(model)


@addQuantityProperty
class Qdot(Heat):

    _displayVars = ['Q', 'cool']
    _units = {'Q': 'POWER'}

    @inputParser
    def __init__(self, name, cool, Q:'POWER'=0):
        self.name = name
        self.cool = cool
        self._Q = Q



class discretizedHeat:


    @inputParser
    def __init__(self, name, Inlet, A:'AREA', L:'LENGTH', Nsections=2):

        self.name = name
        self.Inlet = Inlet
        self._A = A
        self._L = L


# This connects 2 heat exchangers together
# Tin and Q
# Check for only 1 condition
# Can vary T to get pinch Temperature
# Source T => Get Q, Use for State
# Update state after evaluating Q to get upstream state


@addQuantityProperty
class HEX(Heat):

    _displayVars = ['w_hot', 'w_cool']
    _units = {'w_hot': 'MASSFLOW', 'w_cool': 'MASSFLOW', 'Q': 'POWER', 
              'D_tube': 'LENGTH', 'D_shell': 'LENGTH', 'wall_th': 'LENGTH'}

    @inputParser
    def __init__(self, w_hot:'MASSFLOW', w_cool:'MASSFLOW', hot_inlet, cool_inlet, Q:'POWER',
                 D_tube:'LENGTH',
                 D_shell:'LENGTH', n_tubes,
                 wall_th:'LENGTH',
                 k_wall:'CONDUCTIVITY',
                 n_points = 10):
        # Initialize HEX Object
        # Thermo are thermo objects


        self._w_hot = w_hot
        self._w_cool = w_cool

        self._cool_thermo = gt.thermo.from_state(cool_inlet.state)
        self._hot_thermo = gt.thermo.from_state(hot_inlet.state)

        self._Q = Q

        # Discretization
        self.n_points = n_points

        self._D_tube = D_tube
        self._D_shell = D_shell
        self.n_tubes = n_tubes

        self._wall_th = wall_th
        self._k_wall = k_wall

    def initialize(self):
        
        # Calculate flow Areas
        self._A_tube = np.pi*self._D_tube**2/4
        self._A_shell = np.pi*self._D_shell**2/4 - self.n_tubes*self._A_tube

        if self._A_shell < 0:
            from pdb import set_trace
            set_trace()
        

        # Total Enthalpy change for hot and cool streams
        self._dH_cool = self._Q/self._w_cool
        self._dH_hot = self._Q/self._w_hot

        # Initialize a sub model
        # Shell Diameter
        D_shell = np.sqrt(4*self._A_shell/np.pi)


        # Define the thermal resistance layer between tube flow and shell flow
        layers = [gt.CylindricalWallSurface('coolwall', 'cool', D=(self._D_tube, 'm'), L=1),
                  gt.CylindricalWall('wall', k=(self._k_wall, 'W/m/K'),
                                      D=(self._D_tube, 'm'),
                                      t=(self._wall_th, 'm'),
                                      L=1),
                  gt.CylindricalWallSurface('hotwall', 'hot',
                                            D=(self._D_tube+2*self._wall_th, 'm'),
                                            L_ht=D_shell, L=1)]
        
        layers = [layers[0]]
        self.model = gt.Model([gt.flowVol('hot', fluid=self._hot_thermo.Ydict,
                                          T=(self._hot_thermo._T, 'degK'),
                                          P=(self._hot_thermo._P,'Pa'),
                                          w=(self._w_hot, 'kg/s'), 
                                          A=(self._A_shell, 'm**2')),
                               gt.flowVol('cool', fluid=self._cool_thermo.Ydict,
                                          T=(self._cool_thermo._T, 'degK'),
                                          P=(self._cool_thermo._P,'Pa'),
                                          w=(self._w_cool/self.n_tubes, 'kg/s'), 
                                          A=(self._A_tube, 'm**2')),
                               gt.Heatsistor('wall', layers=layers)])
        

        self.model.initialize()

        self.model['wall']._UA_layers

    def _update_thermo(self, dQ, ignore_phase=False):
        
        # Calculate dH for cool and hot
        #dH_cool = dQ/self._w_cool
        #dH_hot = dQ/self._w_hot



        #HP0_cool = self.model['cool'].thermo._HP
        #phase0_cool = self.model['cool'].thermo.phase
        #HP0_hot = self.model['hot'].thermo._HP
        #phase0_hot = self.model['hot'].thermo.phase
        
        #self._coolThermo._HP = self.model['cool'].thermo._H + dH_cool, self._coolThermo._P
        #self._hotThermo._HP = self.model['hot'].thermo._H + dH_hot, self._hotThermo._P

        # Update Thermo
        #self.model['cool'].updateThermo({'H': self.model['cool'].thermo._H + dH_cool,
        #                                 'P': self._coolThermo._P})

        #self.model['hot'].updateThermo({'H': self.model['hot'].thermo._H + dH_hot,
        #                                'P': self._hotThermo._P})


        #T_cool_in = self.model['cool'].thermo._T
        #T_hot_in = self.model['hot'].thermo._T


        # Get dH for cool and hot
        dH_cool = dQ/self._w_cool
        dH_hot = dQ/self._w_hot

        hot_inlet = self.model['hot'].thermo
        cool_inlet = self.model['cool'].thermo

        hot_outlet = self._hot_thermo
        cool_outlet = self._cool_thermo

        hot_outlet._HP = hot_inlet._H + dH_hot, self._hot_thermo._P
        cool_outlet._HP = cool_inlet._H + dH_cool, self._cool_thermo._P
     
        # Get dT
        dT = np.abs(hot_outlet._T - cool_outlet._T)

        # Required heat transfer coefficient for this dQ
        UA = dQ/dT

        if UA < 0:
            from pdb import set_trace
            set_trace()

        

        # Modify to be inlet_phase = outlet_phase
        if not ignore_phase: 
            if cool_inlet.phase != cool_outlet.phase:
                if self.model['cool'].phase == 'liquid':
                    if cool_outlet.phase == 'two-phase':

                        cool_outlet._PQ = cool_outlet._P, 0
                        dQ1 = (cool_outlet._H- cool_inlet._H)*self._w_cool

                        # Update Inlet
                        L1 = self._update_thermo(dQ1, ignore_phase=True)
                        dQ2 = dQ - dQ1

                        L2 = self._update_thermo(dQ2, ignore_phase=True)

                        return L1 + L2
                    else:
                        from pdb import set_trace
                        set_trace()

                elif cool_inlet.phase == 'two-phase':
                    if cool_outlet.phase == 'gas':

                        cool_outlet._PQ = cool_outlet._P, 1
                        dQ1 = (cool_outlet._H- cool_inlet._H)*self._w_cool

                        # Update Inlet
                        L1 = self._update_thermo(dQ1, ignore_phase=True)
                        dQ2 = dQ - dQ1

                        L2 = self._update_thermo(dQ2, ignore_phase=True)

                        return L1 + L2
                    if cool_outlet.phase == 'supercritical-gas':
                        pass
                
                elif cool_inlet.phase == 'gas':
                    if cool_outlet.phase == 'supercritical-gas':
                        pass                    

                else:
                    from pdb import set_trace
                    set_trace()

            if hot_inlet.phase != hot_outlet.phase:
                from pdb import set_trace
                set_trace() 



        L0 = UA/(self.model['wall']._UA*self.n_tubes)
        # Update to thermo state
        self.model['cool'].updateThermo({'H': cool_outlet._H,
                                         'P': cool_outlet._P})

        self.model['hot'].updateThermo({'H': hot_outlet._H,
                                        'P': hot_outlet._P})       


        # Calculate L
        L = UA/(self.model['wall']._UA*self.n_tubes)

        if isinstance(L, complex):
            from pdb import set_trace
            set_trace()

        return L

    def _LMTD(self, dT1, dT2):
        # Calculate Logarithmic mean temperature distribution
        return (dT1 -dT2)/np.log(dT1/dT2)

    def _save_outlet(self):
        from pdb import set_trace
        set_trace()

   # def _update_dQ(self, dQ):

        # Step forward and update states


    def evaluate(self):


        dQ = self._Q/self.n_points


        #hot_out_T = 


        # SET UP INLET FOR COOL, HOT STREAM
        # STEP FORWARD

        # Update to counterflow
        # Pguess
        self._hot_thermo._HP = self._hot_thermo._H - self._Q/self._w_hot, self._hot_thermo._P

        Tin_hot = self._hot_thermo._T
        Tin_cool = self._cool_thermo._T

        # In the big model class update this using upstream conditions
        # For now doing htis
        self.model['hot'].thermo._HP = self._hot_thermo._HP
        self.model['cool'].thermo._HP = self._cool_thermo._HP

        Lvec = []

        # Initialize Dict to store data
        self.hexCalcs = {'L':[],
                         'UA':[],
                         'R':[],
                         'U':[],
                         'dQ':[],
                         'hot': {'T':[],
                                 'P':[]},
                         'cool':{'T':[],
                                 'P':[]}}

        self.model['cool'].thermo._HP = self.model['cool'].thermo._H - dQ/self._w_cool, self.model['cool'].thermo._P
        for i in range(0, self.n_points):

            # Update thermo
            
            L = self._update_thermo(abs(dQ))

            self.hexCalcs['L'].append(L)
            self.hexCalcs['UA'].append(self.model['wall']._UA)
            self.hexCalcs['R'].append(self.model['wall']._R)
            self.hexCalcs['dQ'].append(dQ)
            self.hexCalcs['U'].append(self.model['wall']._U_layers)
            self.hexCalcs['hot']['T'].append(self.model['hot'].thermo._T)
            self.hexCalcs['hot']['P'].append(self.model['hot'].thermo._P)
            #self.hexCalcs['hot']['T'].append(self._hot_thermo._T)
            #self.hexCalcs['hot']['P'].append(self._hot_thermo._P)
            self.hexCalcs['cool']['T'].append(self.model['cool'].thermo._T)
            self.hexCalcs['cool']['P'].append(self.model['cool'].thermo._P)


            # If pressure drop then update with L for pressure
            #self.model['cool'].updateThermo({'H': self._coolThermo._H,
            #                                 'P': self._coolThermo._P})

            #self.model['hot'].updateThermo({'H': self._coolThermo._H,
            #                                'P': self._coolThermo._P})
            


            #T_hot_out = self.model['hot'].thermo._T
            #T_cool_out = self.model['cool'].thermo._T

            #dT1 = T_hot_in - T_cool_out
            #dT2 = T_hot_out - T_cool_in
            #dT1 = T_cool0 - self.model['cool'].thermo._T
            #dT2 = self.model['hot'].thermo._T - T_hot0

            # Logarithmic mean temperature distribution
            #LMTD = (dT1 -dT2)/np.log(dT1/dT2)
            #UA = dQ/LMTD

            #def find_L(L):
            #    if L<0:
            #        return 0-L + 10
                
            #    self.model['wall']._L = L

            #    return self.model['wall']._UA*self.n_tubes - UA

            #sol = fsolve(find_L, 1)
            #L.append(properties[0])

        
        self.hexCalcs['L'] = np.cumsum(self.hexCalcs['L'])
        from matplotlib import pyplot as plt
        plt.plot(self.hexCalcs['L'], self.hexCalcs['cool']['T'])
        plt.plot(self.hexCalcs['L'], self.hexCalcs['hot']['T'])
        plt.figure()
        plt.plot(self.hexCalcs['L'], self.hexCalcs['dQ'])

        from pdb import set_trace
        set_trace()


@addQuantityProperty
class HEX2(Heat):

    _displayVars = ['w_hot', 'w_cool']
    _units = {'w_hot': 'MASSFLOW', 'w_cool': 'MASSFLOW', 'Q': 'POWER', 
              'D_tube': 'LENGTH', 'D_shell': 'LENGTH', 'wall_th': 'LENGTH'}

    @inputParser
    def __init__(self, w_hot:'MASSFLOW', w_cool:'MASSFLOW', hot_inlet, cool_outlet, L:'LENGTH',
                 D_tube:'LENGTH',
                 D_shell:'LENGTH', n_tubes,
                 wall_th:'LENGTH',
                 k_wall:'CONDUCTIVITY',
                 n_points = 10,
                 R_tube=None):
        # Initialize HEX Object
        # Thermo are thermo objects


        self._w_hot = w_hot
        self._w_cool = w_cool

        self._cool_thermo = gt.thermo.from_state(cool_outlet.state)
        self._hot_thermo = gt.thermo.from_state(hot_inlet.state)

        # Length
        self._L = L

        # Discretization
        self.n_points = n_points

        self._D_tube = D_tube
        self._D_shell = D_shell
        self.n_tubes = n_tubes

        self._wall_th = wall_th
        self._k_wall = k_wall
        self._R_tube = R_tube

    def initialize(self):

        # Calculate flow Areas
        self._A_tube = np.pi*self._D_tube**2/4
        # Outer Tube Area
        self._A_tube_outer = np.pi*(self._D_tube+2*self._wall_th)**2/4
        # Shell Flow Area
        self._A_shell = np.pi*self._D_shell**2/4 - self.n_tubes*self._A_tube_outer

        if self._A_shell < 0:
            from pdb import set_trace
            set_trace()


        # Initialize a sub model

        # Shell Perimeter
        Per_shell = np.pi*self._D_shell + self.n_tubes*np.pi*(self._D_tube+2*self._wall_th)
        # Shell Hydraulic Diameter
        self._Dh_shell = 4*self._A_shell/Per_shell

        # Hydraulic Shell Diameter
        #D_shell = np.sqrt(4*self._A_shell/np.pi)


        # Define the thermal resistance layer between tube flow and shell flow
        layers = [gt.CylindricalWallSurface('coolwall', 'cool', D=(self._D_tube, 'm'), L=1),
                  gt.CylindricalWall('wall', k=(self._k_wall, 'W/m/K'),
                                      D=(self._D_tube, 'm'),
                                      t=(self._wall_th, 'm'),
                                      L=1,
                                      R=self._R_tube),
                  gt.CylindricalWallSurface('hotwall', 'hot',
                                            D=(self._D_tube+2*self._wall_th, 'm'),
                                            L_ht=self._Dh_shell, L=1)]

        self.model = gt.Model([gt.flowVol('hot', fluid=self._hot_thermo.Ydict,
                                          T=(self._hot_thermo._T, 'degK'),
                                          P=(self._hot_thermo._P,'Pa'),
                                          w=(self._w_hot, 'kg/s'), 
                                          A=(self._A_shell, 'm**2'),
                                          Per=(Per_shell, 'm')),
                               gt.flowVol('cool', fluid=self._cool_thermo.Ydict,
                                          T=(self._cool_thermo._T, 'degK'),
                                          P=(self._cool_thermo._P,'Pa'),
                                          w=(self._w_cool/self.n_tubes, 'kg/s'),
                                          A=(self._A_tube, 'm**2')),
                               gt.Heatsistor('wall', layers=layers)])


        self.model.initialize()

        self.model['wall']._UA_layers

    def _update_thermo(self, dL):

        hot_inlet = self.model['hot'].thermo
        cool_inlet = self.model['cool'].thermo

        hot_outlet = self._hot_thermo
        cool_outlet = self._cool_thermo

        # Get dT
        dT = np.abs(hot_outlet._T - cool_outlet._T)
        dQ = self.model['wall']._UA*dL*self.n_tubes*dT

        # Calculate friction factor


        # Get dP
        U = self.model['hot']._flowU
        f = gt.utils.friction_factor(hot_inlet, U, self._Dh_shell)


        rho_dynamic = .5*self.model['hot'].thermo._density*U**2
        dP_hot = -f*dL/self._Dh_shell*rho_dynamic

        # Get DP2
        U = self.model['cool']._flowU
        f = gt.utils.friction_factor(cool_inlet, U, self._D_tube)
        rho_dynamic = .5*self.model['cool'].thermo._density*U**2
        dP_cool = -f*dL/self._D_tube*rho_dynamic


        hot_outlet._HP = hot_inlet._H - dQ/self._w_hot, self._hot_thermo._P + dP_hot
        cool_outlet._HP = cool_inlet._H + dQ/self._w_cool, self._cool_thermo._P - dP_cool


        self.model['hot'].thermo._HP = hot_outlet._HP
        self.model['cool'].thermo._HP = cool_outlet._HP


    def evaluate(self):
        dL = self._L/self.n_points



        # In the big model class update this using upstream conditions
        # For now doing htis
        self.model['hot'].thermo._HP = self._hot_thermo._HP
        self.model['cool'].thermo._HP = self._cool_thermo._HP

        Lvec = []

        # Initialize Dict to store data
        self.hexCalcs = {'L':[],
                         'UA':[],
                         'R':[],
                         'U':[],
                         'dQ':[],
                         'hot': {'T':[],
                                 'P':[]},
                         'cool':{'T':[],
                                 'P':[]}}

        #self.model['cool'].thermo._HP = self.model['cool'].thermo._H - dQ/self._w_cool, self.model['cool'].thermo._P
        for i in range(0, self.n_points):

            # Update thermo
            self._update_thermo(abs(dL))

            self.hexCalcs['UA'].append(self.model['wall']._UA)
            self.hexCalcs['R'].append(self.model['wall']._R)
            self.hexCalcs['U'].append(self.model['wall']._U_layers)
            self.hexCalcs['hot']['T'].append(self.model['hot'].thermo._T)
            self.hexCalcs['hot']['P'].append(self.model['hot'].thermo._P)
            #self.hexCalcs['hot']['T'].append(self._hot_thermo._T)
            #self.hexCalcs['hot']['P'].append(self._hot_thermo._P)
            self.hexCalcs['cool']['T'].append(self.model['cool'].thermo._T)
            self.hexCalcs['cool']['P'].append(self.model['cool'].thermo._P)



        self.hexCalcs['L'] =np.linspace(dL, self._L, self.n_points) 

        from matplotlib import pyplot as plt

        plt.plot(self.hexCalcs['L'], self.hexCalcs['cool']['T'])
        plt.plot(self.hexCalcs['L'], self.hexCalcs['hot']['T'])
        plt.figure()
        plt.plot(self.hexCalcs['L'], self.hexCalcs['cool']['P'])
        plt.plot(self.hexCalcs['L'], self.hexCalcs['hot']['P'])
        from pdb import set_trace
        set_trace()


@addQuantityProperty
class HEX2(Heat):

    _displayVars = ['w_tube', 'w_shell']
    _units = {'w_hot': 'MASSFLOW', 'w_cool': 'MASSFLOW', 'Q': 'POWER', 
              'D_tube': 'LENGTH', 'D_shell': 'LENGTH', 'wall_th': 'LENGTH'}

    @inputParser
    def __init__(self, name, w_tube:'MASSFLOW', w_shell:'MASSFLOW', shell_inlet, tube_outlet, L:'LENGTH',
                 D_tube:'LENGTH',
                 D_shell:'LENGTH',
                 n_tubes,
                 wall_th:'LENGTH',
                 k_wall:'CONDUCTIVITY',
                 n_points = 10,
                 R_tube=None):
        # Initialize HEX Object
        # Thermo are thermo objects

        # Component name
        self.name = name
        
        self._w_shell = w_shell
        self._w_tube = w_tube

        # Store shell and tube inlet thermostates
        self._shell_inlet = shell_inlet
        self._tube_outlet = tube_outlet

        # Creat thermo objects that we will use to evaluate thermo
        # Along HEX Length
        self._shell_thermo = gt.thermo.from_state(shell_inlet.state)
        self._tube_thermo = gt.thermo.from_state(tube_outlet.state)

        # Hex Length
        self._L = L

        # Discretization
        self.n_points = n_points

        # Shell and Tube Geometry
        self._D_tube = D_tube
        self._D_shell = D_shell
        self.n_tubes = n_tubes

        self._wall_th = wall_th
        self._k_wall = k_wall
        self._R_tube = R_tube

    def initialize(self):
        # Initialize Shell Object and calculate flow areas/...

        # Tube Flow Area
        self._A_tube = np.pi*self._D_tube**2/4
        # Tube Flow + Wall Area
        self._Ao_tube = np.pi*(self._D_tube+2*self._wall_th)**2/4
        # Shell Flow Area
        self._A_shell = np.pi*self._D_shell**2/4 - self.n_tubes*self._Ao_tube
        # Flow Area Ratio
        self.Arat = self._A_tube*self.n_tubes/self._A_shell

        if self._A_shell < 0:
            logger.critical(f"The Shell & Tube HEX '{self.name}' has a "
                            "negative shell flow area, increase shell Area "
                            "or decrease number of tubes/tube Area")
            from pdb import set_trace
            set_trace()

        # Shell Perimeter
        Per_shell = np.pi*self._D_shell + self.n_tubes*np.pi*(self._D_tube+2*self._wall_th)
        # Shell Hydraulic Diameter
        self._Dh_shell = 4*self._A_shell/Per_shell

        # Create Submodel for solving heat transfer

        # Define the thermal resistance layer between tube flow and shell flow
        layers = [gt.CylindricalWallSurface('inner_tube', 'tube', D=(self._D_tube, 'm'), L=1),
                  gt.CylindricalWall('wall', k=(self._k_wall, 'W/m/K'),
                                      D=(self._D_tube, 'm'),
                                      t=(self._wall_th, 'm'),
                                      L=1,
                                      R=self._R_tube),
                  gt.CylindricalWallSurface('outer_tube', 'shell',
                                            D=(self._D_tube+2*self._wall_th, 'm'),
                                            L_ht=self._Dh_shell, L=1)]

        self.model = gt.Model([gt.flowVol('shell', fluid=self._shell_thermo.Ydict,
                                          T=(self._shell_thermo._T, 'degK'),
                                          P=(self._shell_thermo._P,'Pa'),
                                          w=(self._w_shell, 'kg/s'), 
                                          A=(self._A_shell, 'm**2'),
                                          Per=(Per_shell, 'm')),
                               gt.flowVol('tube', fluid=self._tube_thermo.Ydict,
                                          T=(self._tube_thermo._T, 'degK'),
                                          P=(self._tube_thermo._P,'Pa'),
                                          w=(self._w_tube/self.n_tubes, 'kg/s'),
                                          A=(self._A_tube, 'm**2')),
                               gt.Heatsistor('wall', layers=layers)])

        # Initialize the model and then we are guchi to use it
        self.model.initialize()


    def _update_thermo(self, dL):
        
        shell_inlet = self.model.nodes['shell'].thermo
        tube_outlet = self.model.nodes['tube'].thermo

        shell_outlet = self._shell_thermo
        tube_inlet = self._tube_thermo

        # Calculate dT using Shell Inlet and and cool Outlet
        # Get dT across tube and shell
        dT = np.abs(shell_inlet._T - tube_outlet._T)
        # Calculate dQ
        dQ = self.model.nodes['wall']._UA*dL*self.n_tubes*dT

        # Calculate Pressure Drop
        dP_shell = dP_pipe(thermo=shell_inlet,
                           U=self.model.nodes['shell']._flowU,
                           Dh=self._Dh_shell,
                           L=dL)

        dP_tube = dP_pipe(thermo=tube_outlet,
                          U=self.model.nodes['tube']._flowU,
                          Dh=self._D_tube,
                          L=dL)

        # Update States
        # Step forward
        shell_outlet._HP = shell_inlet._H - dQ/self._w_shell, \
            shell_inlet._P + dP_shell
        # Work backwards
        tube_inlet._HP = tube_outlet._H + dQ/self._w_tube, \
            tube_outlet._P - dP_tube
        
        from pdb import set_trace
        set_trace()
        return shell_outlet, tube_inlet, dQ



    def evaluate(self):

        # Calculate Spacing
        dL = self._L/self.n_points

        # In the big model class update this using upstream conditions
        # For now doing htis
        self.model.nodes['shell'].thermo._HP = self._shell_inlet._HP
        self.model.nodes['tube'].thermo._HP = self._tube_outlet._HP

        from pdb import set_trace
        set_trace()
        # Initialize Dict to store data
        self.hexCalcs = {'L':[],
                         'UA':[],
                         'R':[],
                         'U':[],
                         'Q':[],
                         'hot': {'T':[],
                                 'P':[],
                                 'prandtl':[]},
                         'cool':{'T':[],
                                 'P':[]}}

        #self.model['cool'].thermo._HP = self.model['cool'].thermo._H - dQ/self._w_cool, self.model['cool'].thermo._P
        for i in range(0, self.n_points):

            # Update thermo
            shell_outlet, tube_inlet, dQ = self._update_thermo(abs(dL))

            # Update Submodel
            self.model.nodes['shell'].thermo._HP = shell_outlet._HP
            self.model.nodes['tube'].thermo._HP = tube_inlet._HP

            self.hexCalcs['UA'].append(self.model['wall']._UA)
            self.hexCalcs['Q'].append(dQ)
            self.hexCalcs['R'].append(self.model['wall']._R)
            self.hexCalcs['U'].append(self.model['wall']._U_layers)
            self.hexCalcs['hot']['T'].append(self.model['shell'].thermo._T)
            self.hexCalcs['hot']['P'].append(self.model['shell'].thermo._P)
            #self.hexCalcs['hot']['T'].append(self._hot_thermo._T)
            #self.hexCalcs['hot']['P'].append(self._hot_thermo._P)
            self.hexCalcs['cool']['T'].append(self.model['tube'].thermo._T)
            self.hexCalcs['cool']['P'].append(self.model['tube'].thermo._P)
            self.hexCalcs['hot']['prandtl'].append(self.model['shell'].thermo.Q)




        self.hexCalcs['L'] =np.linspace(dL, self._L, self.n_points) 

        from matplotlib import pyplot as plt

        plt.plot(self.hexCalcs['L'], self.hexCalcs['cool']['T'])
        plt.plot(self.hexCalcs['L'], self.hexCalcs['hot']['T'])
        plt.figure()
        plt.plot(self.hexCalcs['L'], self.hexCalcs['cool']['P'])
        plt.plot(self.hexCalcs['L'], self.hexCalcs['hot']['P'])
        plt.figure()
        plt.plot(self.hexCalcs['L'], self.hexCalcs['Q'])
        plt.figure()
        plt.plot(self.hexCalcs['L'], self.hexCalcs['UA'])
        Uinner = [U['inner_tube'] for U in self.hexCalcs['U']]
        Uoutter = [U['outer_tube'] for U in self.hexCalcs['U']]
        plt.figure()
        plt.plot(self.hexCalcs['L'], self.hexCalcs['hot']['prandtl'])
        from pdb import set_trace
        set_trace()



@addQuantityProperty
class HEX(Heat):

    _displayVars = ['w_tube', 'w_shell', 'Q', 'L']
    _units = {'w_tube': 'MASSFLOW', 'w_shell': 'MASSFLOW', 'Q': 'POWER', 
              'D_tube': 'LENGTH', 'D_shell': 'LENGTH', 'wall_th': 'LENGTH',
              'L': 'LENGTH'}

    @inputParser
    def __init__(self, name, shell, tube, L:'LENGTH',
                 D_tube:'LENGTH',
                 D_shell:'LENGTH',
                 n_tubes,
                 wall_th:'LENGTH',
                 k_wall:'CONDUCTIVITY',
                 n_points = 10,
                 R_tube=None,
                 Q=0):
        # Initialize HEX Object
        # Thermo are thermo objects

        # Component name
        self.name = name
        
        self.shell = shell
        self.tube = tube

        self._Q = Q
        self._dP_shell = -1e-5
        self._dP_tube = -1e-5
        # Creat thermo objects that we will use to evaluate thermo
        # Along HEX Length
        #self._shell_thermo = gt.thermo.from_state(shell_inlet.state)
        #self._tube_thermo = gt.thermo.from_state(tube_outlet.state)

        # Hex Length
        self._L = L

        # Discretization
        self.n_points = n_points

        # Shell and Tube Geometry
        self._D_tube = D_tube
        self._D_shell = D_shell
        self.n_tubes = n_tubes

        self._wall_th = wall_th
        self._k_wall = k_wall
        self._R_tube = R_tube

        self.penalty = False

    def initialize(self, model):
        # Initialize Shell Object and calculate flow areas/...

        # Attach model to class
        self.model = model


        # Shell Flow
        self.shell_node = model.nodes[self.shell]
        self.shell_node.HEX = self
        self.shell_node._Q = self._Q
        # Tube Flow
        self.tube_node = model.nodes[self.tube]
        self.tube_node.HEX = self
        self.tube_node._Q = -self._Q
        
        self.shell_node.initialize(model)
        self.tube_node.initialize(model)

        self.shell_inlet = self.shell_node.US_vol.thermo
        self.shell_outlet = self.shell_node.DS_vol.thermo
        self.tube_inlet = self.tube_node.US_vol.thermo
        self.tube_outlet = self.tube_node.DS_vol.thermo
       # from pdb import set_trace
       # set_trace()
        # Inlet/Outlet
        #self.shell_inlet = self.model.nodes[self.shell_node.US].thermo
        #self.shell_outlet = self.model.nodes[self.shell_node.DS].thermo
        #self.tube_inlet = self.model.nodes[self.tube_node.US].thermo
        #self.tube_outlet = self.model.nodes[self.tube_node.DS].thermo
        # Create temp shell/tube thermo
        self._shell_thermo = gt.thermo.from_state(self.shell_inlet.state)
        self._tube_thermo = gt.thermo.from_state(self.tube_outlet.state)

        # Flow Rates
        self._w_shell = self.shell_node._w
        self._w_tube = self.tube_node._w

        # Tube Flow Area
        self._A_tube = np.pi*self._D_tube**2/4
        # Tube Flow + Wall Area
        self._Ao_tube = np.pi*(self._D_tube+2*self._wall_th)**2/4
        # Shell Flow Area
        self._A_shell = np.pi*self._D_shell**2/4 - self.n_tubes*self._Ao_tube
        # Flow Area Ratio
        self.Arat = self._A_tube*self.n_tubes/self._A_shell

        if self._A_shell < 0:
            logger.critical(f"The Shell & Tube HEX '{self.name}' has a "
                            "negative shell flow area, increase shell Area "
                            "or decrease number of tubes/tube Area")
            from pdb import set_trace
            set_trace()

        # Shell Perimeter
        Per_shell = np.pi*self._D_shell + self.n_tubes*np.pi*(self._D_tube+2*self._wall_th)
        # Shell Hydraulic Diameter
        self._Dh_shell = 4*self._A_shell/Per_shell

        # Create Submodel for solving heat transfer

        # Define the thermal resistance layer between tube flow and shell flow
        layers = [gt.CylindricalWallSurface('inner_tube', 'tube', D=(self._D_tube, 'm'), L=1),
                  gt.CylindricalWall('wall', k=(self._k_wall, 'W/m/K'),
                                      D=(self._D_tube, 'm'),
                                      t=(self._wall_th, 'm'),
                                      L=1,
                                      R=self._R_tube),
                  gt.CylindricalWallSurface('outer_tube', 'shell',
                                            D=(self._D_tube+2*self._wall_th, 'm'),
                                            L_ht=self._Dh_shell, L=1)]

        self.hex = gt.Model([gt.flowVol('shell', fluid=self.shell_inlet.Ydict,
                                          T=(self.shell_inlet._T, 'degK'),
                                          P=(self.shell_inlet._P,'Pa'),
                                          w=(self._w_shell, 'kg/s'), 
                                          A=(self._A_shell, 'm**2'),
                                          Per=(Per_shell, 'm')),
                               gt.flowVol('tube', fluid=self.tube_outlet.Ydict,
                                          T=(self.tube_outlet._T, 'degK'),
                                          P=(self.tube_outlet._P,'Pa'),
                                          w=(self._w_tube/self.n_tubes, 'kg/s'),
                                          A=(self._A_tube, 'm**2')),
                               gt.Heatsistor('wall', layers=layers)])

        # Initialize the model and then we are guchi to use it
        self.hex.initialize()

    def _update_thermo(self, dL):

        shell_inlet = self.hex.nodes['shell'].thermo
        tube_outlet = self.hex.nodes['tube'].thermo

        shell_outlet = self._shell_thermo
        tube_inlet = self._tube_thermo

        # Calculate dT using Shell Inlet and and cool Outlet
        # Get dT across tube and shell
        dT = np.abs(shell_inlet._T - tube_outlet._T)
        # Calculate dQ
        dQ = self.hex.nodes['wall']._UA*dL*self.n_tubes*dT

        # Calculate Pressure Drop
        dP_shell = dP_pipe(thermo=shell_inlet,
                           U=self.hex.nodes['shell']._flowU,
                           Dh=self._Dh_shell,
                           L=dL)

        dP_tube = dP_pipe(thermo=tube_outlet,
                          U=self.hex.nodes['tube']._flowU,
                          Dh=self._D_tube,
                          L=dL)

        # Update States
        # Step forward
        shell_outlet._HP = shell_inlet._H - dQ/self._w_shell, \
            shell_inlet._P + dP_shell
        # Work backwards
        tube_inlet._HP = tube_outlet._H + dQ/self._w_tube, \
            tube_outlet._P - dP_tube

        return shell_outlet, tube_inlet, dQ

    def evaluate(self):
        self.solveOutlet()

    def set_tube_outlet(self, x):
        # This is used to specify tube inlet and step down pipe to get
        x[0]*=1e-5
        x[1]*=1e-5
        self.getInlet(x)

    def getInlet(self, x):

        # Calculate Spacing
        dL = self._L/self.n_points

        # In the big model class update this using upstream conditions
        # For now doing htis

        self.hex.nodes['shell'].thermo._HP = self.shell_inlet._HP

        # Set outlet State
        X0 = self.hex.nodes['tube'].thermo._HP
        try:
            self.hex.nodes['tube'].thermo._HP = x[0]*1e5, x[1]*1e5
        except:
            self.hex.nodes['tube'].thermo._HP = X0
            return (X0-x)*1e5

        # Initialize Dict to store data
        self.hexCalcs = {'L':[],
                         'UA':[],
                         'R':[],
                         'U':[],
                         'Q':[],
                         'shell': {'T':[],
                                 'P':[],
                                 'Q':[]},
                         'tube':{'T':[],
                                 'P':[],
                                 'Q':[]}}

        # Flow Rates
        self._w_shell = self.shell_node._w
        self._w_tube = self.tube_node._w


        for i in range(0, self.n_points):

            # Update thermo
            shell_outlet, tube_inlet, dQ = self._update_thermo(abs(dL))

            # Update Submodel
            self.hex.nodes['shell'].thermo._HP = shell_outlet._HP
            self.hex.nodes['tube'].thermo._HP = tube_inlet._HP
            
            self.hexCalcs['L'] = np.linspace(dL,self._L, self.n_points)
            self.hexCalcs['UA'].append(self.hex['wall']._UA)
            self.hexCalcs['Q'].append(dQ)
            self.hexCalcs['R'].append(self.hex['wall']._R)
            self.hexCalcs['U'].append(self.hex['wall']._U_layers)
            self.hexCalcs['shell']['T'].append(self.hex['shell'].thermo._T)
            self.hexCalcs['shell']['P'].append(self.hex['shell'].thermo._P)
            self.hexCalcs['shell']['Q'].append(self.hex['shell'].thermo.Q)
            self.hexCalcs['tube']['T'].append(self.hex['tube'].thermo._T)
            self.hexCalcs['tube']['P'].append(self.hex['tube'].thermo._P)
            self.hexCalcs['tube']['Q'].append(self.hex['tube'].thermo.Q)

        self._dQ = (shell_outlet._H-self.shell_inlet._H)*self._w_shell
        self._dP_shell_actual = shell_outlet._P - self.shell_inlet._P
        self._dP_tube_actual = self.tube_outlet._P - tube_inlet._P

        error = np.array([self.tube_inlet._T - tube_inlet._T,
                          (self.tube_inlet._P - tube_inlet._P)/self.tube_inlet._P])
        
        self.hexCalcs['Q'] = np.array(self.hexCalcs['Q'])
        self.hexCalcs['shell']['Q'] = np.array(self.hexCalcs['shell']['Q'])
        self.hexCalcs['tube']['Q'] = np.array(self.hexCalcs['tube']['Q'])
        self.hexCalcs['shell']['Q'][self.hexCalcs['shell']['Q'] == -1] = 0
        self.hexCalcs['tube']['Q'][self.hexCalcs['tube']['Q'] == -1] = 0
        self.hexCalcs['shell']['T'] = np.array(self.hexCalcs['shell']['T'])
        self.hexCalcs['tube']['T'] = np.array(self.hexCalcs['tube']['T'])
        self.hexCalcs['shell']['P'] = np.array(self.hexCalcs['shell']['P'])
        self.hexCalcs['tube']['P'] = np.array(self.hexCalcs['tube']['P'])

        return error

    def solveOutlet(self):
        
        if (self.shell_node._w > 1e4 or self.tube_node._w > 1e4):
            print('Triggered Large mass flow')
            return

        n_points = np.copy(self.n_points)
        #self.n_points = 200
        sol = fsolve(self.getInlet, [self.tube_outlet._H*1e-5, self.tube_outlet._P*1e-5], full_output=True)
        self.n_points = n_points

        self._Q = self._dQ


        if not hasattr(self, '_hexCalcs'):
            self._hexCalcs = np.copy(self.hexCalcs).tolist()

        if sol[2] != 1:
            print('No Convergence')
            from matplotlib import pyplot as plt
            plt.plot(self._hexCalcs['L'], self._hexCalcs['hot']['T'])
            plt.plot(self.hexCalcs['L'], self.hexCalcs['hot']['T'])
            from pdb import set_trace
            set_trace()

        self.getInlet(sol[0])


    def updateState(self, x):

        self._Q = x[0]*1e5
        self.shell_node._Q = x[0]*1e5
        self.tube_node._Q = -x[0]*1e5
        self.shell_node._dP = x[1]
        self.tube_node._dP = x[2]

    @property
    def x(self):
        return np.array([self._Q/1e5, self.shell_node._dP, self.tube_node._dP])

    @property
    def error(self):

        #P_shell_out = self.shell_inlet._P + self._dP_shell_actual
        #P_tube_out = self.tube_inlet._P + self._dP_tube_actual

        Pshell_error = self._dP_shell_actual - self.shell_node._dP
        Ptube_error = self._dP_tube_actual - self.tube_node._dP

        return np.array([(self._dQ-self._Q)*1e-5,
                         Pshell_error,
                         Ptube_error])


@addQuantityProperty
class hexFlow(flowNode):

    _displayVars = ['Q', 'dP', 'w']
    _units = {'Q': 'POWER', 'dP': 'PRESSURE', 'w': 'MASSFLOW'}

    def __init__(self, name, US, DS, HEX, w,dP=0):

        self.name = name
        self.US = US
        self.DS = DS
        self._w = w
        self._dP = dP
        self._dH = -1e-5
        self._Q = 0
        self.HEX = None

    def initialize(self, model):

        self.US_vol = model.nodes[self.US]
        self.DS_vol = model.nodes[self.DS]

        return super().initialize(model)

    def evaluate(self):

        self._dH = self._Q/self._w
        pass

    def getOutletState(self):
        # Get the Downstream thermo state
        US = self.model.nodes[self.US].thermo

        return {'H': US._H + self._dH,
                'P': US._P + self._dP}


@addQuantityProperty
class HEXQ(HEX):

    @inputParser
    def __init__(self, name, shell, tube, Q:'POWER',
                 D_tube:'LENGTH',
                 D_shell:'LENGTH', 
                 n_tubes,
                 wall_th:'LENGTH',
                 k_wall:'CONDUCTIVITY',
                 n_points = 10,
                 R_tube=None,
                 L=1):
        # Initialize HEX Object
        # Thermo are thermo objects

        self.name = name

        self.shell = shell
        self.tube = tube

        self._Q = Q

        self._L = L

        # Discretization
        self.n_points = n_points

        # Shell and Tube Geometry
        self._D_tube = D_tube
        self._D_shell = D_shell
        self.n_tubes = n_tubes

        self._wall_th = wall_th
        self._k_wall = k_wall
        self._R_tube = R_tube

        self.penalty = False

    def _update_thermo(self, dQ):

        # Get dH for cool and hot
        dH_shell = dQ/self._w_shell
        dH_tube = dQ/self._w_tube

        shell_inlet = self.hex.nodes['shell'].thermo
        tube_outlet = self.hex.nodes['tube'].thermo

        shell_outlet = self._shell_thermo
        tube_inlet = self._tube_thermo

        # Calculate dT using Shell Inlet and and cool Outlet
        # Get dT across tube and shell
        dT = np.abs(shell_inlet._T - tube_outlet._T)

        # calculate dL
        dL = dQ/(self.hex.nodes['wall']._UA*self.n_tubes*dT)

        # Calculate Pressure Drop
        dP_shell = dP_pipe(thermo=shell_inlet,
                           U=self.hex.nodes['shell']._flowU,
                           Dh=self._Dh_shell,
                           L=dL)

        dP_tube = dP_pipe(thermo=tube_outlet,
                          U=self.hex.nodes['tube']._flowU,
                          Dh=self._D_tube,
                          L=dL)

        # Update States
        # Step forward
        shell_outlet._HP = shell_inlet._H - dQ/self._w_shell, \
            shell_inlet._P + dP_shell
        # Work backwards
        tube_inlet._HP = tube_outlet._H + dQ/self._w_tube, \
            tube_outlet._P - dP_tube


        return shell_outlet, tube_inlet, dL


    def evaluate(self):
        self.solveOutlet()

    def getInlet(self, x):


        dQ = self._Q/self.n_points

        self.hex.nodes['shell'].thermo._HP = self.shell_inlet._HP

        # Set outlet State
        X0 = self.hex.nodes['tube'].thermo._HP
        try:
            self.hex.nodes['tube'].thermo._HP = x[0]*1e5, x[1]*1e5
        except:
            self.hex.nodes['tube'].thermo._HP = X0
            return (X0-x)*1e5

        # Initialize Dict to store data
        self.hexCalcs = {'L':[],
                         'UA':[],
                         'R':[],
                         'U':[],
                         'Q':[],
                         'shell': {'T':[],
                                 'P':[],
                                 'Q':[]},
                         'tube':{'T':[],
                                 'P':[],
                                 'Q':[]}}

        # Flow Rates
        self._w_shell = self.shell_node._w
        self._w_tube = self.tube_node._w


        for i in range(0, self.n_points):

            # Update thermo
            shell_outlet, tube_inlet, dL = self._update_thermo(dQ)

            # Update Submodel
            self.hex.nodes['shell'].thermo._HP = shell_outlet._HP
            self.hex.nodes['tube'].thermo._HP = tube_inlet._HP

            self.hexCalcs['L'].append(dL)
            self.hexCalcs['UA'].append(self.hex['wall']._UA)
            self.hexCalcs['Q'].append(dQ)
            self.hexCalcs['R'].append(self.hex['wall']._R)
            self.hexCalcs['U'].append(self.hex['wall']._U_layers)
            self.hexCalcs['shell']['T'].append(self.hex['shell'].thermo._T)
            self.hexCalcs['shell']['P'].append(self.hex['shell'].thermo._P)
            self.hexCalcs['shell']['Q'].append(self.hex['shell'].thermo.Q)
            self.hexCalcs['tube']['T'].append(self.hex['tube'].thermo._T)
            self.hexCalcs['tube']['P'].append(self.hex['tube'].thermo._P)
            self.hexCalcs['tube']['Q'].append(self.hex['tube'].thermo.Q)

        self._dQ = (shell_outlet._H-self.shell_inlet._H)*self._w_shell
        self._dP_shell_actual = shell_outlet._P - self.shell_inlet._P
        self._dP_tube_actual = self.tube_outlet._P - tube_inlet._P

        error = np.array([self.tube_inlet._T - tube_inlet._T,
                          (self.tube_inlet._P - tube_inlet._P)/self.tube_inlet._P])


        self.hexCalcs['L'] = np.cumsum(self.hexCalcs['L'])
        self.hexCalcs['Q'] = np.array(self.hexCalcs['Q'])
        self.hexCalcs['shell']['Q'] = np.array(self.hexCalcs['shell']['Q'])
        self.hexCalcs['tube']['Q'] = np.array(self.hexCalcs['tube']['Q'])
        self.hexCalcs['shell']['Q'][self.hexCalcs['shell']['Q'] == -1] = 0
        self.hexCalcs['tube']['Q'][self.hexCalcs['tube']['Q'] == -1] = 0
        self.hexCalcs['shell']['T'] = np.array(self.hexCalcs['shell']['T'])
        self.hexCalcs['tube']['T'] = np.array(self.hexCalcs['tube']['T'])
        self.hexCalcs['shell']['P'] = np.array(self.hexCalcs['shell']['P'])
        self.hexCalcs['tube']['P'] = np.array(self.hexCalcs['tube']['P'])

        return error