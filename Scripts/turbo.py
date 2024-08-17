import geoTherm as gt
from pdb import set_trace
import numpy as np


inlet = gt.Boundary(name='inlet', fluid='n-Pentane', P=(9, 'bar'), T=468)
rotor = gt.Rotor(name='rotor', N=3664.18988855444)
rotor2 = gt.Rotor(name='rotor2', N=3664.18988855444)
outlet = gt.Boundary(name='outlet', fluid='n-Pentane', P=(3.15, 'bar'), T=446.7118)
turby = gt.Turbine_NsDs(name='Turb', rotor = 'rotor', Ns=7.4, ds=2.4, PR=2.856, w=91.30935997, axial=False, US='inlet', DS='outlet')
turby2 = gt.Turbine_NsDs(name='Turb2', rotor = 'rotor2', Ns=7.4, ds=2.4, PR=2.856, w=91.30935997, axial=True, US='inlet', DS='outlet')
Model = gt.Model()
Model += inlet
Model += rotor
Model += rotor2
Model += outlet
Model += turby
Model += turby2
Model.initialize()


A = gt.thermo()
A.TPY = 300, 101325, 'acetone'
A.viscosity

#phi = Model['Turb']._vol_flow_out/(Model['Turb'].rotor_node.omega*Model['Turb']._D**3)
#gt.utils.turb_axial_eta(Model['Turb2'].phi, Model['Turb2'].psi)

#gt.utils.turb_axial_eta(0.186696553, -0.867318938, 1.156425251)

psi = Model['Turb2'].psi
phi = Model['Turb2'].phi
psi_opt = Model['Turb2'].psi_is

psi = -0.867318938
phi = 0.186696553
psi_opt = 1.156425251


eta_opt = 0.913 + 0.103*psi-0.0854*psi**2 + 0.0154*psi**3
phi_opt = 0.375 + 0.25*psi_opt
K = 0.375 - 0.125*psi
eta_opt - K*(phi-phi_opt)**2
0.375-0.125*psi

eta_opt - K*(phi-phi_opt)**2
#utip = np.pi*Model['Turb'].rotor_node.N/60

#Q = Model['Turb']._Q_out
#N = Model['Turb'].rotor_node.N
#D = Model['Turb']._D

#utip = Model['Turb']._u_tip

#phii = Q/(N*D**3)*(60/np.pi)

#phi3 = Q/(D**2*utip)



#Cm = Model['Turb']._Q_out/Model['Turb']._D**2

#phi = Cm/utip




#omega = Model['Turb'].rotor_node.omega
#psi = Model['Turb']._dH_is/(N*D)**2*(60/np.pi)**2
#psi2 = Model['Turb']._dH_is/(omega*D/2)**2

#phi5 = Q/((D)**3*omega/2)

#phi2 = Model['Turb']._vol_flow_out/(utip*Model['Turb']._D**3)
#eta = gt.utils.turb_axial_eta(Model['Turb2'].phi, Model['Turb2'].psi)
set_trace()