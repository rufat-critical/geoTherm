from ..HTC import BaseHTC


class Zukauskas(BaseHTC):

    def __init__(self, D, St, Sl, SD, N_L, N_T, L, arrangement='staggered'):
        """
        Initialize the Zukauskas heat transfer correlation for tube bundles.

        Parameters
        ----------
        D : float
            Tube diameter (m)
        St : float
            Transverse pitch - distance between tubes in the same row (m)
        Sl : float
            Longitudinal pitch - distance between tube rows (m)
        SD : float
            Diagonal pitch - distance between tubes in adjacent rows (m)
        N_L : int
            Number of tube rows in the flow direction
        N_T : int
            Number of tube rows in the transverse direction
        arrangement : str, optional
            Tube arrangement type, either 'staggered' or 'in-line', defaults to 'staggered'

        Geometric Parameters Diagram:
        ---------------------------

        In-line Arrangement:
        -------------------
        Flow direction →

        O     O     O     O     O
        |     |     |     |     |
        O     O     O     O     O
        |     |     |     |     |
        O     O     O     O     O

        Where:
        - O represents tubes
        - St = transverse pitch (horizontal distance between tubes)
        - Sl = longitudinal pitch (vertical distance between tube rows)

        Staggered Arrangement:
        ---------------------
        Flow direction →

        O     O     O     O     O
        |     |     |     |     |
          O     O     O     O     O
          |     |     |     |     |
        O     O     O     O     O

        Where:
        - O represents tubes
        - St = transverse pitch (horizontal distance between tubes)
        - Sl = longitudinal pitch (vertical distance between tube rows)
        - SD = diagonal pitch (distance between tubes in adjacent rows)
        """
        self.D = D
        self.St = St
        self.Sl = Sl
        self.SD = SD
        self.N_L = N_L
        self.N_T = N_T
        self.L = L
        self.arrangement = arrangement

    @property
    def flow_area(self):
        return (self.St - self.D) * self.N_T

    @property
    def flow_area(self):
        """
        Estimate the free-flow (minimum) area for air through a bare tube bundle.

        Returns
        -------
        A_flow : float
            Free-flow area (m²)
        """
        gap = (self.St - self.D)*(self.N_T - 1)  # horizontal open area between tubes
        
        return gap * self.L


    def V_max(self, V):
        from pdb import set_trace; set_trace()

    def evaluate(self, thermo, w, V=None, T_s=None):

        if V is None:
            from pdb import set_trace; set_trace()

        V_max = self.V_max(V)

        Re = thermo.density * V_max * self.D / thermo.mu
        Pr = thermo.prandtl

        if T_s is None:
            Pr_s = Pr
        else:
            from pdb import set_trace; set_trace()

        Nu = self.Nu(Re, Pr, Pr_s)

        h = Nu * thermo.conductivity / self.D

        return h


    def Nu(self, Re, Pr, Pr_s=None):
        """
        Calculate the Nusselt number using Zukauskas correlation for flow over a bank of tubes.

        Parameters:
        -----------
        Re : float
            Reynolds number based on maximum velocity
        Pr : float
            Prandtl number of the fluid
        Pr_s : float
            Prandtl number at the surface temperature

        Returns:
        --------
        Nu : float
            Nusselt number
        """

        if Pr_s is None:
            Pr_s = Pr

        # Constants for different regimes
        if self.arrangement == 'staggered':
            if Re < 100:
                C = 0.9
                m = 0.4
            elif Re < 1000:
                C = 0.52
                m = 0.5
            elif Re < 2e5:
                C = 0.27
                m = 0.63
            else:
                C = 0.033
                m = 0.8
        else:  # in-line
            if Re < 100:
                C = 0.8
                m = 0.4
            elif Re < 1000:
                C = 0.52
                m = 0.5
            elif Re < 2e5:
                C = 0.27
                m = 0.63
            else:
                C = 0.021
                m = 0.84

        # Base Nusselt number
        Nu_0 = C * Re**m * Pr**0.36

        # Correction factor for number of rows
        if self.N_L < 20:
            if self.arrangement == 'staggered':
                correction = 1 + 0.6 * (self.N_L - 1) / self.N_L
            else:  # in-line
                correction = 1 + 0.7 * (self.N_L - 1) / self.N_L
        else:
            correction = 1.0

        # Prandtl number correction
        Pr_correction = (Pr / Pr_s)**0.25

        # Final Nusselt number
        Nu = Nu_0 * correction * Pr_correction

        return Nu


class Briggs_Young(BaseHTC):

    def __init__(self, geometry):
        self.geometry = geometry

    def evaluate(self, thermo, w, V_max=None):

        if V_max is None:
            V_max = w/(self.geometry._area_flow_min * thermo._density)

        Re = V_max * self.geometry._Dh * thermo._density / thermo._viscosity
        Pr = thermo.prandtl

        Nu = self.Nu(Re, Pr)

        h = Nu * thermo.conductivity / self.geometry._Dh

        return h

    def Nu(self, Re, Pr):
        return (0.134*Re**0.681*Pr**(1/3.)*
                (self.geometry._dL_tube_exposed/self.geometry._h_fin)**0.2 *
                (self.geometry._dL_tube_exposed/self.geometry._th_fin)**0.1134)


    
def h_Briggs_Young(m, A, A_min, A_increase, A_fin, A_tube_showing,
                   tube_diameter, fin_diameter, fin_thickness, bare_length,
                   rho, Cp, mu, k, k_fin):
    r'''Calculates the air side heat transfer coefficient for an air cooler
    or other finned tube bundle with the formulas of Briggs and Young [1]_,


    '''
    fin_height = 0.5*(fin_diameter - tube_diameter)

    V_max = m/(A_min*rho)

    Re = Reynolds(V=V_max, D=tube_diameter, rho=rho, mu=mu)
    Pr = Prandtl(Cp=Cp, mu=mu, k=k)

    Nu = 0.134*Re**0.681*Pr**(1/3.)*(bare_length/fin_height)**0.2*(bare_length/fin_thickness)**0.1134

    h = k/tube_diameter*Nu
    efficiency = fin_efficiency_Kern_Kraus(Do=tube_diameter, D_fin=fin_diameter,
                                           t_fin=fin_thickness, k_fin=k_fin, h=h)
    h_total_area_basis = (efficiency*A_fin + A_tube_showing)/A*h
    h_bare_tube_basis = h_total_area_basis*A_increase

    return h_bare_tube_basis



def calculate_reynolds_from_mass_flow(m_dot, D, S_T, S_L, rho, mu, arrangement='staggered'):
    """
    Calculate Reynolds number from mass flow rate for a tube bank.
    
    Parameters:
    -----------
    m_dot : float
        Mass flow rate (kg/s)
    D : float
        Tube diameter (m)
    S_T : float
        Transverse pitch (m)
    S_L : float
        Longitudinal pitch (m)
    rho : float
        Air density (kg/m³)
    mu : float
        Dynamic viscosity (Pa·s)
    arrangement : str
        'staggered' or 'in-line'
    
    Returns:
    --------
    Re : float
        Reynolds number
    V_max : float
        Maximum velocity (m/s)
    """
    # Calculate minimum flow area
    if arrangement == 'staggered':
        # For staggered arrangement, minimum area could be either:
        # 1. Between tubes in the same row
        # 2. Between tubes in adjacent rows
        A1 = (S_T - D)  # Area between tubes in same row
        A2 = 2 * (np.sqrt((S_L/2)**2 + (S_T/2)**2) - D)  # Area between tubes in adjacent rows
        A_min = min(A1, A2)
    else:  # in-line
        A_min = (S_T - D)
    
    # Calculate maximum velocity
    V_max = m_dot / (rho * A_min)
    
    # Calculate Reynolds number
    Re = rho * V_max * D / mu
    
    return Re, V_max

# Example usage
if __name__ == "__main__":
    # Example parameters
    m_dot = 0.5  # kg/s
    D = 0.02     # m
    S_T = 0.04   # m
    S_L = 0.04   # m
    
    # Air properties at room temperature (300K)
    rho = 1.177  # kg/m³
    mu = 1.846e-5  # Pa·s
    
    # Calculate Re
    Re, V_max = calculate_reynolds_from_mass_flow(
        m_dot=m_dot,
        D=D,
        S_T=S_T,
        S_L=S_L,
        rho=rho,
        mu=mu,
        arrangement='staggered'
    )
    
    print(f"Maximum velocity: {V_max:.2f} m/s")
    print(f"Reynolds number: {Re:.2f}")
