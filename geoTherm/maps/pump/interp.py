import h5py
import numpy as np
from scipy.interpolate import RegularGridInterpolator, interp1d


import h5py
import numpy as np
from scipy.interpolate import RegularGridInterpolator

class PumpMapGrid:
    def __init__(self, filename):
        """Initialize pump map interpolator from HDF5 file.
        
        Args:
            filename (str): Path to HDF5 file containing pump data
        """
        self.filename = filename
        self.pressures, self.rpms, self.flow_rates = self._load_data()
        self.interpolator = self._create_interpolator()
        
        # Store bounds for validation
        self.rpm_bounds = (np.min(self.rpms), np.max(self.rpms))
        self.flow_bounds = (np.min(self.flow_rates), np.max(self.flow_rates))
        self.pressure_bounds = (np.min(self.pressures), np.max(self.pressures))

    def _load_data(self):
        """Load and structure pump map data from HDF5 file."""
        pressures = []
        rpm_set = set()
        flow_dict = {}

        with h5py.File(self.filename, "r") as f:
            for pressure in f.keys():
                p = float(pressure)
                pressures.append(p)
                rpm_values = np.array(f[pressure]["RPM"])
                Q_values = np.array(f[pressure]["Q"])
                
                rpm_set.update(rpm_values)
                flow_dict[p] = (rpm_values, Q_values)

        # Create structured arrays
        pressures = np.array(sorted(pressures))
        rpms = np.array(sorted(rpm_set))
        flow_rates = np.zeros((len(pressures), len(rpms)))

        # Interpolate flow rates onto regular grid
        for i, pressure in enumerate(pressures):
            existing_rpms, existing_Qs = flow_dict[pressure]
            flow_rates[i, :] = np.interp(rpms, existing_rpms, existing_Qs)

        return pressures, rpms, flow_rates

    def _create_interpolator(self):
        """Create interpolator for the regular grid."""
        return RegularGridInterpolator(
            (self.pressures, self.rpms), 
            self.flow_rates,
            bounds_error=False,
            fill_value=None
        )

    def check_extrapolation(self, rpm_value, mass_flow):
        """Check if point requires extrapolation.
        
        Args:
            rpm_value (float): RPM value to check
            mass_flow (float): Mass flow rate to check [m³/s]
            
        Returns:
            bool: True if point requires extrapolation
        """
        is_extrapolating = False
        
        if rpm_value < self.rpm_bounds[0] or rpm_value > self.rpm_bounds[1]:
            print(f"Warning: RPM {rpm_value} outside bounds {self.rpm_bounds}")
            is_extrapolating = True
            
        if mass_flow < self.flow_bounds[0] or mass_flow > self.flow_bounds[1]:
            print(f"Warning: Flow rate {mass_flow} outside bounds {self.flow_bounds}")
            is_extrapolating = True
            
        return is_extrapolating

    def get_outlet_pressure(self, rpm_value, mass_flow):
        """Get interpolated outlet pressure.
        
        Args:
            rpm_value (float): Pump RPM
            mass_flow (float): Mass flow rate [m³/s]
            
        Returns:
            float: Outlet pressure [bar]
        """
        # Check for extrapolation
        is_extrapolating = self.check_extrapolation(rpm_value, mass_flow)
        
        # Get flow rates at each pressure for given RPM
        flow_at_pressures = self.interpolator(
            (self.pressures, np.full_like(self.pressures, rpm_value))
        )
        
        # Clip mass flow to bounds
        mass_flow = np.clip(mass_flow, np.min(flow_at_pressures), 
                           np.max(flow_at_pressures))
        
        # Create pressure interpolator
        pressure_interpolator = RegularGridInterpolator(
            (flow_at_pressures,),
            self.pressures,
            bounds_error=False
        )
        
        return float(pressure_interpolator([[mass_flow]]))
