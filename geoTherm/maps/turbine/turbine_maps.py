import numpy as np
from scipy.interpolate import LinearNDInterpolator
import matplotlib.pyplot as plt
import os

# Get the directory where this script is located
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# Create the full path to the CSV file
csv_path = os.path.join(SCRIPT_DIR, 'claudio_turbine.csv')

class EtaInterpolator:
    """Interpolator for turbine efficiency map."""
    
    def __init__(self, csv_file):
        """Initialize interpolator directly from CSV file."""
        # Load and parse CSV data
        with open(csv_file, 'r') as f:
            lines = f.readlines()
        
        # Initialize lists to store data
        rpms = []
        prs = []
        etas = []
        
        current_rpm = None
        for line in lines:
            if not line.strip():
                continue
                
            values = line.strip().split(',')
            
            if values[0] == 'RPM':
                current_rpm = float(values[1])
            elif values[0] == 'PR' and current_rpm is not None:
                pr_values = [float(x) for x in values[1:] if x.strip()]
                for pr in pr_values:
                    rpms.append(current_rpm)
                    prs.append(pr)
            elif values[0] == 'eta' and current_rpm is not None:
                eta_values = [float(x) for x in values[1:] if x.strip()]
                etas.extend(eta_values)
        
        # Convert to numpy arrays
        self.raw_rpms = np.array(rpms)
        self.raw_prs = np.array(prs)
        self.raw_etas = np.array(etas)
        
        # Create interpolator
        points = np.column_stack((self.raw_rpms, self.raw_prs))
        self.interpolator = LinearNDInterpolator(points, self.raw_etas)
        
        # Store bounds
        self.rpm_bounds = [self.raw_rpms.min(), self.raw_rpms.max()]
        self.pr_bounds = [self.raw_prs.min(), self.raw_prs.max()]
    
    def get_efficiency(self, rpm, pr):
        """Get interpolated efficiency value."""
        return float(self.interpolator(rpm, pr))
    
    def check_bounds(self, rpm, pr):
        """Check if point is within interpolation bounds."""
        return (self.rpm_bounds[0] <= rpm <= self.rpm_bounds[1] and
                self.pr_bounds[0] <= pr <= self.pr_bounds[1])
    
    def plot_efficiency_map(self):
        """Create visualization of the efficiency map."""
        # Create mesh grid for interpolation
        rpm_range = np.linspace(self.rpm_bounds[0], self.rpm_bounds[1], 100)
        pr_range = np.linspace(self.pr_bounds[0], self.pr_bounds[1], 100)
        RPM, PR = np.meshgrid(rpm_range, pr_range)
        
        # Calculate efficiency at each point
        ETA = np.zeros_like(RPM)
        for i in range(RPM.shape[0]):
            for j in range(RPM.shape[1]):
                ETA[i,j] = self.get_efficiency(RPM[i,j], PR[i,j])
        
        # Create figure with two subplots
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Plot 1: Contour plot
        levels = np.linspace(40, 95, 12)
        contour = ax1.contour(RPM, PR, ETA, levels=levels, colors='black')
        contourf = ax1.contourf(RPM, PR, ETA, levels=levels, cmap='viridis')
        ax1.scatter(self.raw_rpms, self.raw_prs, c='red', s=50, 
                   label='Data points')
        
        ax1.set_xlabel('RPM')
        ax1.set_ylabel('Pressure Ratio')
        ax1.set_title('Efficiency Map (Contour)')
        plt.colorbar(contourf, ax=ax1, label='Efficiency (%)')
        ax1.legend()
        
        # Plot 2: PR sweeps at different RPMs
        actual_rpms = np.unique(self.raw_rpms)
        
        # Create interpolated RPM values
        all_rpms = []
        for i in range(len(actual_rpms)-1):
            # Add 3 interpolated points between each actual RPM
            interp_rpms = np.linspace(actual_rpms[i], actual_rpms[i+1], 5)[:-1]
            all_rpms.extend(interp_rpms)
        all_rpms.append(actual_rpms[-1])
        all_rpms = np.array(all_rpms)
        
        pr_sweep = np.linspace(self.pr_bounds[0], self.pr_bounds[1], 100)
        colors = plt.cm.viridis(np.linspace(0, 1, len(all_rpms)))
        
        # Plot all RPM lines
        for rpm, color in zip(all_rpms, colors):
            linestyle = '-' if rpm in actual_rpms else '--'
            alpha = 1.0 if rpm in actual_rpms else 0.5
            label = f'{rpm:.0f} RPM' if rpm in actual_rpms else None
            
            # Plot interpolated line
            etas = [self.get_efficiency(rpm, pr) for pr in pr_sweep]
            ax2.plot(pr_sweep, etas, linestyle, color=color, 
                    alpha=alpha, label=label)
            
            # Plot actual data points for this RPM
            if rpm in actual_rpms:
                mask = self.raw_rpms == rpm
                ax2.scatter(self.raw_prs[mask], self.raw_etas[mask], 
                           color=color, s=50, marker='o')
        
        ax2.set_xlabel('Pressure Ratio')
        ax2.set_ylabel('Efficiency (%)')
        ax2.set_title('Efficiency vs PR at Different RPMs')
        ax2.grid(True)
        ax2.legend()
        
        plt.tight_layout()
        plt.show()


# Now use the full path
Claudio_Turbine = EtaInterpolator(csv_path)


if __name__ == "__main__":  
    # Create interpolator directly from CSV
    interp = EtaInterpolator(csv_path)
    
    # Plot efficiency map
    interp.plot_efficiency_map()
    
    # Test interpolation
    test_rpm = 35
    test_pr = 3.0
    
    eta = interp.get_efficiency(test_rpm, test_pr)
    print(f"\nTest point:")
    print(f"RPM: {test_rpm}")
    print(f"PR: {test_pr}")
    print(f"Interpolated efficiency: {eta:.2f}%")
    
    print(f"\nValid ranges:")
    print(f"RPM: {interp.rpm_bounds[0]} to {interp.rpm_bounds[1]}")
    print(f"PR: {interp.pr_bounds[0]:.2f} to {interp.pr_bounds[1]:.2f}") 