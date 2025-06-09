import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from geoTherm.units import toSI, fromSI
from geoTherm import units
import geoTherm as gt

# Hardcoded 2" ball valve Cv curve
cv_2in = {
    "angle_deg": np.array([0, 10, 20, 30, 40, 50, 60, 70, 80, 90])*100/90,
    "Cv":        np.array([0, 15, 27, 45, 76, 118, 180, 301, 404, 470])*226/470,
    "Cv_units": "gal/min/psi**(0.5)"
}


class CvMap:
    def __init__(self, cv_curve=cv_2in, fallback='extrapolate'):
        """
        Initialize CvMap with a hardcoded or provided Cv curve dictionary.

        Args:
            cv_curve (dict): Dictionary with keys 'angle_deg' and 'Cv'.
            fallback (str): 'nearest' or 'extrapolate' behavior outside data range.
        """
        self.curve = cv_curve
        self.fallback = fallback
        self.info = {}
        self.angles = self.curve['angle_deg']
        self._cvs = self.curve['Cv']#np.array([toSI((self.curve['Cv'][i], self.curve['Cv_units']), 'FLOWCOEFFICIENT') for i in range(len(self.curve['Cv']))])
        self.cvs_units = self.curve['Cv_units']
        self._process_curve()
        self.create_interpolators()

    def _process_curve(self):
        """Extract min/max info for metadata."""
        self.info['angle_min'] = np.min(self.angles)
        self.info['angle_max'] = np.max(self.angles)
        self.info['cv_min'] = np.min(self._cvs)
        self.info['cv_max'] = np.max(self._cvs)

    def create_interpolators(self):

        """Create 2nd-order interpolation functions for Cv↔angle."""
        extrap = 'extrapolate' if self.fallback == 'extrapolate' else (self.cvs[0], self.cvs[-1])
        self.cv_from_angle = interp1d(
            self.angles, self._cvs,
            kind='quadratic',
            bounds_error=False,
            fill_value=extrap
        )

        extrap_angle = 'extrapolate' if self.fallback == 'extrapolate' else (self.angles[0], self.angles[-1])
        self.angle_from_cv = interp1d(
            self._cvs, self.angles,
            kind='quadratic',
            bounds_error=False,
            fill_value=extrap_angle
        )

    def get_cv(self, angle_deg):
        """Return Cv given angle in degrees."""
        # Convert to input units

        CV_SI = self.cv_from_angle(angle_deg)
        return CV_SI
        if gt.units.output== 'SI':
            return float(CV_SI)
        else:
            return float(fromSI(CV_SI, 'FLOWCOEFFICIENT'))

    def _get_cv(self, angle_deg):
        """Return Cv given angle in degrees."""
        # Convert to input units

        Cv = float(self.cv_from_angle(angle_deg))

        Cv_SI = units.unit_handler.convert(Cv,input_unit=units.unitSystems['ENGLISH'].units['FLOWCOEFFICIENT'],
                                   output_unit=units.unitSystems['SI'].units['FLOWCOEFFICIENT'])
        return Cv_SI

    def get_angle(self, Cv):
        """Return valve position (angle in deg) given Cv."""
        return float(self.angle_from_cv(Cv))
    
    def _get_angle(self, Cv):
        """Return valve position (angle in deg) given Cv."""
        # Convert Cv to english units

        Cv_english = units.unit_handler.convert(Cv,input_unit=units.unitSystems['SI'].units['FLOWCOEFFICIENT'],
                                   output_unit=units.unitSystems['ENGLISH'].units['FLOWCOEFFICIENT'])

        return float(self.angle_from_cv(Cv_english))

    def __str__(self):
        return f"CvMap: Angle {self.info['angle_min']}–{self.info['angle_max']} deg, " \
               f"Cv {self.info['cv_min']}–{self.info['cv_max']}"

    def plot(self, op_points=None):
        """Plot Cv vs angle with optional operating points."""
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.plot(self.angles, self.cvs, label='Cv Curve', lw=2)

        if op_points:
            if not isinstance(op_points, list):
                op_points = [op_points]
            for i, (angle, Cv) in enumerate(op_points):
                ax.plot(angle, Cv, 'ro')
                ax.annotate(f'({angle:.1f}°, {Cv:.1f})', (angle, Cv),
                            textcoords="offset points", xytext=(5, 5))

        ax.set_xlabel('Valve Position (deg)')
        ax.set_ylabel('Cv')
        ax.set_title('Ball Valve Cv Curve')
        ax.grid(True)
        ax.legend()
        plt.tight_layout()
        plt.show()


class CvMapSI:
    def __init__(self, cv_curve=cv_2in, fallback='extrapolate'):
        """
        Initialize CvMap with a hardcoded or provided Cv curve dictionary.
        Converts Cv values to SI units internally.

        Args:
            cv_curve (dict): Dictionary with keys 'angle_deg' and 'Cv'.
            fallback (str): 'nearest' or 'extrapolate' behavior outside data range.
        """
        self.curve = cv_curve
        self.fallback = fallback
        self.info = {}
        self.angles = self.curve['angle_deg']
        # Convert Cv values to SI units
        self._cvs = np.array([toSI((cv, self.curve['Cv_units']), 'FLOWCOEFFICIENT')
                             for cv in self.curve['Cv']])
        self.cvs_units = 'FLOWCOEFFICIENT'  # SI units
        self._process_curve()
        self.create_interpolators()

    def _process_curve(self):
        """Extract min/max info for metadata."""
        self.info['angle_min'] = np.min(self.angles)
        self.info['angle_max'] = np.max(self.angles)
        self.info['cv_min'] = np.min(self._cvs)
        self.info['cv_max'] = np.max(self._cvs)

    def create_interpolators(self):
        """Create 2nd-order interpolation functions for Cv↔angle."""
        extrap = 'extrapolate' if self.fallback == 'extrapolate' else (self._cvs[0], self._cvs[-1])
        self.cv_from_angle = interp1d(
            self.angles, self._cvs,
            kind='quadratic',
            bounds_error=False,
            fill_value=extrap
        )

        extrap_angle = 'extrapolate' if self.fallback == 'extrapolate' else (self.angles[0], self.angles[-1])
        self.angle_from_cv = interp1d(
            self._cvs, self.angles,
            kind='quadratic',
            bounds_error=False,
            fill_value=extrap_angle
        )

    def get_cv(self, angle_deg):
        """Return Cv in SI units given angle in degrees."""
        return float(self.cv_from_angle(angle_deg))

    def get_angle(self, Cv):
        """Return valve position (angle in deg) given Cv in SI units."""
        return float(self.angle_from_cv(Cv))

    def __str__(self):
        return f"CvMap2: Angle {self.info['angle_min']}–{self.info['angle_max']} deg, " \
               f"Cv {self.info['cv_min']:.2f}–{self.info['cv_max']:.2f} (SI units)"

    def plot(self, op_points=None):
        """Plot Cv vs angle with optional operating points."""
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.plot(self.angles, self._cvs, label='Cv Curve (SI)', lw=2)

        if op_points:
            if not isinstance(op_points, list):
                op_points = [op_points]
            for i, (angle, Cv) in enumerate(op_points):
                ax.plot(angle, Cv, 'ro')
                ax.annotate(f'({angle:.1f}°, {Cv:.1f})', (angle, Cv),
                            textcoords="offset points", xytext=(5, 5))

        ax.set_xlabel('Valve Position (deg)')
        ax.set_ylabel('Cv (SI units)')
        ax.set_title('Ball Valve Cv Curve')
        ax.grid(True)
        ax.legend()
        plt.tight_layout()
        plt.show()