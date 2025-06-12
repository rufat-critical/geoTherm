import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from geoTherm.units import toSI, fromSI
from geoTherm import units
import geoTherm as gt
import pandas as pd

# Hardcoded 2" ball valve Cv curve
cv_2in = {
    "Position": np.array([0, 10, 20, 30, 40, 50, 60, 70, 80, 90])*100/90,
    "Cv":        np.array([0, 15, 27, 45, 76, 118, 180, 301, 404, 470])*226/470,
}


class Cv_Map:
    def __init__(self, cv_curve=(cv_2in, 'gal/min/psi**(0.5)')):
        """
        Initialize CvMap with a hardcoded or provided Cv curve dictionary.

        Args:
            cv_curve (dict or tuple): Either:
                - A dictionary with keys 'Position' and 'Cv'
                - A tuple of (csv_path, units) where csv_path is a path to a CSV file with 'Position' and 'Cv' columns
                - A tuple of (dict, units) where dict has 'Position' and 'Cv' keys
            fallback (str): 'nearest' or 'extrapolate' behavior outside data range.
        """
        
        if isinstance(cv_curve, (list, tuple)):
            if isinstance(cv_curve[0], str):
                # Load CSV
                try:
                    df = pd.read_csv(cv_curve[0])
                    if 'Position' not in df.columns or 'Cv' not in df.columns:
                        raise ValueError("CSV file must contain 'Position' and 'Cv' columns")
                    self._cv = toSI((df['Cv'].values, cv_curve[1]), 'FLOWCOEFFICIENT')
                    self._positions = df['Position'].values
                except Exception as e:
                    raise ValueError(f"Error loading CSV file: {str(e)}")
            elif isinstance(cv_curve[0], dict):
                self._cv = toSI((cv_curve[0]['Cv'], cv_curve[1]), 'FLOWCOEFFICIENT')
                self._positions = cv_curve[0]['Position']
            else:
                raise ValueError("Invalid input type for cv_curve")
        else:
            if isinstance(cv_curve, dict):
                self._cv = toSI(cv_curve['Cv'], 'FLOWCOEFFICIENT')
                self._positions = cv_curve['Position']
            elif isinstance(cv_curve, str):
                # Load CSV
                try:
                    df = pd.read_csv(cv_curve)
                    if 'Position' not in df.columns or 'Cv' not in df.columns:
                        raise ValueError("CSV file must contain 'Position' and 'Cv' columns")
                    self._cv = toSI(df['Cv'].values, 'FLOWCOEFFICIENT')
                    self._positions = df['Position'].values
                except Exception as e:
                    raise ValueError(f"Error loading CSV file: {str(e)}")
            else:
                raise ValueError("Invalid input type for cv_curve")

        self.fallback = 'extrapolate'
        self.initialize()

    def initialize(self):
        """Create 2nd-order interpolation functions for Cv↔angle."""
        extrap = 'extrapolate' if self.fallback == 'extrapolate' else (self._cv[0], self._cv[-1])
        self.cv_from_position = interp1d(
            self._positions, self._cv,
            kind='quadratic',
            bounds_error=False,
            fill_value=extrap
        )

        extrap_position = 'extrapolate' if self.fallback == 'extrapolate' else (self._positions[0], self._positions[-1])
        self.position_from_cv = interp1d(
            self._cv, self._positions,
            kind='quadratic',
            bounds_error=False,
            fill_value=extrap_position
        )

    @property
    def cv_curve(self):
        return fromSI(self._cv, 'FLOWCOEFFICIENT')

    def _Cv(self, position):
        """Return Cv given position."""
        # Convert to input units

        return self.cv_from_position(position)

    def _position(self, Cv):
        """Return position given Cv."""
        # Convert to input units
        return self.position_from_cv(Cv)

    def Cv(self, position):
        """Return Cv given position."""
        return fromSI(self._Cv(position), 'FLOWCOEFFICIENT')

    def position(self, Cv):
        """Return position given Cv."""
        return self._position(toSI(Cv, 'FLOWCOEFFICIENT'))

    def __str__(self):
        return f"CvMap: Angle {self.info['angle_min']}–{self.info['angle_max']} deg, " \
               f"Cv {self.info['cv_min']}–{self.info['cv_max']}"

    def plot(self, op_points=None):
        """Plot Cv vs position with optional operating points."""
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.plot(self._positions, self.cv_curve, label='Cv Curve', lw=2)

        if op_points:
            if not isinstance(op_points, list):
                op_points = [op_points]
            for i, (pos, cv) in enumerate(op_points):
                ax.plot(pos, cv, 'ro')
                ax.annotate(f'({pos:.1f}%, {cv:.1f})', (pos, cv),
                            textcoords="offset points", xytext=(5, 5))

        Cv_units = units.units._output_units_for_display['FLOWCOEFFICIENT']
        ax.set_xlabel('Valve Position (%)')
        ax.set_ylabel(f'Cv ({Cv_units})')
        ax.set_title('Ball Valve Cv Curve')
        ax.grid(True)
        ax.legend()
        plt.tight_layout()
        plt.show()

