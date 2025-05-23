import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch
from ..thermostate import thermo
from ..units import unit_handler, units
from ..logger import logger
import numpy as np


# Colors for plot lines. ChatGPT gave me these color
# codes to make them unique and easy to differentiate
# on plots
colors = [
    '#1f77b4',  # Blue
    '#ff7f0e',  # Orange
    '#2ca02c',  # Green
    '#d62728',  # Red
    '#9467bd',  # Purple
    '#8c564b',  # Brown
    '#e377c2',  # Pink
    '#7f7f7f',  # Gray
    '#bcbd22',  # Olive
    '#17becf',  # Cyan
    '#aec7e8',  # Light Blue
    '#ffbb78',  # Light Orange
    '#98df8a',  # Light Green
    '#ff9896',  # Light Red
    '#c5b0d5',  # Light Purple
    '#c49c94',  # Light Brown
    '#f7b6d2',  # Light Pink
    '#c7c7c7',  # Light Gray
    '#dbdb8d',  # Light Olive
    '#9edae5',  # Light Cyan
]


class thermoPlotter:
    """Class to plot thermodynamic property diagrams."""

    def __init__(self, fluid):
        """
        Initialize the plotter with a specific fluid.

        Parameters:
        fluid (str): The fluid to be used for thermodynamic calculations.
        """
        self._vapor = {}
        self.state_points = {}
        self.iso_lines = {}
        self.process_lines = {}
        self.update_fluid(fluid)

    def add_fluid(self, fluid):
        """Add or update the fluid for thermodynamic calculations."""
        self.update_fluid(fluid)

    def update_fluid(self, fluid):
        """
        Update the fluid and associated thermodynamic properties.

        Parameters:
        fluid (str): The fluid to be used for thermodynamic calculations.
        """

        if isinstance(fluid, thermo):
            self.__thermo = thermo.from_state(fluid.state)
            self.fluid = fluid.species_names[0]
        else:
            self.fluid = fluid
            # Make thermo a "private attribute"
            self.__thermo = thermo(fluid=self.fluid)

        # Get the critical point values
        self._T_critical = self.__thermo.pObj.cpObj.T_critical()
        self._P_critical = self.__thermo.pObj.cpObj.p_critical()
        # Get the Min/Max T data
        self._Tmin = self.__thermo.pObj.cpObj.Tmin()
        self._Tmax = self.__thermo.pObj.cpObj.Tmax()

        # Sweep through the vapor dome to get properties
        self.sweep_vapor_dome()


    def add_state_point(self, name,
                        state=None,
                        fluid=None,
                        color=None,
                        marker=None):
        """
        Add a thermodynamic state point to the plot.

        Parameters:
        name (str): The name of the state point.
        state (dict, thermo object): The state of the fluid, provided
            as a dictionary or geoTherm thermo object
        fluid (dict): The composition of the fluid (optional).
        color (str): The color to use for this state point (optional).
        marker (str): The marker style for this state point (optional).
        """

        if state is None:
            logger.warn(f'No state info specified for {self.name}')
            return

        # Assign default color and marker if not provided
        if color is None:
            color = colors[len(self.state_points)]
        if marker is None:
            marker = 'o'

        # Store the HP properties of the state-point
        # This allows us to capture states inside and outside
        # vapor dome
        if isinstance(state, dict):
            self.thermo.update_state(state, composition=fluid)
            state = {'H': self.thermo._H, 'P': self.thermo._P}

        elif isinstance(state, thermo):
            state = {'H': state._H, 'P': state._P}

        if name in self.state_points:
            logger.warn(f"Statepoint '{name}' has already been "
                        "defined in ThermoPlotter. Overwritting "
                        "previous definition")

        # Store the state point
        self.state_points[name] = {'state': state,
                                   'fluid': self.thermo.Ydict,
                                   'color': color,
                                   'marker': marker}

    def add_isoline(self, isoline, state_val, fluid=None,
                    color=None, line_style=None):
        """
        Add an isoline to the plot.

        Parameters:
        isoline (str): The property for which the isoline is generated
            (e.g., 'T' for temperature).
        state_val (float/str): The value of the property or the name of
            a state point to use.
        fluid (dict): The composition of the fluid (optional).
        color (str): The color to use for this isoline (optional).
        line_style (str): The line style for this isoline (optional).
        """

        if isinstance(state_val, (int, float, list, tuple)):
            val = state_val
            state_point = None
        elif isinstance(state_val, str):
            val = None
            state_point = state_val
        else:
            from pdb import set_trace
            set_trace()

        if fluid is None:
            fluid = self.thermo.Ydict

        # Store isoline data
        iso_dat = {'val': val,
                   'state_point': state_point,
                   'fluid': fluid,
                   'color': color,
                   'line_style': line_style}

        if isoline in self.iso_lines:
            self.iso_lines[isoline].append(iso_dat)
        else:
            self.iso_lines[isoline] = [iso_dat]

    def add_process_line(self, name, US, DS, process_type, color=None, line_style=None):
        """
        Add a process line connecting two state points.

        Parameters:
        name (str): The name of the process line.
        US (str): The name of the upstream state point.
        DS (str): The name of the downstream state point.
        process_type (str): The type of process (e.g., isobaric, isothermal).
        color (str, optional): The color to use for the process line (default: None).
        line_style (str, optional): The line style for the process line (default: None).
        """

        # Ensure that the upstream (US) and downstream (DS) state points are defined
        if US not in self.state_points:
            logger.critical(f"'{US}' has not been defined as a state point. "
                            "Define it first before defining the process line.")
        if DS not in self.state_points:
            logger.critical(f"'{DS}' has not been defined as a state point. "
                            "Define it first before defining the process line.")

        if color is None:
            color = self.state_points[US]['color']
        if line_style is None:
            line_style = ':'

        # Add the process line to the process_lines dictionary
        self.process_lines[name] = {
            'US': US,
            'DS': DS,
            'type': process_type,
            'color': color,
            'line_style': line_style
        }

    def remove_iso_lines(self):
        logger.warn('Removing Existing Iso lines')
        self.iso_lines = {}

    def remove_process_lines(self):
        logger.warn('Removing Existing Process Lines')
        self.process_lines = {}

    def generate_process_line(self, process_var, US, DS, x_prop, y_prop,
                              n_points=500):
        """
        Generate data for a process line between two state points. The process
        var will be linearly interpolated between upstream and downstream states.

        Parameters:
        process_var (str): The variable to interpolate (e.g., temperature).
        upstream_state (str): The name of the upstream state point.
        downstream_state (str): The name of the downstream state point.
        x_prop (str): The x-axis property for the plot.
        y_prop (str): The y-axis property for the plot.

        Returns:
        tuple: Arrays for x and y values representing the process line.
        """

        # Get the Upstream State point x, y and process variable
        self.thermo._update_state(self.state_points[US]['state'])
        _x0 = self.evaluate_thermo(x_prop, SI=True)
        x0 = self.evaluate_thermo(x_prop)
        y0 = self.evaluate_thermo(y_prop)
        _p0 = self.evaluate_thermo(process_var, SI=True)
        # Get the Downstream State point x, y and process variable
        self.thermo._update_state(self.state_points[DS]['state'])
        _x1 = self.evaluate_thermo(x_prop, SI=True)
        x1 = self.evaluate_thermo(x_prop)
        y1 = self.evaluate_thermo(y_prop)
        _p1 = self.evaluate_thermo(process_var, SI=True)

        # If process var is the same as x or y properties
        # Then we can simply linearly interpolate
        if (process_var == x_prop or
                process_var == y_prop):
            return [x0, x1], [y0, y1]

        # Generate line to sweep
        x_sweep = np.linspace(_x0, _x1, n_points)
        # Interpolate process variable p(x_sweep)
        p_sweep = (_p1 - _p0) / (_x1 - _x0+1e-10) * (x_sweep - _x0) + _p0
        # Initialize Array to store values
        x_vals = np.full_like(x_sweep, np.nan)
        y_vals = np.full_like(x_sweep, np.nan)

        for i, x in enumerate(x_sweep):
            try:
                self._update_thermo({x_prop: x,
                                    process_var: p_sweep[i]},
                                    SI=True)
                x_vals[i] = self.evaluate_thermo(x_prop)
                y_vals[i] = self.evaluate_thermo(y_prop)
            except Exception:
                continue

        x_vals = x_vals[~np.isnan(x_vals)]
        y_vals = y_vals[~np.isnan(y_vals)]

        return x_vals, y_vals

    def generate_iso_line(self, iso_prop, iso_val, _iso_val, fluid,
                          x_prop, y_prop, n_points=500):
        """
        Generate data for a specific isoline.

        Parameters:
        iso_prop (str): The property for the isoline
            (e.g., 'T' for temperature).
        iso_val (float): The value of the property for the isoline.
        _iso_val (float): The SI value of the property for the isoline.
        fluid (dict): The fluid composition.
        x_prop (str): The x-axis property for the plot.
        y_prop (str): The y-axis property for the plot.
        n_points (int): The number of points to generate for the isoline.

        Returns:
        dict: A dictionary containing x and y values for the isoline.
        """

        # These are cases where iso_prop matches x_prop or y_prop
        # The line will be horizontal or vertical
        if iso_prop == x_prop:
            return {x_prop: iso_val, y_prop: None}
        elif iso_prop == y_prop:
            return {y_prop: iso_val, x_prop: None}

        # Get the sweeping variable and range
        if iso_prop == 'T':
            x_min, x_max = 1e-5, 1e4
            prop_sweep_var = 'D'
        else:
            prop_sweep_var = 'T'
            x_min, x_max = self._Tmin, self._Tmax

        # Initialize arrays for storing results
        x_sweep = np.linspace(x_min, x_max, n_points)
        x_vals = np.full_like(x_sweep, np.nan)
        y_vals = np.full_like(x_sweep, np.nan)

        # Update fluid
        self.thermo.Y = fluid

        # Sweep through the specified range and compute the properties
        for i, x in enumerate(x_sweep):
            try:
                # Update thermo via SI update
                self.thermo._update_state({prop_sweep_var: x,
                                           iso_prop: _iso_val})
                x_vals[i] = self.evaluate_thermo(x_prop, SI=False)
                y_vals[i] = self.evaluate_thermo(y_prop, SI=False)
            except Exception:
                pass

        return {x_prop: x_vals, y_prop: y_vals}

    def make_iso_lines(self, x_prop, y_prop,
                       n_points=500):
        """
        Generate iso-line sweeps for given x and y properties.

        Parameters:
        x_prop (str): The x-axis property for the plot.
        y_prop (str): The y-axis property for the plot.
        n_points (int): The number of points to generate for each isoline.
        """

        self.iso_dat = {}

        for iso_prop, lines in self.iso_lines.items():
            # Get the iso_state quantity
            q = self.quantities[iso_prop]

            for line in lines:

                if line['val'] is not None:
                    # SI Value
                    _iso_val = unit_handler.parse_units(line['val'], q)
                    # Output Value
                    iso_val = unit_handler.convert(_iso_val, units.SI_units[q],
                                                   units.output_units[q])
                    label = f"{iso_prop} = {line['val']}"
                    color = colors[len(self.iso_dat)]
                    # Do Unit Conversion here
                else:
                    # Statepoint
                    name = line['state_point']
                    if name in self.state_points:
                        # Get state point state ^and color
                        state = self.state_points[name]['state']
                        color = self.state_points[name]['color']
                        fluid = self.state_points[name]['fluid']
                        self.thermo._update_state(state,
                                                  composition=fluid)

                        # Get Value
                        iso_val = self.evaluate_thermo(iso_prop, SI=False)
                        _iso_val = self.evaluate_thermo(iso_prop, SI=True)
                        label = f"{iso_prop}: {name}"
                    else:
                        from pdb import set_trace
                        set_trace()

                fluid = line['fluid']
                # Check line style and color

                iso_line = self.generate_iso_line(iso_prop, iso_val, _iso_val,
                                                  fluid, x_prop, y_prop)

                # Assign color and line style if provided
                color = line['color'] or color
                line_style = line['line_style'] or ':'

                # Ensure unique labeling for multiple fluids
                if label in self.iso_dat:
                    label = f'{fluid}: {label}'

                self.iso_dat[label] = iso_line
                self.iso_dat[label].update({'color': color,
                                            'line_style': line_style})

    def make_process_lines(self, x_prop, y_prop):

        self.process_dat = {}
        for process, data in self.process_lines.items():
            x, y = self.generate_process_line(data['type'], data['US'], data['DS'], x_prop, y_prop)
            self.process_dat[process] = {'x': x, 'y': y}

    def sweep_vapor_dome(self, fluid=None, n_points=500):
        """
        Sweep the vapor dome boundary and retrieve properties.

        Parameters:
        fluid (str): The fluid to be used for thermodynamic calculations.
        n_points (int): The number of points to generate along the vapor dome.
        """

        if fluid is not None:
            self.update_fluid(fluid)

        # Use temperature as sweeping variable
        T_sweep = np.linspace(self._Tmin, self._T_critical, n_points)

        # Initialize vapor data dictionary using list/dict comprehension
        self.vapor = {i: {prop: np.full(len(T_sweep), np.nan) for prop
                          in ['T', 'P', 'S', 'H', 'D', 'v']}
                      for i in range(2)}

        # Sweep thru temperature and get properties
        for i, T in enumerate(T_sweep):
            for j in range(2):
                self.thermo._TQ = T, j
                self.vapor[j]['T'][i] = self.thermo.T
                self.vapor[j]['P'][i] = self.thermo.P
                self.vapor[j]['S'][i] = self.thermo.S
                self.vapor[j]['H'][i] = self.thermo.H
                self.vapor[j]['D'][i] = self.thermo.density
                self.vapor[j]['v'][i] = 1/self.thermo.density

        # Add to vapor data object
        self._vapor[self.fluid] = self.vapor

    def evaluate_thermo(self, property, SI=False):
        """
        Evaluate and return the thermodynamic property.

        Parameters:
        property (str): The thermodynamic property to evaluate
            (e.g., 'T', 'P').
        SI (bool): Whether to return the property in SI units (default: False).

        Returns:
        float: The value of the requested thermodynamic property.
        """
        if SI:
            if property == 'v':
                return 1/self.thermo._density
            elif property == 'D':
                return self.thermo._density
            else:
                return self.thermo._get_property(property)
        else:
            if property == 'v':
                return 1/self.thermo.density
            elif property == 'D':
                return self.thermo.density
            else:
                return getattr(self.thermo, property)

    def _update_thermo(self, state, SI=False):

        if 'v' in state:
            state['D'] = 1/state['v']
            del(state['v'])

        if SI:
            self.thermo._update_state(state)
        else:
            self.thermo.update_state(state)

    def _plot_state_points(self, x_prop, y_prop):
        """
        Helper function to plot state points.

        Parameters:
        x_prop (str): The x-axis property for the plot.
        y_prop (str): The y-axis property for the plot.
        """
        for name, state in self.state_points.items():
            # Update internal thermo state object
            self.thermo._update_state(state['state'],
                                      composition=state['fluid'])
            # get x, y values and plot
            x = self.evaluate_thermo(x_prop)
            y = self.evaluate_thermo(y_prop)
            # Plot state point
            plt.plot(x, y, marker=state['marker'],
                     color=state['color'])

            # Add statepoint name over state point
            plt.text(x, y, name)

    def _plot_process_lines(self, ax, x_prop, y_prop):

        self.make_process_lines(x_prop, y_prop)

        for name, data in self.process_dat.items():
            plt.plot(data['x'], data['y'],
                     self.process_lines[name]['line_style'],
                     color=self.process_lines[name]['color'])

            if len(data['x']) == 2:
                x_mid = data['x'][-1] - data['x'][0]
                y_mid = data['y'][-1] - data['y'][0]
                x1 = data['x'][0] + x_mid*(.99)/2
                x2 = data['x'][0] + x_mid*(1.01)/2
                y1 = data['y'][0] + y_mid*(.99)/2
                y2 = data['y'][0] + y_mid*(1.01)/2         
            else:
                icenter = len(data['x'])//2
                x_mid = data['x'][icenter]
                y_mid = data['x'][icenter]
                x1 = data['x'][icenter-1]
                x2 = data['x'][icenter+1]
                y1 = data['y'][icenter-1]
                y2 = data['y'][icenter+1]

            arrow = FancyArrowPatch(
                    (x1, y1), (x2, y2),
                    #arrowstyle='->',  # arrow style
                    mutation_scale=15,  # size of the arrow
                    color=self.process_lines[name]['color'],  # color of the arrow
                    linewidth=2.5,  # linewidth of the arrow
                    )

            ax.add_patch(arrow)

    def _plot_isolines(self, ax, x_prop, y_prop):
        """
        Helper function to plot isolines.

        Parameters:
        ax (matplotlib.axes.Axes): The axes on which to plot the isolines.
        x_prop (str): The x-axis property for the plot.
        y_prop (str): The y-axis property for the plot.
        """

        self.make_iso_lines(x_prop, y_prop)

        for name, iso_line in self.iso_dat.items():
            if iso_line[x_prop] is None:
                ax.axhline(y=iso_line[y_prop],
                           linestyle=iso_line['line_style'],
                           color=iso_line['color'],
                           label=name)
            elif iso_line[y_prop] is None:
                ax.axvline(x=iso_line[x_prop],
                           linestyle=iso_line['line_style'],
                           color=iso_line['color'],
                           label=name)
            else:
                plt.plot(iso_line[x_prop], iso_line[y_prop],
                         linestyle=iso_line['line_style'],
                         color=iso_line['color'],
                         label=name)

    def _plot_vapor_dome(self, x_prop, y_prop):
        """
        Helper function to plot the vapor dome.

        Parameters:
        x_prop (str): The x-axis property for the plot.
        y_prop (str): The y-axis property for the plot.
        """

        for i, (name, vapor) in enumerate(self._vapor.items()):
            for j in range(2):
                self.vapor = vapor

                # Plot vapor dome with a single color if we
                # are only plotting one fluid vapor, otherwise
                # plot with multiple
                color = 'black' if len(self._vapor) == 1 else colors[i]

                plt.plot(self.vapor[j][x_prop],
                         self.vapor[j][y_prop],
                         color=color,
                         label=name)

    def plot(self, plot_type,
             xscale='linear',
             yscale='linear',
             xlim=None,
             ylim=None,
             legend=None,
             show=True):
        """
        Plot the thermodynamic diagram.

        Parameters:
        plot_type (str): The type of plot to generate, specified as a string
            of the form 'yx'.
        xscale (str): The scale of the x-axis ('linear' or 'log').
        yscale (str): The scale of the y-axis ('linear' or 'log').
        xlim (tuple): The limits for the x-axis (optional).
        ylim (tuple): The limits for the y-axis (optional).
        legend (bool): Whether to display a legend (optional).
        show (bool): Whether to display the plot immediately (default: True).

        Returns:
        matplotlib.figure.Figure: The generated figure.
        """

        # Get the x and y properties to plot
        x_prop, y_prop = plot_type[1], plot_type[0]

        # Create Plot
        fig, ax = plt.subplots()

        # Set axis scales
        plt.xscale(xscale)
        plt.yscale(yscale)

        # Plot Vapor Dome and state points and process_lines
        self._plot_vapor_dome(x_prop, y_prop)
        self._plot_state_points(x_prop, y_prop)
        self._plot_process_lines(ax, x_prop, y_prop)

        # Get initial plot limits
        x0_lim = ax.get_xlim()
        y0_lim = ax.get_ylim()

        # Plot isolines
        self._plot_isolines(ax, x_prop, y_prop)

        # Configure legend
        if legend is None:
            legend = len(self._vapor) > 1

        if legend:
            plt.legend()
            # Use `handles` and `labels` to ensure only unique labels are shown
            handles, labels = plt.gca().get_legend_handles_labels()
            # Get the x and y properties to plot
            unique_labels = dict(zip(labels, handles))
            plt.legend(unique_labels.values(), unique_labels.keys())

        # Set x and y limits
        plt.xlim(xlim or x0_lim)
        plt.ylim(ylim or y0_lim)

        # Set labels
        plt.xlabel(self.labels[x_prop])
        plt.ylabel(self.labels[y_prop])

        # Add major and minor gridlines
        ax.minorticks_on()
        ax.grid(which='major', linestyle='-', linewidth=0.75, alpha=0.7)
        ax.grid(which='minor', linestyle=':', linewidth=0.5, alpha=0.5)

        if show is True:
            fig.show()

        return fig

    @property
    def thermo(self):
        """Return the thermo object."""

        return self.__thermo

    @property
    def labels(self):
        """Return axis labels with units."""

        return {'T': f"Temperature [{units.output_units['TEMPERATURE']}]",
                'P': f"Pressure [{units.output_units['PRESSURE']}]",
                'H': f"Enthalpy [{units.output_units['SPECIFICENERGY']}]",
                'S': f"Entropy [{units.output_units['SPECIFICENTROPY']}]",
                'D': f"Density [{units.output_units['DENSITY']}]",
                'v': ("Specific Volume "
                      f"[{units.output_units['SPECIFICVOLUME']}]")}

    @property
    def quantities(self):
        """Return a dictionary of quantities mapped to their unit names."""

        return {'T': 'TEMPERATURE',
                'P': 'PRESSURE',
                'H': 'SPECIFICENERGY',
                'S': 'SPECIFICENTROPY',
                'D': 'DENSITY',
                'v': 'SPECIFICVOLUME'}
