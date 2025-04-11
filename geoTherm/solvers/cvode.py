from assimulo.problem import Explicit_Problem
from assimulo.solvers import CVode
import numpy as np


def CVode_solver(f, y0, t_points):
    """
    Integrate a user-specified function using the Assimulo package
    and return the solution at specified time points.

    Parameters:
    - f: Callable
        The function to be integrated. It should be of the form f(t, y).
    - y0: array-like
        Initial condition(s).
    - t_points: array-like
        Specific time points at which the solution is required.

    Returns:
    - t: array
        Time points at which the solution was computed (same as t_points).
    - y: array
        Solution values at the corresponding time points.
    """
    # Create an Explicit_Problem instance
    problem = Explicit_Problem(f, y0, t_points[0])

    #problem.display_progress = False
    # Create a solver instance (e.g., CVode)
    solver = CVode(problem)
    #solver.report_continuously = True
    #solver.display_progress=True
    # Integrate and get results at the specified time points
    # Solver Verbosity
    solver.verbosity = 50




    t, y = solver.simulate(t_points[-1])

    # Integrate and store the results at specified points

    return np.array(t), y