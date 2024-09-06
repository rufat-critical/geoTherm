import pytest
import numpy as np
import geoTherm as gt
import math
from geoTherm.flow_funcs import (
    _total_to_static, _static_to_total, _dH_isentropic, _dH_isentropic_perfect, _dH_incompressible,
    _w_incomp, _w_comp, _w_isen, perfect_ratio_from_Mach
)


@pytest.fixture
def setup_thermo_states():
    """
    Fixture to set up common thermodynamic states for testing.
    """
    liquid = gt.thermo(fluid='H2O', state={'T': 300, 'P': 101325}, model='coolprop')
    gas = gt.thermo(fluid='H2O', state={'T': 500, 'P': 101325}, model='coolprop')
    return liquid, gas


def test_static_total(setup_thermo_states):
    """
    Test _total_to_static function for expected output.
    """
    _, gas = setup_thermo_states
    w_flux = 100  # Example mass flux value
    static_state = _total_to_static(gas, w_flux)

    total_state = _static_to_total(static_state, w_flux)

    assert math.isclose(gas._H, total_state._H, abs_tol=1e-3)
    assert math.isclose(total_state._H, 2928507.767, abs_tol=1e-3)
    assert math.isclose(total_state._P, gas._P, abs_tol=1e-3)
