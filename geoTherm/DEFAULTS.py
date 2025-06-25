from dataclasses import dataclass
from typing import Literal

@dataclass
class ThermSettings:
    """Default settings for geoTherm thermodynamic calculations."""
    input_units: Literal['SI'] = 'SI'
    output_units: Literal['SI'] = 'SI'
    EoS: Literal['REFPROP', 'HEOS'] = 'HEOS'
    model: Literal['coolprop'] = 'coolprop'

# Create the default instance
DEFAULTS = ThermSettings()

