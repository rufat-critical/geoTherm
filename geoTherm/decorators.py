# Decorators to modify/adopt geoTherm functionality
from functools import wraps
import numpy as np
from geoTherm.units import units, unit_converter

def make_serializable(obj):
    """
    Recursively convert numpy arrays, tuples, and numpy scalars
    into native Python types that are safe for copying, logging, or saving.
    """
    if isinstance(obj, dict):
        return {make_serializable(k): make_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [make_serializable(i) for i in obj]
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, (np.generic,)):
        return obj.item()
    else:
        return obj

def state_dict(method):
    """
    Decorator that:
    - Walks MRO and collects all _state_dict returns into 'config'
    - Adds 'Node_Type' and optionally 'x'
    - Validates that the decorated method does not use super()._state_dict
    """

    @wraps(method)
    def wrapper(self):
        config = {}

        # Traverse MRO bottom-up
        for cls in reversed(self.__class__.mro()):
            if cls is object:
                continue
            prop = cls.__dict__.get('_state_dict')
            if isinstance(prop, property):
                raw_func = getattr(prop.fget, '__wrapped__', prop.fget)
                if raw_func is method:
                    continue
                update = raw_func(self)
                if update:
                    config.update(make_serializable(update))
        # Add this class's contribution
        own = method(self)
        if own:
            own = make_serializable(own)
            config.update(own)

        
        if 'fluid' in config:
            state = config['fluid']['state']

            if 'T' in state:
                config['fluid']['state']['T'] = [unit_converter(state['T'][0], state['T'][1],
                                                               units.output_units['TEMPERATURE']),
                                                units.output_units['TEMPERATURE']]

            if 'P' in state:
                config['fluid']['state']['P'] = [unit_converter(state['P'][0], state['P'][1],
                                                               units.output_units['PRESSURE']),
                                                units.output_units['PRESSURE']]

            if 'H' in state:
                config['fluid']['state']['H'] = [unit_converter(state['H'][0], state['H'][1],
                                                               units.output_units['SPECIFICENERGY']),
                                                units.output_units['SPECIFICENERGY']]
            
            if 'D' in state:
                config['fluid']['state']['D'] = [unit_converter(state['D'][0], state['D'][1],
                                                               units.output_units['DENSITY']),
                                                units.output_units['DENSITY']]

        out = {
            'Node_Type': type(self).__name__,
            'config': config
        }

        if hasattr(self, 'x'):
            out['x'] = make_serializable(getattr(self, 'x'))

        return out

    return property(wrapper)
