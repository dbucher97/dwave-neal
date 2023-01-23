from dataclasses import dataclass


@dataclass
class Timer:
    preprocess: float = 0.0
    initial_states: float = 0.0
    to_numpy: float = 0.0
    beta_range: float = 0.0
    simulated_annealing: float = 0.0
    response: float = 0.0
    cpp_init: float = 0.0
    cpp_sa: float = 0.0
    cpp_full: float = 0.0
    sample_call: float = 0.0
    full_call: float = 0.0
