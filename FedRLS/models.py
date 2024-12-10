from typing import NamedTuple
import ex_fuzzy.fuzzy_sets as fs

class FRBCmodel(NamedTuple):
    """A namedtuple subclass to hold ex-fuzzy's model data."""
    n_gen: int = 30
    n_pop: int = 50
    nRules: int = 20
    nAnts: int = 4
    fz_type_studied: fs.FUZZY_SETS = fs.FUZZY_SETS.t1
    tolerance: float = 0.001
    runner: int = 1
    random_seed: int = 23
    class_names: list[str] = None
    verbose: bool = False


class FedRLSmodel(NamedTuple):
    """A namedtuple compModelDto to hold fedRLS' specific model data."""
    sim_threshold: float = 0.7
    contradictory_factor: float = 0.8
    max_retrains: int = 0
    adaptative: bool = False