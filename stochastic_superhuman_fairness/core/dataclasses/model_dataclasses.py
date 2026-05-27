from dataclasses import dataclass
from typing import Union, Literal

@dataclass
class CouplingConfig:
    solver: Literal['mosek', 'sinkhorn'] = "mosek"
    normalize_s_matrix: bool = True,
    row_constraints: bool = True
    row_constraints: Literal['rev_ranking', 'ranking', 'policy_probs', None] = 'rev_ranking',
    tau: float = 1.0
    #  ot_temperature: float = 1.0,
    gamma_smoothing_ema: float = 0.9
    #  gamma_temperature: float = 1.0,
    #  lambda_reg: float = 1e-8
