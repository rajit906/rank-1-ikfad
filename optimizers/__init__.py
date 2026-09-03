# optimizers/__init__.py

from .ikfad import iKFAD          # per-parameter adaptive friction (baseline)
from .ikfad_r1 import iKFAD_R1    # rank-1 factored friction (this work)

__all__ = ['iKFAD', 'iKFAD_R1']
