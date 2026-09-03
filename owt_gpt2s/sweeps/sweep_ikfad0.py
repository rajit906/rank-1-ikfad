# sweeps/sweep_ikfad0.py
from sweep_engine import run_optimizer_sweep

def ikfad0_params(trial):
    return {
        "h": trial.suggest_float("h", 1e-6, 1e-1, log=True),
        "alpha": trial.suggest_float("alpha", 1e-3, 1, log=True),
        "mu": trial.suggest_float("mu", 1e-9, 1e-2, log=True),
        "gamma": trial.suggest_categorical("gamma", [0.]),
    }

if __name__ == "__main__":
    run_optimizer_sweep("sweep_ikfad0", "iKFAD", ikfad0_params)
