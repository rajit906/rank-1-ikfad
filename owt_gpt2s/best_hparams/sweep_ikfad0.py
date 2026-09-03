# sweeps/sweep_ikfad0.py
from sweep_engine import run_optimizer_sweep

def ikfad0_params(trial):
    return {
        "h": trial.suggest_categorical("h", [0.09375160123937598]),
        "alpha": trial.suggest_categorical("alpha", [0.1933458819969178]),
        "mu": trial.suggest_categorical("mu", [4.272612928109156e-07]),
        "gamma": trial.suggest_categorical("gamma", [0.]),
    }

if __name__ == "__main__":
    run_optimizer_sweep("sweep_ikfad0", "iKFAD", ikfad0_params)
