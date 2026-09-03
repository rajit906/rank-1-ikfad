# sweeps/sweep_adam.py
from sweep_engine import run_optimizer_sweep

def adam_params(trial):
    return {
        "lr": trial.suggest_float("lr", 1e-7, 1e-2, log=True),
        "beta_1": trial.suggest_categorical("beta_1", [0.9]),
        "beta_2": trial.suggest_float("beta_2", 0.9, 0.999, log=True),
        "eps": 1e-8,
    }

if __name__ == "__main__":
    run_optimizer_sweep("sweep_adam", "ADAM", adam_params)
