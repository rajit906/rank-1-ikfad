# best_hparams/sweep_rikfad0.py
# TODO: fill in best hyperparameters from sweeps/sweep_rikfad0 once sweep completes.
# Run: bash scripts/sweep_rikfad0_best.sh
from sweep_engine import run_optimizer_sweep

def rikfad0_params(trial):
    return {
        "h":     trial.suggest_categorical("h",     [0.09181014353588908]),   # TODO: replace with best h
        "alpha": trial.suggest_categorical("alpha", [0.05793444718851863]),    # TODO: replace with best alpha
        "mu":    trial.suggest_categorical("mu",    [1.3637001131696629e-06]),   # TODO: replace with best mu
        "gamma": trial.suggest_categorical("gamma", [0.]),
    }

if __name__ == "__main__":
    run_optimizer_sweep("sweep_rikfad0", "iKFAD_R1", rikfad0_params)
