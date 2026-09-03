# best_hparams/sweep_adafactor.py
# TODO: fill in best hyperparameters from sweeps/sweep_adafactor once sweep completes.
# Run: bash scripts/sweep_adafactor_best.sh
from sweep_engine import run_optimizer_sweep

def adafactor_params(trial):
    return {
        "lr":              trial.suggest_categorical("lr",    [0.0007277712804656839]),
        "beta1":           trial.suggest_categorical("beta1", [0.8864730411941274]),
        "scale_parameter": False,
        "relative_step":   False,
        "warmup_init":     False,
    }

if __name__ == "__main__":
    run_optimizer_sweep("sweep_adafactor", "Adafactor", adafactor_params)
