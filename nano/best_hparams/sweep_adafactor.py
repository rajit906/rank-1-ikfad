from sweep_engine import run_optimizer_sweep

def adafactor_params(trial):
    return {
        "lr":              trial.suggest_categorical("lr",    [0.0029923041538933324]),
        "beta1":           trial.suggest_categorical("beta1", [0.8893482889171225]),
        "scale_parameter": False,
        "relative_step":   False,
        "warmup_init":     False,
    }

if __name__ == "__main__":
    run_optimizer_sweep("sweep_adafactor", "Adafactor", adafactor_params, n_trials=5)
