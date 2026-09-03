from sweep_engine import run_optimizer_sweep

def adafactor_params(trial):
    return {
        "lr": trial.suggest_float("lr", 1e-5, 1e-2, log=True),
        "beta1": trial.suggest_float("beta1", 0.85, 0.999, log=True),
        "scale_parameter": False,
        "relative_step": False,
        "warmup_init": False,
    }

if __name__ == "__main__":
    run_optimizer_sweep("sweep_adafactor", "Adafactor", adafactor_params)
