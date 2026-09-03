from sweep_engine import run_optimizer_sweep

def rikfad_params(trial):
    return {
        "h": trial.suggest_categorical("h", [0.48421557604419196]),
        "alpha": trial.suggest_categorical("alpha", [2.847475250828921]),
        "mu": trial.suggest_categorical("mu", [1.940675514888073e-06]),
        "gamma": trial.suggest_categorical("gamma", [0.])
    }

if __name__ == "__main__":
    run_optimizer_sweep("sweep_rikfad0", "iKFAD_R1", rikfad_params, n_trials=5)
