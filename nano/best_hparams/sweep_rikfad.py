from sweep_engine import run_optimizer_sweep

def rikfad_params(trial):
    return {
        "h": trial.suggest_categorical("h", [0.4943374576911439]),
        "alpha": trial.suggest_categorical("alpha", [0.697079252867011]),
        "mu": trial.suggest_categorical("mu", [4.95129118269127e-06]),
        "gamma": trial.suggest_categorical("gamma", [1.0012253781587678e-06])
    }

if __name__ == "__main__":
    run_optimizer_sweep("sweep_rikfad", "iKFAD_R1", rikfad_params, n_trials=5)
