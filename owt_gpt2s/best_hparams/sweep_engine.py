# sweeps/sweep_engine.py
import sys
import os
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

import torch
import optuna
from config.default_config import get_default_config
from train import GPTTrainer

def run_optimizer_sweep(study_name, optimizer_name, param_selector_fn, n_trials=1):
    """
    optimizer_name: str matching the class name in algorithms.py
    param_selector_fn: function(trial) -> dict of hyperparameters
    """

    def objective(trial):
        config = get_default_config()
        optimizer_kwargs = param_selector_fn(trial)
        # Infer the optimizer's base LR for the cosine schedule (lr for Adam/Adafactor, h for iKFAD/R-iKFAD)
        opt_lr = optimizer_kwargs.get('lr', optimizer_kwargs.get('h', 6e-4))

        config.update({
            # --- CORE SETTINGS ---
            'optimizer_name': optimizer_name,
            'optimizer_params': optimizer_kwargs,
            'weight_decay': 0.0,

            # --- SYSTEM ---
            'out_dir': f'out-best_hparams/{optimizer_name}/trial_{trial.number}',
            'dataset': 'openwebtext',
            'device': 'cuda',
            'dtype': 'float16',
            'compile': False,
            'grad_clip': 1.0,
            'seed': 1337 + trial.number,

            # --- MODEL (GPT2-S) ---
            'n_layer': 12, 'n_head': 12, 'n_embd': 768,
            'block_size': 1024, 'batch_size': 12,
            'dropout': 0., 'bias': False, 'flash_attn': True,
            'gradient_accumulation_steps': 1,

            # --- DURATION ---
            'max_iters': 200_000,
            'eval_interval': 1000,
            'eval_iters': 200,
            'log_interval': 1000,
            'always_save_checkpoint': False,
            'ckpt_interval': 200_001,

            # --- LR SCHEDULE (cosine, Rae et al. 2021: min_lr = 0.1 * lr) ---
            'learning_rate': opt_lr,
            'decay_lr': False,
            'warmup_iters': 200_001,
            'lr_decay_iters': 200_001,
            'min_lr': 0.1 * opt_lr,
        })

        try:
            print(f"--- {optimizer_name} Trial {trial.number} ---")
            print(f"Params: {optimizer_kwargs}")
            trainer = GPTTrainer(config)
            final_loss = trainer.train()

            if final_loss == float('inf'):
                print("Received infinity from trainer. Pruning trial.")
                raise optuna.TrialPruned()

            del trainer
            torch.cuda.empty_cache()
            return final_loss

        except Exception as e:
            print(f"Trial failed: {e}")
            import traceback
            traceback.print_exc()
            return float('inf')

    study = optuna.create_study(
        direction="minimize",
        study_name=study_name,
    )

    study.optimize(objective, n_trials=n_trials)

    print(f"Best {optimizer_name} params: {study.best_params}")
