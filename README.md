# Code For Reproducing Paper "Low-Rank Adaptive Friction for Memory-Efficient LLM Pretraining".

This repository contains code for the paper:

>**[Low-Rank Adaptive Friction for Memory-Efficient LLM Pretraining](https://rajit906.github.io/Rank_1_iKFAD_NeurIPS_OPT_2026.pdf)**<br>
>Rajit Rajpal, Benedict Leimkuhler<br>
>**Abstract:** "iKFAD is a recently proposed optimizer that closes the Adam–SGD performance gap observed in transformer model pretraining by replacing adaptive learning rates with adaptive friction in the momentum dynamics. Its limitation is that the full friction tensor ξ ∈ R^(m×n) carries the same O(nm) memory overhead per layer as Adam's second-moment buffer. Here we exploit the rank-1 compression idea that Adafactor applies to Adam's second moment, replacing iKFAD's friction tensor ξ with a rank-1 outer-product factorization built from row and column momentum statistics, resulting in Rank-1 iKFAD (R-iKFAD). This reduces friction memory footprint from O(nm) to O(n+m) per layer which approximately halves iKFAD's total optimizer state. Despite this massive decrease in signal, we show that R-iKFAD maintains parity in performance with iKFAD, and consequently Adam on transformer pretraining tasks. Experiments on GPT2-Nano, TinyViT, DistilBERT, and GPT2-S confirm that R-iKFAD matches or exceeds iKFAD while nearly halving the memory footprint and being comparably robust to hyperparameters. We analyze the continuous-time dynamics in two damping regimes. For γ>0 (linear damping) we prove exponential convergence under strong convexity. For γ=0, the preferred option in our experiments, the friction is generated entirely based on past momentum and it switches off as the momentum vanishes, so geometric convergence cannot be shown. We nonetheless prove unconditional convergence to the minimizer together with explicit algebraic rates, O(t⁻¹) and O(t⁻¹ᐟ²) according to the regularization scale. To our knowledge this is the first convergence rate for a rank-1 factored optimizer in continuous time, and the first such result that does not require positive damping."<br>

The project includes four experiments with optimizer implementations (iKFAD and R-iKFAD) in the `optimizers/` directory. Adam is taken from `torch.optim` and Adafactor from `transformers.optimization` with its optional first moment enabled. Hyperparameter sweep drivers and per-optimizer launch scripts are available inside each experiment folder — `nano/` (GPT2-Nano on Shakespeare), `vit/` (TinyViT on CIFAR-10), `sst2/` (DistilBERT on SST-2) and `owt_gpt2s/` (GPT2-S on OpenWebText) — and each provides a conda environment configuration file. The notebook `results.ipynb` regenerates the loss, accuracy and ablation figures from the per-seed curves in `*/sweep-table/`, without retraining.

```bash
jupyter nbconvert --execute --inplace results.ipynb   # writes the figures into figs/
```

```bash
conda env create -f vit/environment.yml && conda activate vit
python vit/sweep.py --optimizer rikfad0 --n_trials 100      # hyperparameter search
python vit/run_best_seeds.py --optimizer rikfad0            # seed replicates
```

Launch scripts under `*/scripts/` are SLURM wrappers with the cluster-specific directives removed; adapt the partition and account lines to your site.

### MIT License

```
Copyright (c) 2026 Rajit Rajpal

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```
