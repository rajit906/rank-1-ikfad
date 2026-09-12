# Theory numerics — algebraic decay at `gamma = 0`

Numerical verification of the continuous-time analysis at `gamma = 0`. With the
reduced energy

    E(t) = f(X(t)) - f(X*) + 0.5*||P(t)||_F^2

the decay is algebraic, and which exponent applies is decided by `epsilon_stab`:
`t^-1` when `epsilon_stab = 0`, `t^-1/2` when `epsilon_stab > 0`, with a crossover
at `E ~ alpha*mu*epsilon_stab`.

![Energy decay at gamma = 0](gamma0_decay.png)

BACD splitting with `m=4`, `n=3`, `alpha=mu=1`, `h=0.02`, `T=1e7`, `R(0)=C(0)=0`,
seed 1. The blue dashed curve has `epsilon_stab=0`, the orange dash-dotted curve
`epsilon_stab=0.1`; the pale lines have slopes `-1` and `-1/2`. This is the figure
that appears in the paper, regenerated from the shipped trajectories.

## Setup

Both implementations use the quadratic objective

    f(X) = 0.5 * X' * A * X,    A = B*B'/(m*n) + 0.8*I

with `B` having independent standard normal entries, so the smallest eigenvalue of
`A` is at least `0.8`. Initial `X` and `P` entries are independent normals with
standard deviations `0.4` and `0.3`. When `epsilon_stab = 0` and `sum(R) = 0`, both
implementations define the friction to be zero; no positive numerical floor is
added.

## Two integrators

Short-horizon parameter sweeps use MATLAB's adaptive `ode89`. The long runs to
`T = 1e7` use the BACD splitting of the method itself, implemented in C, because
adaptive integration is impractical at that horizon. On their common range the two
agree to about `0.01` in every measured slope.

    # adaptive, MATLAB  (R2026a, ode89, RelTol=1e-10, AbsTol=1e-12, MaxStep unset)
    matlab -batch validate_rates

    # BACD splitting, C
    cc -O3 -o rikfad rikfad.c -lm
    ./rikfad 4 3 0   1 1 1e7 0.02 1 0 long_4x3_e0.dat
    ./rikfad 4 3 0.1 1 1 1e7 0.02 1 0 long_4x3_e01.dat

The arguments are `m`, `n`, `epsilon_stab`, `alpha`, `mu`, `T`, `h`, `seed`, `S0`
and the output file; initial factors are `R_i = S0/m` and `C_j = S0/n`. Each data
file holds time, energy and `sum(R)`.

## Reproducing the figure

    matlab -batch plot_gamma0_decay

Reads `long_4x3_e0.dat` and `long_4x3_e01.dat` and writes
`gamma0_decay_reproduced.pdf`, which is the figure above.

## Fitted slopes

`slopes.py` fits `log E` against `log t` by ordinary least squares over the given
intervals. No fitting uncertainties are recorded.

    python3 slopes.py long_4x3_e0.dat 1e6:1e7

Final-decade slopes over `[1e6, 1e7]`, across the three dimensions:

| | `epsilon_stab = 0` (expect `-1`) | `epsilon_stab = 0.1` (expect `-1/2`) |
|---|---|---|
| `m=2, n=2` | `-0.989` | `-0.497` |
| `m=4, n=3` | `-0.998` | `-0.501` |
| `m=6, n=5` | `-0.999` | `-0.503` |

Both exponents are independent of the matrix dimensions. The approach to `-1/2` is
slower for larger `m*n`, because the second regime begins only once
`E <~ alpha*mu*epsilon_stab` and the first-regime constant grows with `m*n`.

## Parameter sweeps

    matlab -batch validate_rates

Runs four checks: **(A)** the exponents at `m=3, n=2`; **(B)** whether
`t*E/(alpha*mu)` is constant across `alpha, mu` at `epsilon_stab = 0`; **(C)**
whether `E*sqrt(t)/(alpha*mu*sqrt(epsilon_stab/2))` is constant at
`epsilon_stab > 0`; and **(D)** whether the crossover sits at
`E ~ alpha*mu*epsilon_stab`. Check (D) reports mean `0.87` with `max/min = 1.21`.
A final check repeats the scalar case `m = n = 1`, which is scalar KFAD.

## Step-size refinement

    matlab -batch validate_bacd_steps

Repeats the `h = 0.02` versus `h = 0.01` comparison entirely in MATLAB. The
`hcheck_*.dat` files preserve the same rerun made with the C source; recompute
their slopes with `slopes.py`, for example

    python3 slopes.py hcheck_e0_h02.dat 1e2:1e3 1e3:1e4 1e4:3e4

Halving the step changes every fitted slope by less than `8.9e-4`.

## Files

| file | purpose |
|---|---|
| `rikfad.c` | BACD splitting integrator used for the long runs |
| `rikfad_rhs.m`, `energy.m` | vector field and energy for the MATLAB integrator |
| `validate_rates.m` | the four parameter sweeps (A)–(D) |
| `validate_bacd_steps.m` | step-size refinement check |
| `plot_gamma0_decay.m` | regenerates the figure above |
| `slopes.py` | least-squares log-log slope fits |
| `long_{2x2,4x3,6x5}_{e0,e01}.dat` | long runs to `T = 1e7`; the two `4x3` files are the figure |
| `hcheck_{e0,e01}_{h01,h02}.dat` | step-refinement runs at `h = 0.02` and `h = 0.01` |
