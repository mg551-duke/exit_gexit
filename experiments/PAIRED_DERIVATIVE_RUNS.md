# Paired Derivative EXIT Runs

This note records how `experiments/exit_curve_experiments.py` was used to
generate the paired derivative plots for the surface-code EXIT curves.

## What The Paired Derivative Mode Does

The original EXIT output estimates an average entropy curve at each sampled
erasure probability `p`, then applies a finite difference to the averaged
curve. The paired derivative mode instead draws one threshold vector per Monte
Carlo sample,

```text
u_i ~ Uniform(0, 1)
E(p) = {i : u_i < p},
```

and reuses that same nested erasure sample for nearby `p` values. It then
forms the finite difference per sample and averages those differences:

```text
((H_CX(p_high) / n) - (H_CX(p_low) / n)) / (p_high - p_low).
```

This targets the same finite-difference derivative but usually has lower
Monte Carlo variance because the two entropy values are strongly correlated.

## Grid And Sampling

The paired derivative runs below used the centered grid enabled by
`--centered-ps`. The default grid is sparse near the edges and keeps the
original-like `0.02` spacing near the transition:

```text
0.0, 0.1, 0.2, 0.25, 0.3, 0.35,
0.37, 0.39, 0.41, 0.43, 0.45, 0.47,
0.49, 0.5, 0.51, 0.53, 0.55, 0.57,
0.59, 0.61, 0.63, 0.65, 0.7, 0.75,
0.8, 0.9, 1.0
```

The goal is to spend fewer evaluations near `p = 0` and `p = 1`, where the
curve is less informative, and spend the saved compute on more Monte Carlo
samples.

## Commands Used

Surface 5 was run with both exact and peeling quantities so the plot shows the
exact, aided-peeling, and unassisted-peeling derivative curves:

```powershell
.\.venv\Scripts\python.exe experiments\exit_curve_experiments.py --code codes\surface5_HxHzLxLz.npz --runs 5000 --centered-ps --seed 0 --paired-derivative --plot
```

Surface 7 and surface 9 were run with exact quantities only. For these surface
code runs, the exact `H(C_X | E,S)` curve and the stabilizer-aided peeling
class curve coincide, and exact-only was faster in this implementation:

```powershell
.\.venv\Scripts\python.exe experiments\exit_curve_experiments.py --code codes\surface7_HxHzLxLz.npz --runs 5000 --centered-ps --seed 0 --paired-derivative --exact-only --plot
```

```powershell
.\.venv\Scripts\python.exe experiments\exit_curve_experiments.py --code codes\surface9_HxHzLxLz.npz --runs 5000 --centered-ps --seed 0 --paired-derivative --exact-only --plot
```

For the scaled derivative-family comparison plots, surface 5, 7, and 9 were
rerun with three times as many samples on the same centered p-grid to reduce
visible Monte Carlo roughness near `p = 0.5`:

```powershell
.\.venv\Scripts\python.exe experiments\exit_curve_experiments.py --code codes\surface5_HxHzLxLz.npz --runs 15000 --centered-ps --seed 0 --paired-derivative --exact-only --plot
```

```powershell
.\.venv\Scripts\python.exe experiments\exit_curve_experiments.py --code codes\surface7_HxHzLxLz.npz --runs 15000 --centered-ps --seed 0 --paired-derivative --exact-only --plot
```

```powershell
.\.venv\Scripts\python.exe experiments\exit_curve_experiments.py --code codes\surface9_HxHzLxLz.npz --runs 15000 --centered-ps --seed 0 --paired-derivative --exact-only --plot
```

Surface 11 was run twice. First, the older three-panel output was generated
from the averaged entropy curve, so its derivative panel uses the original
average-then-difference method:

```powershell
.\.venv\Scripts\python.exe experiments\exit_curve_experiments.py --code codes\surface11_HxHzLxLz.npz --centered-ps --seed 0 --exact-only --plot
```

Second, the derivative-only paired plot was generated:

```powershell
.\.venv\Scripts\python.exe experiments\exit_curve_experiments.py --code codes\surface11_HxHzLxLz.npz --centered-ps --seed 0 --paired-derivative --exact-only --plot
```

For future three-panel plots, the middle derivative panel should preferably
use the paired derivative estimate rather than the derivative of independently
sampled averaged curves.

Surface 15 was also run with exact quantities only and the same centered grid.
The script defaulted to `400` runs for this `n = 421` code:

```powershell
.\.venv\Scripts\python.exe experiments\exit_curve_experiments.py --code codes\surface15_HxHzLxLz.npz --centered-ps --seed 0 --exact-only --plot
```

```powershell
.\.venv\Scripts\python.exe experiments\exit_curve_experiments.py --code codes\surface15_HxHzLxLz.npz --centered-ps --seed 0 --paired-derivative --exact-only --plot
```

Surface 21 was generated with `scripts/generate_surface_code.py` and then run
with exact quantities only. The script defaulted to `200` runs for this
`n = 841` code:

```powershell
.\.venv\Scripts\python.exe scripts\generate_surface_code.py 21
```

```powershell
.\.venv\Scripts\python.exe experiments\exit_curve_experiments.py --code codes\surface21_HxHzLxLz.npz --centered-ps --seed 0 --exact-only --plot
```

```powershell
.\.venv\Scripts\python.exe experiments\exit_curve_experiments.py --code codes\surface21_HxHzLxLz.npz --centered-ps --seed 0 --paired-derivative --exact-only --plot
```

Surface 45 was generated and then run only in paired derivative mode with
`1000` samples. For this larger code, the paired derivative run used the
surface-code fast path. This computes `H(C_X | E,S)` as the indicator that
erased qubit-edges contain a top-to-bottom X logical path, which was validated
against the rank-based exact calculation on smaller generated surface codes:

```powershell
.\.venv\Scripts\python.exe scripts\generate_surface_code.py 45
```

```powershell
.\.venv\Scripts\python.exe experiments\exit_curve_experiments.py --code codes\surface45_HxHzLxLz.npz --runs 1000 --centered-ps --seed 0 --paired-derivative --surface-fast-x-class --plot
```

Surface 75 was generated and run only in paired derivative mode with `1000`
samples. The p-grid was restricted to the transition window `[0.4, 0.6]` with
step `0.0025`, and the paired output records both the derivative and the
logical-normalized class curve:

```powershell
.\.venv\Scripts\python.exe scripts\generate_surface_code.py 75
```

```powershell
$ps = 0..80 | ForEach-Object { [Math]::Round(0.4 + 0.0025 * $_, 4) }
.\.venv\Scripts\python.exe experiments\exit_curve_experiments.py --code codes\surface75_HxHzLxLz.npz --runs 1000 --seed 0 --paired-derivative --surface-fast-x-class --plot --ps $ps
```

Surface 99 was generated and run only in paired derivative mode with `3000`
samples. The p-grid used exactly 25 points, sparse near `0.4` and `0.6` and
finer near `0.5`:

```powershell
.\.venv\Scripts\python.exe scripts\generate_surface_code.py 99
```

```powershell
$ps = @(0.4, 0.43, 0.45, 0.455, 0.46) + (0..14 | ForEach-Object { [Math]::Round(0.465 + 0.005 * $_, 3) }) + @(0.54, 0.545, 0.55, 0.57, 0.6)
$ps = $ps | Sort-Object -Unique
.\.venv\Scripts\python.exe experiments\exit_curve_experiments.py --code codes\surface99_HxHzLxLz.npz --runs 3000 --seed 0 --paired-derivative --surface-fast-x-class --plot --ps $ps
```

Surface 151 was run only in paired derivative mode with `5000` samples and 15
p-points. A dense `.npz` was intentionally not generated for this distance
because the dense `Hx`/`Hz` arrays would be multi-gigabyte in memory. The
surface fast path infers `d=151` from the code filename and does not load the
file:

```powershell
$ps = @(0.4, 0.45, 0.47, 0.485, 0.4925, 0.495, 0.4975, 0.5, 0.5025, 0.505, 0.5075, 0.515, 0.53, 0.55, 0.6)
.\.venv\Scripts\python.exe experiments\exit_curve_experiments.py --code codes\surface151_HxHzLxLz.npz --runs 5000 --seed 0 --paired-derivative --surface-fast-x-class --plot --ps $ps
```

Surface 251 was run the same way: paired derivative only, no bit-scale curve,
and no dense `.npz` generation. The 15-point p-grid is restricted to
`[0.4, 0.6]`:

```powershell
$ps = @(0.4, 0.45, 0.47, 0.485, 0.4925, 0.495, 0.4975, 0.5, 0.5025, 0.505, 0.5075, 0.515, 0.53, 0.55, 0.6)
.\.venv\Scripts\python.exe experiments\exit_curve_experiments.py --code codes\surface251_HxHzLxLz.npz --runs 5000 --seed 0 --paired-derivative --surface-fast-x-class --plot --ps $ps
```

## Sparse Bivariate-Bicycle Runs

The larger bivariate-bicycle examples were generated directly in sparse COO
format with `scripts/generate_bivariate_bicycle_code.py`. The generator uses
the standard BB form `H_X=[A,B]`, `H_Z=[B^T,A^T]`.

The `gross` family reproduces the existing `gross_HxHzLxLz.npz` parity checks
at `ell=12, m=6`:

```text
A = y + y^2 + x^3
B = y^3 + x + x^2
```

The `two_gross` family reproduces the existing `two_gross_HxHzLxLz.npz`
parity checks at `ell=12, m=12`:

```text
A = y^2 + y^7 + x^3
B = y^3 + x + x^2
```

Generated sparse code files:

```powershell
.\.venv\Scripts\python.exe scripts\generate_bivariate_bicycle_code.py --family gross --ell 24 --m 12
.\.venv\Scripts\python.exe scripts\generate_bivariate_bicycle_code.py --family two_gross --ell 24 --m 24
.\.venv\Scripts\python.exe scripts\generate_bivariate_bicycle_code.py --family gross --ell 48 --m 24
.\.venv\Scripts\python.exe scripts\generate_bivariate_bicycle_code.py --family two_gross --ell 48 --m 48
```

The paired derivative runs used the same 15-point `[0.4, 0.6]` grid as
surface251, no bit-scale run, and `5000` samples:

```powershell
.\.venv\Scripts\python.exe scripts\run_bb_sparse_paired_exit_batch.py `
  codes\bb_gross_l24_m12_n576_sparse.npz `
  codes\bb_two_gross_l24_m24_n1152_sparse.npz `
  codes\bb_gross_l48_m24_n2304_sparse.npz `
  codes\bb_two_gross_l48_m48_n4608_sparse.npz `
  --runs 5000
```

These sparse BB files currently store `H_X` and `H_Z` only, not `L_X/L_Z`.
The exact class calculation therefore uses the kernel/stabilizer rank identity
rather than the logical-rank shortcut.

The generated and existing BB-family parameter values are recorded in
`data/experiments/bb_code_parameters.csv`. In that table, `k` is exact from
`n - rank(H_X) - rank(H_Z)`. The recorded `d` values are explicitly labeled as
upper bounds unless exact optimality has been proved. For the generated larger
BB codes, exact distance certification was not practical in this run; the
recorded bounds come from explicit nontrivial logical operators found either
by the MILP distance formulation or from a qLDPC logical basis.

## Outputs

Each paired derivative run writes files with the `_paired_derivative` suffix in
`data/experiments/exit_curves/`:

```text
<code>_exit_rule1_paired_derivative.json
<code>_exit_rule1_paired_derivative.csv
<code>_exit_rule1_paired_derivative.png
```

The JSON and CSV contain derivative columns such as:

```text
exact_x_class_component_norm_dp
peel_x_aided_guess_component_norm_dp
peel_x_unassisted_guess_component_norm_dp
```

The PNG reports the area under each plotted derivative curve. For
`exact_x_class_component_norm_dp`, this area is the endpoint change in
`H(C_X | E,S) / n`. For the one-logical-qubit surface codes here, that should
be approximately:

```text
k / n = 1 / n.
```

So the plot annotation `area = 0.011765 ~= 1.00/n`, for example, means the
integrated EXIT curve recovers the expected logical entropy scale for
surface7, where `n = 85` and `k = 1`.
