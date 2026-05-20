# Quantum EXIT/GEXIT Diagnostics for CSS Codes

This repository studies EXIT-style entropy diagnostics for CSS stabilizer
codes. The erasure channel is the exact anchor: for an erasure pattern `E`,
the one-sided correction-class entropy

```text
H(C_X | E,S)
```

is computable by GF(2) rank identities and has the component-normalized area
`k/n`. The broader research program treats this as the quantum analogue of the
classical erasure EXIT area theorem and extends the same diagnostic language to
mixed erasure-Pauli, biased CSS Pauli, depolarizing, and eventually
circuit-level noise.

## Main Experiment Families

- `experiments/exit_curve_experiments.py`: exact erasure EXIT curves, paired
  derivative estimates, and peeling/stabilizer-peeling proxies.
- `experiments/beyond_erasure_diagnostics.py`: posterior class-entropy
  diagnostics beyond pure erasure. This currently supports one CSS component
  at a time, with exact affine-space enumeration when feasible and a clearly
  labelled MCMC/list approximation otherwise.

## Recommended Smoke Runs

```powershell
.\.venv\Scripts\python.exe experiments\exit_curve_experiments.py --code codes\surface5_HxHzLxLz.npz --runs 100 --centered-ps --exact-only --plot
```

```powershell
.\.venv\Scripts\python.exe experiments\beyond_erasure_diagnostics.py --code codes\surface5_HxHzLxLz.npz --component x --runs 5 --p-erasure-grid 0 0.2 --p-error-grid 0 0.02 --max-exact-affine-dim 12 --approx-samples 200 --plot
```

The second command intentionally allows list approximation for quick smoke
testing. For paper-facing numbers, prefer exact rows (`exact_runs == runs`) or
report list-restricted rows as approximate.

## Roadmap

The current roadmap is in
[`docs/EXIT_GEXIT_ROADMAP.md`](docs/EXIT_GEXIT_ROADMAP.md).
