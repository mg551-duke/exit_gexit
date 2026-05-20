# Surface BSC GEXIT Cluster Jobs

These jobs run the sampled exact-probability BSC/GEXIT method on an entropy-axis
centered grid.  The sampled grid is uniform-like in `t = h2(p)`, sparse near
`t=0` and `t=1`, and densest around `t=0.5`.

Submit the full distance sweep from the repository root:

```bash
sbatch experiments/cluster_gexit/run_surface_gexit.sbatch
```

The Slurm script is a five-task array for `d=7,11,15,21,45`.  Each array task
requests 20 CPUs and the Python runner uses `SLURM_CPUS_PER_TASK` worker
processes across independent p-grid points.

Outputs are written under:

```text
data/experiments/gexit_curves/entropy_centered_surface_jobs/
```

Set `PYTHON_BIN` if the cluster Python executable is not named `python`.
