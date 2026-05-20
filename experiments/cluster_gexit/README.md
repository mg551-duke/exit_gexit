# Surface BSC GEXIT Cluster Jobs

These jobs run the sampled exact-probability BSC/GEXIT method on an entropy-axis
centered grid.  The sampled grid is uniform-like in `t = h2(p)`, sparse near
`t=0` and `t=1`, and densest around `t=0.5`.

Submit the full distance sweep from the repository root:

```bash
bash experiments/cluster_gexit/submit_surface_gexit_all.sh
```

This submits separate Slurm jobs so each distance can request different CPU
counts.  The current resource requests are:

```text
d=7:  20 CPUs, 64G
d=11: 20 CPUs, 64G
d=15: 20 CPUs, 64G
d=21:  6 CPUs, 64G
d=45:  4 CPUs, 64G
```

The Python runner uses `SLURM_CPUS_PER_TASK` worker processes across coupled
sample batches.  Each Monte Carlo sample is reused across the whole p-grid
before derivatives are averaged, which is the BSC analogue of the paired EXIT
derivative estimator.  The `d=21` and `d=45` submissions use fewer processes to
avoid OOM from large factor-contraction state.

To submit one distance directly:

```bash
sbatch --cpus-per-task=6 --mem=64G experiments/cluster_gexit/run_surface_gexit.sbatch experiments/cluster_gexit/surface_gexit_d21.py
```

Outputs are written under:

```text
data/experiments/gexit_curves/entropy_centered_surface_jobs/
```

Set `PYTHON_BIN` if the cluster Python executable is not named `python`.
