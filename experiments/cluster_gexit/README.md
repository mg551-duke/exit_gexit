# Surface BSC GEXIT Cluster Jobs

These jobs run the sampled exact-probability BSC/GEXIT method on an entropy-axis
centered grid.  The sampled grid is uniform-like in `t = h2(p)`, sparse near
`t=0` and `t=1`, and densest around `t=0.5`.

Submit the full distance sweep from the repository root:

```bash
bash experiments/cluster_gexit/submit_surface_gexit_all.sh
```

This submits separate Slurm jobs so each distance can request different CPU
counts.  The current layout is:

```text
d=7:  one 20-CPU, 64G job
d=11: one 20-CPU, 64G job
d=15: 20 one-CPU, 64G shard jobs, 750 samples each
d=21: 20 one-CPU, 64G shard jobs, 500 samples each
d=45: 20 one-CPU, 64G shard jobs, 250 samples each
```

The Python runner uses `SLURM_CPUS_PER_TASK` worker processes across coupled
sample batches.  Each Monte Carlo sample is reused across the whole p-grid
before derivatives are averaged, which is the BSC analogue of the paired EXIT
derivative estimator.  The `d=21` and `d=45` submissions intentionally use very
few processes because each coupled worker evaluates the full p-grid and carries
large factor-contraction state.  For larger distances, the preferred
parallelism is therefore across many one-worker Slurm array jobs, then merging
the independent shards afterward.

Progress is written to the Slurm `.out` file as flushed newline bars, one per
worker batch, at roughly 5% increments.

To add 80,000 independent coupled samples to completed `d=7` and `d=11` runs,
then merge the repeat into the main result files:

```bash
bash experiments/cluster_gexit/submit_surface_gexit_addons.sh
```

Each add-on job first writes its repeat under
`data/experiments/gexit_curves/entropy_centered_surface_jobs/addons/`, then
rewrites the merged main JSON/CSV/PNG/TikZ outputs in
`entropy_centered_surface_jobs/`.

To submit one distance directly:

```bash
sbatch --cpus-per-task=2 --mem=64G experiments/cluster_gexit/run_surface_gexit.sbatch experiments/cluster_gexit/surface_gexit_d21.py
```

To submit one add-on directly:

```bash
sbatch --cpus-per-task=20 --mem=64G experiments/cluster_gexit/run_surface_gexit.sbatch experiments/cluster_gexit/surface_gexit_d7_addon.py
```

To submit only the sharded large-distance jobs:

```bash
bash experiments/cluster_gexit/submit_surface_gexit_shards.sh
```

After every shard in an array has completed, merge the shards:

```bash
python experiments/cluster_gexit/merge_surface_gexit_shards.py --distance 15
python experiments/cluster_gexit/merge_surface_gexit_shards.py --distance 21
python experiments/cluster_gexit/merge_surface_gexit_shards.py --distance 45
```

To add shards to an existing completed main result instead of replacing it,
pass `--include-existing` to the merge command.

Outputs are written under:

```text
data/experiments/gexit_curves/entropy_centered_surface_jobs/
```

Set `PYTHON_BIN` if the cluster Python executable is not named `python`.
