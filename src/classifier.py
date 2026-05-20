# classifier.py  (DCC driver / harness)

from __future__ import annotations
import argparse
import json
import os
import time
from pathlib import Path
from typing import Optional

import numpy as np

# non-interactive backend for DCC
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ── your existing modules ────────────────────────────────────────────
from algorithm_sparse import ERASURE
from decoder_sparse import SparseDecoder
# ─────────────────────────────────────────────────────────────────────


def classify_from_decoder_output(
    out,                  # DecodeResult
    Hz_dense: np.ndarray,
    Lz: np.ndarray | None = None,
) -> str:
    """
    EXACTLY like your old script.
    """
    corr = out.correction
    orig = out.answer

    # 1) if decoder still has erasures -> nonconv
    if np.any(corr == ERASURE):
        return "nonconv"

    # 2) residual
    residual = (corr % 2 + orig % 2) % 2

    # 3) Hz check
    if not np.all((Hz_dense @ residual) % 2 == 0):
        return "nonconv"

    # 4) exact recovery
    if residual.sum() == 0:
        return "same"

    # 5) logical vs true fail
    if Lz is not None and Lz.size:
        if np.all((Lz @ residual) % 2 == 0):
            return "degen"
        else:
            return "logic"

    return "logic"


def run_once(
    decoder: SparseDecoder,
    Hz_dense: np.ndarray,
    p: float,
    seed: int,
    mode: int,
    Lz: Optional[np.ndarray] = None,
):
    rng = np.random.default_rng(seed)
    n = Hz_dense.shape[1]

    msg = np.zeros(n, dtype=np.int8)
    erased_mask = rng.random(n) < p
    msg[erased_mask] = ERASURE

    out = decoder.decode(msg, mode)

    cls = classify_from_decoder_output(out, Hz_dense, Lz=Lz)

    inact_guesses = out.inactivation_guesses or 0
    stab_guesses = out.stabilizer_guesses or 0

    return cls, inact_guesses, stab_guesses, out.mode5_stats


def sim(
    decoder: SparseDecoder,
    Hz_dense: np.ndarray,
    ps: list[float],
    runs: int,
    mode: int,
    Lz: Optional[np.ndarray] = None,
    seed_offset: int = 0,
):
    results = []
    guess_details: dict[float, dict[str, list]] = {}

    for p in ps:
        counts = dict(
            same=0,
            degen=0,
            logic=0,
            nonconv=0,
            inact_guesses_total=0,
            stab_guesses_total=0,
            total_time=0.0,
        )

        per_run_inact: list[int] = []
        per_run_stab: list[int] = []
        per_run_cls: list[str] = []

        # NEW: mode5 aggregation bucket (only for mode==5)
        mode5_agg = None
        if mode == 5:
            mode5_agg = dict(
                cycles_total=0,
                sd_calls_total=0,
                peel_calls_total=0,
                sd_helped_cycles_total=0,
                peel_helped_cycles_total=0,
                both_helped_cycles_total=0,
                neither_helped_cycles_total=0,
                erasures_start_total=0,
                erasures_end_total=0,
                total_stab_moves_total=0,
                stop_solved=0,
                stop_stalled=0,
                stop_max_iters=0,
                cycles_hist={},  # {cycles: count}
                extra_cycles_helped_total=0,
                runs_with_extra_help=0,
            )

        t0 = time.perf_counter()
        for i in range(runs):
            t1 = time.perf_counter()

            cls, inact_g, stab_g, m5 = run_once(
                decoder, Hz_dense, p, seed=i + seed_offset, mode=mode, Lz=Lz
            )

            counts[cls] += 1
            counts["inact_guesses_total"] += inact_g
            counts["stab_guesses_total"] += stab_g
            counts["total_time"] += time.perf_counter() - t1

            per_run_inact.append(inact_g)
            per_run_stab.append(stab_g)
            per_run_cls.append(cls)

            # NEW: accumulate mode5 stats
            if mode == 5 and m5 is not None and mode5_agg is not None:
                mode5_agg["cycles_total"] += m5.cycles
                mode5_agg["sd_calls_total"] += m5.sd_calls
                mode5_agg["peel_calls_total"] += m5.peel_calls
                mode5_agg["sd_helped_cycles_total"] += m5.sd_helped_cycles
                mode5_agg["peel_helped_cycles_total"] += m5.peel_helped_cycles
                mode5_agg["both_helped_cycles_total"] += m5.both_helped_cycles
                mode5_agg["neither_helped_cycles_total"] += m5.neither_helped_cycles
                mode5_agg["erasures_start_total"] += m5.erasures_start
                mode5_agg["erasures_end_total"] += m5.erasures_end
                mode5_agg["total_stab_moves_total"] += m5.total_stab_moves

                h = mode5_agg["cycles_hist"]
                h[m5.cycles] = h.get(m5.cycles, 0) + 1

                if m5.stop_reason == "solved":
                    mode5_agg["stop_solved"] += 1
                elif m5.stop_reason == "stalled":
                    mode5_agg["stop_stalled"] += 1
                elif m5.stop_reason == "max_iters":
                    mode5_agg["stop_max_iters"] += 1
                mode5_agg["extra_cycles_helped_total"] += m5.extra_cycles_helped
                if m5.extra_cycles_helped > 0:
                    mode5_agg["runs_with_extra_help"] += 1


        t_end = time.perf_counter()

        counts["runs"] = runs
        counts["avg_time_per_run"] = counts["total_time"] / runs
        counts["total_time_all_runs"] = t_end - t0

        # attach mode5_agg into results so it goes into JSON
        if mode == 5 and mode5_agg is not None:
            counts["mode5"] = mode5_agg

        results.append((p, counts))
        guess_details[p] = {"inact": per_run_inact, "stab": per_run_stab, "cls": per_run_cls}

    return results, guess_details



# ─────────────────────────────────────────────────────────────
# Histogram helpers
# ─────────────────────────────────────────────────────────────

def plot_inactivation_histograms(
    sparse_guess_details: dict[int, dict[float, dict[str, list]]],
    ps: list[float],
    out_prefix: str = "inact_hist_stacked",
):
    """
    For each p in ps, make one figure with up to two subplots:
      - mode 2  stacked success/failure by # inactivation guesses
      - mode 4  stacked success/failure by # inactivation guesses

    Success = 'same' or 'degen'
    Failure = 'logic' or 'nonconv'
    """
    success_labels = {"same", "degen"}

    have2 = 2 in sparse_guess_details
    have4 = 4 in sparse_guess_details
    if not (have2 or have4):
        return

    def build_counts(data):
        inact = data["inact"]
        cls = data["cls"]
        if not inact:
            return np.array([0]), np.array([0]), np.array([0])

        max_g = max(inact)
        success_counts = np.zeros(max_g + 1, dtype=int)
        fail_counts = np.zeros(max_g + 1, dtype=int)

        for g, c in zip(inact, cls):
            if c in success_labels:
                success_counts[g] += 1
            else:
                fail_counts[g] += 1

        x = np.arange(max_g + 1)
        return x, success_counts, fail_counts

    for p in ps:
        # figure out how many subplots we actually need for this p
        modes_for_p = []
        if have2 and p in sparse_guess_details[2]:
            modes_for_p.append(2)
        if have4 and p in sparse_guess_details[4]:
            modes_for_p.append(4)

        if not modes_for_p:
            continue

        n_sub = len(modes_for_p)
        fig, axes = plt.subplots(1, n_sub, figsize=(5 * n_sub, 4), sharey=True)
        if n_sub == 1:
            axes = [axes]

        for ax, mode in zip(axes, modes_for_p):
            data = sparse_guess_details[mode][p]
            x, succ, fail = build_counts(data)

            ax.bar(x, succ, label="success")
            ax.bar(x, fail, bottom=succ, label="failure")
            ax.set_title(f"Mode {mode} – inactivation guesses (p={p:.2f})")
            ax.set_xlabel("# inactivation guesses")
            ax.set_ylabel("run count")
            ax.legend()

        fig.suptitle(f"Inactivation guesses: success vs failure (p={p:.2f})")
        fig.tight_layout()

        fig.savefig(f"{out_prefix}_p{p:.2f}.png", dpi=150)
        plt.close(fig)


def plot_mode4_stabilizer_histograms(
    sparse_guess_details: dict[int, dict[float, dict[str, list]]],
    ps: list[float],
    out_prefix: str = "stab_hist",
):
    """
    For each p in ps, plot histogram of stabilizer moves in mode 4.
    Uses sparse_guess_details[4][p]["stab"].
    """
    if 4 not in sparse_guess_details:
        return

    for p in ps:
        if p not in sparse_guess_details[4]:
            continue

        stab_mode4 = sparse_guess_details[4][p]["stab"]
        if not stab_mode4:
            continue

        max_guess = max(stab_mode4)
        bins = np.arange(0, max_guess + 2)

        plt.figure(figsize=(5, 4))
        plt.hist(stab_mode4, bins=bins, align="left", rwidth=0.8)
        plt.title(f"Mode 4 – stabilizer moves (p={p:.2f})")
        plt.xlabel("# stabilizer moves")
        plt.ylabel("count")
        plt.tight_layout()

        plt.savefig(f"{out_prefix}_p{p:.2f}.png", dpi=150)
        plt.close()


# ─────────────────────────────────────────────────────────────
# main / CLI
# ─────────────────────────────────────────────────────────────

def main(p, runs):
    ap = argparse.ArgumentParser(description="Sparse decoder simulation (DCC harness)")
    ap.add_argument("--npz", default="BB_n360_k12_l30_m6_Ax9_y1_y2_By3_x25_x26.npz",
                    help="Path to npz with Hx, Hz, (opt) Lz")
    ap.add_argument("--ps", type=float, nargs="+", default=p,
                    help="Erasure probabilities")
    ap.add_argument("--runs", type=int, default=runs, help="Runs per p")
    ap.add_argument("--modes", type=int, nargs="+", default=[3, 4, 5],
                    help="Decoder modes to run")
    ap.add_argument("--guess-cap", type=int, default=None,
                    help="Guess cap for inactivation (None = unlimited)")
    ap.add_argument("--seed-offset", type=int, default=0,
                    help="Global seed offset (use SLURM array offset)")
    ap.add_argument("--out", default=None,
                    help="Output JSON path (default: results/auto.json)")
    ap.add_argument("--results-dir", default="results",
                    help="Directory to store JSON if --out not given")
    ap.add_argument("--tag", default=None,
                    help="Optional tag to include in filename")
    ap.add_argument("--rule1-only", action="store_true",
                    help="Use only known-column weight-2 dual peeling; intended for surface-code ML tests")
    args = ap.parse_args()

    # Load matrices once
    npz = np.load(args.npz, allow_pickle=True)
    Hz = (npz["Hz"] % 2).astype(np.uint8)
    Hx = (npz["Hx"] % 2).astype(np.uint8)
    Lz = (
        (npz["Lz"] % 2).astype(np.uint8)
        if "Lz" in npz
        else np.empty((0, Hz.shape[1]), dtype=np.uint8)
    )

    # Build decoder once (sparse conversion stays the same)
    sparse_dec = SparseDecoder(
        Hz,
        Hx,
        guess_cap=args.guess_cap,
        dual_rule2=not args.rule1_only,
    )

    # Run all modes
    sparse_summary: dict[int, list[tuple[float, dict]]] = {}
    sparse_guess_details: dict[int, dict[float, dict[str, list]]] = {}

    for mode in args.modes:
        res, guess_det = sim(
            sparse_dec,
            Hz,
            args.ps,
            args.runs,
            mode,
            Lz=Lz,
            seed_offset=args.seed_offset,
        )
        sparse_summary[mode] = res
        sparse_guess_details[mode] = guess_det

    # Pretty print (identical summary format to your script)
    print("\n================ SPARSE ================")
    for mode in args.modes:
        print(f"\n=== Sparse mode: {mode} ===")
        for p, c in sparse_summary[mode]:
            total_fail = c["logic"] + c["nonconv"]
            success = c["same"] + c["degen"]
            err_rate = total_fail / c["runs"]
            avg_inact = c["inact_guesses_total"] / c["runs"]
            avg_stab = c["stab_guesses_total"] / c["runs"]
            avg_time = c.get("avg_time_per_run", 0.0)
            print(
                f"p={p:.2f}  runs={c['runs']:4d}  "
                f"same={c['same']:4d}  degen={c['degen']:4d}  "
                f"logic={c['logic']:4d}  nonconv={c['nonconv']:4d}  "
                f"success={success:4d}  "
                f"error_rate={err_rate:.4f}  "
                f"avg_inact_guesses={avg_inact:.2f}  "
                f"avg_stabilizer_moves={avg_stab:.2f}  "
                f"avg_time={avg_time*1000:.2f} ms"
            )

            if mode == 5 and "mode5" in c:
                m5 = c["mode5"]
                runs = c["runs"]

                avg_cycles = m5["cycles_total"] / runs
                pct_cycles_gt1 = 100.0 * (runs - m5["cycles_hist"].get(1, 0)) / runs

                avg_sd_helped = m5["sd_helped_cycles_total"] / runs
                avg_peel_helped = m5["peel_helped_cycles_total"] / runs
                avg_neither = m5["neither_helped_cycles_total"] / runs

                avg_er_start = m5["erasures_start_total"] / runs
                avg_er_end = m5["erasures_end_total"] / runs

                avg_extra_help = m5["extra_cycles_helped_total"] / runs
                pct_runs_extra_help = 100.0 * m5["runs_with_extra_help"] / runs


                print(
                f"  [mode5 overall] avg_cycles={avg_cycles:.3f}  pct(cycles>1)={pct_cycles_gt1:.1f}%  "
                f"avg_sd_helped_cycles={avg_sd_helped:.3f}  avg_peel_helped_cycles={avg_peel_helped:.3f}  "
                f"avg_neither_cycles={avg_neither:.3f}  "
                f"stop: solved={m5['stop_solved']} stalled={m5['stop_stalled']} max_iters={m5['stop_max_iters']}  "
                f"avg_erasures: start={avg_er_start:.2f} end={avg_er_end:.2f}  "
                f"extra-cycle progress: avg_extra_cycles_helped={avg_extra_help:.3f} "
                f"pct_runs_with_extra_help={pct_runs_extra_help:.1f}%"
                )


                # short histogram preview
                hist_items = sorted(m5["cycles_hist"].items())
                hist_str = " ".join(f"{k}:{v}" for k, v in hist_items[:12])
                if len(hist_items) > 12:
                    hist_str += " ..."
                print(f"  [mode5 cycles_hist] {hist_str}")
                


    # Persist JSON (one file per array task is fine)
    ts = time.strftime("%Y%m%d-%H%M%S")
    tag = f"_{args.tag}" if args.tag else ""
    default_name = (
        f"sparse_modes{''.join(map(str, args.modes))}"
        f"_runs{args.runs}_seedoff{args.seed_offset}{tag}_{ts}.json"
    )
    out_path = args.out or str(Path(args.results_dir) / default_name)
    os.makedirs(Path(out_path).parent, exist_ok=True)

    # make JSON-able (UNCHANGED)
    serializable = {
        "npz": args.npz,
        "ps": args.ps,
        "runs": args.runs,
        "modes": args.modes,
        "guess_cap": args.guess_cap,
        "seed_offset": args.seed_offset,
        "rule1_only": args.rule1_only,
        "results": {
            int(mode): [
                {"p": float(p), **counts} for (p, counts) in sparse_summary[mode]
            ]
            for mode in args.modes
        },
    }
    with open(out_path, "w") as f:
        json.dump(serializable, f, indent=2)
    print(f"\nSaved JSON -> {out_path}")

    # ── NEW: generate histograms for modes 2 and 4 ───────────────
    # Put all PNGs in a "histograms" subfolder next to the JSON.
    out_path = Path(out_path)
    hist_dir = out_path.parent / "histograms"
    os.makedirs(hist_dir, exist_ok=True)

    # base name without extension, e.g. "sparse_modes24_runs20000_seedoff0_..."
    base_name = out_path.stem

    # prefixes inside histograms/
    inact_prefix = str(hist_dir / f"{base_name}_inact")
    stab_prefix = str(hist_dir / f"{base_name}_stab")

    # inactivation histograms (modes 2 and/or 4 if present)
    if any(m in sparse_guess_details for m in (2, 4)):
        plot_inactivation_histograms(
            sparse_guess_details,
            args.ps,
            out_prefix=inact_prefix,
        )
        print(f"Saved inactivation histograms with prefix {inact_prefix}_p*.png")

    # stabilizer-move histograms (mode 4 only)
    if 4 in sparse_guess_details:
        plot_mode4_stabilizer_histograms(
            sparse_guess_details,
            args.ps,
            out_prefix=stab_prefix,
        )
        print(f"Saved mode-4 stabilizer histograms with prefix {stab_prefix}_p*.png")



if __name__ == "__main__":
    p = [0.24, 0.3, 0.1, 0.38, 0.42, 0.46, 0.48]
    runs = 5
    main(p, runs)
