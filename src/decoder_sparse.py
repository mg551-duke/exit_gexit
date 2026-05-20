from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np

from algorithm_sparse import (
    INT,
    ERASURE,
    SparseMatrix,
    structured_decode,
    build_hz_system_and_inactivate,
    hz_peel_only,
    syndrome,
)

@dataclass
class Mode5Stats:
    cycles: int = 0
    sd_calls: int = 0
    peel_calls: int = 0

    sd_helped_cycles: int = 0
    peel_helped_cycles: int = 0
    both_helped_cycles: int = 0
    neither_helped_cycles: int = 0
    extra_cycles_helped: int = 0  # cycles > 1 that reduce erasures


    solved: bool = False
    stop_reason: str = ""      # "solved" | "stalled" | "max_iters"
    erasures_start: int = 0
    erasures_end: int = 0

    total_stab_moves: int = 0  # sum stab_used across all SD calls


@dataclass
class DecodeResult:
    answer: np.ndarray
    correction: np.ndarray
    stabilizer_guesses: int
    inactivation_guesses: Optional[int]
    mode: int
    mode5_stats: Optional[Mode5Stats] = None

def adjusted_syndrome_for_known_bits(
    Hz: SparseMatrix,
    msg: np.ndarray,
    s: np.ndarray,
) -> np.ndarray:
    """
    Given original constraint Hz * x = s (GF(2)), and a partially-known x in msg
    (0/1/ERASURE), compute an equivalent RHS for the remaining unknowns after
    eliminating known variables:

        s_adj = s XOR (Hz_known * x_known)

    where "known" means msg[i] != ERASURE.

    This is necessary if msg contains known 1s (e.g. produced by Hz peeling),
    otherwise removing known columns would silently assume those bits are 0.
    """
    msg = np.asarray(msg, dtype=INT)
    s_adj = (np.asarray(s, dtype=INT) & 1).copy()

    # For each check row i: toggle RHS if an odd number of known-1 bits appear in that row.
    for i, row in enumerate(Hz.rows):
        parity = 0
        for col in row:
            v = msg[col]
            if v != ERASURE and (int(v) & 1):
                parity ^= 1
        s_adj[i] ^= parity

    return s_adj.astype(INT)




class SparseDecoder:
    """Sparse set-based decoder following the dense algorithm exactly."""

    def __init__(
        self,
        Hz: np.ndarray | SparseMatrix,
        Hx: np.ndarray | SparseMatrix,
        *,
        n_deg: int | None = None,
        guess_cap: int | None = None,
        dual_rule2: bool = True,
    ) -> None:
        self.Hz = Hz if isinstance(Hz, SparseMatrix) else SparseMatrix.from_numpy(Hz)
        self.Hx = Hx if isinstance(Hx, SparseMatrix) else SparseMatrix.from_numpy(Hx)
        self.n_deg = n_deg
        self.guess_cap = guess_cap
        self.dual_rule2 = dual_rule2

    def _hz_peel_with_adjusted_syndrome(self, msg: np.ndarray, s_orig: np.ndarray) -> np.ndarray:
        """
        Run Hz peeling but with the RHS adjusted for any already-known bits in msg.
        """
        s_adj = adjusted_syndrome_for_known_bits(self.Hz, msg, s_orig)
        return hz_peel_only(self.Hz, msg, s_adj)

    def _hz_inactivate_with_adjusted_syndrome(
        self, msg: np.ndarray, s_orig: np.ndarray
    ) -> Tuple[np.ndarray, int]:
        """
        Run Hz inactivation but with the RHS adjusted for any already-known bits in msg.
        """
        s_adj = adjusted_syndrome_for_known_bits(self.Hz, msg, s_orig)
        return build_hz_system_and_inactivate(
            self.Hz,
            msg,
            s_adj,
            guess_cap=self.guess_cap,
        )
    def _mode5_iterate_structured_then_peel(
        self,
        msg: np.ndarray,
        s_orig: np.ndarray,
        *,
        max_iters: int = 10_000,
        debug: bool = False,
        debug_every: int = 1,
    ) -> Tuple[np.ndarray, int, Mode5Stats]:
        msg_work = msg.copy()
        stab_total = 0

        stats = Mode5Stats()
        stats.erasures_start = int(np.sum(msg_work == ERASURE))

        for it in range(max_iters):
            er_before = int(np.sum(msg_work == ERASURE))
            if er_before == 0:
                stats.solved = True
                stats.stop_reason = "solved"
                break

            # one full alternation cycle
            stats.cycles += 1

            # --- structured decode ---
            stats.sd_calls += 1
            msg_sd, stab_used = structured_decode(
                msg_work.copy(), self.Hx, use_rule2=self.dual_rule2
            )
            stab_total += int(stab_used)
            stats.total_stab_moves = stab_total
            er_after_sd = int(np.sum(msg_sd == ERASURE))

            # --- peel ---
            stats.peel_calls += 1
            msg_peel = self._hz_peel_with_adjusted_syndrome(msg_sd, s_orig)
            er_after_peel = int(np.sum(msg_peel == ERASURE))

            # per-cycle contribution flags
            sd_helped = (er_after_sd < er_before)
            peel_helped = (er_after_peel < er_after_sd)

            if sd_helped:
                stats.sd_helped_cycles += 1
            if peel_helped:
                stats.peel_helped_cycles += 1
            if sd_helped and peel_helped:
                stats.both_helped_cycles += 1
            if (not sd_helped) and (not peel_helped):
                stats.neither_helped_cycles += 1

            filled_by_sd = er_before - er_after_sd
            filled_by_peel = er_after_sd - er_after_peel
            net_filled = er_before - er_after_peel
            if stats.cycles > 1 and (er_after_peel < er_before):
                stats.extra_cycles_helped += 1


            if debug and (it % debug_every == 0):
                print(
                    f"[mode5] it={it:4d}  "
                    f"er: {er_before:5d} -> {er_after_sd:5d} -> {er_after_peel:5d}  "
                    f"(+sd={filled_by_sd:+d}, +peel={filled_by_peel:+d}, net={net_filled:+d})  "
                    f"stab_used={int(stab_used):3d} stab_total={stab_total:5d}  "
                    f"cycles={stats.cycles} sd_calls={stats.sd_calls} peel_calls={stats.peel_calls}"
                )

            msg_work = msg_peel

            # stall = no net improvement after full cycle
            if er_after_peel >= er_before:
                stats.stop_reason = "stalled"
                break
        else:
            stats.stop_reason = "max_iters"

        stats.erasures_end = int(np.sum(msg_work == ERASURE))
        if stats.erasures_end == 0:
            stats.solved = True
            if stats.stop_reason == "":
                stats.stop_reason = "solved"

        return msg_work, stab_total, stats




    def decode(self, msg: np.ndarray, mode: int) -> DecodeResult:
        msg = np.asarray(msg, dtype=INT)
        s_orig, answer = syndrome(self.Hz, msg)

        if mode == 1:
            # peeling only (needs adjusted syndrome even here if msg could contain 1s,
            # but in your harness msg non-erased bits are 0 so it doesn't matter)
            corr = self._hz_peel_with_adjusted_syndrome(msg, s_orig)
            return DecodeResult(
                answer=answer,
                correction=corr,
                stabilizer_guesses=0,
                inactivation_guesses=None,
                mode=mode,
            )

        if mode == 2:
            # inactivation only
            final_corr, inact_g = self._hz_inactivate_with_adjusted_syndrome(msg, s_orig)
            return DecodeResult(
                answer=answer,
                correction=final_corr,
                stabilizer_guesses=0,
                inactivation_guesses=inact_g,
                mode=mode,
            )
        
        if mode == 6:
            msg_work = msg.copy()
            hard_guesses = 0

            while True:
                # Peel to exhaustion (with RHS adjusted for any known/guessed bits)
                msg_work = self._hz_peel_with_adjusted_syndrome(msg_work, s_orig)

                # Solved
                er_idxs = np.flatnonzero(msg_work == ERASURE)
                if er_idxs.size == 0:
                    return DecodeResult(
                        answer=answer,
                        correction=msg_work,
                        stabilizer_guesses=0,
                        inactivation_guesses=hard_guesses,  # reuse this field for "hard guesses"
                        mode=mode,
                    )

                guess_pos = int(er_idxs[0])  
                msg_work[guess_pos] = np.random.randint(0, 2, dtype=INT)
                hard_guesses += 1

        # Do one structured pass (Hx side)
        stab_msg, stab_used = structured_decode(
            msg.copy(), self.Hx, use_rule2=self.dual_rule2
        )

        if mode == 3:
            corr = self._hz_peel_with_adjusted_syndrome(stab_msg, s_orig)
            return DecodeResult(
                answer=answer,
                correction=corr,
                stabilizer_guesses=stab_used,
                inactivation_guesses=None,
                mode=mode,
            )

        if mode == 4:
            final_corr, inact_g = self._hz_inactivate_with_adjusted_syndrome(stab_msg, s_orig)
            return DecodeResult(
                answer=answer,
                correction=final_corr,
                stabilizer_guesses=stab_used,
                inactivation_guesses=inact_g,
                mode=mode,
            )
        
        if mode == 5:
            # set debug=False for large sims; turn on only when manually testing
            iter_msg, stab_used_total, m5 = self._mode5_iterate_structured_then_peel(
                msg.copy(), s_orig, debug=False, debug_every=1
            )

            # If fully solved by alternation, skip inactivation
            if not np.any(iter_msg == ERASURE):
                return DecodeResult(
                    answer=answer,
                    correction=iter_msg,
                    stabilizer_guesses=stab_used_total,
                    inactivation_guesses=0,
                    mode=mode,
                    mode5_stats=m5,
                )

            final_corr, inact_g = self._hz_inactivate_with_adjusted_syndrome(iter_msg, s_orig)
            return DecodeResult(
                answer=answer,
                correction=final_corr,
                stabilizer_guesses=stab_used_total,
                inactivation_guesses=inact_g,
                mode=mode,
                mode5_stats=m5,
            )
        

        if mode == 7:
            msg_work = msg.copy()
            hard_guesses = 0

            while True:
                # Peel to exhaustion (with RHS adjusted for any known/guessed bits)
                msg_work = self._hz_peel_with_adjusted_syndrome(msg_work, s_orig)

                # Solved
                er_idxs = np.flatnonzero(msg_work == ERASURE)
                if er_idxs.size == 0:
                    return DecodeResult(
                        answer=answer,
                        correction=msg_work,
                        stabilizer_guesses=0,
                        inactivation_guesses=hard_guesses,  # reuse this field for "hard guesses"
                        mode=mode,
                    )

                guess_pos = int(er_idxs[0])  
                msg_work[guess_pos] = np.random.randint(0, 2, dtype=INT)
                hard_guesses += 1


        raise ValueError(f"Unsupported mode: {mode}")
