from __future__ import annotations

"""Generate standard planar surface-code `.npz` files compatible with this repo.

Example:
    python scripts/generate_surface_code.py 15 25

The output files are written to `codes/` by default and are named
`surface<d>_HxHzLxLz.npz`.
"""

import argparse
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]


def primary_index(row: int, col: int, d: int) -> int:
    """Index of a qubit in the d-by-d primary grid."""
    return row * d + col


def interior_index(row: int, col: int, d: int) -> int:
    """Index of a qubit in the (d-1)-by-(d-1) interior grid."""
    return d * d + row * (d - 1) + col


def row_from_support(support: list[int], n: int) -> np.ndarray:
    """Build one binary parity-check row from a list of qubit indices."""
    row = np.zeros(n, dtype=np.uint8)
    row[support] = 1
    return row


def rank_gf2(matrix: np.ndarray) -> int:
    """Dense Gaussian elimination over GF(2), used only for validation."""
    matrix = (matrix.copy() % 2).astype(np.uint8)
    if matrix.size == 0:
        return 0

    m, n = matrix.shape
    rank = 0
    for col in range(n):
        pivots = np.flatnonzero(matrix[rank:, col])
        if pivots.size == 0:
            continue
        pivot = int(pivots[0] + rank)
        if pivot != rank:
            matrix[[rank, pivot]] = matrix[[pivot, rank]]
        for row in range(m):
            if row != rank and matrix[row, col]:
                matrix[row] ^= matrix[rank]
        rank += 1
        if rank == m:
            break
    return rank


def build_planar_surface_code(d: int) -> dict[str, np.ndarray]:
    """
    Build the same standard planar surface-code layout used by the existing files.

    Column layout:
      1. d-by-d primary qubits, row-major: indices 0 through d^2 - 1.
      2. (d-1)-by-(d-1) interior qubits, row-major: remaining indices.

    Check layout:
      Hx has one row for every horizontal bond in the primary grid. The check
      touches the two primary qubits on that bond and the adjacent interior
      qubits above/below it when those interior qubits exist.

      Hz has one row for every vertical bond in the primary grid. The check
      touches the two primary qubits on that bond and the adjacent interior
      qubits left/right of it when those interior qubits exist.

    This reproduces the existing surface5/7/9/11 convention, including row
    ordering and logical operators:
      Lx = left column of the primary grid.
      Lz = top row of the primary grid.
    """
    if d < 2:
        raise ValueError("distance d must be at least 2")

    n = d * d + (d - 1) * (d - 1)
    hx_rows: list[np.ndarray] = []
    hz_rows: list[np.ndarray] = []

    # X checks: horizontal bonds between primary qubits.
    for row in range(d):
        for col in range(d - 1):
            support = [
                primary_index(row, col, d),
                primary_index(row, col + 1, d),
            ]
            if row > 0:
                support.append(interior_index(row - 1, col, d))
            if row < d - 1:
                support.append(interior_index(row, col, d))
            hx_rows.append(row_from_support(support, n))

    # Z checks: vertical bonds between primary qubits.
    for row in range(d - 1):
        for col in range(d):
            support = [
                primary_index(row, col, d),
                primary_index(row + 1, col, d),
            ]
            if col > 0:
                support.append(interior_index(row, col - 1, d))
            if col < d - 1:
                support.append(interior_index(row, col, d))
            hz_rows.append(row_from_support(support, n))

    hx = np.vstack(hx_rows).astype(np.uint8)
    hz = np.vstack(hz_rows).astype(np.uint8)

    lx = np.zeros((1, n), dtype=np.uint8)
    lz = np.zeros((1, n), dtype=np.uint8)
    for row in range(d):
        lx[0, primary_index(row, 0, d)] = 1
    for col in range(d):
        lz[0, primary_index(0, col, d)] = 1

    return {"Hx": hx, "Hz": hz, "Lx": lx, "Lz": lz}


# Backward-compatible alias for any local scratch commands that used the older
# name before the terminology was corrected.
build_rotated_surface_code = build_planar_surface_code


def validate_code(code: dict[str, np.ndarray], d: int) -> None:
    """Check CSS commutation, logical anticommutation, and k=1."""
    hx = code["Hx"]
    hz = code["Hz"]
    lx = code["Lx"]
    lz = code["Lz"]
    n = hx.shape[1]

    if hx.shape != (d * (d - 1), n):
        raise ValueError(f"unexpected Hx shape: {hx.shape}")
    if hz.shape != (d * (d - 1), n):
        raise ValueError(f"unexpected Hz shape: {hz.shape}")
    if lx.shape != (1, n) or lz.shape != (1, n):
        raise ValueError(f"unexpected logical shapes: Lx={lx.shape}, Lz={lz.shape}")

    if np.any((hx @ hz.T) % 2):
        raise ValueError("Hx and Hz do not commute")
    if np.any((hz @ lx.T) % 2):
        raise ValueError("Lx is not in the Hz nullspace")
    if np.any((hx @ lz.T) % 2):
        raise ValueError("Lz is not in the Hx nullspace")
    if int((lx @ lz.T)[0, 0] % 2) != 1:
        raise ValueError("Lx and Lz should anticommute")

    rank_hx = rank_gf2(hx)
    rank_hz = rank_gf2(hz)
    k = n - rank_hx - rank_hz
    if k != 1:
        raise ValueError(f"expected k=1, got k={k}")


def write_code(d: int, out_dir: Path, *, overwrite: bool = False) -> Path:
    """Generate and write one surface code npz file."""
    code = build_planar_surface_code(d)
    validate_code(code, d)

    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"surface{d}_HxHzLxLz.npz"
    if path.exists() and not overwrite:
        raise FileExistsError(f"{path} already exists; pass --overwrite to replace it")

    np.savez(path, Hx=code["Hx"], Hz=code["Hz"], Lx=code["Lx"], Lz=code["Lz"])
    return path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate standard planar surface-code Hx/Hz/Lx/Lz npz files."
    )
    parser.add_argument("distances", type=int, nargs="+", help="Code distances to generate")
    parser.add_argument("--out-dir", type=Path, default=ROOT / "codes")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    for d in args.distances:
        path = write_code(d, args.out_dir, overwrite=args.overwrite)
        code = np.load(path)
        print(
            f"wrote {path} "
            f"Hx={code['Hx'].shape} Hz={code['Hz'].shape} "
            f"Lx={code['Lx'].shape} Lz={code['Lz'].shape}"
        )


if __name__ == "__main__":
    main()
