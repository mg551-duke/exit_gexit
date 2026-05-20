# EXIT/GEXIT Diagnostics Roadmap

## Positioning

The project is best framed as a quantum EXIT/GEXIT diagnostic program. Pure
erasure is the exact solvable anchor: the one-sided CSS class entropy
`H(C_X | E,S)` is a rank quantity, its endpoint at `p=1` is `k` bits, and the
component-normalized derivative area is `k/n`.

Beyond erasure, the same object becomes a posterior or free-energy diagnostic:

```text
H(C_X | Y,S)
```

where `Y` is the full channel observation. For mixed erasure-Pauli noise, `Y`
contains erasure locations and the syndrome. For Pauli-only channels, `Y` is
primarily the channel model plus syndrome. For circuit-level noise, `Y` should
eventually become the detector-event observation.

## Literature Map

- Classical EXIT and erasure area theorem: Ashikhmin, Kramer, and ten Brink,
  "Extrinsic Information Transfer Functions: Model and Erasure Channel
  Properties", IEEE TIT 2004:
  <https://portal.fis.tum.de/en/publications/extrinsic-information-transfer-functions-model-and-erasure-channe/>
- Generalized area/GEXIT viewpoint for non-erasure channels:
  <https://web.stanford.edu/~montanar/RESEARCH/FILEPAP/gatpap.pdf>
- General-channel coding context: Richardson and Urbanke, Modern Coding
  Theory, especially BMS and general-channel chapters:
  <https://www.cambridge.org/core/books/modern-coding-theory/general-channels/90AAD56273CD6540BCDE97663146F487>
- Quantum erasure capacity baseline:
  <https://journals.aps.org/prl/abstract/10.1103/PhysRevLett.78.3217>
- Low-density stabilizer bounds on the quantum erasure channel:
  <https://arxiv.org/abs/1205.7036>
- Surface-code ML erasure decoder:
  <https://journals.aps.org/prresearch/abstract/10.1103/PhysRevResearch.2.033042>
- HGP erasure decoder:
  <https://quantum-journal.org/papers/q-2024-08-27-1450/>
- Decoder baselines beyond erasure: BP-SI, BPGD, BP+LSD, and degenerate
  erasure decoding:
  <https://arxiv.org/abs/2205.06125>,
  <https://arxiv.org/abs/2411.08177>,
  <https://pmc.ncbi.nlm.nih.gov/articles/PMC12405505/>,
  <https://www.nature.com/articles/s41534-026-01212-3>
- Bivariate-bicycle/QLDPC motivation:
  <https://www.nature.com/articles/s41586-024-07107-7>

## Research Phases

1. Exact erasure foundation.
   Keep `H(C_X | E,S)` as the theorem-facing quantity. Report area `k/n`,
   peak location, transition width, and exact-vs-peeling gaps. Surface-code
   ML comparisons should use rule 1 only.

2. Mixed erasure plus Pauli.
   Study erasure probability `p_e` and component Pauli probability `p_P`.
   The current executable diagnostic is
   `experiments/beyond_erasure_diagnostics.py`, which estimates
   `H(C_X | Y,S)` or `H(C_Z | Y,S)` for one CSS component.

3. Biased CSS Pauli channels.
   Set `p_e=0` and sweep different X/Z component error rates. Plot
   `H(C_X | Y,S)` and `H(C_Z | Y,S)` separately before attempting a coupled
   depolarizing analysis.

4. Depolarizing noise.
   Treat this as a harder posterior problem because Y errors couple the CSS
   components. Use exact enumeration only for tiny codes and decoder-list
   approximations for larger codes. Claims here should be diagnostic rather
   than theorem-level until exact posterior computation is available.

5. Circuit-level noise.
   Replace qubit error classes by detector-event fault classes. Compare MWPM,
   BP-OSD/LSD, BPGD, and stabilizer-inactivation style post-processing through
   entropy of logical fault classes conditioned on detector syndrome.

## Experiment Priorities

- Channel interpolation heatmaps over `(p_e, p_P)` for
  `H(C_X | Y,S)/k`.
- Degeneracy gain curves:
  `H(x | Y,S) - H(C_X | Y,S)`.
- Decoder approximation gaps: exact/list entropy vs peeling, BP-OSD, BP-SI,
  BPGD, or BP+LSD outputs on the same samples.
- Bias asymmetry: compare X-side and Z-side curves under biased CSS Pauli
  noise.
- Finite-size split: use large codes for exact erasure; use exact small-code
  and list-restricted larger-code diagnostics beyond erasure.
- Representation sensitivity: exact entropy should be invariant under
  generator changes; algorithmic curves can reveal basis dependence.

## Reporting Rules

- Label pure erasure rank results as exact.
- Label non-erasure rows as exact only when every sample used
  `exact_affine_enumeration`.
- Label MCMC/list rows as list-restricted estimates; do not use them as ML
  theorem evidence.
- Keep the paper-facing default one-sided for now:
  `H(C_X | E,S)` and, beyond erasure, `H(C_X | Y,S)`.
