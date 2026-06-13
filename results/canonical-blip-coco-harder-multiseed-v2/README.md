# Harder multi-seed BLIP evaluation v2

This evaluation consumes the real 3 × 256 held-out confidence records and OOD
metrics produced by Vision Language Workbench on an A100. ECE is computed with
the repository's existing ten-bin calibration definition; NLL is correctness
NLL for the top-1 image-to-text decision.

## Integrated scorecard

| Variant | Params % | Clean R@1 | Mean OOD R@1 | Retention | Worst retention | ECE | NLL | CKA | Probe |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Base | 0.0000 | 0.85612 | 0.75303 | 0.87951 | 0.41030 | 0.72612 | 1.73744 | 1.000000 | 0.78646 |
| LoRA r=8 | 0.1318 | 0.84766 | 0.74966 | 0.88431 | 0.41134 | 0.67632 | 1.52217 | 0.999477 | 0.78646 |
| LoRA r=16 | 0.2636 | 0.84701 | 0.74556 | 0.88019 | 0.41620 | 0.64459 | 1.39361 | 0.998371 | 0.78646 |
| LoRA r=8 + hard negative | 0.1318 | 0.84766 | 0.75050 | 0.88532 | 0.41284 | 0.68054 | 1.53745 | 0.999543 | 0.78646 |

## Calibration stability

Paired deltas are current minus reference, reported as mean ± Student-t 95% CI
half-width over three seeds.

| Contrast | ECE delta | NLL delta | Interpretation |
|---|---:|---:|---|
| r8 − Base | -0.04980 ± 0.02333 | -0.21526 ± 0.05677 | Improved on all three seeds; CI excludes zero |
| r16 − Base | -0.08153 ± 0.03272 | -0.34383 ± 0.09898 | Improved on all three seeds; CI excludes zero |
| r8+HN − r8 | +0.00422 ± 0.00860 | +0.01528 ± 0.03125 | Slightly worse on all three seeds; inconclusive at n=3 |

The first canonical study's LoRA calibration advantage reproduces. Relative
retention for r8 also improves on all three seeds, but its paired 95% CI crosses
zero. r16 retention changes direction across seeds. Hard-negative training no
longer recovers clean performance; it ties r8 exactly and makes only a tiny OOD
gain while slightly worsening calibration.

Occlusion severity 4–5 separates absolute accuracy from relative stability.
Base has the best severity-4 R@1 (0.44336); r16 has the best severity-5 R@1 and
retention (0.35286 / 0.41620). r8 has the best severity-4 retention (0.51829).
All high-severity confidence intervals overlap, so these rankings are
descriptive rather than significant.

All four variants remain on the Pareto frontier under the declared clean, OOD,
calibration, and parameter objectives: no single variant dominates every
trade-off. More seeds are warranted for OOD retention, hard-negative effects,
and severity-4/5 occlusion; the calibration direction is already consistent.

## Contents

- `calibration-by-seed.json` and aggregate summary JSON/CSV.
- `calibration-paired-comparisons.json`.
- `retrieval-paired-comparisons.json` and `occlusion-severity-4-5.json`.
- `integrated-scorecard.json` / `.csv`.
- `decision-analysis.json` / `.md` using the existing Pareto implementation.
