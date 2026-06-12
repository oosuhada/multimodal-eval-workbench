# Canonical BLIP/COCO evaluation integration

These reports were computed with the existing calibration and automatic
failure-slice implementations from 80 real COCO 2017 held-out predictions per
variant. Source model outputs are from Vision Language Workbench commit
`cebdb46dd2b2cb3871644ea83c2a26cda5e93264` and its
`results/canonical-blip-coco-small-v1` directory.

Calibration records represent top-1 image-to-text retrieval confidence and
correctness. Consequently, `correctness_nll` is Bernoulli NLL for the top-1
decision; `retrieval_nll` in `evaluation-comparison.json` is the separate
80-way target-caption NLL produced by the model experiment.

Failure slicing uses the matching 256-dimensional fused representation. The
failure count is small (2 Base, 3 r=8, 3 r=16, 2 r=8 + hard negative), so the
three combined clusters are exploratory rather than population-level claims.

`integrated-scorecard.{json,csv,md}` joins trainable-parameter cost, clean and
15-condition OOD retrieval, top-1 calibration, CKA, cosine drift, geometry, and
linear-probe results. `ood/` stores the exact-checkpoint summary and provenance
copied from Vision Language Workbench commit
`984106906c1f443057a4d5c0d77e6766bcce6f58`.

## Decision analysis

`decision-analysis.{json,md}` treats clean R@1, absolute OOD R@1, mean OOD
retention, worst-case retention, ECE, correctness NLL, and trainable parameter
cost as explicit competing objectives. All four variants remain on the Pareto
frontier, so this study does not support a single universal winner.

- Base wins clean R@1 (`0.95625`) and absolute mean OOD R@1 (`0.91625`).
- LoRA r=8 wins mean OOD retention (`0.97333`).
- LoRA r=16 wins worst-case retention (`0.80795`), ECE (`0.59122`), and
  correctness NLL (`1.00461`).
- LoRA r=8 + hard negative uses the same `0.1318%` trainable-parameter budget
  as ordinary r=8 while recovering clean R@1 from `0.93750` to `0.95000` and
  absolute OOD R@1 from `0.91250` to `0.91542`.

The representation probe is saturated (`1.0` for every variant), R@5/R@10 are
also saturated, and all CKA values remain above `0.99985`. The next study should
therefore increase discriminative difficulty rather than add more orchestration:
use a larger held-out retrieval pool, substantially more confusable hard
negatives, and an occlusion-focused shift set. Multi-seed repeats are also
needed before treating the observed calibration/retention improvements as
population-level effects.
