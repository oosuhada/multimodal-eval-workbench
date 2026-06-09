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
failure count is small (2 Base, 3 r=8, 3 r=16), so the clusters are exploratory
rather than population-level claims. No OOD/corruption result is included in
this directory.
