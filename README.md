# Multimodal Eval Workbench

Multimodal Eval Workbench keeps the full lmms-eval task, model-adapter, and
evaluation engine available while adding a compact suite layer for repeatable
multimodal benchmark runs.

This project does not reimplement the evaluator. A suite manifest is translated
directly into the existing `lmms_eval` CLI so the upstream benchmark ecosystem
remains the execution engine.

## What is added

- YAML/JSON benchmark suite manifests.
- One command plan for multiple image/video/audio/text tasks.
- Deterministic suite fingerprints.
- A lightweight run ledger and result summary layer.
- Filesystem task catalog search without importing heavyweight models.
- Calibration, uncertainty, and selective-risk analysis.

## Quick start

```yaml
name: compact-multimodal-smoke
model: qwen2_5_vl
model_args:
  pretrained: Qwen/Qwen2.5-VL-3B-Instruct
tasks:
  - mme
  - textvqa_val_lite
  - videomme
batch_size: 1
limit: 8
output_path: outputs/compact-multimodal-smoke
```

Render the exact lmms-eval command without loading a model:

```bash
mm-eval-workbench plan suites/compact-multimodal-smoke.yaml
```

Search and validate the bundled benchmark catalog without initializing Torch or
loading model weights:

```bash
mm-eval-workbench catalog videomme
mm-eval-workbench validate suites/compact-multimodal-smoke.yaml
```

Run the same suite through the original evaluator and keep a compact provenance
ledger outside Git history:

```bash
mm-eval-workbench run suites/compact-multimodal-smoke.yaml
mm-eval-workbench history --limit 10
```

Turn an existing lmms-eval result JSON into a small task/metric summary:

```bash
mm-eval-workbench summarize outputs/result.json

# build one suite from reusable image/video task fragments
mm-eval-workbench compose suites/compact-composed.yaml

# optionally persist the resolved suite as a normal manifest
mm-eval-workbench compose suites/compact-composed.yaml --write artifacts/compact-composed-resolved.yaml

# enforce explicit task/metric regression budgets against a previous result
mm-eval-workbench gate baseline.json current.json --policy gates/compact-regression.yaml

# cluster failed samples from saved embeddings into recurring failure slices
mm-eval-workbench failure-slices failures.jsonl --max-clusters 8

# reveal clean/OOD/calibration trade-offs between two model revisions
mm-eval-workbench compare-experiments profiles/base.yaml profiles/current.yaml \
  --policy gates/compact-regression.yaml
```

Failure slice discovery L2-normalizes sample embeddings, optionally reduces
them with PCA, selects K with silhouette score, then reports cluster prevalence,
dominant tasks/error tags, mean confidence, and centroid-nearest representative
failures. This turns a flat error list into data-driven OCR, temporal, counting,
reasoning, or other recurring failure groups without requiring manual labels for
the clustering itself.

Evaluation profiles can bind a clean lmms-eval result, confidence predictions,
and any number of matched corruption/OOD condition results. `compare-experiments`
then produces one scorecard covering directional clean-metric change, ECE/NLL
change, and OOD retention change. This makes improvements that trade clean
accuracy for worse calibration or robustness visible instead of collapsing the
research decision into one benchmark number.

## Measured BLIP calibration and failure slices

The first cross-workbench result consumes the 80 held-out confidence records
and fused representations from Vision Language Workbench's real COCO 2017
canonical study. Calibration here means the confidence of the top-1
image-to-text retrieval decision; it is not a synthetic classifier fixture.

| Variant | I→T accuracy | Mean retrieval R@1 | ECE | Correctness NLL | Brier | AURC | Pair failures |
|---|---:|---:|---:|---:|---:|---:|---:|
| Pretrained Base | 0.9750 | 0.95625 | 0.63706 | 1.13433 | 0.44055 | 0.00523 | 2 |
| LoRA Q/V r=8 | 0.9625 | 0.93750 | 0.60593 | 1.04428 | 0.40514 | 0.00503 | 3 |
| LoRA Q/V r=16 | 0.9625 | 0.94375 | 0.59122 | 1.00461 | 0.38794 | 0.00507 | 3 |

The models are strongly under-confident in the 80-way retrieval setting. LoRA
reduces ECE and NLL but adds one top-1 pair error, so calibration improves while
retrieval accuracy regresses. The eight failures across variants form three
embedding clusters: two persistent within-`dog` caption confusions and one
`person` confusion introduced by both LoRA ranks. This also explains why the
class probe remains perfect while exact-pair retrieval is not. OOD retention is
not reported here because no corrupted-image run was executed in this first
canonical pass.

The measured reports and failure inputs are stored under
[`results/canonical-blip-coco-small-v1`](results/canonical-blip-coco-small-v1).

## Calibration and uncertainty

For models or adapters that expose answer confidence, store predictions as
JSONL using either a direct correctness confidence:

```json
{"confidence":0.91,"correct":true}
{"confidence":0.88,"correct":false}
```

or a full class probability vector:

```json
{"probabilities":[0.05,0.15,0.80],"target":2}
```

Then run:

```bash
mm-eval-workbench calibration artifacts/predictions.jsonl --bins 10
```

The report includes expected/adaptive/max calibration error (ECE), Brier score,
negative log likelihood, normalized predictive entropy, confidence on errors,
reliability bins, area under the risk-coverage curve (AURC), and selective
accuracy at multiple coverage levels. This separates raw benchmark accuracy
from whether a multimodal model knows when it is likely to be wrong.

### Post-hoc temperature scaling

On a **held-out calibration split**, save classifier scores as logits:

```json
{"logits":[3.2,0.7,-1.1],"target":0}
{"logits":[2.8,1.4,0.2],"target":1}
```

Fit a single positive temperature by minimizing multiclass NLL:

```bash
mm-eval-workbench temperature-scale artifacts/calibration-logits.jsonl \
  --write-probabilities artifacts/calibrated.jsonl
```

The command reports calibration metrics before/after scaling plus the fitted
temperature and changes in NLL, ECE, and Brier score. The optimizer is a small
bounded one-dimensional search, so it adds no training-framework dependency and
does not retrain the underlying multimodal model.

## OOD / corruption robustness profiles

Once the same suite has been evaluated on clean and shifted inputs, aggregate
the degradation using the same task/metric direction definitions already used
by regression gates:

```bash
mm-eval-workbench robustness clean.json \
  --condition blur=blur.json \
  --condition noise=noise.json \
  --condition low-light=low-light.json \
  --policy gates/compact-regression.yaml
```

The profile reports per-metric absolute and relative regression, retention,
worst corruption, per-condition mean/worst regression, and overall worst-case
robustness. This keeps clean accuracy separate from distribution-shift
reliability and is ready to consume future corruption-generated benchmark runs.

The workbench does not duplicate benchmark scoring logic; it only orchestrates
and summarizes the evaluator that is already bundled in this repository.

The underlying evaluator remains `lmms_eval` and its existing task/model code.

## Repository shape

```text
lmms_eval/             Original evaluator, tasks, model adapters, CLI and TUI
configs/               Existing lmms-eval configuration examples
examples/              Existing usage examples
eval_workbench/        Oosu suite planning, provenance and reporting layer
suites/                Reusable benchmark suite manifests
```

## Origin and licensing

This repository was created from a clean source snapshot rather than a fork and
contains no upstream Git history. The bundled lmms-eval code remains under its
MIT license. See `LICENSE` and `THIRD_PARTY_NOTICE.md`.
