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
```

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
