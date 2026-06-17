# Multimodal Eval Workbench — Reproducible Multimodal Model Evaluation

Multimodal Eval Workbench keeps the full lmms-eval task, model-adapter and
evaluation engine available while adding reproducible calibration, OOD,
failure-slice, paired-statistics and model-decision analysis.

Multimodal Eval Workbench는 lmms-eval의 task, model adapter, evaluation engine을
그대로 활용하면서 **calibration, OOD robustness, failure slice, paired
statistics, model decision 분석을 재현 가능하게 연결하는 multimodal evaluation
workbench**입니다.

This project does not reimplement the evaluator. The existing `lmms_eval` CLI
remains the benchmark execution engine; this repository adds the research and
decision layer around measured outputs.

이 프로젝트는 evaluator를 다시 구현하지 않습니다. 기존 `lmms_eval` CLI를
benchmark execution engine으로 유지하고, 그 결과 위에 연구·의사결정 계층을
추가합니다.

## Overview / 개요

The workbench is organized around four evaluation questions.

이 워크벤치는 다음 네 가지 평가 질문을 중심으로 구성됩니다.

- **Ⅰ Measure** — did the model actually improve on clean benchmark metrics?<br>
  모델이 clean benchmark에서 실제로 개선됐는가?
- **Ⅱ Calibrate** — does model confidence track correctness, and can post-hoc calibration help?<br>
  모델 confidence가 correctness와 일치하며 post-hoc calibration으로 개선 가능한가?
- **Ⅲ Stress** — how much performance survives corruption/OOD shift and where do failures cluster?<br>
  corruption/OOD shift에서 성능이 얼마나 유지되고 failure가 어디에 군집되는가?
- **Ⅳ Decide** — which model is Pareto-preferred once clean score, OOD, calibration and parameter cost are considered together?<br>
  clean, OOD, calibration, parameter cost를 함께 볼 때 어떤 모델 선택이 합리적인가?

## Problem / 문제

Multimodal benchmark numbers are easy to compare but hard to interpret in
isolation. A model can gain calibration while losing clean retrieval, improve
relative OOD retention from a weaker baseline, or appear better because one
seed happened to be favorable.

멀티모달 benchmark 수치는 비교하기 쉽지만 단독으로 해석하기는 어렵습니다.
모델은 calibration이 좋아지는 동시에 clean retrieval이 떨어질 수 있고,
약해진 clean baseline 때문에 상대 OOD retention만 높아질 수 있으며,
특정 seed가 유리해서 좋아 보일 수도 있습니다.

This workbench turns raw benchmark outputs into evidence for model decisions by
combining calibration, paired seed statistics, OOD stress, failure slices and
Pareto trade-offs.

이 워크벤치는 raw benchmark output에 calibration, paired seed statistics,
OOD stress, failure slice, Pareto trade-off를 결합해 **model decision을 위한
근거**로 변환합니다.

## Evaluation walkthrough / 프로젝트 화면

This repository is also a CLI/research workbench rather than a web product.
Its actual user-facing surfaces are benchmark plans, compact metric reports,
failure-slice reports and integrated decision scorecards. Colab screenshots
remain with the source training experiment instead of being duplicated here.

이 레포 역시 웹 제품이 아니라 CLI/research workbench입니다. 실제 프로젝트
화면은 **benchmark plan, compact metric report, failure-slice report,
integrated decision scorecard**이며, Colab 캡처는 source training experiment에
보존하고 이 레포에서는 중복하지 않습니다.

### 1. Benchmark planning / 벤치마크 계획

```bash
mm-eval-workbench plan suites/compact-multimodal-smoke.yaml
mm-eval-workbench validate suites/compact-multimodal-smoke.yaml
```

### 2. Calibration and selective risk / Calibration 분석

```bash
mm-eval-workbench calibration predictions.jsonl --bins 10
mm-eval-workbench temperature-scale logits.jsonl --bins 10
```

### 3. OOD and failure discovery / OOD·failure 분석

```bash
mm-eval-workbench robustness clean.json \
  --condition blur=blur.json --condition occlusion=occlusion.json \
  --policy gates/compact-regression.yaml

mm-eval-workbench failure-slices failures.jsonl --max-clusters 8
```

### 4. Model decision report / 모델 의사결정 리포트

The canonical result directory combines clean/OOD retrieval, ECE/NLL,
representation drift, paired seed deltas and Pareto analysis into one measured
research record.

canonical result directory는 clean/OOD retrieval, ECE/NLL, representation
drift, paired seed delta, Pareto analysis를 하나의 실측 연구 기록으로
통합합니다.

[`results/canonical-blip-coco-harder-multiseed-v2`](results/canonical-blip-coco-harder-multiseed-v2)

## Current capabilities / 현재 기능

| Capability / 기능 | Current implementation / 현재 구현 |
|---|---|
| Benchmark orchestration / 벤치마크 실행 | suite manifests, task catalog, composition, provenance ledger |
| Calibration / 보정 | ECE, adaptive ECE, MCE, Brier, NLL, entropy, AURC, selective accuracy |
| Post-hoc calibration / 사후 보정 | scalar temperature scaling on held-out logits |
| OOD robustness / OOD 강건성 | clean-to-shift retention, worst condition, severity analysis |
| Failure discovery / 실패 분석 | PCA(SVD) + K-Means++ + silhouette automatic failure slices |
| Decision analysis / 의사결정 분석 | paired seed deltas, regression gates, integrated scorecards, Pareto frontier |
| Research lineage / 연구 계보 | source experiment commit, artifact hash, predecessor and replication status |

## Evaluation loop / 평가 루프

```text
vision-language-workbench
  measured model / representation / OOD artifacts
          ↓
multimodal-eval-workbench
  calibration + NLL + selective risk
          ↓
  paired seed statistics + OOD retention
          ↓
  failure slices + Pareto decision analysis
          ↓
  replicated / inconclusive / not replicated
          ↓
  next experiment decision
```

The evaluation repository is intentionally downstream of the source training
experiment. It does not duplicate model training evidence; it records how that
evidence changes the research conclusion.

이 평가 레포는 source training experiment의 downstream으로 의도적으로
분리되어 있습니다. 학습 증거를 중복 보관하지 않고, 그 증거가 연구 결론을
어떻게 바꾸는지를 기록합니다.

## Quick start / 빠른 시작

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

Render the exact lmms-eval command without loading a model / 모델 로드 없이 실행 계획 확인:

```bash
mm-eval-workbench plan suites/compact-multimodal-smoke.yaml
```

Search and validate the bundled benchmark catalog / bundled task 검색 및 검증:

```bash
mm-eval-workbench catalog videomme
mm-eval-workbench validate suites/compact-multimodal-smoke.yaml
```

Run the same suite and keep provenance / 동일 suite 실행 및 provenance 기록:

```bash
mm-eval-workbench run suites/compact-multimodal-smoke.yaml
mm-eval-workbench history --limit 10
```

Turn an lmms-eval result into compact analysis / lmms-eval 결과를 compact 분석으로 변환:

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

## Measured BLIP evaluation / BLIP 실측 평가

### Cross-workbench research flow / 교차 워크벤치 연구 흐름

The measured BLIP studies are produced in `vision-language-workbench` and then
consumed here as evaluation evidence. The two repositories form one research
pipeline rather than two independent demos:

BLIP 실측 결과는 `vision-language-workbench`에서 생성되고 이 레포에서 평가
근거로 소비됩니다. 두 레포는 독립적인 demo가 아니라 하나의 research
pipeline입니다.

```text
vision-language-workbench
  pretrained BLIP / LoRA / hard negatives
          ↓
  clean + OOD + representation artifacts
          ↓
multimodal-eval-workbench
  calibration + NLL + paired seed statistics
          ↓
  Pareto / replication analysis
          ↓
  next experiment decision
```

For the harder multi-seed v2 study, the source experiment was executed on a
Google Colab `NVIDIA A100-SXM4-40GB` over seeds `42`, `1337`, and `2026`.
`research-lineage.json` in the result directory records the source repository,
source result commit, artifact SHA-256, predecessor study, and the conclusions
that replicated, remained inconclusive, or failed to replicate.

The human execution evidence stays with the source experiment rather than
being duplicated into this evaluation repository. The Vision Language
Workbench canonical result directories preserve the Colab screenshots for the
first study and the harder v2 run, including A100 selection, active execution,
three-seed validation/result packaging, and final runtime deletion. This repo
keeps the derived calibration, OOD, paired-statistics, and decision artifacts.

The v2 result is especially useful because it distinguishes a one-run effect
from a repeatable direction: Base's clean retrieval advantage and LoRA's
calibration improvement replicated, while the v1 hard-negative clean recovery
did not. This is why the evaluation layer is treated as part of the experiment,
not as a presentation-only post-processing step.

### Harder multi-seed v2 / 더 어려운 3-seed 재현 평가

The 256-probe, three-seed follow-up reproduces the calibration direction: r8
reduces ECE from 0.72612 to 0.67632 and r16 to 0.64459, with paired 95% CIs
excluding zero. Base keeps the best clean R@1 (0.85612). r8 relative retention
is directionally higher on all three seeds, but its CI crosses zero. The mined
hard-negative variant ties ordinary r8 on clean R@1 and gains only +0.00084 OOD
R@1; that recovery is inconclusive at n=3. Detailed calibration, OOD severity
4–5, Pareto, and integrated scorecards are in
[`results/canonical-blip-coco-harder-multiseed-v2`](results/canonical-blip-coco-harder-multiseed-v2).

v2에서는 LoRA의 calibration 개선이 세 seed에서 반복됐고 r8/r16 모두 ECE가
Base보다 낮았습니다. 반면 hard-negative의 v1 clean recovery는 재현되지
않았고, r8 대비 OOD 증가는 `+0.00084`로 n=3에서 결론을 내리기 어렵습니다.

### First canonical study / 첫 canonical 평가

The first cross-workbench result consumes the 80 held-out confidence records
and fused representations from Vision Language Workbench's real COCO 2017
canonical study. Calibration here means the confidence of the top-1
image-to-text retrieval decision; it is not a synthetic classifier fixture.

첫 canonical 평가는 Vision Language Workbench의 실제 80개 held-out prediction과
fused representation을 사용합니다. 여기서 calibration은 synthetic classifier가
아니라 실제 image-to-text top-1 retrieval decision의 confidence를 의미합니다.

| Variant | I→T accuracy | Mean retrieval R@1 | ECE | Correctness NLL | Brier | AURC | Pair failures |
|---|---:|---:|---:|---:|---:|---:|---:|
| Pretrained Base | 0.9750 | 0.95625 | 0.63706 | 1.13433 | 0.44055 | 0.00523 | 2 |
| LoRA Q/V r=8 | 0.9625 | 0.93750 | 0.60593 | 1.04428 | 0.40514 | 0.00503 | 3 |
| LoRA Q/V r=16 | 0.9625 | 0.94375 | 0.59122 | 1.00461 | 0.38794 | 0.00507 | 3 |
| LoRA Q/V r=8 + mined hard negative | 0.9750 | 0.95000 | 0.61704 | 1.07672 | 0.41641 | 0.00509 | 2 |

The models are strongly under-confident in the 80-way retrieval setting. LoRA
reduces ECE and NLL but adds one top-1 pair error, so calibration improves while
retrieval accuracy regresses. Mined negatives restore Base's image-to-text
accuracy and improve ECE versus Base, while mean bidirectional R@1 remains
0.00625 lower. The ten failures across four variants form three
embedding clusters: two persistent within-`dog` caption confusions and one
`person` confusion introduced by both LoRA ranks. This also explains why the
class probe remains perfect while exact-pair retrieval is not.

The exact saved adapters were also evaluated over Gaussian blur/noise, JPEG,
low light, and occlusion at severity 1–3. The integrated scorecard is:

| Variant | Params % | Clean R@1 | Mean OOD R@1 | OOD retention | ECE | Correctness NLL | CKA | Probe |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Base | 0.0000% | 0.95625 | 0.91625 | 0.95817 | 0.63706 | 1.13433 | 1.000000 | 1.00000 |
| LoRA r=8 | 0.1318% | 0.93750 | 0.91250 | 0.97333 | 0.60593 | 1.04428 | 0.999953 | 1.00000 |
| LoRA r=16 | 0.2636% | 0.94375 | 0.91458 | 0.96909 | 0.59122 | 1.00461 | 0.999850 | 1.00000 |
| LoRA r=8 + hard negative | 0.1318% | 0.95000 | 0.91542 | 0.96360 | 0.61704 | 1.07672 | 0.999956 | 1.00000 |

Base keeps the highest absolute clean and mean OOD R@1. Ordinary LoRA improves
relative OOD retention and calibration but does so from a lower clean score;
mined negatives recover most of the clean/OOD gap while retaining some of the
calibration improvement. Severity-3 occlusion is the worst condition for every
variant.

The measured reports and failure inputs are stored under
[`results/canonical-blip-coco-small-v1`](results/canonical-blip-coco-small-v1).

## Calibration and uncertainty / Calibration·불확실성

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

### Post-hoc temperature scaling / 사후 temperature scaling

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

## OOD / corruption robustness / OOD·corruption 강건성

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

## Architecture / 아키텍처

```text
lmms_eval/             Original evaluator, tasks, model adapters, CLI and TUI
configs/               Existing lmms-eval configuration examples
examples/              Existing usage examples
eval_workbench/        Oosu suite planning, provenance and reporting layer
suites/                Reusable benchmark suite manifests
```

## Origin and licensing / 출처 및 라이선스

This repository was created from a clean source snapshot rather than a fork and
contains no upstream Git history. The bundled lmms-eval code remains under its
MIT license. See `LICENSE` and `THIRD_PARTY_NOTICE.md`.
