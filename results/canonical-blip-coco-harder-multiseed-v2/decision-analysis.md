# Canonical decision analysis

Baseline: `base`

## Objective winners

| Objective | Direction | Winner | Value |
|---|---|---|---:|
| clean_mean_r_at_1 | max | base | 0.856120 |
| ood_mean_r_at_1 | max | base | 0.753025 |
| ood_mean_retention | max | lora-r8-hard-negative | 0.885318 |
| ood_worst_retention | max | lora-r16 | 0.416201 |
| ece | min | lora-r16 | 0.644591 |
| correctness_nll | min | lora-r16 | 1.393605 |
| trainable_parameter_pct | min | base | 0.000000 |

## Pareto frontier

`base`, `lora-r8`, `lora-r16`, `lora-r8-hard-negative`

Multiple frontier variants mean there is no universal winner under the declared objectives.

## Deltas vs base

| Variant | Clean R@1 Δ | OOD R@1 Δ | Retention Δ | Worst retention Δ | ECE improvement | NLL improvement | Params % Δ |
|---|---:|---:|---:|---:|---:|---:|---:|
| lora-r8 | -0.00846 | -0.00337 | +0.00480 | +0.00104 | +0.04980 | +0.21526 | +0.1318 |
| lora-r16 | -0.00911 | -0.00747 | +0.00069 | +0.00590 | +0.08153 | +0.34383 | +0.2636 |
| lora-r8-hard-negative | -0.00846 | -0.00253 | +0.00581 | +0.00254 | +0.04558 | +0.19998 | +0.1318 |
