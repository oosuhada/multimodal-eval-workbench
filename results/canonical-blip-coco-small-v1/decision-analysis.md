# Canonical decision analysis

Baseline: `base`

## Objective winners

| Objective | Direction | Winner | Value |
|---|---|---|---:|
| clean_mean_r_at_1 | max | base | 0.956250 |
| ood_mean_r_at_1 | max | base | 0.916250 |
| ood_mean_retention | max | lora-r8 | 0.973333 |
| ood_worst_retention | max | lora-r16 | 0.807947 |
| ece | min | lora-r16 | 0.591222 |
| correctness_nll | min | lora-r16 | 1.004610 |
| trainable_parameter_pct | min | base | 0.000000 |

## Pareto frontier

`base`, `lora-r8`, `lora-r16`, `lora-r8-hard-negative`

Multiple frontier variants mean there is no universal winner under the declared objectives.

## Deltas vs base

| Variant | Clean R@1 Δ | OOD R@1 Δ | Retention Δ | Worst retention Δ | ECE improvement | NLL improvement | Params % Δ |
|---|---:|---:|---:|---:|---:|---:|---:|
| lora-r8 | -0.01875 | -0.00375 | +0.01516 | +0.01569 | +0.03114 | +0.09005 | +0.1318 |
| lora-r16 | -0.01250 | -0.00167 | +0.01092 | +0.02363 | +0.04584 | +0.12972 | +0.2636 |
| lora-r8-hard-negative | -0.00625 | -0.00083 | +0.00543 | +0.00516 | +0.02003 | +0.05761 | +0.1318 |
