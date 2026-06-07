# Phase 1 Post-hoc Calibration — Full Results

Variant: D (Qwen3-8B-Base + LoRA, BioRED-only + loss reweight)
Methods: P2P / LA / PAS / TECP / P2P+TECP

## Micro F1 summary

| Dataset | n_pairs | Baseline F1 | Best F1 | Δ | Method |
|---|---|---|---|---|---|
| BioRED dev (processed_test) | 20376 | 0.5095 | **0.5576** | **+0.0481** | P2P_oracle |
| BC8 test | 83314 | 0.3947 | **0.5046** | **+0.1099** | P2P_oracle |
| cdr (BC5CDR) | 75334 | 0.3895 | **0.4591** | **+0.0697** | LA_tau0.5 |
| disgenet | 3912 | 0.7955 | **0.8335** | **+0.0380** | P2P_oracle |
| pharmgkb | 88448 | 0.1928 | **0.2764** | **+0.0835** | TECP |

## Method ranking by mean Δ across 5 datasets

| Method | Mean Δ | n datasets | per-dataset Δ |
|---|---|---|---|
| P2P_oracle | **+0.0571** | 5 | +0.048, +0.110, +0.011, +0.038, +0.078 |
| P2P_uniform | **-0.0804** | 5 | -0.279, -0.120, +0.050, +0.029, -0.082 |
| LA_tau0.5 | **-0.0861** | 5 | -0.232, -0.135, +0.070, -0.006, -0.127 |
| TECP | **-0.1127** | 5 | -0.198, -0.138, -0.252, -0.059, +0.084 |
| LA_tau1.0 | **-0.1283** | 5 | -0.332, -0.198, +0.051, -0.012, -0.151 |
| P2P+TECP | **-0.1471** | 5 | -0.271, -0.132, -0.230, -0.101, -0.002 |
| LA_tau2.0 | **-0.1978** | 5 | -0.406, -0.246, -0.159, -0.025, -0.153 |

## Per-class breakdown (best method per dataset)

### BioRED dev (processed_test) (best=P2P_oracle)

| Class | Baseline P/R/F1 | Best P/R/F1 |
|---|---|---|
| Association | 0.330 / 0.767 / 0.462 | 0.714 / 0.339 / 0.459 |
| Positive_Correlation | 0.833 / 0.521 / 0.642 | 0.669 / 0.626 / 0.647 |
| Negative_Correlation | 0.712 / 0.550 / 0.620 | 0.618 / 0.673 / 0.644 |
| Comparison | 0.385 / 0.833 / 0.526 | 1.000 / 0.667 / 0.800 |
| Bind | 1.000 / 0.222 / 0.364 | 0.800 / 0.444 / 0.571 |
| Conversion | 0.000 / 0.000 / 0.000 | 0.333 / 1.000 / 0.500 |
| Cotreatment | 0.833 / 0.357 / 0.500 | 0.750 / 0.643 / 0.692 |
| Drug_Interaction | 0.333 / 0.500 / 0.400 | 0.000 / 0.000 / 0.000 |

### BC8 test (best=P2P_oracle)

| Class | Baseline P/R/F1 | Best P/R/F1 |
|---|---|---|
| Association | 0.249 / 0.823 / 0.383 | 0.577 / 0.344 / 0.431 |
| Negative_Correlation | 0.809 / 0.406 / 0.541 | 0.645 / 0.610 / 0.627 |
| Positive_Correlation | 0.713 / 0.220 / 0.337 | 0.512 / 0.499 / 0.506 |
| Comparison | 0.200 / 0.615 / 0.302 | 0.000 / 0.000 / 0.000 |
| Bind | 0.367 / 0.132 / 0.195 | 0.405 / 0.375 / 0.389 |
| Cotreatment | 0.883 / 0.485 / 0.626 | 0.866 / 0.643 / 0.738 |
| Conversion | 0.000 / 0.000 / 0.000 | 1.000 / 0.538 / 0.700 |

### cdr (BC5CDR) (best=LA_tau0.5)

| Class | Baseline P/R/F1 | Best P/R/F1 |
|---|---|---|
| CID | 0.529 / 0.308 / 0.389 | 0.402 / 0.534 / 0.459 |

### disgenet (best=P2P_oracle)

| Class | Baseline P/R/F1 | Best P/R/F1 |
|---|---|---|
| Association | 0.665 / 0.990 / 0.796 | 0.729 / 0.973 / 0.833 |


## Effective prior diagnostic

p_eff (model's self-estimated prior) vs p_target (oracle from gold):

**BioRED dev (processed_test)**
  - p_eff:    `[0.707, 0.243, 0.029, 0.016, 0.002, 0.0, 0.0, 0.001, 0.002]`
  - p_target: `[0.885, 0.062, 0.032, 0.017, 0.001, 0.001, 0.0, 0.001, 0.0]`

**BC8 test**
  - p_eff:    `[0.671, 0.277, 0.019, 0.025, 0.003, 0.003, 0.002, 0.0]`
  - p_target: `[0.855, 0.066, 0.029, 0.042, 0.0, 0.003, 0.004, 0.0]`

**cdr (BC5CDR)**
  - p_eff:    `[0.923, 0.077]`
  - p_target: `[0.917, 0.083]`

**disgenet**
  - p_eff:    `[0.321, 0.679]`
  - p_target: `[0.561, 0.439]`

**pharmgkb**
  - p_eff:    `[0.731, 0.269]`
  - p_target: `[0.98, 0.02]`
