# Phase 1 Post-hoc Calibration — Full Results

Variant: D (Qwen3-8B-Base + LoRA, BioRED-only + loss reweight)
Methods: P2P / LA / PAS / TECP / P2P+TECP

## Micro F1 summary

| Dataset | n_pairs | Baseline F1 | Best F1 | Δ | Method |
|---|---|---|---|---|---|
| BioRED dev (processed_test) | 20376 | 0.6487 | **0.6487** | **+0.0000** | baseline |
| BC8 test | 83314 | 0.5654 | **0.5704** | **+0.0050** | P2P_oracle |
| cdr (BC5CDR) | 75332 | 0.4714 | **0.5095** | **+0.0382** | LA_tau0.5 |
| disgenet | 3912 | 0.8485 | **0.8489** | **+0.0005** | LA_tau2.0 |
| pharmgkb | 88448 | 0.2552 | **0.2552** | **+0.0000** | baseline |

## Method ranking by mean Δ across 5 datasets

| Method | Mean Δ | n datasets | per-dataset Δ |
|---|---|---|---|
| P2P_oracle | **+0.0025** | 5 | -0.006, +0.005, +0.018, -0.002, -0.002 |
| LA_tau0.5 | **-0.0029** | 5 | -0.020, -0.018, +0.038, +0.000, -0.015 |
| P2P_uniform | **-0.0356** | 5 | -0.090, -0.059, -0.009, +0.000, -0.020 |
| LA_tau1.0 | **-0.0422** | 5 | -0.094, -0.070, -0.001, +0.000, -0.046 |
| LA_tau2.0 | **-0.1888** | 5 | -0.384, -0.292, -0.089, +0.000, -0.179 |
| P2P+TECP | **-0.3856** | 5 | -0.311, -0.339, -0.211, -0.838, -0.229 |
| TECP | **-0.4051** | 5 | -0.331, -0.357, -0.272, -0.838, -0.228 |

## Per-class breakdown (best method per dataset)

### BC8 test (best=P2P_oracle)

| Class | Baseline P/R/F1 | Best P/R/F1 |
|---|---|---|
| Association | 0.445 / 0.654 / 0.529 | 0.455 / 0.635 / 0.530 |
| Negative_Correlation | 0.670 / 0.683 / 0.677 | 0.642 / 0.711 / 0.675 |
| Positive_Correlation | 0.592 / 0.511 / 0.549 | 0.566 / 0.544 / 0.555 |
| Comparison | 0.115 / 0.231 / 0.154 | 0.154 / 0.154 / 0.154 |
| Bind | 0.610 / 0.551 / 0.579 | 0.586 / 0.625 / 0.605 |
| Cotreatment | 0.904 / 0.602 / 0.723 | 0.883 / 0.661 / 0.756 |
| Conversion | 1.000 / 0.462 / 0.632 | 0.800 / 0.615 / 0.696 |

### cdr (BC5CDR) (best=LA_tau0.5)

| Class | Baseline P/R/F1 | Best P/R/F1 |
|---|---|---|
| CID | 0.555 / 0.410 / 0.471 | 0.445 / 0.595 / 0.510 |

### disgenet (best=LA_tau2.0)

| Class | Baseline P/R/F1 | Best P/R/F1 |
|---|---|---|
| Association | 0.749 / 0.978 / 0.848 | 0.748 / 0.981 / 0.849 |


## Effective prior diagnostic

p_eff (model's self-estimated prior) vs p_target (oracle from gold):

**BioRED dev (processed_test)**
  - p_eff:    `[0.899, 0.049, 0.031, 0.019, 0.001, 0.001, 0.0, 0.001, 0.0]`
  - p_target: `[0.885, 0.062, 0.032, 0.017, 0.001, 0.001, 0.0, 0.001, 0.0]`

**BC8 test**
  - p_eff:    `[0.86, 0.079, 0.025, 0.031, 0.001, 0.003, 0.002, 0.0]`
  - p_target: `[0.855, 0.066, 0.029, 0.042, 0.0, 0.003, 0.004, 0.0]`

**cdr (BC5CDR)**
  - p_eff:    `[0.937, 0.063]`
  - p_target: `[0.917, 0.083]`

**disgenet**
  - p_eff:    `[0.449, 0.551]`
  - p_target: `[0.561, 0.439]`

**pharmgkb**
  - p_eff:    `[0.916, 0.084]`
  - p_target: `[0.98, 0.02]`
