# Phase 1 Post-hoc Calibration — Full Results

Variant: D (Qwen3-8B-Base + LoRA, BioRED-only + loss reweight)
Methods: P2P / LA / PAS / TECP / P2P+TECP

## Micro F1 summary

| Dataset | n_pairs | Baseline F1 | Best F1 | Δ | Method |
|---|---|---|---|---|---|
| BioRED dev (processed_test) | 20263 | 0.6062 | **0.6068** | **+0.0005** | P2P_oracle |
| BC8 test | 82392 | 0.5519 | **0.5647** | **+0.0128** | P2P_oracle |
| cdr (BC5CDR) | 75313 | 0.3842 | **0.4577** | **+0.0735** | LA_tau0.5 |
| disgenet | 3912 | 0.8657 | **0.8672** | **+0.0015** | P2P_uniform |
| pharmgkb | 88336 | 0.2606 | **0.2794** | **+0.0188** | P2P_oracle |

## Method ranking by mean Δ across 5 datasets

| Method | Mean Δ | n datasets | per-dataset Δ |
|---|---|---|---|
| P2P_oracle | **+0.0108** | 5 | +0.001, +0.013, +0.021, +0.001, +0.019 |
| LA_tau0.5 | **-0.0081** | 5 | -0.015, -0.017, +0.073, -0.004, -0.077 |
| P2P_uniform | **-0.0408** | 5 | -0.108, -0.066, +0.047, +0.001, -0.079 |
| LA_tau1.0 | **-0.0533** | 5 | -0.110, -0.087, +0.054, -0.007, -0.115 |
| LA_tau2.0 | **-0.1859** | 5 | -0.373, -0.298, -0.054, -0.007, -0.198 |
| TECP | **-0.3022** | 5 | -0.214, -0.280, -0.276, -0.643, -0.098 |
| P2P+TECP | **-0.3067** | 5 | -0.213, -0.269, -0.262, -0.643, -0.146 |

## Per-class breakdown (best method per dataset)

### BioRED dev (processed_test) (best=P2P_oracle)

| Class | Baseline P/R/F1 | Best P/R/F1 |
|---|---|---|
| Association | 0.614 / 0.488 / 0.544 | 0.623 / 0.485 / 0.546 |
| Positive_Correlation | 0.720 / 0.623 / 0.668 | 0.705 / 0.632 / 0.667 |
| Negative_Correlation | 0.684 / 0.712 / 0.697 | 0.686 / 0.706 / 0.696 |
| Comparison | 0.667 / 0.667 / 0.667 | 0.667 / 0.667 / 0.667 |
| Bind | 0.857 / 0.667 / 0.750 | 0.778 / 0.778 / 0.778 |
| Conversion | 0.000 / 0.000 / 0.000 | 0.000 / 0.000 / 0.000 |
| Cotreatment | 0.889 / 0.571 / 0.696 | 0.556 / 0.714 / 0.625 |
| Drug_Interaction | 0.000 / 0.000 / 0.000 | 0.000 / 0.000 / 0.000 |

### BC8 test (best=P2P_oracle)

| Class | Baseline P/R/F1 | Best P/R/F1 |
|---|---|---|
| Association | 0.466 / 0.589 / 0.521 | 0.519 / 0.518 / 0.518 |
| Negative_Correlation | 0.755 / 0.603 / 0.670 | 0.709 / 0.640 / 0.673 |
| Positive_Correlation | 0.687 / 0.431 / 0.530 | 0.621 / 0.504 / 0.557 |
| Comparison | 0.125 / 0.077 / 0.095 | 0.500 / 0.077 / 0.133 |
| Bind | 0.625 / 0.404 / 0.491 | 0.603 / 0.559 / 0.580 |
| Cotreatment | 0.936 / 0.515 / 0.664 | 0.911 / 0.596 / 0.721 |
| Conversion | 1.000 / 0.308 / 0.471 | 0.364 / 0.615 / 0.457 |

### cdr (BC5CDR) (best=LA_tau0.5)

| Class | Baseline P/R/F1 | Best P/R/F1 |
|---|---|---|
| CID | 0.525 / 0.303 / 0.384 | 0.407 / 0.522 / 0.458 |

### pharmgkb (best=P2P_oracle)

| Class | Baseline P/R/F1 | Best P/R/F1 |
|---|---|---|
| Association | 0.159 / 0.719 / 0.261 | 0.190 / 0.525 / 0.279 |


## Effective prior diagnostic

p_eff (model's self-estimated prior) vs p_target (oracle from gold):

**BioRED dev (processed_test)**
  - p_eff:    `[0.885, 0.065, 0.03, 0.017, 0.001, 0.001, 0.0, 0.001, 0.0]`
  - p_target: `[0.885, 0.062, 0.032, 0.017, 0.001, 0.001, 0.0, 0.001, 0.0]`

**BC8 test**
  - p_eff:    `[0.851, 0.092, 0.023, 0.029, 0.001, 0.002, 0.002, 0.0]`
  - p_target: `[0.855, 0.066, 0.029, 0.042, 0.0, 0.003, 0.004, 0.0]`

**cdr (BC5CDR)**
  - p_eff:    `[0.93, 0.07]`
  - p_target: `[0.917, 0.083]`

**disgenet**
  - p_eff:    `[0.442, 0.558]`
  - p_target: `[0.561, 0.439]`

**pharmgkb**
  - p_eff:    `[0.889, 0.111]`
  - p_target: `[0.98, 0.02]`
