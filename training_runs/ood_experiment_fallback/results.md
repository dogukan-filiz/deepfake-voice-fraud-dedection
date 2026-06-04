# OOD Degradation Experiment

Threshold: 0.49

| Variant | Real FA | FA rate | Fake miss | Miss rate | mean Δp_real | real p̄_real | fake p̄_real |
|---|---|---|---|---|---|---|---|
| original | 50 | 100% | 2 | 4% | +0.000 | 0.024 | 0.0709 |
| opus_24k | 49 | 98% | 2 | 4% | -0.000 | 0.0309 | 0.0635 |
| opus_12k | 50 | 100% | 1 | 2% | -0.019 | 0.0142 | 0.0433 |
| gain_minus12 | 5 | 10% | 38 | 76% | +0.708 | 0.7987 | 0.7123 |
| gain_plus6 | 50 | 100% | 3 | 6% | +0.000 | 0.0054 | 0.0901 |
| headset_sim | 50 | 100% | 2 | 4% | +0.004 | 0.0235 | 0.0803 |
| resample_8k | 50 | 100% | 0 | 0% | -0.034 | 0.0141 | 0.0126 |
