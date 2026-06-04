# OOD Degradation Experiment (with preprocessing)

Threshold: 0.49

| Variant | Real FA | FA rate | Fake miss | Miss rate | mean dp_real | real mean p_real | fake mean p_real |
|---|---|---|---|---|---|---|---|
| original | 50 | 100% | 1 | 2% | +0.000 | 0.0297 | 0.0712 |
| opus_24k | 49 | 98% | 2 | 4% | -0.004 | 0.0291 | 0.0648 |
| opus_12k | 50 | 100% | 1 | 2% | -0.027 | 0.0105 | 0.0362 |
| gain_minus12 | 50 | 100% | 1 | 2% | -0.000 | 0.0296 | 0.0712 |
| gain_plus6 | 50 | 100% | 1 | 2% | +0.000 | 0.0293 | 0.0725 |
| headset_sim | 49 | 98% | 2 | 4% | +0.005 | 0.0292 | 0.0812 |
| resample_8k | 50 | 100% | 0 | 0% | -0.037 | 0.0148 | 0.0115 |
