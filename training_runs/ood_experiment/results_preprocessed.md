# OOD Degradation Experiment (with preprocessing)

Threshold: 0.49

| Variant | Real FA | FA rate | Fake miss | Miss rate | mean dp_real | real mean p_real | fake mean p_real |
|---|---|---|---|---|---|---|---|
| original | 0 | 0% | 0 | 0% | +0.000 | 0.9578 | 0.0738 |
| opus_24k | 0 | 0% | 0 | 0% | -0.031 | 0.9073 | 0.0617 |
| opus_12k | 18 | 36% | 0 | 0% | -0.219 | 0.5571 | 0.0359 |
| gain_minus12 | 0 | 0% | 0 | 0% | -0.000 | 0.9579 | 0.0733 |
| gain_plus6 | 0 | 0% | 0 | 0% | -0.000 | 0.9577 | 0.0737 |
| headset_sim | 0 | 0% | 3 | 6% | +0.042 | 0.9543 | 0.1611 |
| resample_8k | 0 | 0% | 1 | 2% | +0.005 | 0.9476 | 0.0942 |
