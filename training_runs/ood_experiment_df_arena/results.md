# OOD Degradation Experiment

Threshold: 0.5

| Variant | Real FA | FA rate | Fake miss | Miss rate | mean dp_real | real mean p_real | fake mean p_real |
|---|---|---|---|---|---|---|---|
| original | 0 | 0% | 3 | 15% | +0.000 | 0.9674 | 0.2129 |
| opus_24k | 0 | 0% | 2 | 10% | -0.050 | 0.9609 | 0.1191 |
| opus_12k | 1 | 5% | 2 | 10% | -0.114 | 0.8469 | 0.1051 |
| gain_minus12 | 0 | 0% | 6 | 30% | +0.100 | 0.9827 | 0.3973 |
| gain_plus6 | 0 | 0% | 0 | 0% | -0.068 | 0.9459 | 0.0992 |
| headset_sim | 0 | 0% | 7 | 35% | +0.109 | 0.9677 | 0.4314 |
| resample_8k | 0 | 0% | 2 | 10% | -0.045 | 0.9642 | 0.1263 |
