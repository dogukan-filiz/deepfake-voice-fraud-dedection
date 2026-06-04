# OOD Degradation Experiment

Threshold: 0.49

| Variant | Real FA | FA rate | Fake miss | Miss rate | mean dp_real | real mean p_real | fake mean p_real |
|---|---|---|---|---|---|---|---|
| original | 0 | 0% | 0 | 0% | +0.000 | 0.9576 | 0.0833 |
| opus_24k | 0 | 0% | 2 | 4% | -0.026 | 0.9052 | 0.0839 |
| opus_12k | 17 | 34% | 2 | 4% | -0.218 | 0.5422 | 0.0618 |
| gain_minus12 | 3 | 6% | 6 | 12% | -0.056 | 0.7644 | 0.1642 |
| gain_plus6 | 0 | 0% | 1 | 2% | +0.021 | 0.9754 | 0.1081 |
| headset_sim | 0 | 0% | 6 | 12% | +0.046 | 0.9553 | 0.177 |
| resample_8k | 0 | 0% | 3 | 6% | +0.019 | 0.9487 | 0.1307 |
