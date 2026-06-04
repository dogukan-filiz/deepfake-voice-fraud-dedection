# full_run_004 ep1_best — Final Candidate Evaluation

- **Checkpoint:** `/workspace/deepfake-voice-fraud-dedection/training_runs/full_run_004/best_head.pth`
- **Threshold:** 0.47
- **Date:** 2026-06-03

## Results

| Benchmark | n | Acc | Macro F1 | Real Recall | Fake Recall | AUC | TP | TN | FP | FN |
|-----------|---|-----|---------|------------|------------|-----|----|----|----|-----|
| raw_digital | 100 | 0.68 | 0.6484 | 0.98 | 0.38 | 0.9184 | 49 | 19 | 31 | 1 |
| callnorm_digital_g711 | 100 | 0.59 | 0.5464 | 0.9 | 0.28 | 0.7644 | 45 | 14 | 36 | 5 |
| callnorm_digital_opus | 100 | 0.7 | 0.6956 | 0.82 | 0.58 | 0.796 | 41 | 29 | 21 | 9 |
| raw_mic | 22 | 0.7273 | 0.5417 | 0.8824 | 0.2 | 0.7529 | 15 | 1 | 4 | 2 |
| callnorm_mic_g711 | 22 | 0.9091 | 0.8854 | 0.8824 | 1.0 | 0.9294 | 15 | 5 | 0 | 2 |
| callnorm_mic_opus | 22 | 0.7273 | 0.6857 | 0.7059 | 0.8 | 0.8235 | 12 | 4 | 1 | 5 |

## False Positives / Negatives

### raw_digital
- FP (31): fake_000.wav(0.483985), fake_006.wav(0.479794), fake_007.wav(0.479439), fake_008.wav(0.477672), fake_009.wav(0.479712), fake_010.wav(0.522258), fake_012.wav(0.531593), fake_013.wav(0.487856), fake_016.wav(0.476603), fake_017.wav(0.491339), fake_019.wav(0.475726), fake_020.wav(0.574619), fake_021.wav(0.986045), fake_022.wav(0.482923), fake_023.wav(0.547038), fake_024.wav(0.737152), fake_025.wav(0.489078), fake_026.wav(0.977645), fake_027.wav(0.47497), fake_028.wav(0.627144), fake_029.wav(0.488939), fake_030.wav(0.497163), fake_032.wav(0.477856), fake_035.wav(0.536075), fake_038.wav(0.48375), fake_039.wav(0.470289), fake_041.wav(0.501685), fake_043.wav(0.478091), fake_044.wav(0.482918), fake_045.wav(0.485214), fake_046.wav(0.487436)
- FN (1): real_037.wav(0.453635)

### callnorm_digital_g711
- FP (36): fake_000_g711.wav(0.487165), fake_001_g711.wav(0.487531), fake_004_g711.wav(0.475681), fake_005_g711.wav(0.471402), fake_006_g711.wav(0.480157), fake_007_g711.wav(0.477331), fake_008_g711.wav(0.471526), fake_009_g711.wav(0.483701), fake_010_g711.wav(0.48703), fake_012_g711.wav(0.528428), fake_013_g711.wav(0.47244), fake_014_g711.wav(0.47368), fake_016_g711.wav(0.475679), fake_019_g711.wav(0.484952), fake_020_g711.wav(0.908555), fake_021_g711.wav(0.582795), fake_022_g711.wav(0.470555), fake_023_g711.wav(0.479602), fake_024_g711.wav(0.89571), fake_025_g711.wav(0.477468), fake_027_g711.wav(0.536004), fake_029_g711.wav(0.476792), fake_030_g711.wav(0.472342), fake_031_g711.wav(0.474389), fake_032_g711.wav(0.470091), fake_034_g711.wav(0.472808), fake_035_g711.wav(0.479493), fake_037_g711.wav(0.49432), fake_038_g711.wav(0.513786), fake_039_g711.wav(0.480164), fake_041_g711.wav(0.49205), fake_042_g711.wav(0.47833), fake_044_g711.wav(0.484264), fake_045_g711.wav(0.483856), fake_048_g711.wav(0.483608), fake_049_g711.wav(0.485459)
- FN (5): real_008_g711.wav(0.468335), real_019_g711.wav(0.468134), real_025_g711.wav(0.455608), real_033_g711.wav(0.466423), real_042_g711.wav(0.469994)

### callnorm_digital_opus
- FP (21): fake_004_opus.wav(0.472971), fake_007_opus.wav(0.475302), fake_008_opus.wav(0.470329), fake_009_opus.wav(0.485686), fake_010_opus.wav(0.485377), fake_011_opus.wav(0.478772), fake_012_opus.wav(0.488754), fake_014_opus.wav(0.474274), fake_019_opus.wav(0.478672), fake_020_opus.wav(0.492557), fake_021_opus.wav(0.482315), fake_022_opus.wav(0.473113), fake_023_opus.wav(0.475024), fake_024_opus.wav(0.479011), fake_025_opus.wav(0.472053), fake_029_opus.wav(0.472328), fake_031_opus.wav(0.476273), fake_041_opus.wav(0.480236), fake_043_opus.wav(0.481037), fake_045_opus.wav(0.471201), fake_046_opus.wav(0.47507)
- FN (9): real_008_opus.wav(0.465056), real_019_opus.wav(0.463443), real_025_opus.wav(0.462164), real_029_opus.wav(0.463283), real_032_opus.wav(0.453579), real_034_opus.wav(0.457776), real_037_opus.wav(0.455722), real_042_opus.wav(0.467678), real_047_opus.wav(0.459346)

### raw_mic
- FP (4): fake_replay_tts_000.wav(0.478494), fake_replay_tts_001.wav(0.531797), fake_replay_tts_002.wav(0.485769), fake_replay_tts_004.wav(0.477584)
- FN (2): real_mic_003.wav(0.461713), real_mic_005.wav(0.450651)

### callnorm_mic_g711
- FP (0): none
- FN (2): real_mic_005_g711.wav(0.453379), real_mic_014_g711.wav(0.455751)

### callnorm_mic_opus
- FP (1): fake_replay_tts_001_opus.wav(0.470874)
- FN (5): real_mic_000_opus.wav(0.459269), real_mic_003_opus.wav(0.460247), real_mic_005_opus.wav(0.450741), real_mic_014_opus.wav(0.451404), real_mic_015_opus.wav(0.451355)
