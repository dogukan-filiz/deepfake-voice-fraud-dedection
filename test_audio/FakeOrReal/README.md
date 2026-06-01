# Fake-or-Real (FoR) Dataset

**Source:** https://zenodo.org/record/4234322  
**License:** Creative Commons Attribution 4.0 International  
**Paper:** Reimao & Tzerpos, *FoR: A Dataset for Synthetic Speech Detection*,
ICCST 2019.

## Description

The FoR dataset contains real and TTS-synthesised speech samples for binary
deepfake detection. Two variants are provided:

| Variant    | Description                              |
|------------|------------------------------------------|
| for-original | Raw recordings and TTS output          |
| for-norm   | Amplitude-normalised version (–23 LUFS) |

## Directory Layout

```
FakeOrReal/
  for-norm/
    testing/
      real/   genuine speech utterances (test split)
      fake/   TTS-synthesised utterances (test split)
    validation/
      real/
      fake/
```

## File Naming

`for_real_NNN.wav` — genuine speech, normalised  
`for_fake_NNN.wav` — TTS output, normalised  

## Statistics (this subset)

| Split   | Real | Fake |
|---------|------|------|
| testing | 16   | 16   |
| Total   | 16   | 16   |
