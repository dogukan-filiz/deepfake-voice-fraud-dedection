# SSL+AASIST (XLSR) Integration — Design Spec

**Date:** 2026-05-28
**Author:** Doğukan Filiz (w/ Claude Code)
**Status:** Draft → awaiting user review

## Context

The current production model in `backend/model_wrapper.py` chains:
`Speech-Arena-2025/DF_Arena_1B_V_1` (HuggingFace) → AASIST baseline → Heuristic fallback.

The HF `DF_Arena_1B_V_1` model gives unsatisfactory results on real bank-call-center
audio in our test set. We initially considered RawNet2 (eurecom-asp/rawnet2-antispoofing),
but the repo ships no pretrained weights — training on ASVspoof2019 LA from scratch
is out of scope.

This spec replaces `DF_Arena_1B_V_1` with **SSL+AASIST** (XLSR-300M frontend +
AASIST classifier), the SOTA Tak et al. model. It achieves 0.82% EER on ASVspoof
LA21 and 2.85% on DF21 — substantially stronger generalization than vanilla AASIST.

Source: https://github.com/TakHemlata/SSL_Anti-spoofing (MIT license).

## Goals

- Replace `DF_Arena_1B_V_1` with SSL+AASIST as the primary inference model.
- Keep vanilla AASIST and Heuristic as fallbacks (preserve graceful degradation).
- Avoid the `fairseq` dependency — load XLSR via HuggingFace `transformers`.
- Reuse existing chunking (64600-sample windows) and risk-weighted aggregation
  (60% mean + 40% worst-case) in `backend/model_wrapper.py`.
- Verify accuracy on the existing 100-sample test library before declaring done.

## Non-Goals

- Training or fine-tuning. We use TakHemlata's published weights as-is.
- Adding `fairseq` to dependencies.
- Frontend or API changes (request/response shape unchanged).
- Multi-model ensemble (single primary, sequential fallbacks only).

## Architecture

### Model Stack (new)

```
SSL+AASIST (XLSR-300M + AASIST head)   ← primary
        ↓ (load failure / inference exception)
AASIST baseline (clovaai weights)       ← fallback
        ↓
HeuristicFallbackModel                  ← last resort
```

### Components

1. **XLSR frontend** — `facebook/wav2vec2-xls-r-300m` loaded via
   `transformers.Wav2Vec2Model.from_pretrained(...)`. Auto-downloads to HF cache
   on first run (~1.2 GB). Produces `last_hidden_state` of shape `[B, T, 1024]`
   for input waveforms.

2. **AASIST classifier head** — ported from TakHemlata/SSL_Anti-spoofing `model.py`
   into `backend/aasist/models/SSLAASIST.py`. Same graph-attention architecture
   as our existing `backend/aasist/models/AASIST.py`, but configured for
   1024-dim XLSR features (vs raw waveform input).

3. **Wrapper class** — new `XLSRAASISTModel` in `backend/model_wrapper.py`,
   following the existing `DeepfakeVoiceModel` interface
   (`predict(waveform) → {p_real, p_fake, ...}`).

### File Layout

```
backend/
  aasist/models/
    AASIST.py              # existing, unchanged
    SSLAASIST.py           # NEW — ported from TakHemlata repo
  model_wrapper.py         # MODIFIED — strip HF DF_Arena, add XLSRAASISTModel
  config.py                # MODIFIED — add LOCAL_SSL_MODEL_DIR env var
models/
  aasist_baseline/         # existing
  ssl_aasist/              # NEW (gitignored)
    weights.pth            # manual download from TakHemlata Google Drive
    meta.json              # model metadata
docs/superpowers/specs/
  2026-05-28-ssl-aasist-integration-design.md  # this file
```

### Loader Priority

`get_model()` in `backend/model_wrapper.py`:

1. Try `XLSRAASISTModel`:
   - Resolve weights via `LOCAL_SSL_MODEL_DIR` env var, else `models/ssl_aasist/`.
   - If `weights.pth` missing → log warning, skip.
   - Load XLSR via `transformers`; load AASIST head; load `.pth` state dict.
   - GPU if `torch.cuda.is_available()`, else CPU.
2. Else `DeepfakeVoiceModel` (vanilla AASIST, existing logic).
3. Else `HeuristicFallbackModel`.

### Inference Flow (unchanged)

`backend/audio_processing.py` chunking stays as-is (16 kHz mono, 64600-sample
non-overlapping windows, silent-chunk filter). `XLSRAASISTModel.predict()` runs
XLSR + AASIST head per chunk, returns per-chunk `p_fake`. Aggregation in the
wrapper (60% mean + 40% max) stays as-is. Spectral-anomaly adjustment above 0.6
stays as-is.

## Dependencies

- `transformers >= 4.30` (already in `requirements.txt`; pin version)
- `torch >= 2.0` (already in)
- `soundfile`, `librosa` (already in)
- **No** `fairseq`

## Risks & Mitigations

| Risk | Mitigation |
|------|-----------|
| `transformers` Wav2Vec2 output shape/scale differs from `fairseq` XLSR features → trained head misbehaves | Both use the same `facebook/wav2vec2-xls-r-300m` checkpoint. Verify by running on 100-sample test set and confirming accuracy ≥ vanilla AASIST. If divergence, add a feature-extraction shim. |
| XLSR-300M slow on CPU (~2-3s per 4s chunk) | Auto-detect GPU. Document expected latency in README. If unacceptable, consider distilled XLSR-128. |
| 1.2GB weight download blocks first startup | Pre-download in setup script; document in README; cache survives across runs. |
| TakHemlata weights are a single Google Drive link (no HF mirror) | Add explicit manual-download instructions in README + `models/ssl_aasist/README.md`. Verify SHA256 in `meta.json`. |
| License attribution | SSL_Anti-spoofing repo is MIT. Preserve copyright header in ported `SSLAASIST.py`. |

## Verification Plan

End-to-end checks before considering the work complete:

1. **Unit smoke** — `python tests/smoke_test_prechecks.py` passes (silence/duration rejection unaffected).
2. **Health endpoint** — `MODEL_SELFTEST=1` + `GET /health` returns `model_selftest.ok=true` with new model name `xlsr_aasist`.
3. **Integration accuracy** — `python tests/comprehensive_test.py` on 50 real + 50 fake samples:
   - Target: accuracy ≥ current AASIST baseline, F1 improvement on fake class.
   - Save results to `docs/model_evaluation_results.json` for comparison.
4. **Threshold sweep** — `python scripts/evaluation/enhanced_evaluate.py` with thresholds 0.30–0.40, pick best F1.
5. **Manual UI smoke** — start backend + frontend, upload a known-fake WAV and a known-real WAV, confirm verdict banner and risk classification.
6. **GPU/CPU parity** — if GPU available, confirm CPU fallback path also runs (set `CUDA_VISIBLE_DEVICES=`).

## Out of Scope (future work)

- Ensemble with a second model (e.g., MelodyMachine V2) for analyst-review flagging.
- Streaming inference (current pipeline is file-at-a-time).
- Fine-tuning SSL+AASIST on Turkish bank-call audio.
