# full_run_002 — Konsolide Benchmark Raporu

**Model:** full_run_002 (XLSR-300M frozen backbone + AASIST head)
**Base:** models/ssl_aasist/weights.pth
**Head:** training_runs/full_run_002/best_head.pth
**Threshold:** 0.49 | **CALL_CHANNEL_MODE:** false
**Tarih:** 2026-06-05

## Sonuç Tablosu

| Test seti | Domain | n | Accuracy | Macro F1 | Real recall | Fake recall |
|-----------|--------|---|----------|----------|-------------|-------------|
| test_audio (50+50) | in-domain dijital | 100 | **%100.0** | 1.000 | 1.00 | 1.00 |
| FoR for-original | OOD dijital | 399 | **%83.2** | 0.829 | 0.70 | 0.96 |
| FoR for-rerecorded | OOD mic/replay | 400 | **%55.5** | 0.451 | 0.12 | 0.99 |

> FoR ölçümlerinde AUDIO_MIN_DURATION_SEC=1.0 (FoR klipleri 1-2s). test_audio klipleri >2s.
> Her domain dengeli örneklem: for-* = 200 real + 200 fake.

## Confusion Matrix

**test_audio:** TP=50 TN=50 FP=0 FN=0
**FoR for-original:** TP=192 TN=140 FP=59 FN=8
**FoR for-rerecorded:** TP=198 TN=24 FP=176 FN=2

## Skor Dağılımı (p_real)

| Domain | Real mean | Fake mean | Ayrışma |
|--------|-----------|-----------|---------|
| test_audio | 0.958 | 0.083 | net |
| for-original | 0.632 | 0.083 | orta |
| for-rerecorded | 0.238 | 0.125 | çökme |

## Yorum

1. **In-domain mükemmel:** test_audio %100, skorlar net ayrışıyor.
2. **OOD dijital sağlam:** Sahte tespiti %96, fakat real recall %70'e düşüyor — 59 gerçek ses sahte damgalanıyor.
3. **OOD mic/replay çöküş:** Real recall %12 (176/200 gerçek → sahte). Model bu domain'de neredeyse her şeyi sahte sayıyor; accuracy %55.5 (şans seviyesine yakın).
4. **Genel örüntü:** Model güçlü bir SAHTE detektörü (fake recall domainlerde 1.00/0.96/0.99 kararlı) ama domain dışına çıktıkça gerçek sesleri aşırı sahte-işaretliyor (real recall 1.00 → 0.70 → 0.12).

## Tez/Doküman için defansif ifade

> "Model in-domain test setinde %100 doğruluk sağlamış; bağımsız FoR dijital alt kümesinde %83.2 (sahte duyarlılığı %96) doğruluğa ulaşmıştır. Hoparlörden yeniden kaydedilmiş (replay) alt kümede performans %55.5'e düşmüştür; bu, modelin mikrofon/replay domain'ine genelleme sınırını gösterir ve gelecek çalışma olarak domain-çeşitli eğitim (full_run_004 call-channel hattı) önerilmektedir."

## Kayıtlar
- training_runs/full_run_002/for_eval/for_eval_summary.json
- training_runs/full_run_002/for_eval/for_eval_per_file.csv
- training_runs/full_run_002/final_local_test_audio_eval.json
