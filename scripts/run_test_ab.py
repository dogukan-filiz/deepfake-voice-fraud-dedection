"""
scripts/run_test_ab.py
A/B evaluation: digital TTS vs replayed TTS pairs.
Outputs CSV, JSON summary, score-gap table.
Usage: python3 scripts/run_test_ab.py [--output_dir training_runs/test_ab_eval]
"""
import sys, os, json, csv, argparse
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)


def load_model(threshold=0.49):
    os.environ.setdefault("LOCAL_SSL_MODEL_DIR", "models/ssl_aasist")
    os.chdir(ROOT)
    from backend.model_wrapper import get_model
    model = get_model()
    return model, threshold


def analyze_wav(model, wav_path):
    import soundfile as sf
    data, sr = sf.read(wav_path)
    if data.ndim > 1:
        data = data.mean(axis=1)
    if sr != 16000:
        import librosa
        data = librosa.resample(data, orig_sr=sr, target_sr=16000)
        sr = 16000
    data = data.astype(np.float32)
    out = model.predict({"waveform": data, "sr": sr})
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--digital_dir", default="test_ab/a_digital")
    parser.add_argument("--replayed_dir", default="test_ab/b_replayed")
    parser.add_argument("--threshold", type=float, default=0.49)
    parser.add_argument("--output_dir", default="training_runs/test_ab_eval")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    model, thr = load_model(args.threshold)
    print("Model loaded. Threshold={}".format(thr))

    digital_dir = os.path.join(ROOT, args.digital_dir)
    replayed_dir = os.path.join(ROOT, args.replayed_dir)

    digital_files = sorted([f for f in os.listdir(digital_dir) if f.endswith(".wav")])
    replayed_files = sorted([f for f in os.listdir(replayed_dir) if f.endswith(".wav")])

    n_pairs = min(len(digital_files), len(replayed_files))
    print("\nFound {} digital, {} replayed files ({} pairs)\n".format(
        len(digital_files), len(replayed_files), n_pairs))

    header = "{:<4} {:<28} {:>7} {:<8} | {:<28} {:>7} {:<8} {:>8}".format(
        "#", "Digital File", "p_real", "Label", "Replayed File", "p_real", "Label", "Gap")
    print(header)
    print("-" * 100)

    results = []
    pairs = []

    for i in range(n_pairs):
        df = digital_files[i]
        rf = replayed_files[i]
        d_path = os.path.join(digital_dir, df)
        r_path = os.path.join(replayed_dir, rf)

        d_out = analyze_wav(model, d_path)
        r_out = analyze_wav(model, r_path)

        d_score = d_out["p_real"]
        r_score = r_out["p_real"]
        gap = r_score - d_score

        d_label = "FRAUD" if d_score < thr else "CLEAN"
        r_label = "FRAUD" if r_score < thr else "CLEAN"
        d_correct = d_label == "FRAUD"
        r_correct = r_label == "FRAUD"

        print("{:<4} {:<28} {:>7.3f} {:<8} | {:<28} {:>7.3f} {:<8} {:>+8.3f}".format(
            i, df, d_score, d_label, rf, r_score, r_label, gap))

        pairs.append({
            "pair_id": i,
            "digital_file": df,
            "digital_p_real": round(d_score, 4),
            "digital_label": d_label,
            "digital_correct": d_correct,
            "replayed_file": rf,
            "replayed_p_real": round(r_score, 4),
            "replayed_label": r_label,
            "replayed_correct": r_correct,
            "gap": round(gap, 4),
        })

    # All per-file rows (re-run to populate results list cleanly)
    for f in digital_files:
        out = analyze_wav(model, os.path.join(digital_dir, f))
        results.append({
            "file": f, "domain": "digital", "label_true": "fake",
            "p_real": round(out["p_real"], 4),
            "p_fake": round(out["p_fake"], 4),
            "predicted": "FRAUD" if out["p_real"] < thr else "CLEAN",
            "correct": out["p_real"] < thr,
        })
    for f in replayed_files:
        out = analyze_wav(model, os.path.join(replayed_dir, f))
        results.append({
            "file": f, "domain": "replayed", "label_true": "fake",
            "p_real": round(out["p_real"], 4),
            "p_fake": round(out["p_fake"], 4),
            "predicted": "FRAUD" if out["p_real"] < thr else "CLEAN",
            "correct": out["p_real"] < thr,
        })

    digital_results = [r for r in results if r["domain"] == "digital"]
    replayed_results = [r for r in results if r["domain"] == "replayed"]
    d_acc = sum(r["correct"] for r in digital_results) / max(len(digital_results), 1)
    r_acc = sum(r["correct"] for r in replayed_results) / max(len(replayed_results), 1)
    d_scores = [r["p_real"] for r in digital_results]
    r_scores = [r["p_real"] for r in replayed_results]
    gaps = [p["gap"] for p in pairs]

    summary = {
        "threshold": thr,
        "digital": {
            "n": len(digital_results),
            "correctly_fraud": sum(r["correct"] for r in digital_results),
            "accuracy": round(d_acc, 4),
            "mean_p_real": round(float(np.mean(d_scores)), 4),
            "max_p_real": round(float(max(d_scores)), 4),
        },
        "replayed": {
            "n": len(replayed_results),
            "correctly_fraud": sum(r["correct"] for r in replayed_results),
            "accuracy": round(r_acc, 4),
            "mean_p_real": round(float(np.mean(r_scores)), 4),
            "max_p_real": round(float(max(r_scores)), 4),
        },
        "domain_gap": {
            "mean_gap": round(float(np.mean(gaps)), 4),
            "max_gap": round(float(max(gaps)), 4),
            "false_negatives_replayed": sum(1 for p in pairs if not p["replayed_correct"]),
        },
        "pairs": pairs,
    }

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print("Digital TTS:  {}/{} correct ({:.0f}%)  mean_p_real={:.3f}".format(
        summary["digital"]["correctly_fraud"], summary["digital"]["n"],
        d_acc * 100, summary["digital"]["mean_p_real"]))
    print("Replayed TTS: {}/{} correct ({:.0f}%)  mean_p_real={:.3f}".format(
        summary["replayed"]["correctly_fraud"], summary["replayed"]["n"],
        r_acc * 100, summary["replayed"]["mean_p_real"]))
    print("Mean domain gap: {:+.3f}".format(summary["domain_gap"]["mean_gap"]))
    print("False negatives (replayed): {}/{}".format(
        summary["domain_gap"]["false_negatives_replayed"], n_pairs))

    csv_path = os.path.join(args.output_dir, "test_ab_results.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=results[0].keys())
        writer.writeheader()
        writer.writerows(results)

    json_path = os.path.join(args.output_dir, "test_ab_summary.json")
    with open(json_path, "w") as f:
        json.dump(summary, f, indent=2)

    print("\nOutputs written:")
    print("  " + csv_path)
    print("  " + json_path)


if __name__ == "__main__":
    main()
