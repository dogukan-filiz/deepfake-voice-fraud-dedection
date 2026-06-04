"""
scripts/evaluate_microphone_benchmark.py
Evaluate model on microphone_benchmark/ dataset.
Domains: real_mic (label=real) and fake_replayed (label=fake).
Outputs CSV, JSON summary, confusion matrix.
Usage: python3 scripts/evaluate_microphone_benchmark.py [--output_dir training_runs/mic_eval]
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


def print_confusion_matrix(tp, fp, tn, fn):
    print("              Pred CLEAN  Pred FRAUD")
    print("  True REAL:     {:>5}      {:>5}".format(tn, fp))
    print("  True FAKE:     {:>5}      {:>5}".format(fn, tp))
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall_fake = tp / (tp + fn) if (tp + fn) > 0 else 0
    recall_real = tn / (tn + fp) if (tn + fp) > 0 else 0
    f1 = 2 * precision * recall_fake / (precision + recall_fake) if (precision + recall_fake) > 0 else 0
    print("\n  Fake recall (sensitivity): {:.1f}%  ({}/{} fakes caught)".format(
        recall_fake * 100, tp, tp + fn))
    print("  Real recall (specificity): {:.1f}%  ({}/{} reals correctly clean)".format(
        recall_real * 100, tn, tn + fp))
    print("  Precision (fraud):         {:.1f}%".format(precision * 100))
    print("  Macro F1:                  {:.4f}".format(f1))
    return {"precision": round(precision, 4), "recall_fake": round(recall_fake, 4),
            "recall_real": round(recall_real, 4), "macro_f1": round(f1, 4)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--real_dir", default="microphone_benchmark/real")
    parser.add_argument("--fake_dir", default="microphone_benchmark/fake_replayed")
    parser.add_argument("--threshold", type=float, default=0.49)
    parser.add_argument("--output_dir", default="training_runs/mic_eval")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    model, thr = load_model(args.threshold)
    print("Model loaded. Threshold={}".format(thr))

    real_dir = os.path.join(ROOT, args.real_dir)
    fake_dir = os.path.join(ROOT, args.fake_dir)

    results = []
    tp = fp = tn = fn = 0

    for label_true, d in [("real", real_dir), ("fake", fake_dir)]:
        if not os.path.isdir(d):
            print("WARN: {} not found, skipping".format(d))
            continue
        files = sorted([f for f in os.listdir(d) if f.endswith(".wav")])
        print("\n--- {} ({} files) ---".format(label_true.upper(), len(files)))
        for fname in files:
            path = os.path.join(d, fname)
            try:
                out = analyze_wav(model, path)
                p_real = out["p_real"]
                predicted = "FRAUD" if p_real < thr else "CLEAN"
                expected = "FRAUD" if label_true == "fake" else "CLEAN"
                correct = predicted == expected

                if label_true == "fake":
                    if predicted == "FRAUD":
                        tp += 1
                    else:
                        fn += 1
                else:
                    if predicted == "CLEAN":
                        tn += 1
                    else:
                        fp += 1

                marker = "OK" if correct else "WRONG"
                print("  {:<35} p_real={:.3f}  {:<6}  [{}]".format(
                    fname, p_real, predicted, marker))

                results.append({
                    "file": fname,
                    "domain": "real_mic" if label_true == "real" else "fake_rir",
                    "label_true": label_true,
                    "p_real": round(p_real, 4),
                    "p_fake": round(out["p_fake"], 4),
                    "predicted": predicted,
                    "correct": correct,
                    "spectral_residual": round(float(out.get("spectral_residual", 0)), 4),
                })
            except Exception as e:
                print("  ERROR {}: {}".format(fname, e))
                results.append({
                    "file": fname, "domain": "error", "label_true": label_true,
                    "p_real": -1, "p_fake": -1, "predicted": "ERROR",
                    "correct": False, "spectral_residual": -1,
                })

    total = tp + fp + tn + fn
    acc = (tp + tn) / total if total > 0 else 0

    print("\n" + "=" * 60)
    print("CONFUSION MATRIX (threshold={})".format(thr))
    print("=" * 60)
    metrics = print_confusion_matrix(tp, fp, tn, fn)

    real_results = [r for r in results if r["label_true"] == "real"]
    fake_results = [r for r in results if r["label_true"] == "fake"]

    print("\n" + "=" * 60)
    print("DOMAIN GAP ANALYSIS")
    print("=" * 60)
    if real_results:
        r_scores = [r["p_real"] for r in real_results]
        print("Real mic p_real: mean={:.3f}  min={:.3f}  max={:.3f}".format(
            np.mean(r_scores), min(r_scores), max(r_scores)))
        fps_list = [r for r in real_results if not r["correct"]]
        print("False positives (real as FRAUD): {}/{}".format(len(fps_list), len(real_results)))
        for r in fps_list:
            print("  FP: {}  p_real={}".format(r["file"], r["p_real"]))

    if fake_results:
        f_scores = [r["p_real"] for r in fake_results]
        print("\nFake RIR p_real: mean={:.3f}  min={:.3f}  max={:.3f}".format(
            np.mean(f_scores), min(f_scores), max(f_scores)))
        fns_list = [r for r in fake_results if not r["correct"]]
        print("False negatives (fake as CLEAN): {}/{}".format(len(fns_list), len(fake_results)))
        for r in fns_list:
            print("  FN: {}  p_real={}".format(r["file"], r["p_real"]))

    all_real_scores = [r["p_real"] for r in real_results] if real_results else []
    all_fake_scores = [r["p_real"] for r in fake_results] if fake_results else []

    summary = {
        "threshold": thr,
        "total_files": total,
        "overall_accuracy": round(acc, 4),
        "metrics": metrics,
        "real_mic": {
            "n": len(real_results),
            "correctly_clean": tn,
            "false_positives": fp,
            "accuracy": round(tn / max(len(real_results), 1), 4),
            "p_real_mean": round(float(np.mean(all_real_scores)), 4) if all_real_scores else None,
            "p_real_min": round(float(min(all_real_scores)), 4) if all_real_scores else None,
            "p_real_max": round(float(max(all_real_scores)), 4) if all_real_scores else None,
        },
        "fake_rir": {
            "n": len(fake_results),
            "correctly_fraud": tp,
            "false_negatives": fn,
            "accuracy": round(tp / max(len(fake_results), 1), 4),
            "p_real_mean": round(float(np.mean(all_fake_scores)), 4) if all_fake_scores else None,
        },
        "confusion_matrix": {"TP": tp, "FP": fp, "TN": tn, "FN": fn},
        "per_file": results,
    }

    csv_path = os.path.join(args.output_dir, "mic_benchmark_results.csv")
    with open(csv_path, "w", newline="") as f:
        if results:
            writer = csv.DictWriter(f, fieldnames=results[0].keys())
            writer.writeheader()
            writer.writerows(results)

    json_path = os.path.join(args.output_dir, "mic_benchmark_summary.json")
    with open(json_path, "w") as f:
        json.dump(summary, f, indent=2)

    print("\nOutputs written:")
    print("  " + csv_path)
    print("  " + json_path)


if __name__ == "__main__":
    main()
