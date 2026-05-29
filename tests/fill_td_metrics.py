#!/usr/bin/env python3
"""
Read T1/T2/T3 + format_and_validation JSON results, write a markdown snippet
containing the filled tables for TD §7.2. Print to stdout for splice into TD.md.
"""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RES_DIR = Path(__file__).resolve().parent / "results"

T1_PATH = ROOT / "deepfake_test_results.json"
T2_PATH = RES_DIR / "T2_results.json"
T3_PATH = RES_DIR / "T3_results.json"
FV_PATH = RES_DIR / "format_and_validation_results.json"


def fmt(x, digits=4):
    if isinstance(x, float):
        return f"{x:.{digits}f}"
    return str(x)


def load(p: Path):
    if not p.exists():
        return None
    return json.loads(p.read_text(encoding="utf-8"))


def t1_block(d):
    if not d:
        return "_T1 results missing_"
    m = d["metrics"]
    # comprehensive_test.py uses sklearn default labels = sorted alphabetical = ['fake','real']
    cm = m["confusion_matrix"]
    fake_row, real_row = cm[0], cm[1]
    return f"""| Accuracy | {fmt(m['accuracy'])} |
| F1 (weighted) | {fmt(m['f1_score'])} |
| Real precision / recall | {fmt(m['real_metrics']['precision'])} / {fmt(m['real_metrics']['recall'])} |
| Fake precision / recall | {fmt(m['fake_metrics']['precision'])} / {fmt(m['fake_metrics']['recall'])} |
| Karışıklık matrisi (rows=actual, cols=pred [fake,real]) | fake:[{fake_row[0]}, {fake_row[1]}], real:[{real_row[0]}, {real_row[1]}] |
| Toplam istek | {d['total_files']} |
| Başarılı / Başarısız | {d['successful_analyses']} / {d['total_files'] - d['successful_analyses']} |"""


def dataset_block(d):
    if not d:
        return "_results missing_"
    m = d["metrics"]
    cm = m["confusion_matrix"]
    lab = m["confusion_matrix_labels"]
    rep = m["classification_report"]
    lat = d["latency_s"]
    real = rep.get("real", {})
    fake = rep.get("fake", {})
    return f"""| Accuracy | {fmt(m['accuracy'])} |
| F1 (weighted) | {fmt(m['f1_weighted'])} |
| Real precision / recall / F1 | {fmt(real.get('precision', 0))} / {fmt(real.get('recall', 0))} / {fmt(real.get('f1-score', 0))} |
| Fake precision / recall / F1 | {fmt(fake.get('precision', 0))} / {fmt(fake.get('recall', 0))} / {fmt(fake.get('f1-score', 0))} |
| Karışıklık matrisi (labels={lab}, rows=actual) | real:[{cm[0][0]}, {cm[0][1]}], fake:[{cm[1][0]}, {cm[1][1]}] |
| Toplam (denenen / başarılı / başarısız) | {d['n_total_attempted']} / {d['n_success']} / {d['n_failed']} |
| Latency mean / p50 / p95 (sn) | {fmt(lat['mean'], 3)} / {fmt(lat['p50'], 3)} / {fmt(lat['p95'], 3)} |
| Latency min / max (sn) | {fmt(lat['min'], 3)} / {fmt(lat['max'], 3)} |"""


def fv_block(d):
    if not d:
        return "_format/validation results missing_"
    lines = []
    for c in d["cases"]:
        sym = "✅" if c.get("passed") else "❌"
        lines.append(f"| {c['case']} | {sym} | status={c.get('status_code', '-')} |")
    return "\n".join(lines)


def main():
    t1 = load(T1_PATH)
    t2 = load(T2_PATH)
    t3 = load(T3_PATH)
    fv = load(FV_PATH)

    print("# TD §7.2 Filled Metrics (auto-generated)")
    print("\n## 7.2.1 T1 Smoke (50 + 50)\n")
    print("| Metrik | Değer |\n|---|---|")
    print(t1_block(t1))
    print("\n## 7.2.2 T2 In-domain\n")
    print("| Metrik | Değer |\n|---|---|")
    print(dataset_block(t2))
    print("\n## 7.2.3 T3 Cross-domain\n")
    print("| Metrik | Değer |\n|---|---|")
    print(dataset_block(t3))
    print("\n## 7.2.4 Format & Validation\n")
    print("| Test | Sonuç | Detay |\n|---|---|---|")
    print(fv_block(fv))


if __name__ == "__main__":
    main()
