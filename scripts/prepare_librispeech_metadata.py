"""Yeni veri yapisindan `data/metadata.csv` ureten script.

Beklenen duzen:

    dataset_root/
        for-original/
            training/
                real|fake/*.wav
            validation/
                real|fake/*.wav
            testing/
                real|fake/*.wav
        for-rerecorded/
            training/
                real|fake/*.wav
            validation/
                real|fake/*.wav
            testing/
                real|fake/*.wav

Uretilen metadata kolonlari:

    path,label,split,source

Etiketler:

    real -> 0
    fake -> 1

Kullanim (proje kokunden):

    python prepare_librispeech_metadata.py

Istersen farkli veri koku ve cikis dosyasi da verebilirsin:

    python prepare_librispeech_metadata.py --dataset_root "C:\\Users\\DOGUKAN\\OneDrive\\Masaüstü\\dataset" --output_csv data/metadata.csv
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Dict, Iterable, List

PROJECT_ROOT = Path(__file__).resolve().parent
DATASET_ROOT = Path(r"C:\Users\DOGUKAN\OneDrive\Masaüstü\dataset")
OUTPUT_CSV = PROJECT_ROOT / "data" / "metadata.csv"
COLLECTIONS = ("for-original", "for-rerecorded")
SPLITS = ("training", "validation", "testing")
CLASS_TO_LABEL = {"real": "0", "fake": "1"}
ALLOWED_AUDIO_SUFFIXES = {".wav", ".flac", ".mp3", ".ogg", ".m4a", ".webm", ".aac", ".opus"}


def _collect_audio_rows(dataset_root: Path) -> List[Dict[str, str]]:
    if not dataset_root.is_dir():
        raise FileNotFoundError(f"Veri koku bulunamadi: {dataset_root}")

    rows: List[Dict[str, str]] = []
    for collection in COLLECTIONS:
        collection_dir = dataset_root / collection
        if not collection_dir.is_dir():
            print(f"[UYARI] Klasor bulunamadi, atliyorum: {collection_dir}")
            continue

        for split in SPLITS:
            split_dir = collection_dir / split
            if not split_dir.is_dir():
                print(f"[UYARI] Split klasoru bulunamadi, atliyorum: {split_dir}")
                continue

            for class_name, label in CLASS_TO_LABEL.items():
                class_dir = split_dir / class_name
                if not class_dir.is_dir():
                    print(f"[UYARI] Sinif klasoru bulunamadi, atliyorum: {class_dir}")
                    continue

                for audio_path in class_dir.rglob("*"):
                    if not audio_path.is_file():
                        continue
                    if audio_path.suffix.lower() not in ALLOWED_AUDIO_SUFFIXES:
                        continue

                    rows.append(
                        {
                            "path": str(audio_path.resolve()),
                            "label": label,
                            "split": split,
                            "source": collection,
                        }
                    )

    return rows


def _log_counts(rows: Iterable[Dict[str, str]]) -> None:
    rows_list = list(rows)
    counts = {"0": 0, "1": 0}
    split_counts: Dict[str, Dict[str, int]] = {}
    for row in rows_list:
        label = row["label"]
        split = row["split"]
        counts[label] = counts.get(label, 0) + 1
        split_counts.setdefault(split, {"0": 0, "1": 0})
        split_counts[split][label] = split_counts[split].get(label, 0) + 1

    print(f"Sinif dagilimi -> real(0): {counts.get('0', 0)}, fake(1): {counts.get('1', 0)}")
    for split, split_count in split_counts.items():
        print(f"  {split}: real(0)={split_count.get('0', 0)}, fake(1)={split_count.get('1', 0)}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Yeni dataset yapisindan metadata uret")
    parser.add_argument("--dataset_root", type=str, default=str(DATASET_ROOT), help="for-original ve for-rerecorded klasorlerini iceren veri koku")
    parser.add_argument("--output_csv", type=str, default=str(OUTPUT_CSV), help="Yazilacak metadata CSV yolu")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dataset_root = Path(args.dataset_root)
    output_csv = Path(args.output_csv)

    print(f"Proje kok: {PROJECT_ROOT}")
    print(f"Veri koku: {dataset_root}")

    rows = _collect_audio_rows(dataset_root)
    if not rows:
        raise RuntimeError(f"{dataset_root} altinda metadata olusturulacak audio dosyasi bulunamadi.")

    rows.sort(key=lambda row: (row["split"], row["source"], row["label"], row["path"]))
    print(f"Toplam metadata satiri: {len(rows)}")
    _log_counts(rows)

    print("Ilk 20 metadata satiri:")
    for row in rows[:20]:
        print("  ", row["path"], "label=", row["label"], "split=", row["split"], "source=", row["source"])

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["path", "label", "split", "source"])
        writer.writeheader()
        writer.writerows(rows)

    print(f"Metadata yazildi: {output_csv}")


if __name__ == "__main__":
    main()
