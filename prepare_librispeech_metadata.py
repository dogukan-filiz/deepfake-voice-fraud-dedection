"""LibriSpeech -> data/metadata.csv hazirlayan script.

- LibriSpeech'i hiyerarsisini bozmadan `data/real/LibriSpeech` altina koydugun varsayilir.
- SPEAKERS.TXT'ten speaker -> cinsiyet (M/F) haritasi okunur.
- Secilen subset'lerdeki (.flac) dosyalar taranir ve
  `data/metadata.csv` dosyasina path,label,gender,source kolonlariyla yazilir.

Bu script sadece REAL (label=1) ornekleri yazar. TTS ile uretecegin
FAKE (label=0) ornekleri daha sonra metadata.csv'ye ekleyebilirsin.

Kullanim (proje kokunden):

    python prepare_librispeech_metadata.py

Gerekirse ayarlari dosya basindaki sabitlerden degistirebilirsin.
"""

from __future__ import annotations

import csv
import os
from pathlib import Path
from typing import Dict, List, Optional

# Proje kokune gore ayarlar
PROJECT_ROOT = Path(__file__).resolve().parent
DATA_DIR = PROJECT_ROOT / "data"
LIBRISPEECH_ROOT = DATA_DIR / "real" / "LibriSpeech"
OUTPUT_CSV = DATA_DIR / "metadata.csv"

# Hangi subset'leri kullanacagimizi burada belirliyoruz.
# Ornek: sadece test-other, istersen buraya "train-clean-100" vb. ekleyebilirsin.
SUBSETS: List[str] = [
    "test-other",
]


def load_speakers(path: Path) -> Dict[str, str]:
    """SPEAKERS.TXT'ten speaker_id -> gender (M/F) haritasi dondur.

    Dosyada genelde yorum satirlari ';' ile baslar, onlari atliyoruz.
    Beklenen format kabaca:

        367 M ...
        8461 F ...
    """

    if not path.is_file():
        raise FileNotFoundError(f"SPEAKERS.TXT bulunamadi: {path}")

    speaker_gender: Dict[str, str] = {}
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            raw = line.rstrip("\n")
            if not raw or raw.lstrip().startswith(";"):
                continue

            # Beklenen format:
            # ID  |SEX| SUBSET           |MINUTES| NAME
            # 14   | F | train-clean-360  | 25.03 | Kristin LeMoine
            # Bu nedenle once '|' ile ayiralim.
            parts = [p.strip() for p in raw.split("|")]
            if len(parts) < 2:
                continue

            spk_id = parts[0].split()[0]  # '14   ' -> '14'
            gender = parts[1].upper()     # 'F' veya 'M'
            if gender not in {"M", "F"}:
                gender = "U"  # unknown
            speaker_gender[spk_id] = gender
    if not speaker_gender:
        raise RuntimeError("SPEAKERS.TXT bos veya cozumlenemedi.")
    return speaker_gender


def _load_transcript_for_utterance(flac_path: Path) -> Optional[str]:
    """Verilen .flac dosyasi icin ilgili .trans.txt dosyasindan transcript'i bul.

    Ornek yol:
        data/real/LibriSpeech/test-other/367/130732/367-130732-0000.flac

    Transcript dosyasi:
        data/real/LibriSpeech/test-other/367/130732/367-130732.trans.txt

    Satir formati:
        367-130732-0000 LOBSTERS AND LOBSTERS
    """

    chapter_dir = flac_path.parent  # .../130732
    stem = flac_path.stem  # 367-130732-0000
    trans_path = chapter_dir / f"{stem.rsplit('-', 1)[0]}.trans.txt"  # 367-130732.trans.txt

    if not trans_path.is_file():
        return None

    utt_id = stem  # 367-130732-0000
    try:
        with trans_path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                if line.startswith(utt_id + " "):
                    # ilk bosluktan sonrasini text olarak al
                    return line.split(" ", 1)[1].strip()
    except OSError:
        return None

    return None


def collect_librispeech_rows() -> List[Dict[str, str]]:
    """LibriSpeech altindaki .flac dosyalari icin metadata satirlari olustur.

    Artik her satir icin soyle bir yapi donuyoruz:

        {"path", "label", "gender", "source", "text"}

    text alanini ilgili .trans.txt dosyasindan okumaya calisiriz; bulunamazsa
    bos string olarak birakiriz.
    """

    if not LIBRISPEECH_ROOT.exists():
        raise FileNotFoundError(
            f"LibriSpeech klasoru bulunamadi: {LIBRISPEECH_ROOT}.\n"
            "Lutfen LibriSpeech'i data/real/LibriSpeech altina koy." 
        )

    speakers_txt = LIBRISPEECH_ROOT / "SPEAKERS.TXT"
    speaker_gender = load_speakers(speakers_txt)

    rows: List[Dict[str, str]] = []

    for subset in SUBSETS:
        subset_dir = LIBRISPEECH_ROOT / subset
        if not subset_dir.is_dir():
            print(f"[UYARI] Subset klasoru bulunamadi, atliyorum: {subset_dir}")
            continue

        for flac_path in subset_dir.rglob("*.flac"):
            # Ornek yol: .../test-other/367/130732/367-130732-0000.flac
            # Burada speaker id, dosyanin bir ust klasorunun adidir ("367").
            speaker_dir = flac_path.parent.parent  # 130732 klasorunun ustu -> 367
            speaker_id = speaker_dir.name
            gender = speaker_gender.get(speaker_id, "U")

            transcript = _load_transcript_for_utterance(flac_path) or ""

            # metadata.csv icinde proje kokune gore goreli path kullanalim
            try:
                rel_path = flac_path.relative_to(PROJECT_ROOT).as_posix()
            except ValueError:
                # Her ihtimale karsi, relative_to basarisiz olursa normal path yaz
                rel_path = flac_path.as_posix()

            rows.append(
                {
                    "path": rel_path,
                    "label": "1",  # real
                    "gender": gender,
                    "source": "real-librispeech",
                    "text": transcript,
                }
            )

    if not rows:
        raise RuntimeError("LibriSpeech altinda hic .flac dosyasi bulunamadi.")

    return rows


def merge_with_existing_metadata(new_rows: List[Dict[str, str]]) -> List[Dict[str, str]]:
    """Var olan metadata.csv'yi (varsa) okur ve yeni satirlarla birlestirir.

    - Eski dosyada sadece path,label kolonlari olabilir. gender/source yoksa
      'U' ve 'unknown' olarak doldururuz.
    - path'i ayni olan satirlari tekrar etmeyiz (onceki kayit onecelikli kalir).
    """

    if not OUTPUT_CSV.is_file():
        return new_rows

    existing: Dict[str, Dict[str, str]] = {}
    with OUTPUT_CSV.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(row for row in f if not row.lstrip().startswith("#"))
        for row in reader:
            path = row.get("path")
            if not path:
                continue
            normalized = {
                "path": path,
                "label": row.get("label", "1"),
                "gender": row.get("gender", "U"),
                "source": row.get("source", "unknown"),
                "text": row.get("text", ""),
            }
            existing[path] = normalized

    for r in new_rows:
        p = r["path"]
        if p not in existing:
            existing[p] = r

    return list(existing.values())


def main() -> None:
    print(f"Proje kok: {PROJECT_ROOT}")
    print(f"LibriSpeech kok: {LIBRISPEECH_ROOT}")

    rows = collect_librispeech_rows()
    print(f"LibriSpeech'ten {len(rows)} adet REAL ornek toplandi.")

    merged = merge_with_existing_metadata(rows)
    print(f"Toplam metadata satiri (eski + yeni): {len(merged)}")

    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT_CSV.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["path", "label", "gender", "source", "text"],
        )
        writer.writeheader()
        for r in merged:
            writer.writerow(r)

    print(f"Metadata yazildi: {OUTPUT_CSV}")
    print("Bu dosyaya daha sonra TTS ile uretilen FAKE ornekleri (label=0) ekleyebilirsin.")


if __name__ == "__main__":
    main()
