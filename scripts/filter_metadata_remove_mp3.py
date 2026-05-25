"""
Metadata'dan MP3 satırlarını kaldır (WAV'lar ile devam et).
"""

import csv
from pathlib import Path

metadata_path = Path(r"D:\Workspace\deepfake-voice-fraud-dedection\data\metadata.csv")
backup_path = metadata_path.with_stem(f"{metadata_path.stem}_backup")

if not backup_path.exists():
    metadata_path.rename(backup_path)
    print(f"Backup: {backup_path}")
else:
    print(f"Backup zaten var: {backup_path}")

# WAV'ları filter et
rows = []
with backup_path.open('r', encoding='utf-8') as f:
    reader = csv.DictReader(f)
    fieldnames = reader.fieldnames
    for row in reader:
        path = row['path'].strip()
        if not path.lower().endswith('.mp3'):
            rows.append(row)

print(f"Total satır: {len(rows)} (MP3'ler kaldırıldı)")

# Yeni metadata yaz
with metadata_path.open('w', newline='', encoding='utf-8') as f:
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(rows)

print(f"Yazıldı: {metadata_path}")
