"""
MP3 dosyalarını WAV'a dönüştür.
"""

import csv
import sys
from pathlib import Path

try:
    import librosa
    import soundfile as sf
except ImportError:
    print("librosa ve soundfile gerekli: pip install librosa soundfile")
    sys.exit(1)

SAMPLE_RATE = 16000


def convert_metadata_mp3_to_wav(metadata_path: str, dry_run: bool = False) -> None:
    """Metadata'daki MP3 dosyalarını WAV'a dönüştür."""
    
    metadata_path = Path(metadata_path).expanduser().resolve()
    metadata_dir = metadata_path.parent
    
    conversions = []
    
    with metadata_path.open('r', encoding='utf-8') as f:
        reader = csv.DictReader(row for row in f if not row.lstrip().startswith("#"))
        for row in reader:
            audio_path_raw = row['path'].strip()
            audio_path = Path(audio_path_raw)
            if not audio_path.is_absolute():
                audio_path = (metadata_dir / audio_path).resolve()
            
            if str(audio_path).lower().endswith('.mp3'):
                wav_path = audio_path.with_suffix('.wav')
                conversions.append((audio_path, wav_path))
    
    if not conversions:
        print("MP3 dosyası bulunamadı.")
        return
    
    print(f"Dönüştürülecek MP3 dosyası: {len(conversions)}")
    
    if dry_run:
        print("[DRY RUN] Dönüştürmeler:")
        for mp3, wav in conversions[:10]:
            print(f"  {mp3.name} -> {wav.name}")
        return
    
    failed = []
    for i, (mp3_path, wav_path) in enumerate(conversions):
        try:
            if wav_path.exists():
                print(f"[{i+1}/{len(conversions)}] {mp3_path.name} -> WAV zaten var, skip")
                continue
            
            print(f"[{i+1}/{len(conversions)}] {mp3_path.name} dönüştürülüyor...", end=' ')
            audio, sr = librosa.load(str(mp3_path), sr=SAMPLE_RATE, mono=True)
            sf.write(str(wav_path), audio, sr)
            print("OK")
        except Exception as e:
            print(f"FAIL: {e}")
            failed.append((mp3_path, str(e)))
    
    if failed:
        print(f"\n{len(failed)} dosya başarısız:")
        for path, err in failed[:20]:
            print(f"  {path.name}: {err}")
    
    print(f"\nDönüştürülen: {len(conversions) - len(failed)}/{len(conversions)}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--metadata", default="data/metadata.csv")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    
    convert_metadata_mp3_to_wav(args.metadata, dry_run=args.dry_run)
