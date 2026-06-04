"""
scripts/prepare_balanced_dataset_v3.py

Builds metadata_v3.csv for full_run_003 training.

Supported sources (pass via --sources flag or auto-discovered):
  - test_audio/real          → real_digital
  - test_audio/fake          → fake_digital
  - data/common_voice_tr/    → real_microphone (after download)
  - data/asvspoof_pa/        → fake_replayed   (after download)
  - data/augmented/rir/      → fake_replayed   (RIR-augmented, generated)
  - data/augmented/codec/    → real_microphone (codec-chain augmented)
  - microphone_benchmark/real/ → real_microphone (HOLDOUT ONLY, never train)

Leakage prevention rules:
  - All 17 WhatsApp files → original_recording_group=whatsapp_local_real_group_001
    → assigned to split=holdout, never train or val
  - Speaker-level grouping for Common Voice (speaker_id from filename)
  - ASVspoof PA: speaker groups from official speaker list
  - Same original_recording_group never split across train/val/test

Usage:
  python3 scripts/prepare_balanced_dataset_v3.py \\
    --output_dir training_runs/full_run_003 \\
    --max_per_domain 2000 \\
    --val_ratio 0.15 \\
    --test_ratio 0.15 \\
    --dry_run          # prints stats without writing file
"""
import os, sys, json, csv, random, argparse, hashlib
from pathlib import Path
from collections import defaultdict

ROOT = Path(__file__).resolve().parent.parent
# External datasets root (outside project dir on RunPod)
DATASETS_ROOT = Path(os.environ.get("DATASETS_ROOT", str(ROOT.parent / "datasets")))


# ---------------------------------------------------------------------------
# Source definitions
# ---------------------------------------------------------------------------

SOURCE_DEFS = [
    # (dir_relative_to_ROOT, label, channel_type, codec, generation_type,
    #  language, split_override, original_recording_group_prefix)

    # --- HOLDOUT BENCHMARKS (never train/val/test) ---
    # test_audio is the final digital benchmark from full_run_002 onwards.
    # It must never enter training or validation splits.
    {
        "dir": "test_audio/real",
        "label": "real",
        "channel_type": "digital",
        "codec": "wav_pcm",
        "generation_type": "human_speech",
        "language": "mixed",
        "split_override": "holdout",
        "group_prefix": "digital_benchmark_real",
        "benchmark_name": "digital_benchmark",
    },
    {
        "dir": "test_audio/fake",
        "label": "fake",
        "channel_type": "digital",
        "codec": "wav_pcm",
        "generation_type": "tts_digital",
        "language": "en",
        "split_override": "holdout",
        "group_prefix": "digital_benchmark_fake",
        "benchmark_name": "digital_benchmark",
    },
    {
        "dir": "microphone_benchmark/real",
        "label": "real",
        "channel_type": "microphone",
        "codec": "opus_ogg",
        "generation_type": "human_speech",
        "language": "tr",
        "split_override": "holdout",
        "group_prefix": "whatsapp_local_real_group_001",
        "benchmark_name": "microphone_benchmark",
    },
    {
        "dir": "microphone_benchmark/fake_replayed",
        "label": "fake",
        "channel_type": "replayed_rir",
        "codec": "wav_pcm",
        "generation_type": "tts_replayed_rir",
        "language": "tr",
        "split_override": "holdout",
        "group_prefix": "local_fake_rir",
        "benchmark_name": "microphone_benchmark",
    },
    # --- TRAINING SOURCES (populated after downloads) ---

    # real_digital + fake_digital: balanced_v2 (LJSpeech, garystafford, WaveFake, shunyalabs_tr)
    # balanced_v2/metadata.csv has train=400+val=50 per label (no test split).
    # v3 re-splits via hash to produce train=350/val=50/test=50 per label.
    # balanced_v2/metadata.csv is NOT modified — only read for label/language/generation_type.
    # Located at DATASETS_ROOT/balanced_v2/ (outside project dir on RunPod)
    {
        "dir": "__ext__:balanced_v2",
        "label": None,           # read from metadata.csv
        "channel_type": "digital",
        "codec": "wav_pcm",
        "generation_type": None,  # read from metadata.csv
        "language": None,         # read from metadata.csv
        "split_override": None,
        "group_prefix": "balanced_v2",
        "benchmark_name": None,
        "metadata_csv": "__ext__:balanced_v2/metadata.csv",  # read label/gen_type only
        "force_hash_split": True,  # ignore metadata.csv splits; rank-based exact split
        "exact_val_count": 50,     # exactly 50 val per label (hash-rank order)
        "exact_test_count": 50,    # exactly 50 test per label (hash-rank order)
    },
    # real_microphone: FLEURS TR (CC-BY-4.0, ~2526 train clips, browser-mic quality)
    {
        "dir": "data/fleurs_tr",
        "label": "real",
        "channel_type": "microphone",
        "codec": "wav_pcm",
        "generation_type": "human_speech",
        "language": "tr",
        "split_override": None,
        "group_prefix": "fleurs_tr",
        "benchmark_name": None,
    },
    # real_microphone: Common Voice TR (gated, needs HF token; CC0 license)
    {
        "dir": "data/common_voice_tr",
        "label": "real",
        "channel_type": "microphone",
        "codec": "mp3",
        "generation_type": "human_speech",
        "language": "tr",
        "split_override": None,
        "group_prefix": "cv_tr",
        "benchmark_name": None,
    },
    # ASVspoof 2019 PA: protocol-based loader (no file copying).
    # Reads train+dev protocols → original FLAC paths.
    # PA eval → holdout only (never train/val/test).
    # bonafide → real/microphone, spoof → fake/replayed.
    {
        "source_type": "asvspoof_pa_protocol",
        "pa_root": "data/asvspoof_pa/PA",
        "group_prefix": "asvspoof_pa",
        "channel_type_bonafide": "microphone",
        "channel_type_spoof": "replayed",
        "codec": "flac",
        "language": "en",
    },
    # fake_replayed: RIR-augmented TTS (generated from balanced_v2/fake)
    {
        "dir": "data/augmented/rir",
        "label": "fake",
        "channel_type": "augmented_rir",
        "codec": "wav_pcm",
        "generation_type": "tts_augmented_rir",
        "language": "mixed",
        "split_override": None,
        "group_prefix": "aug_rir",
        "benchmark_name": None,
    },
    # real_microphone: codec-chain augmented real speech (Opus encode-decode)
    {
        "dir": "data/augmented/codec",
        "label": "real",
        "channel_type": "augmented_codec",
        "codec": "opus_wav",
        "generation_type": "human_speech_augmented_codec",
        "language": "mixed",
        "split_override": None,
        "group_prefix": "aug_codec",
        "benchmark_name": None,
    },
]

AUDIO_EXTS = {".wav", ".flac", ".mp3", ".ogg", ".m4a"}
COLUMNS = [
    "path", "label", "split", "source", "language",
    "generation_type", "channel_type", "codec",
    "original_recording_id", "original_recording_group",
    "segment_id", "duration_sec", "clipping_flag", "silence_ratio",
    "benchmark_name", "notes",
]


def get_duration_and_quality(filepath):
    """Returns (duration_sec, clipping_flag, silence_ratio) or (None, None, None)."""
    try:
        import soundfile as sf
        import numpy as np
        data, sr = sf.read(str(filepath))
        if data.ndim > 1:
            data = data.mean(axis=1)
        dur = len(data) / sr
        peak = float(np.abs(data).max())
        clipping = peak > 0.99
        silence_ratio = float((np.abs(data) < 0.001).mean())
        return round(dur, 2), clipping, round(silence_ratio, 4)
    except Exception:
        return None, None, None


def stable_hash(s):
    """Deterministic short hash for reproducible splits."""
    return int(hashlib.md5(s.encode()).hexdigest(), 16) % 10000


def assign_split(group_key, val_ratio, test_ratio, split_override):
    if split_override:
        return split_override
    h = stable_hash(group_key)
    val_cut = int(val_ratio * 10000)
    test_cut = int((val_ratio + test_ratio) * 10000)
    if h < val_cut:
        return "val"
    elif h < test_cut:
        return "test"
    return "train"


def extract_speaker_id(fname, group_prefix):
    """Best-effort speaker ID extraction from filename."""
    stem = Path(fname).stem
    # Common Voice: common_voice_tr_XXXXXXXX → XXXXXXXX as ID isn't speaker
    # Speaker ID not in filename for CV; use hash bucket grouping instead
    # ASVspoof PA: LA_E_XXXXXXX pattern
    if "LA_E_" in stem or "PA_E_" in stem:
        parts = stem.split("_")
        for i, p in enumerate(parts):
            if p in ("E", "T", "D") and i + 1 < len(parts):
                return "spk_" + parts[i + 1][:4]
    return None


def resolve_dir(dir_str):
    """Resolve __ext__:path as DATASETS_ROOT/path, else ROOT/path."""
    if dir_str.startswith("__ext__:"):
        return DATASETS_ROOT / dir_str[len("__ext__:"):]
    return ROOT / dir_str


def load_metadata_csv(csv_path_str):
    """Load an existing metadata CSV; return dict keyed by filename stem."""
    csv_path = resolve_dir(csv_path_str) if csv_path_str.startswith("__ext__:") \
               else ROOT / csv_path_str
    if not csv_path.is_file():
        return {}
    mapping = {}
    with open(csv_path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            p = Path(row["path"])
            mapping[p.name] = row      # key = filename only
            mapping[p.stem] = row      # key = stem without ext
    return mapping


def collect_asvspoof_pa_rows(src_def, max_per_domain, val_ratio, test_ratio, dry_run=False):
    """Protocol-based ASVspoof PA loader. No file copying — uses original FLAC paths."""
    pa_root = ROOT / src_def["pa_root"]
    if not pa_root.is_dir():
        return [], 0

    proto_dir  = pa_root / "ASVspoof2019_PA_cm_protocols"
    train_flac = pa_root / "ASVspoof2019_PA_train" / "flac"
    dev_flac   = pa_root / "ASVspoof2019_PA_dev" / "flac"
    eval_flac  = pa_root / "ASVspoof2019_PA_eval" / "flac"

    if not proto_dir.is_dir():
        return [], 0

    group_prefix = src_def["group_prefix"]

    # Parse train + dev only. PA eval excluded from metadata (Seçenek A).
    entries = []
    for proto_name, audio_dir, split_override in [
        ("ASVspoof2019.PA.cm.train.trn.txt", train_flac, None),
        ("ASVspoof2019.PA.cm.dev.trl.txt",   dev_flac,   None),
    ]:
        proto = proto_dir / proto_name
        if not proto.is_file():
            continue
        with open(proto) as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) < 5:
                    continue
                speaker_id, file_id, env_id, attack_id, key = parts[0], parts[1], parts[2], parts[3], parts[4]
                entries.append((speaker_id, file_id, env_id, attack_id, key, audio_dir, split_override))

    if not entries:
        return [], 0

    random.shuffle(entries)

    # Cap bonafide and spoof separately at max_per_domain (upper bound only)
    bonafide_train = [e for e in entries if e[4] == "bonafide"][:max_per_domain]
    spoof_train    = [e for e in entries if e[4] == "spoof"][:max_per_domain]
    selected = bonafide_train + spoof_train

    rows = []
    skipped = 0

    for i, (speaker_id, file_id, env_id, attack_id, key, audio_dir, _) in enumerate(selected):
        label        = "real" if key == "bonafide" else "fake"
        channel_type = src_def["channel_type_bonafide"] if label == "real" else src_def["channel_type_spoof"]
        gen_type     = "human_speech" if label == "real" else "tts_replayed_physical"

        fp = audio_dir / (file_id + ".flac")

        group_key = "{}_spk_{}".format(group_prefix, speaker_id)
        split = assign_split(group_key, val_ratio, test_ratio, None)
        rec_id = "{}_{}".format(group_prefix, file_id)

        if not dry_run and fp.is_file():
            dur, clip, sil = get_duration_and_quality(fp)
            if dur is not None and dur < 1.0:
                skipped += 1
                continue
        else:
            dur, clip, sil = None, None, None

        rows.append({
            "path": str(fp.relative_to(ROOT)),
            "label": label,
            "split": split,
            "source": "asvspoof_pa",
            "language": src_def["language"],
            "generation_type": gen_type,
            "channel_type": channel_type,
            "codec": src_def["codec"],
            "original_recording_id": rec_id,
            "original_recording_group": group_key,
            "segment_id": i,
            "duration_sec": dur if dur is not None else "",
            "clipping_flag": int(clip) if clip is not None else "",
            "silence_ratio": sil if sil is not None else "",
            "benchmark_name": "",
            "notes": "env={} attack={}".format(env_id, attack_id),
        })

    return rows, skipped


def collect_rows(src_def, max_per_domain, val_ratio, test_ratio, dry_run=False):
    d = resolve_dir(src_def["dir"])
    if not d.is_dir():
        return [], 0

    # Load pre-existing metadata CSV if specified (balanced_v2 case)
    meta_map = {}
    if src_def.get("metadata_csv"):
        meta_map = load_metadata_csv(src_def["metadata_csv"])

    files = sorted([f for f in d.rglob("*") if f.suffix.lower() in AUDIO_EXTS])
    if not files:
        return [], 0

    random.shuffle(files)
    files = files[:max_per_domain]
    is_external = src_def["dir"].startswith("__ext__:")
    group_prefix = src_def["group_prefix"]

    # For force_hash_split: rank files by hash per label → assign exact val/test counts.
    # This guarantees exact split sizes regardless of hash distribution variance.
    split_map = {}
    if src_def.get("force_hash_split"):
        from collections import defaultdict as _dd
        exact_val  = src_def.get("exact_val_count", 0)
        exact_test = src_def.get("exact_test_count", 0)
        by_label = _dd(list)
        for fp in files:
            stem = fp.stem
            meta = meta_map.get(fp.name) or meta_map.get(stem) or {}
            label = meta.get("label") or src_def["label"]
            h = stable_hash("{}_{}".format(group_prefix, stem))
            by_label[label].append((h, fp))
        for label, items in by_label.items():
            for rank, (_, fp) in enumerate(sorted(items)):
                if rank < exact_val:
                    split_map[fp] = "val"
                elif rank < exact_val + exact_test:
                    split_map[fp] = "test"
                else:
                    split_map[fp] = "train"

    rows = []
    skipped = 0

    for i, fp in enumerate(files):
        stem = fp.stem

        meta = meta_map.get(fp.name) or meta_map.get(stem) or {}

        label = meta.get("label") or src_def["label"]
        language = meta.get("language") or src_def["language"] or "unknown"
        gen_type = meta.get("generation_type") or src_def["generation_type"] or "unknown"

        group_key = "{}_{}".format(group_prefix, stem)

        if src_def.get("force_hash_split"):
            split = split_map.get(fp, "train")
        elif meta.get("split") and not src_def["split_override"]:
            split = meta["split"]
        else:
            sv_ratio = src_def.get("source_val_ratio", val_ratio)
            st_ratio = src_def.get("source_test_ratio", test_ratio)
            spk = extract_speaker_id(fp.name, group_prefix)
            group_key = "{}_{}".format(group_prefix, spk if spk else stem)
            split = assign_split(group_key, sv_ratio, st_ratio, src_def["split_override"])

        rec_id = "{}_{}".format(group_prefix, stem)

        if not dry_run:
            dur, clip, sil = get_duration_and_quality(fp)
            if dur is not None and dur < 1.0:
                skipped += 1
                continue
        else:
            dur, clip, sil = None, None, None

        rows.append({
            "path": str(fp) if is_external else str(fp.relative_to(ROOT)),
            "label": label,
            "split": split,
            "source": src_def["dir"],
            "language": language,
            "generation_type": gen_type,
            "channel_type": src_def["channel_type"],
            "codec": src_def["codec"],
            "original_recording_id": rec_id,
            "original_recording_group": group_key if not src_def["split_override"]
                                        else src_def["group_prefix"],
            "segment_id": i,
            "duration_sec": dur if dur is not None else "",
            "clipping_flag": int(clip) if clip is not None else "",
            "silence_ratio": sil if sil is not None else "",
            "benchmark_name": src_def.get("benchmark_name") or "",
            "notes": "",
        })

    return rows, skipped


def print_stats(rows):
    from collections import Counter
    splits = Counter(r["split"] for r in rows)
    domains = Counter((r["channel_type"], r["label"]) for r in rows)
    sources = Counter(r["source"] for r in rows)
    channel_types = Counter(r["channel_type"] for r in rows)

    print("\n--- Split distribution ---")
    for s in ("train", "val", "test", "holdout"):
        print("  {:10s}: {:>5}".format(s, splits.get(s, 0)))

    print("\n--- Channel type distribution (all splits) ---")
    for ch, cnt in sorted(channel_types.items()):
        print("  {:<30}: {:>5}".format(ch, cnt))

    print("\n--- Domain distribution (channel_type × label) ---")
    for (ch, lab), cnt in sorted(domains.items()):
        print("  {:<28} {}: {:>5}".format(ch, lab, cnt))

    print("\n--- Source distribution ---")
    for src, cnt in sorted(sources.items()):
        print("  {:<50}: {:>5}".format(src, cnt))

    # ASVspoof attack type distribution (from notes field)
    pa_rows = [r for r in rows if r["source"] == "asvspoof_pa" and r["notes"]]
    if pa_rows:
        attack_counter = Counter()
        env_counter = Counter()
        for r in pa_rows:
            # notes format: "env=aaa attack=AA"
            parts = dict(p.split("=") for p in r["notes"].split() if "=" in p)
            attack_counter[parts.get("attack", "?")] += 1
            env_counter[parts.get("env", "?")] += 1
        print("\n--- ASVspoof PA attack_id distribution ---")
        for k, v in sorted(attack_counter.items()):
            print("  {}: {:>5}".format(k, v))
        print("\n--- ASVspoof PA env_id top-10 ---")
        for k, v in env_counter.most_common(10):
            print("  {}: {:>5}".format(k, v))

    train_rows = [r for r in rows if r["split"] == "train"]
    if train_rows:
        train_labels = Counter(r["label"] for r in train_rows)
        real_n = train_labels.get("real", 0)
        fake_n = train_labels.get("fake", 0)
        ratio = real_n / fake_n if fake_n else float("inf")
        train_channels = Counter(r["channel_type"] for r in train_rows)
        print("\n--- Train balance ---")
        print("  real: {}  fake: {}  ratio: {:.2f}".format(real_n, fake_n, ratio))
        if ratio < 0.5 or ratio > 2.0:
            print("  WARN: imbalanced! Consider --max_per_domain adjustment.")
        print("\n--- Train channel_type distribution ---")
        for ch, cnt in sorted(train_channels.items()):
            print("  {:<30}: {:>5}".format(ch, cnt))


def check_leakage(rows):
    """Verify no recording group appears in multiple non-holdout splits."""
    from collections import defaultdict
    group_splits = defaultdict(set)
    for r in rows:
        if r["split"] != "holdout":
            group_splits[r["original_recording_group"]].add(r["split"])
    leaks = {g: s for g, s in group_splits.items() if len(s) > 1}
    if leaks:
        print("\nWARN: Potential leakage detected ({} groups span multiple splits):".format(
            len(leaks)))
        for g, s in list(leaks.items())[:10]:
            print("  {} → {}".format(g, s))
    else:
        print("\nLeakage check: OK (no group spans multiple splits)")
    return leaks


def balance_by_domain(rows, domain_targets):
    """Subsample non-holdout rows to hit per-(label×channel_type×split) targets.

    domain_targets: dict keyed by (label, channel_type, split) → int target.
    Returns (balanced_rows, report, any_insufficient).
    No oversampling — if insufficient, reports and uses all available.
    """
    from collections import defaultdict

    buckets = defaultdict(list)
    holdout_rows = []
    for r in rows:
        if r["split"] == "holdout":
            holdout_rows.append(r)
        else:
            key = (r["label"], r["channel_type"], r["split"])
            buckets[key].append(r)

    balanced = []
    report = {}
    any_insufficient = False

    # All (label, channel) combos present in collected data
    all_combos = sorted(set((lab, ch) for (lab, ch, _) in buckets.keys()))
    for (label, channel) in all_combos:
        for split in ("train", "val", "test"):
            key = (label, channel, split)
            target = domain_targets.get(key, 0)
            if target == 0:
                continue
            avail = buckets.get(key, [])
            if len(avail) >= target:
                selected = random.sample(avail, target)
                status = "OK"
            else:
                selected = avail
                status = "INSUFFICIENT ({}/{})".format(len(avail), target)
                any_insufficient = True
            report[key] = {"available": len(avail), "selected": len(selected), "status": status}
            balanced.extend(selected)

    return balanced + holdout_rows, report, any_insufficient


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", default="training_runs/full_run_003")
    parser.add_argument("--max_per_domain", type=int, default=2000,
                        help="Upper-bound cap per source before domain balancing")
    parser.add_argument("--train_per_domain", type=int, default=350,
                        help="Target train samples per (label × channel_type)")
    parser.add_argument("--val_per_domain", type=int, default=75,
                        help="Target val samples per non-digital (label × channel_type)")
    parser.add_argument("--test_per_domain", type=int, default=75,
                        help="Target test samples per non-digital (label × channel_type)")
    parser.add_argument("--digital_val_per_domain", type=int, default=50,
                        help="Target val samples for digital channel_type")
    parser.add_argument("--digital_test_per_domain", type=int, default=50,
                        help="Target test samples for digital channel_type")
    parser.add_argument("--val_ratio", type=float, default=0.15)
    parser.add_argument("--test_ratio", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dry_run", action="store_true",
                        help="Print stats without writing CSV")
    args = parser.parse_args()

    random.seed(args.seed)
    os.makedirs(ROOT / args.output_dir, exist_ok=True)

    all_rows = []
    total_skipped = 0

    for src in SOURCE_DEFS:
        # Protocol-based ASVspoof PA loader
        if src.get("source_type") == "asvspoof_pa_protocol":
            pa_root = ROOT / src["pa_root"]
            if not pa_root.is_dir():
                print("SKIP (not found): {}".format(src["pa_root"]))
                continue
            rows, skipped = collect_asvspoof_pa_rows(
                src, args.max_per_domain, args.val_ratio, args.test_ratio,
                dry_run=args.dry_run)
            total_skipped += skipped
            print("Collected {:>5} rows from {}  (skipped {} too-short)".format(
                len(rows), src["pa_root"], skipped))
            all_rows.extend(rows)
            continue

        d = resolve_dir(src["dir"])
        if not d.is_dir():
            print("SKIP (not found): {}".format(src["dir"]))
            continue
        rows, skipped = collect_rows(src, args.max_per_domain,
                                     args.val_ratio, args.test_ratio,
                                     dry_run=args.dry_run)
        total_skipped += skipped
        print("Collected {:>5} rows from {}  (skipped {} too-short)".format(
            len(rows), src["dir"], skipped))
        all_rows.extend(rows)

    if not all_rows:
        print("\nNo rows collected. Check that source directories exist.")
        print("Expected sources:")
        for s in SOURCE_DEFS:
            src_id = s.get("pa_root") or s.get("dir", "?")
            exists = "EXISTS" if (ROOT / src_id).is_dir() else "MISSING"
            print("  [{:>7}] {}".format(exists, src_id))
        return

    from collections import Counter as _Counter
    pre = _Counter(r["split"] for r in all_rows)
    print("\n--- Pre-balance totals (raw collected) ---")
    for s in ("train", "val", "test", "holdout"):
        print("  {:10s}: {:>5}".format(s, pre.get(s, 0)))

    # Build per-(label × channel_type × split) targets.
    # digital uses separate val/test targets; all others use val_per_domain / test_per_domain.
    DIGITAL_CHANNEL_TYPES = {"digital"}
    domain_targets = {}
    for label in ("real", "fake"):
        for channel in ("digital", "microphone", "replayed", "replayed_rir",
                        "augmented_rir", "augmented_codec"):
            for split in ("train", "val", "test"):
                if split == "train":
                    t = args.train_per_domain
                elif channel in DIGITAL_CHANNEL_TYPES:
                    t = args.digital_val_per_domain if split == "val" else args.digital_test_per_domain
                else:
                    t = args.val_per_domain if split == "val" else args.test_per_domain
                domain_targets[(label, channel, split)] = t

    all_rows, balance_report, any_insufficient = balance_by_domain(all_rows, domain_targets)

    print("\n--- Domain balance report ---")
    print("  {:5} {:<12} {:<12} {:<6} {:>8} → {:>8}".format(
        "FLAG", "channel_type", "label", "split", "avail", "selected"))
    for (label, channel, split), info in sorted(balance_report.items()):
        flag = "INSUF" if "INSUF" in info["status"] else "OK   "
        print("  {} {:<12} {:<12} {:<6} {:>8} → {:>8}  {}".format(
            flag, channel, label, split,
            info["available"], info["selected"], info["status"]))
    if any_insufficient:
        print("\n  WARN: Insufficient data in some domains. No oversampling applied.")
        print("  Approve action before training.")

    print_stats(all_rows)
    leaks = check_leakage(all_rows)

    # Holdout verification
    print("\n--- Holdout verification ---")
    holdout_sources = _Counter(r["source"] for r in all_rows if r["split"] == "holdout")
    ta_real  = sum(1 for r in all_rows if r["source"] == "test_audio/real"  and r["split"] == "holdout")
    ta_fake  = sum(1 for r in all_rows if r["source"] == "test_audio/fake"  and r["split"] == "holdout")
    mb_real  = sum(1 for r in all_rows if r["source"] == "microphone_benchmark/real"         and r["split"] == "holdout")
    mb_fake  = sum(1 for r in all_rows if r["source"] == "microphone_benchmark/fake_replayed" and r["split"] == "holdout")
    pa_train = sum(1 for r in all_rows if r["source"] == "asvspoof_pa" and r["split"] != "holdout")
    print("  test_audio:           real={} fake={}  (must be holdout ✓/✗)".format(ta_real, ta_fake))
    print("  microphone_benchmark: real={} fake={}  (must be holdout ✓/✗)".format(mb_real, mb_fake))
    print("  asvspoof_pa in train/val/test: {}".format(pa_train))
    ta_in_train = sum(1 for r in all_rows
                      if r["source"] in ("test_audio/real", "test_audio/fake")
                      and r["split"] != "holdout")
    mb_in_train = sum(1 for r in all_rows
                      if "microphone_benchmark" in r["source"]
                      and r["split"] != "holdout")
    if ta_in_train == 0 and mb_in_train == 0:
        print("  HOLDOUT INTEGRITY: OK — no benchmark data in training splits")
    else:
        print("  HOLDOUT INTEGRITY: FAIL — ta_in_train={} mb_in_train={}".format(
            ta_in_train, mb_in_train))

    if not args.dry_run:
        out_path = ROOT / args.output_dir / "metadata_v3.csv"
        with open(out_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=COLUMNS)
            writer.writeheader()
            writer.writerows(all_rows)
        print("\nCSV written: {} ({} rows)".format(out_path, len(all_rows)))

        sources_found   = [s.get("pa_root") or s["dir"] for s in SOURCE_DEFS
                           if (ROOT / (s.get("pa_root") or s["dir"])).is_dir()]
        sources_missing = [s.get("pa_root") or s["dir"] for s in SOURCE_DEFS
                           if not (ROOT / (s.get("pa_root") or s["dir"])).is_dir()]
        manifest = {
            "total_rows": len(all_rows),
            "skipped_too_short": total_skipped,
            "sources_found": sources_found,
            "sources_missing": sources_missing,
            "balance_report": {str(k): v for k, v in balance_report.items()},
            "leakage_groups": list(leaks.keys()),
            "args": vars(args),
        }
        mpath = ROOT / args.output_dir / "metadata_v3_manifest.json"
        with open(mpath, "w") as f:
            json.dump(manifest, f, indent=2)
        print("Manifest written: {}".format(mpath))
    else:
        print("\n[DRY RUN] CSV not written.")


if __name__ == "__main__":
    main()
