"""Fine-tune AASIST head with frozen XLSR-300M backbone."""

import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "backend" / "aasist"))


class AudioDataset(Dataset):
    WINDOW = 64600
    SR = 16000

    def __init__(self, df, split):
        self.rows = df[df["split"] == split].reset_index(drop=True)

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        import librosa
        row = self.rows.iloc[idx]
        label = 1 if row["label"] == "real" else 0  # 1=bonafide, 0=spoof

        try:
            audio, _ = librosa.load(row["path"], sr=self.SR, mono=True)
        except Exception:
            audio = np.zeros(self.WINDOW, dtype=np.float32)

        if len(audio) == 0:
            audio = np.zeros(self.WINDOW, dtype=np.float32)

        if len(audio) < self.WINDOW:
            reps = math.ceil(self.WINDOW / len(audio))
            audio = np.tile(audio, reps)[: self.WINDOW]
        else:
            audio = audio[: self.WINDOW]

        audio = audio.astype(np.float32)
        peak = np.max(np.abs(audio))
        if peak > 1.5:
            audio = audio / peak

        return torch.from_numpy(audio), torch.tensor(label, dtype=torch.long)


class FocalLoss(nn.Module):
    def __init__(self, gamma=2.0, alpha=None):
        super().__init__()
        self.gamma = gamma
        self.register_buffer("alpha", alpha)

    def forward(self, logits, targets):
        ce = F.cross_entropy(logits, targets, weight=self.alpha, reduction="none")
        pt = torch.exp(-ce)
        loss = ((1 - pt) ** self.gamma) * ce
        return loss.mean()


def evaluate(model, loader, device):
    model.eval()
    all_preds, all_labels, all_scores = [], [], []
    with torch.no_grad():
        for audio, labels in loader:
            audio, labels = audio.to(device), labels.to(device)
            logits = model(audio)
            probs = torch.softmax(logits, dim=-1)
            preds = torch.argmax(logits, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_scores.extend(probs[:, 1].cpu().numpy())  # p_real

    from sklearn.metrics import f1_score, recall_score, roc_auc_score
    preds_arr = np.array(all_preds)
    labels_arr = np.array(all_labels)
    scores_arr = np.array(all_scores)

    macro_f1 = f1_score(labels_arr, preds_arr, average="macro", zero_division=0)
    real_recall = recall_score(labels_arr, preds_arr, pos_label=1, zero_division=0)
    fake_recall = recall_score(labels_arr, preds_arr, pos_label=0, zero_division=0)
    try:
        auc = roc_auc_score(labels_arr, scores_arr)
    except Exception:
        auc = 0.0

    return {
        "macro_f1": round(macro_f1, 4),
        "real_recall": round(real_recall, 4),
        "fake_recall": round(fake_recall, 4),
        "auc_roc": round(auc, 4),
    }


def main():
    parser = argparse.ArgumentParser(description="Fine-tune AASIST head (frozen XLSR backbone)")
    parser.add_argument("--metadata", required=True, help="Path to metadata.csv")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--focal_loss", action="store_true",
                        help="Use focal loss (recommended for real recall improvement)")
    parser.add_argument("--patience", type=int, default=5,
                        help="Early stopping patience on val macro F1")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--resume", default=None,
                        help="Path to checkpoint .pth to resume from")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"[train] Device: {device}")

    # Load data
    df = pd.read_csv(args.metadata)
    train_set = AudioDataset(df, "train")
    val_set = AudioDataset(df, "val")
    print(f"[train] Train: {len(train_set)} samples, Val: {len(val_set)} samples")

    if len(train_set) == 0:
        print("[train] ERROR: No training samples found. Check metadata.csv splits.")
        sys.exit(1)

    train_loader = DataLoader(train_set, batch_size=args.batch_size,
                              shuffle=True, num_workers=0, pin_memory=True)
    val_loader = DataLoader(val_set, batch_size=args.batch_size,
                            shuffle=False, num_workers=0, pin_memory=True)

    # Load model
    print("[train] Loading XLSRAASISTModel (will download XLSR-300M from HuggingFace if not cached)...")
    t0 = time.time()
    from backend.model_wrapper_ssl import XLSRAASISTModel
    wrapper = XLSRAASISTModel(device=str(device))
    model = wrapper._model  # nn.Module: SSLModel + AASIST head
    print(f"[train] Model loaded in {time.time()-t0:.1f}s")

    # Freeze SSL backbone
    for param in model.ssl_model.parameters():
        param.requires_grad = False
    model.ssl_model.eval()

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[train] Params: {trainable_params:,} trainable / {total_params:,} total "
          f"({100*trainable_params/total_params:.1f}%)")

    # Resume
    start_epoch = 0
    if args.resume:
        ckpt = torch.load(args.resume, map_location=device)
        state = ckpt.get("model_state_dict", ckpt)
        missing, unexpected = model.load_state_dict(state, strict=False)
        print(f"[train] Resumed from {args.resume} (missing={len(missing)}, unexpected={len(unexpected)})")
        start_epoch = ckpt.get("epoch", 0) + 1

    # Loss
    if args.focal_loss:
        # Weight real class higher to improve real recall
        alpha = torch.tensor([0.4, 0.6], dtype=torch.float32).to(device)
        criterion = FocalLoss(gamma=2.0, alpha=alpha)
        print("[train] Using Focal Loss (gamma=2.0, alpha=[fake=0.4, real=0.6])")
    else:
        criterion = nn.CrossEntropyLoss()
        print("[train] Using CrossEntropyLoss")

    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=2
    )

    best_val_f1 = 0.0
    best_path = output_dir / "best_head.pth"
    patience_counter = 0
    log_rows = []

    print(f"\n[train] Starting training for {args.epochs} epochs...\n")

    for epoch in range(start_epoch, args.epochs):
        # Train
        model.train()
        model.ssl_model.eval()  # Keep backbone frozen

        train_loss = 0.0
        n_batches = 0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs}", leave=False)
        for audio, labels in pbar:
            audio, labels = audio.to(device), labels.to(device)
            optimizer.zero_grad()
            logits = model(audio)
            loss = criterion(logits, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                filter(lambda p: p.requires_grad, model.parameters()), 1.0
            )
            optimizer.step()
            model.ssl_model.eval()  # Re-ensure after step
            train_loss += loss.item()
            n_batches += 1
            pbar.set_postfix(loss=f"{loss.item():.4f}")

        avg_train_loss = train_loss / max(n_batches, 1)

        # Validate
        val_metrics = evaluate(model, val_loader, device)
        scheduler.step(val_metrics["macro_f1"])

        row = {
            "epoch": epoch + 1,
            "train_loss": round(avg_train_loss, 4),
            **val_metrics,
        }
        log_rows.append(row)

        print(f"Epoch {epoch+1:3d} | loss={avg_train_loss:.4f} | "
              f"val_f1={val_metrics['macro_f1']:.4f} | "
              f"real_recall={val_metrics['real_recall']:.4f} | "
              f"fake_recall={val_metrics['fake_recall']:.4f}")

        # Save latest checkpoint (for resume); delete previous latest to save space
        head_state = {k: v for k, v in model.state_dict().items()
                      if not k.startswith("ssl_model.")}
        ckpt_path = output_dir / f"checkpoint_epoch_{epoch+1}.pth"
        torch.save({"epoch": epoch + 1, "model_state_dict": head_state,
                    "val_metrics": val_metrics, "args": vars(args)}, ckpt_path)
        # Delete previous epoch checkpoint (keep only latest + best)
        prev_ckpt = output_dir / f"checkpoint_epoch_{epoch}.pth"
        if prev_ckpt.exists() and prev_ckpt != best_path:
            prev_ckpt.unlink()

        if val_metrics["macro_f1"] > best_val_f1:
            best_val_f1 = val_metrics["macro_f1"]
            torch.save({"epoch": epoch + 1, "model_state_dict": head_state,
                        "val_metrics": val_metrics, "args": vars(args)}, best_path)
            print(f"  ✓ New best val F1: {best_val_f1:.4f} → saved {best_path.name}")
            # Also remove latest if it is the best (best_head.pth already saved it)
            if ckpt_path.exists() and ckpt_path != best_path:
                ckpt_path.unlink()
            ckpt_path = best_path  # so next epoch does not delete best
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= args.patience:
                print(f"[train] Early stopping (patience={args.patience})")
                break

        # Save log
        pd.DataFrame(log_rows).to_csv(output_dir / "training_log.csv", index=False)

    # GPU memory report
    if torch.cuda.is_available():
        mem_alloc = torch.cuda.memory_allocated(device) / 1e9
        mem_max = torch.cuda.max_memory_allocated(device) / 1e9
        print(f"\n[train] GPU memory: current={mem_alloc:.2f}GB  peak={mem_max:.2f}GB")

    print(f"\n[train] Done. Best val macro F1: {best_val_f1:.4f}")
    print(f"  Outputs: {output_dir}/")
    print(f"  Best checkpoint: {best_path}")


if __name__ == "__main__":
    main()
