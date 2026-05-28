# SSL+AASIST (XLSR) Integration Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the HuggingFace `DF_Arena_1B_V_1` primary model with TakHemlata's SSL+AASIST (XLSR-300M frontend + AASIST head), loaded via HuggingFace `transformers` (no fairseq).

**Architecture:** Port TakHemlata/SSL_Anti-spoofing `model.py` into `backend/aasist/models/SSLAASIST.py`. Swap the fairseq XLSR loader for `Wav2Vec2Model.from_pretrained("facebook/wav2vec2-xls-r-300m")`. Wrap in `XLSRAASISTModel` matching the existing `DeepfakeVoiceModel.predict()` contract. Loader chain becomes SSL+AASIST → AASIST baseline → Heuristic.

**Tech Stack:** Python 3.11, PyTorch ≥2.0, transformers ≥4.30, librosa, FastAPI, pytest.

**Spec:** `docs/superpowers/specs/2026-05-28-ssl-aasist-integration-design.md`

---

## File Structure

| File | Responsibility | Status |
|------|----------------|--------|
| `backend/aasist/models/SSLAASIST.py` | Ported model class: `SSLModel` (HF XLSR wrapper) + `Model` (AASIST head on XLSR features). | NEW |
| `backend/model_wrapper_ssl.py` | `XLSRAASISTModel`: load weights, chunk audio, run inference, aggregate. | NEW |
| `backend/model_wrapper.py` | Loader chain. Replace `DF_Arena` branch with SSL+AASIST branch. | MODIFY |
| `backend/model_wrapper_hf.py` | DEAD CODE — remove. | DELETE |
| `backend/config.py` | Add `LOCAL_SSL_MODEL_DIR` env var. | MODIFY |
| `models/ssl_aasist/README.md` | Manual download instructions for `weights.pth`. | NEW |
| `models/ssl_aasist/meta.json` | Model metadata (name, source URL, sha256). | NEW |
| `.gitignore` | Ensure `models/ssl_aasist/*.pth` is ignored (already covered by `models/`). | VERIFY |
| `requirements.txt` | Pin `transformers>=4.30`. | MODIFY |
| `tests/test_xlsr_aasist.py` | Unit tests: instantiation, dummy-waveform forward shape. | NEW |
| `README.md` | Update model section. | MODIFY |

---

## Task 1: Add LOCAL_SSL_MODEL_DIR setting

**Files:**
- Modify: `backend/config.py`

- [ ] **Step 1: Write the failing test**

Create `tests/test_config_ssl.py`:

```python
import os
import importlib
import sys


def test_ssl_model_dir_defaults_none(monkeypatch):
    monkeypatch.delenv("LOCAL_SSL_MODEL_DIR", raising=False)
    if "backend.config" in sys.modules:
        del sys.modules["backend.config"]
    cfg = importlib.import_module("backend.config")
    assert cfg.settings.LOCAL_SSL_MODEL_DIR is None


def test_ssl_model_dir_reads_env(monkeypatch, tmp_path):
    monkeypatch.setenv("LOCAL_SSL_MODEL_DIR", str(tmp_path))
    if "backend.config" in sys.modules:
        del sys.modules["backend.config"]
    cfg = importlib.import_module("backend.config")
    assert cfg.settings.LOCAL_SSL_MODEL_DIR == str(tmp_path)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.\.venv\Scripts\python.exe -m pytest tests/test_config_ssl.py -v`
Expected: FAIL with `AttributeError: 'Settings' object has no attribute 'LOCAL_SSL_MODEL_DIR'`.

- [ ] **Step 3: Implement**

Edit `backend/config.py`:

```python
from typing import Optional

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    AUTH_THRESHOLD: float = 0.35

    AUDIO_MIN_DURATION_SEC: float = 2.0
    AUDIO_MIN_PEAK_ABS: float = 5e-4
    AUDIO_MIN_RMS: float = 1e-4

    LOCAL_SSL_MODEL_DIR: Optional[str] = None

    class Config:
        env_file = ".env"


settings = Settings()
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.\.venv\Scripts\python.exe -m pytest tests/test_config_ssl.py -v`
Expected: 2 passed.

- [ ] **Step 5: Commit**

```powershell
git add backend/config.py tests/test_config_ssl.py
git commit -m "feat(config): add LOCAL_SSL_MODEL_DIR for SSL+AASIST weights override"
```

---

## Task 2: Create models/ssl_aasist directory with download README

**Files:**
- Create: `models/ssl_aasist/README.md`
- Create: `models/ssl_aasist/meta.json`

- [ ] **Step 1: Create directory and README**

Create `models/ssl_aasist/README.md` with the following exact content:

```markdown
# SSL+AASIST Model Weights

Source: https://github.com/TakHemlata/SSL_Anti-spoofing (MIT license)

## Manual Download Required

The pretrained weights are **not committed to this repo** (1.2 GB).

1. Download `Best_LA_model_for_DF.pth` (or `LA_model.pth`) from the TakHemlata
   Google Drive link in the project README:
   https://github.com/TakHemlata/SSL_Anti-spoofing#pre-trained-models
2. Rename to `weights.pth` and place in this directory:
   `models/ssl_aasist/weights.pth`
3. Verify SHA-256 matches `meta.json`.

The XLSR backbone (`facebook/wav2vec2-xls-r-300m`, ~1.2 GB) is auto-downloaded
by `transformers` on first run and cached in
`%USERPROFILE%\.cache\huggingface\hub\`.

## Citation

Tak, H., Todisco, M., Wang, X., Jung, J., Yamagishi, J., & Evans, N. (2022).
"Automatic speaker verification spoofing and deepfake detection using
wav2vec 2.0 and data augmentation." Proc. The Speaker and Language
Recognition Workshop (Odyssey 2022).
```

- [ ] **Step 2: Create meta.json**

Create `models/ssl_aasist/meta.json`:

```json
{
  "name": "ssl_aasist_la",
  "source": "https://github.com/TakHemlata/SSL_Anti-spoofing",
  "license": "MIT",
  "license_attribution": "Copyright (c) Hemlata Tak (tak@eurecom.fr), EURECOM",
  "backbone": "facebook/wav2vec2-xls-r-300m",
  "ssl_out_dim": 1024,
  "input_sample_rate": 16000,
  "input_window_samples": 64600,
  "id2label": {"0": "spoof", "1": "bonafide"},
  "real_label_id": 1,
  "fake_label_id": 0,
  "weights_file": "weights.pth",
  "weights_sha256": "TBD-fill-after-download"
}
```

Note: `weights_sha256` will be filled in by the operator after downloading the file.
This is operator-supplied data, not a code placeholder.

- [ ] **Step 3: Verify gitignore covers weights**

Run: `git check-ignore -v models/ssl_aasist/weights.pth`
Expected: Output shows ignore rule (e.g., `.gitignore:N:models/  models/ssl_aasist/weights.pth`).
If NOT ignored, add `models/ssl_aasist/*.pth` to `.gitignore`.

- [ ] **Step 4: Commit**

```powershell
git add models/ssl_aasist/README.md models/ssl_aasist/meta.json
git commit -m "docs(models): add SSL+AASIST weights directory with download instructions"
```

---

## Task 3: Port SSL+AASIST model code (HF XLSR swap)

**Files:**
- Create: `backend/aasist/models/SSLAASIST.py`

- [ ] **Step 1: Create the file**

Create `backend/aasist/models/SSLAASIST.py` with the full ported code below.
This is a verbatim port of TakHemlata/SSL_Anti-spoofing `model.py` (MIT-licensed)
with **only the `SSLModel` class swapped** to use HuggingFace
`Wav2Vec2Model.from_pretrained("facebook/wav2vec2-xls-r-300m")` instead of
fairseq.

```python
"""SSL+AASIST model for audio anti-spoofing.

Ported from TakHemlata/SSL_Anti-spoofing (MIT license).
Original author: Hemlata Tak <tak@eurecom.fr>, EURECOM.

Modifications:
- Replaced fairseq-based SSLModel with HuggingFace transformers Wav2Vec2Model
  loader to remove the fairseq dependency.
"""

from typing import Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class SSLModel(nn.Module):
    """XLSR-300M frontend loaded via HuggingFace transformers."""

    def __init__(self, device):
        super().__init__()
        from transformers import Wav2Vec2Model

        self.model = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-xls-r-300m")
        self.device = device
        self.out_dim = 1024

    def extract_feat(self, input_data):
        # Ensure backbone follows input device/dtype
        if (next(self.model.parameters()).device != input_data.device
                or next(self.model.parameters()).dtype != input_data.dtype):
            self.model.to(input_data.device, dtype=input_data.dtype)

        if input_data.ndim == 3:
            input_data = input_data[:, :, 0]

        outputs = self.model(input_data)
        return outputs.last_hidden_state  # [B, T, 1024]


class GraphAttentionLayer(nn.Module):
    def __init__(self, in_dim, out_dim, **kwargs):
        super().__init__()
        self.att_proj = nn.Linear(in_dim, out_dim)
        self.att_weight = self._init_new_params(out_dim, 1)
        self.proj_with_att = nn.Linear(in_dim, out_dim)
        self.proj_without_att = nn.Linear(in_dim, out_dim)
        self.bn = nn.BatchNorm1d(out_dim)
        self.input_drop = nn.Dropout(p=0.2)
        self.act = nn.SELU(inplace=True)
        self.temp = kwargs.get("temperature", 1.0)

    def forward(self, x):
        x = self.input_drop(x)
        att_map = self._derive_att_map(x)
        x = self._project(x, att_map)
        x = self._apply_BN(x)
        x = self.act(x)
        return x

    def _pairwise_mul_nodes(self, x):
        nb_nodes = x.size(1)
        x = x.unsqueeze(2).expand(-1, -1, nb_nodes, -1)
        x_mirror = x.transpose(1, 2)
        return x * x_mirror

    def _derive_att_map(self, x):
        att_map = self._pairwise_mul_nodes(x)
        att_map = torch.tanh(self.att_proj(att_map))
        att_map = torch.matmul(att_map, self.att_weight)
        att_map = att_map / self.temp
        return F.softmax(att_map, dim=-2)

    def _project(self, x, att_map):
        x1 = self.proj_with_att(torch.matmul(att_map.squeeze(-1), x))
        x2 = self.proj_without_att(x)
        return x1 + x2

    def _apply_BN(self, x):
        org_size = x.size()
        x = x.view(-1, org_size[-1])
        x = self.bn(x)
        return x.view(org_size)

    def _init_new_params(self, *size):
        out = nn.Parameter(torch.FloatTensor(*size))
        nn.init.xavier_normal_(out)
        return out


class HtrgGraphAttentionLayer(nn.Module):
    def __init__(self, in_dim, out_dim, **kwargs):
        super().__init__()
        self.proj_type1 = nn.Linear(in_dim, in_dim)
        self.proj_type2 = nn.Linear(in_dim, in_dim)
        self.att_proj = nn.Linear(in_dim, out_dim)
        self.att_projM = nn.Linear(in_dim, out_dim)
        self.att_weight11 = self._init_new_params(out_dim, 1)
        self.att_weight22 = self._init_new_params(out_dim, 1)
        self.att_weight12 = self._init_new_params(out_dim, 1)
        self.att_weightM = self._init_new_params(out_dim, 1)
        self.proj_with_att = nn.Linear(in_dim, out_dim)
        self.proj_without_att = nn.Linear(in_dim, out_dim)
        self.proj_with_attM = nn.Linear(in_dim, out_dim)
        self.proj_without_attM = nn.Linear(in_dim, out_dim)
        self.bn = nn.BatchNorm1d(out_dim)
        self.input_drop = nn.Dropout(p=0.2)
        self.act = nn.SELU(inplace=True)
        self.temp = kwargs.get("temperature", 1.0)

    def forward(self, x1, x2, master=None):
        num_type1 = x1.size(1)
        num_type2 = x2.size(1)
        x1 = self.proj_type1(x1)
        x2 = self.proj_type2(x2)
        x = torch.cat([x1, x2], dim=1)
        if master is None:
            master = torch.mean(x, dim=1, keepdim=True)
        x = self.input_drop(x)
        att_map = self._derive_att_map(x, num_type1, num_type2)
        master = self._update_master(x, master)
        x = self._project(x, att_map)
        x = self._apply_BN(x)
        x = self.act(x)
        x1 = x.narrow(1, 0, num_type1)
        x2 = x.narrow(1, num_type1, num_type2)
        return x1, x2, master

    def _update_master(self, x, master):
        att_map = self._derive_att_map_master(x, master)
        return self._project_master(x, master, att_map)

    def _pairwise_mul_nodes(self, x):
        nb_nodes = x.size(1)
        x = x.unsqueeze(2).expand(-1, -1, nb_nodes, -1)
        x_mirror = x.transpose(1, 2)
        return x * x_mirror

    def _derive_att_map_master(self, x, master):
        att_map = x * master
        att_map = torch.tanh(self.att_projM(att_map))
        att_map = torch.matmul(att_map, self.att_weightM)
        att_map = att_map / self.temp
        return F.softmax(att_map, dim=-2)

    def _derive_att_map(self, x, num_type1, num_type2):
        att_map = self._pairwise_mul_nodes(x)
        att_map = torch.tanh(self.att_proj(att_map))
        att_board = torch.zeros_like(att_map[:, :, :, 0]).unsqueeze(-1)
        att_board[:, :num_type1, :num_type1, :] = torch.matmul(
            att_map[:, :num_type1, :num_type1, :], self.att_weight11)
        att_board[:, num_type1:, num_type1:, :] = torch.matmul(
            att_map[:, num_type1:, num_type1:, :], self.att_weight22)
        att_board[:, :num_type1, num_type1:, :] = torch.matmul(
            att_map[:, :num_type1, num_type1:, :], self.att_weight12)
        att_board[:, num_type1:, :num_type1, :] = torch.matmul(
            att_map[:, num_type1:, :num_type1, :], self.att_weight12)
        att_map = att_board / self.temp
        return F.softmax(att_map, dim=-2)

    def _project(self, x, att_map):
        x1 = self.proj_with_att(torch.matmul(att_map.squeeze(-1), x))
        x2 = self.proj_without_att(x)
        return x1 + x2

    def _project_master(self, x, master, att_map):
        x1 = self.proj_with_attM(torch.matmul(
            att_map.squeeze(-1).unsqueeze(1), x))
        x2 = self.proj_without_attM(master)
        return x1 + x2

    def _apply_BN(self, x):
        org_size = x.size()
        x = x.view(-1, org_size[-1])
        x = self.bn(x)
        return x.view(org_size)

    def _init_new_params(self, *size):
        out = nn.Parameter(torch.FloatTensor(*size))
        nn.init.xavier_normal_(out)
        return out


class GraphPool(nn.Module):
    def __init__(self, k: float, in_dim: int, p: Union[float, int]):
        super().__init__()
        self.k = k
        self.sigmoid = nn.Sigmoid()
        self.proj = nn.Linear(in_dim, 1)
        self.drop = nn.Dropout(p=p) if p > 0 else nn.Identity()
        self.in_dim = in_dim

    def forward(self, h):
        Z = self.drop(h)
        weights = self.proj(Z)
        scores = self.sigmoid(weights)
        return self.top_k_graph(scores, h, self.k)

    def top_k_graph(self, scores, h, k):
        _, n_nodes, n_feat = h.size()
        n_nodes = max(int(n_nodes * k), 1)
        _, idx = torch.topk(scores, n_nodes, dim=1)
        idx = idx.expand(-1, -1, n_feat)
        h = h * scores
        return torch.gather(h, 1, idx)


class Residual_block(nn.Module):
    def __init__(self, nb_filts, first=False):
        super().__init__()
        self.first = first
        if not self.first:
            self.bn1 = nn.BatchNorm2d(num_features=nb_filts[0])
        self.conv1 = nn.Conv2d(in_channels=nb_filts[0], out_channels=nb_filts[1],
                               kernel_size=(2, 3), padding=(1, 1), stride=1)
        self.selu = nn.SELU(inplace=True)
        self.bn2 = nn.BatchNorm2d(num_features=nb_filts[1])
        self.conv2 = nn.Conv2d(in_channels=nb_filts[1], out_channels=nb_filts[1],
                               kernel_size=(2, 3), padding=(0, 1), stride=1)
        if nb_filts[0] != nb_filts[1]:
            self.downsample = True
            self.conv_downsample = nn.Conv2d(in_channels=nb_filts[0], out_channels=nb_filts[1],
                                             padding=(0, 1), kernel_size=(1, 3), stride=1)
        else:
            self.downsample = False

    def forward(self, x):
        identity = x
        if not self.first:
            out = self.bn1(x)
            out = self.selu(out)
        else:
            out = x
        out = self.conv1(x)
        out = self.bn2(out)
        out = self.selu(out)
        out = self.conv2(out)
        if self.downsample:
            identity = self.conv_downsample(identity)
        out += identity
        return out


class Model(nn.Module):
    """SSL+AASIST: XLSR-300M frontend + AASIST graph-attention head.

    Output: logits shape [B, 2]. Index 0 = spoof, index 1 = bonafide
    (matches the existing AASIST baseline convention in this repo).
    """

    def __init__(self, args=None, device=None):
        super().__init__()
        self.device = device

        filts = [128, [1, 32], [32, 32], [32, 64], [64, 64]]
        gat_dims = [64, 32]
        pool_ratios = [0.5, 0.5, 0.5, 0.5]
        temperatures = [2.0, 2.0, 100.0, 100.0]

        self.ssl_model = SSLModel(self.device)
        self.LL = nn.Linear(self.ssl_model.out_dim, 128)

        self.first_bn = nn.BatchNorm2d(num_features=1)
        self.first_bn1 = nn.BatchNorm2d(num_features=64)
        self.drop = nn.Dropout(0.5, inplace=True)
        self.drop_way = nn.Dropout(0.2, inplace=True)
        self.selu = nn.SELU(inplace=True)

        self.encoder = nn.Sequential(
            nn.Sequential(Residual_block(nb_filts=filts[1], first=True)),
            nn.Sequential(Residual_block(nb_filts=filts[2])),
            nn.Sequential(Residual_block(nb_filts=filts[3])),
            nn.Sequential(Residual_block(nb_filts=filts[4])),
            nn.Sequential(Residual_block(nb_filts=filts[4])),
            nn.Sequential(Residual_block(nb_filts=filts[4])))

        self.attention = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=(1, 1)),
            nn.SELU(inplace=True),
            nn.BatchNorm2d(128),
            nn.Conv2d(128, 64, kernel_size=(1, 1)),
        )

        self.pos_S = nn.Parameter(torch.randn(1, 42, filts[-1][-1]))
        self.master1 = nn.Parameter(torch.randn(1, 1, gat_dims[0]))
        self.master2 = nn.Parameter(torch.randn(1, 1, gat_dims[0]))

        self.GAT_layer_S = GraphAttentionLayer(filts[-1][-1], gat_dims[0], temperature=temperatures[0])
        self.GAT_layer_T = GraphAttentionLayer(filts[-1][-1], gat_dims[0], temperature=temperatures[1])

        self.HtrgGAT_layer_ST11 = HtrgGraphAttentionLayer(gat_dims[0], gat_dims[1], temperature=temperatures[2])
        self.HtrgGAT_layer_ST12 = HtrgGraphAttentionLayer(gat_dims[1], gat_dims[1], temperature=temperatures[2])
        self.HtrgGAT_layer_ST21 = HtrgGraphAttentionLayer(gat_dims[0], gat_dims[1], temperature=temperatures[2])
        self.HtrgGAT_layer_ST22 = HtrgGraphAttentionLayer(gat_dims[1], gat_dims[1], temperature=temperatures[2])

        self.pool_S = GraphPool(pool_ratios[0], gat_dims[0], 0.3)
        self.pool_T = GraphPool(pool_ratios[1], gat_dims[0], 0.3)
        self.pool_hS1 = GraphPool(pool_ratios[2], gat_dims[1], 0.3)
        self.pool_hT1 = GraphPool(pool_ratios[2], gat_dims[1], 0.3)
        self.pool_hS2 = GraphPool(pool_ratios[2], gat_dims[1], 0.3)
        self.pool_hT2 = GraphPool(pool_ratios[2], gat_dims[1], 0.3)

        self.out_layer = nn.Linear(5 * gat_dims[1], 2)

    def forward(self, x):
        x_ssl_feat = self.ssl_model.extract_feat(x.squeeze(-1))
        x = self.LL(x_ssl_feat)

        x = x.transpose(1, 2)
        x = x.unsqueeze(dim=1)
        x = F.max_pool2d(x, (3, 3))
        x = self.first_bn(x)
        x = self.selu(x)

        x = self.encoder(x)
        x = self.first_bn1(x)
        x = self.selu(x)

        w = self.attention(x)

        w1 = F.softmax(w, dim=-1)
        m = torch.sum(x * w1, dim=-1)
        e_S = m.transpose(1, 2) + self.pos_S

        gat_S = self.GAT_layer_S(e_S)
        out_S = self.pool_S(gat_S)

        w2 = F.softmax(w, dim=-2)
        m1 = torch.sum(x * w2, dim=-2)
        e_T = m1.transpose(1, 2)

        gat_T = self.GAT_layer_T(e_T)
        out_T = self.pool_T(gat_T)

        master1 = self.master1.expand(x.size(0), -1, -1)
        master2 = self.master2.expand(x.size(0), -1, -1)

        out_T1, out_S1, master1 = self.HtrgGAT_layer_ST11(out_T, out_S, master=self.master1)
        out_S1 = self.pool_hS1(out_S1)
        out_T1 = self.pool_hT1(out_T1)
        out_T_aug, out_S_aug, master_aug = self.HtrgGAT_layer_ST12(out_T1, out_S1, master=master1)
        out_T1 = out_T1 + out_T_aug
        out_S1 = out_S1 + out_S_aug
        master1 = master1 + master_aug

        out_T2, out_S2, master2 = self.HtrgGAT_layer_ST21(out_T, out_S, master=self.master2)
        out_S2 = self.pool_hS2(out_S2)
        out_T2 = self.pool_hT2(out_T2)
        out_T_aug, out_S_aug, master_aug = self.HtrgGAT_layer_ST22(out_T2, out_S2, master=master2)
        out_T2 = out_T2 + out_T_aug
        out_S2 = out_S2 + out_S_aug
        master2 = master2 + master_aug

        out_T1 = self.drop_way(out_T1)
        out_T2 = self.drop_way(out_T2)
        out_S1 = self.drop_way(out_S1)
        out_S2 = self.drop_way(out_S2)
        master1 = self.drop_way(master1)
        master2 = self.drop_way(master2)

        out_T = torch.max(out_T1, out_T2)
        out_S = torch.max(out_S1, out_S2)
        master = torch.max(master1, master2)

        T_max, _ = torch.max(torch.abs(out_T), dim=1)
        T_avg = torch.mean(out_T, dim=1)
        S_max, _ = torch.max(torch.abs(out_S), dim=1)
        S_avg = torch.mean(out_S, dim=1)

        last_hidden = torch.cat([T_max, T_avg, S_max, S_avg, master.squeeze(1)], dim=1)
        last_hidden = self.drop(last_hidden)
        output = self.out_layer(last_hidden)
        return output
```

- [ ] **Step 2: Verify the module imports cleanly**

Run: `.\.venv\Scripts\python.exe -c "import sys; sys.path.insert(0, 'backend/aasist'); from models.SSLAASIST import Model, SSLModel; print('OK')"`
Expected: `OK` (no ImportError). Module-level imports do NOT instantiate the model, so no HF download happens here.

- [ ] **Step 3: Commit**

```powershell
git add backend/aasist/models/SSLAASIST.py
git commit -m "feat(model): port SSL+AASIST architecture from TakHemlata (HF XLSR loader)"
```

---

## Task 4: Write XLSRAASISTModel wrapper

**Files:**
- Create: `backend/model_wrapper_ssl.py`
- Create: `tests/test_xlsr_aasist.py`

- [ ] **Step 1: Write the failing test**

Create `tests/test_xlsr_aasist.py`:

```python
"""Unit tests for XLSRAASISTModel.

These tests use a MockSSLModel monkeypatch to avoid the 1.2 GB
facebook/wav2vec2-xls-r-300m download in CI. The full model is exercised
in tests/comprehensive_test.py (integration suite).
"""
import sys
from pathlib import Path

import numpy as np
import pytest
import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "backend" / "aasist"))


class MockSSLModel(nn.Module):
    def __init__(self, device):
        super().__init__()
        self.device = device
        self.out_dim = 1024
        self.proj = nn.Linear(1, 1024)

    def extract_feat(self, x):
        if x.ndim == 3:
            x = x[:, :, 0]
        t = x.shape[1] // 320
        return torch.randn(x.shape[0], t, self.out_dim, device=x.device)


def test_model_forward_shape(monkeypatch):
    from models import SSLAASIST as ssl_mod
    monkeypatch.setattr(ssl_mod, "SSLModel", MockSSLModel)

    model = ssl_mod.Model(device=torch.device("cpu"))
    model.eval()
    waveform = torch.randn(1, 64600)
    with torch.no_grad():
        out = model(waveform)
    assert out.shape == (1, 2), f"expected [1, 2], got {tuple(out.shape)}"


def test_wrapper_predict_returns_schema(monkeypatch, tmp_path):
    from models import SSLAASIST as ssl_mod
    monkeypatch.setattr(ssl_mod, "SSLModel", MockSSLModel)

    weights = ssl_mod.Model(device=torch.device("cpu")).state_dict()
    weights_path = tmp_path / "weights.pth"
    torch.save(weights, weights_path)
    (tmp_path / "meta.json").write_text('{"name": "test", "id2label": {"0": "spoof", "1": "bonafide"}}', encoding="utf-8")

    from backend.model_wrapper_ssl import XLSRAASISTModel
    m = XLSRAASISTModel(model_dir=tmp_path, device="cpu")

    features = {
        "waveform": np.random.randn(16000 * 3).astype(np.float32),
        "sr": 16000,
        "spectral_residual_score": 0.2,
    }
    result = m.predict(features)
    for key in ("p_real", "p_fake", "spectral_residual", "num_chunks", "max_chunk_p_fake"):
        assert key in result, f"missing key {key}"
    assert 0.0 <= result["p_real"] <= 1.0
    assert 0.0 <= result["p_fake"] <= 1.0
    assert abs((result["p_real"] + result["p_fake"]) - 1.0) < 1e-5
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.\.venv\Scripts\python.exe -m pytest tests/test_xlsr_aasist.py -v`
Expected: FAIL — `test_model_forward_shape` may PASS once SSLAASIST.py is in place from Task 3; `test_wrapper_predict_returns_schema` MUST FAIL with `ModuleNotFoundError: No module named 'backend.model_wrapper_ssl'`.

- [ ] **Step 3: Implement the wrapper**

Create `backend/model_wrapper_ssl.py`:

```python
"""SSL+AASIST (XLSR-300M + AASIST) inference wrapper."""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import librosa
import numpy as np
import torch

_AASIST_DIR = Path(__file__).resolve().parent / "aasist"
if str(_AASIST_DIR) not in sys.path:
    sys.path.insert(0, str(_AASIST_DIR))


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _resolve_ssl_model_dir(model_dir: str | Path | None = None) -> Path:
    env_dir = os.getenv("LOCAL_SSL_MODEL_DIR")
    if model_dir is not None:
        chosen = Path(model_dir)
    elif env_dir:
        chosen = Path(env_dir)
    else:
        chosen = Path("models/ssl_aasist")
    if not chosen.is_absolute():
        chosen = (_repo_root() / chosen).resolve()
    return chosen


class XLSRAASISTModel:
    """SSL+AASIST inference wrapper.

    Expected directory contents:
      weights.pth   - state dict from TakHemlata pretrained checkpoint
      meta.json     - metadata (optional)
    """

    def __init__(
        self,
        model_dir: str | Path | None = None,
        device: Optional[str] = None,
    ) -> None:
        self.model_dir = _resolve_ssl_model_dir(model_dir)
        if not self.model_dir.exists():
            raise FileNotFoundError(f"SSL model directory not found: {self.model_dir}")

        meta_path = self.model_dir / "meta.json"
        if meta_path.exists():
            with open(meta_path, "r", encoding="utf-8") as f:
                self.meta = json.load(f)
        else:
            self.meta = {}

        self.device = torch.device(device) if device is not None else torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )

        self.sampling_rate: int = 16000
        self.window_sec: float = 64600 / self.sampling_rate
        self.stride_sec: float = self.window_sec

        self.id2label: Dict[int, str] = {0: "spoof", 1: "bonafide"}
        self.real_label_id: int = 1
        self.fake_label_id: int = 0

        from models.SSLAASIST import Model as SSLAASISTNet  # type: ignore

        self._model = SSLAASISTNet(device=self.device).to(self.device)

        weights_path = self.model_dir / "weights.pth"
        self.weights_path = weights_path
        if not weights_path.exists():
            raise FileNotFoundError(
                f"SSL+AASIST weights file not found: {weights_path}. "
                f"See {self.model_dir / 'README.md'} for download instructions."
            )

        state = torch.load(weights_path, map_location=self.device)
        if isinstance(state, dict) and "state_dict" in state:
            state = state["state_dict"]
        # Some checkpoints prefix keys with 'module.' (DataParallel)
        if any(k.startswith("module.") for k in state.keys()):
            state = {k.replace("module.", "", 1): v for k, v in state.items()}
        self._model.load_state_dict(state, strict=False)
        self._model.eval()

    @staticmethod
    def _chunk_audio(audio: np.ndarray, window_samples: int, stride_samples: int) -> List[np.ndarray]:
        if len(audio) <= window_samples:
            pad = window_samples - len(audio)
            if pad > 0:
                reps = int(np.ceil(window_samples / max(len(audio), 1)))
                audio = np.tile(audio, reps)[:window_samples]
            return [audio]
        starts = list(range(0, len(audio) - window_samples + 1, stride_samples))
        last_start = len(audio) - window_samples
        if starts[-1] != last_start:
            starts.append(last_start)
        return [audio[s : s + window_samples] for s in starts]

    @torch.inference_mode()
    def predict(self, features: Dict[str, Any]) -> Dict[str, float]:
        waveform = np.asarray(features["waveform"], dtype=np.float32)
        sr = int(features["sr"])
        spectral_resid = float(features.get("spectral_residual_score", 0.0))

        if waveform.ndim != 1:
            waveform = waveform.reshape(-1)
        if waveform.size == 0:
            raise ValueError("Empty waveform - no audio data to process.")
        if not np.isfinite(waveform).all():
            raise ValueError("Audio data contains NaN or Inf values.")

        if sr != self.sampling_rate:
            waveform = librosa.resample(waveform, orig_sr=sr, target_sr=self.sampling_rate)

        peak = float(np.max(np.abs(waveform))) if waveform.size else 0.0
        if peak > 1.5:
            waveform = waveform / peak

        window_samples = int(self.window_sec * self.sampling_rate)
        stride_samples = int(self.stride_sec * self.sampling_rate)
        chunks = self._chunk_audio(waveform, window_samples, stride_samples)

        p_real_chunks: List[float] = []
        p_fake_chunks: List[float] = []

        for chunk in chunks:
            x = torch.from_numpy(chunk.astype(np.float32)).unsqueeze(0).to(self.device)
            logits = self._model(x)
            probs = torch.softmax(logits, dim=-1).detach().cpu().numpy()
            p_real_chunks.append(float(probs[0, self.real_label_id]))
            p_fake_chunks.append(float(probs[0, self.fake_label_id]))

        mean_p_fake = float(np.mean(p_fake_chunks))
        max_p_fake = float(np.max(p_fake_chunks))
        p_fake = 0.6 * mean_p_fake + 0.4 * max_p_fake
        p_real = 1.0 - p_fake

        if spectral_resid > 0.6:
            adjustment = 0.05 * (spectral_resid - 0.6)
            p_fake = min(p_fake + adjustment, 1.0)
            p_real = 1.0 - p_fake

        return {
            "p_real": round(p_real, 6),
            "p_fake": round(p_fake, 6),
            "spectral_residual": spectral_resid,
            "num_chunks": len(chunks),
            "max_chunk_p_fake": round(max_p_fake, 6),
        }
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.\.venv\Scripts\python.exe -m pytest tests/test_xlsr_aasist.py -v`
Expected: 2 passed.

- [ ] **Step 5: Commit**

```powershell
git add backend/model_wrapper_ssl.py tests/test_xlsr_aasist.py
git commit -m "feat(model): add XLSRAASISTModel inference wrapper with unit tests"
```

---

## Task 5: Wire SSL+AASIST into loader chain

**Files:**
- Modify: `backend/model_wrapper.py`
- Delete: `backend/model_wrapper_hf.py`

- [ ] **Step 1: Write the failing test**

Append to `tests/test_xlsr_aasist.py`:

```python
def test_get_model_status_reports_ssl_primary(monkeypatch):
    import importlib
    import backend.model_wrapper as mw
    importlib.reload(mw)
    monkeypatch.setattr(mw, "_model_instance", None)
    monkeypatch.setattr(mw, "_model_load_error", None)
    status = mw.get_model_status()
    assert status["model_type"] == "ssl_aasist", (
        f"loader must advertise ssl_aasist as primary, got {status['model_type']}"
    )
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.\.venv\Scripts\python.exe -m pytest tests/test_xlsr_aasist.py::test_get_model_status_reports_ssl_primary -v`
Expected: FAIL — current `model_type` is `"huggingface"`.

- [ ] **Step 3: Replace HF DF_Arena loader with SSL+AASIST**

Edit `backend/model_wrapper.py`. Replace the bottom section (from `_USE_HF_MODEL` flag through end of file) with:

```python
_model_instance: Optional[Any] = None  # XLSRAASISTModel | DeepfakeVoiceModel | HeuristicFallbackModel
_model_load_error: Optional[str] = None


def get_model_status() -> Dict:
    """Return model status without triggering load."""
    model_info: Dict[str, Any] = {
        "model_type": "ssl_aasist",
        "backbone": "facebook/wav2vec2-xls-r-300m",
    }

    if _model_instance is None:
        return {"loaded": False, "type": None, "error": _model_load_error, **model_info}

    t = type(_model_instance).__name__
    if t == "DeepfakeVoiceModel":
        model_info["model_type"] = "aasist_fallback"
    elif t == "HeuristicFallbackModel":
        model_info["model_type"] = "heuristic_fallback"

    status: Dict[str, Any] = {
        "loaded": True,
        "type": t,
        "error": _model_load_error,
        **model_info,
    }

    for attr in ("model_dir", "weights_path", "id2label", "real_label_id", "fake_label_id"):
        if hasattr(_model_instance, attr):
            value = getattr(_model_instance, attr)
            status[attr] = str(value) if attr in ("model_dir", "weights_path") else value

    return status


def get_model() -> Any:
    global _model_instance, _model_load_error
    if _model_instance is not None:
        return _model_instance

    # Primary: SSL+AASIST (XLSR-300M + AASIST head)
    try:
        from .model_wrapper_ssl import XLSRAASISTModel  # type: ignore
    except ImportError:
        from model_wrapper_ssl import XLSRAASISTModel  # type: ignore

    try:
        _model_instance = XLSRAASISTModel()
        _model_load_error = None
        print("[MODEL] Loaded SSL+AASIST (XLSR-300M + AASIST head) as primary")
        return _model_instance
    except Exception as e:
        ssl_err = f"SSL+AASIST load failed: {type(e).__name__}: {e}"
        print("[MODEL]", ssl_err)

    # Fallback: vanilla AASIST baseline
    try:
        _model_instance = DeepfakeVoiceModel()
        _model_load_error = ssl_err
        print("[MODEL] Falling back to AASIST baseline")
        return _model_instance
    except Exception as e2:
        aasist_err = f"AASIST load failed: {type(e2).__name__}: {e2}"
        _model_instance = HeuristicFallbackModel()
        _model_load_error = f"{ssl_err}; {aasist_err}"
        print("[MODEL] Falling back to heuristic model.")
        print("[MODEL]", _model_load_error)
        return _model_instance
```

Also remove the now-unused `_USE_HF_MODEL` flag and all `_USE_HF_MODEL` branches.

- [ ] **Step 4: Delete the dead HF DF_Arena wrapper**

```powershell
git rm backend/model_wrapper_hf.py
```

- [ ] **Step 5: Run test to verify it passes**

Run: `.\.venv\Scripts\python.exe -m pytest tests/test_xlsr_aasist.py -v`
Expected: 3 passed.

- [ ] **Step 6: Commit**

```powershell
git add backend/model_wrapper.py tests/test_xlsr_aasist.py
git commit -m "feat(model): swap loader chain to SSL+AASIST primary, drop DF_Arena_1B"
```

---

## Task 6: Pin transformers version

**Files:**
- Modify: `requirements.txt`

- [ ] **Step 1: Modify requirements**

Replace the `transformers` line in `requirements.txt`:

```
transformers>=4.30,<5.0
```

- [ ] **Step 2: Verify installed version satisfies pin**

Run: `.\.venv\Scripts\python.exe -m pip show transformers`
Expected: Version >= 4.30. If older, run `pip install --upgrade "transformers>=4.30,<5.0"`.

- [ ] **Step 3: Commit**

```powershell
git add requirements.txt
git commit -m "chore(deps): pin transformers>=4.30 for Wav2Vec2Model.from_pretrained"
```

---

## Task 7: Smoke tests still pass

**Files:** (no changes — verification only)

- [ ] **Step 1: Pre-checks smoke**

Run: `.\.venv\Scripts\python.exe tests/smoke_test_prechecks.py`
Expected: All assertions pass (silence/duration rejection unchanged).

If FAIL: stop and debug — `backend/audio_processing.py` was not supposed to change.

- [ ] **Step 2: Run unit suite**

Run: `.\.venv\Scripts\python.exe -m pytest tests/test_config_ssl.py tests/test_xlsr_aasist.py -v`
Expected: all green.

---

## Task 8: Backend health endpoint reports SSL model

**Files:** (no changes — verification only)

- [ ] **Step 1: Download weights**

Operator action — follow `models/ssl_aasist/README.md`. Place `weights.pth` in `models/ssl_aasist/`.
Update `weights_sha256` field in `models/ssl_aasist/meta.json` with the actual hash:

```powershell
Get-FileHash models/ssl_aasist/weights.pth -Algorithm SHA256
```

- [ ] **Step 2: Start backend with self-test**

```powershell
$env:MODEL_SELFTEST='1'
.\.venv\Scripts\python.exe -m uvicorn backend.main:app --host 127.0.0.1 --port 8010
```

- [ ] **Step 3: Curl /health from a second shell**

Run: `curl http://127.0.0.1:8010/health`
Expected JSON contains:

```json
{
  "model": {
    "loaded": true,
    "type": "XLSRAASISTModel",
    "model_type": "ssl_aasist"
  }
}
```

Plus `model_selftest.ok = true`.

If `type` is `DeepfakeVoiceModel`, primary failed — check `error` field and console logs.

- [ ] **Step 4: Stop the backend**

Press `Ctrl+C` in the uvicorn shell.

---

## Task 9: Accuracy verification on 100-sample test set

**Files:** (no code changes — measurement only)

- [ ] **Step 1: Re-start backend (no self-test flag needed)**

```powershell
.\.venv\Scripts\python.exe -m uvicorn backend.main:app --host 127.0.0.1 --port 8010
```

- [ ] **Step 2: Run comprehensive test**

In a second shell:

```powershell
.\.venv\Scripts\python.exe tests/comprehensive_test.py
```

Expected: Accuracy/precision/recall/F1 printed for the 50 real + 50 fake set.
Results JSON written under `docs/`.

Acceptance criteria (per design spec verification plan):
- Accuracy on the 100-sample set ≥ accuracy of vanilla AASIST baseline
  (whatever the existing `docs/model_evaluation_results.json` reports).
- F1 for the `fake` class strictly improves vs. the baseline.

If acceptance fails: STOP. Do not proceed to Task 10. Investigate
feature-extraction mismatch between HF `Wav2Vec2Model.last_hidden_state`
and what TakHemlata's classifier head was trained on. Log findings, consult
TakHemlata repo issues, decide whether to adjust the SSLModel wrapper.

- [ ] **Step 3: Threshold sweep**

```powershell
.\.venv\Scripts\python.exe scripts/evaluation/enhanced_evaluate.py
```

Pick the threshold with the highest F1 on the fake class. If it differs from
the current `AUTH_THRESHOLD=0.35`, note it for a follow-up commit (do NOT
change config in this plan — threshold tuning is a separate, traceable change).

- [ ] **Step 4: Commit results**

```powershell
git add docs/model_evaluation_results.json docs/detailed_model_evaluation.json
git commit -m "docs(eval): record SSL+AASIST accuracy on 100-sample test set"
```

---

## Task 10: UI smoke

**Files:** (no code changes — verification only)

- [ ] **Step 1: Start the full stack**

```powershell
.\run_fullstack.bat
```

- [ ] **Step 2: In the browser, exercise the golden paths**

Open http://127.0.0.1:5173 and verify:

1. Health indicator shows "model: ssl_aasist" or equivalent.
2. Upload `test_audio/real/real_000.wav` — verdict banner GREEN, risk LOW or MEDIUM.
3. Upload `test_audio/fake/fake_000.wav` — verdict banner RED with pulse, risk HIGH or CRITICAL.
4. Click test library modal, run one real + one fake from the modal, confirm consistent verdicts.
5. Try live microphone recording — confirm it analyzes without error (verdict accuracy not asserted).

If any step fails: STOP, debug, re-run Task 7-8 unit checks.

- [ ] **Step 3: Stop both servers**

Close the two cmd windows opened by `run_fullstack.bat`.

---

## Task 11: README update

**Files:**
- Modify: `README.md`

- [ ] **Step 1: Update model section**

Find the section describing the model stack and replace any mention of
`Speech-Arena-2025/DF_Arena_1B_V_1` with the new primary:

```markdown
### Model Stack

1. **Primary**: SSL+AASIST — XLSR-300M (`facebook/wav2vec2-xls-r-300m`) frontend
   + AASIST graph-attention head, weights from TakHemlata/SSL_Anti-spoofing
   (MIT). See `models/ssl_aasist/README.md` for the manual weights download.
2. **Fallback**: AASIST baseline (clovaai, local weights in `models/aasist_baseline/`).
3. **Last resort**: Heuristic (sigmoid on spectral anomaly score).
```

- [ ] **Step 2: Commit**

```powershell
git add README.md
git commit -m "docs(readme): describe SSL+AASIST primary model and fallback chain"
```

---

## Self-Review Checklist (run before opening a PR)

- [ ] All tests in `tests/test_config_ssl.py` and `tests/test_xlsr_aasist.py` pass.
- [ ] `tests/smoke_test_prechecks.py` passes.
- [ ] `tests/comprehensive_test.py` accuracy ≥ baseline, F1(fake) > baseline.
- [ ] `GET /health` shows `model_type: ssl_aasist` and `loaded: true`.
- [ ] No leftover references to `Speech-Arena-2025/DF_Arena_1B_V_1` in code or docs.
- [ ] `models/ssl_aasist/weights.pth` is git-ignored.
- [ ] `meta.json` has the real SHA-256 of the downloaded weights file.
