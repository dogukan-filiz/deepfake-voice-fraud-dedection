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
