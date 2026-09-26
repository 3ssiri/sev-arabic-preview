# Changelog

## 0.1.0 — 2026-09-26 — Research Preview

- First public research preview of Sev (Arabic-first typed decisions).
- Supported question types: `choice`, `noul`. `score` is rejected by the runtime and carries no quality claim.
- Checkpoint: Track E (mmBERT-base + LoRA + typed marker head), seed 42, frozen hashes in `sev_preview_config.json`.
- Per-type temperature calibration (choice T=1.3642, noul T=1.5122).
- Licenses: code Apache-2.0; model weights CC BY-SA 4.0.
- Runtime refuses to load weights whose SHA-256 differs from the manifest.
- Known limits: noul trained as label verification only; 384-token input; no dedicated Saudi/Gulf, OOD or code-switch evaluation.
