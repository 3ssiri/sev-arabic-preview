# Sev Arabic Preview v0.1

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.22979051.svg)](https://zenodo.org/doi/10.5281/zenodo.22979051)

> **Research Preview** — not the final Sev model, not production-ready.

Sev is an Arabic-first **typed decision engine**. You give it a state (text or JSON) and a few typed questions; one
encoder pass returns bounded answers with calibrated probabilities. No text is generated.

| type | you give | you get |
|---|---|---|
| `choice` | instruction + options | the chosen option + a probability per option |
| `noul` | a label-verification question, next to its `choice` question | `P(true)` |

`score` is not supported in v0.1.

## Install

```bash
pip install -r requirements.txt
```

Model files (adapter, head, tokenizer, config) are in the gated Hugging Face repository
[3ssiri/Sev-Arabic-Preview-v0.1](https://huggingface.co/3ssiri/Sev-Arabic-Preview-v0.1). Access is approved
automatically: open the page, accept the terms, then log in and download:

```bash
hf auth login
hf download 3ssiri/Sev-Arabic-Preview-v0.1 --local-dir Sev-Arabic-Preview-v0.1
```

The mmBERT-base encoder is downloaded automatically at its pinned revision.

## Use

```python
from sev_preview import SevPreview

sev = SevPreview("path/to/Sev-Arabic-Preview-v0.1")   # checks file hashes before loading
out = sev.decide(
    "ابغى احول ٥٠٠ ريال لأخوي",
    {"intent": {"type": "choice", "instructions": "ما نية المستخدم؟",
                "criteria": {"balance": "الاستعلام عن الرصيد", "transfer": "تحويل أموال", "other": "طلب آخر"}}},
)
# {"model": "Sev Arabic Preview v0.1", "version": "0.1.0",
#  "answers": {"intent": {"type": "choice", "choice": "transfer", "probabilities": {...}}},
#  "meta": {"input_tokens": ..., "state_truncated": false}}
```

More in `examples/`. Rules the runtime enforces: at most 8 questions per request; `choice` needs at least 2 options;
`noul` criteria, when given, are exactly `false` / `true`.

**noul in v0.1** was trained only as a verification question ("is this label correct for the request?") with explicit
Arabic criteria, asked in the same request as the `choice` question it verifies — see `examples/noul_example.py`.
Standalone or free-form yes/no questions are not reliable in this preview.

## Check your install

```bash
python smoke_test.py --device cpu     # or --device cuda
```

## Evidence and limits

Development results (94.53% mean accuracy over 3 seeds, ECE ≈ 0.0053 after calibration) come from Sev's own
development split and are **not** an external benchmark; they cannot be compared with other systems' published numbers.
Choice accuracy on the model card is 92.82% with the paired verification question. Without that question the same
checkpoint scores 90.94% on 6,228 held-out rows (92.95% when the question is included on those rows).
On a broader, harder development set (Saudi-dialect call-centre dialogues and other Arabic sources, one question per
request; measured 2026-10-01) the same checkpoint scores only 57.3% on `choice` and 56.9% on standalone yes/no
questions, so the 94.53% figure holds only for its own development split. Newer, unreleased development models reach
81–87% on those rows.
Limitations, data sources and licensing are in the model card.

## License

- Code: **Apache-2.0** (`LICENSE`), © 2026 Ali Asiri. `sev_preview/_jevlite/` is Apache-2.0, © 2026 Intikhab Azam
  (unmodified; see `NOTICE`).
- Model weights (adapter + head): **CC BY-SA 4.0** (`LICENSE-WEIGHTS`), © 2026 Ali Asiri — share adaptations under the
  same license. The weights were fine-tuned on data including ArBanking77 (CC BY-SA 4.0).
- Base encoder mmBERT-base: MIT, not redistributed.

## Contact

[@3li3 on X](https://x.com/3li3) · assiri@gmail.com

## Citation

See `CITATION.cff`. DOI (all versions): [10.5281/zenodo.22979051](https://zenodo.org/doi/10.5281/zenodo.22979051) · v0.1.0: [10.5281/zenodo.22979052](https://zenodo.org/records/22979052)

© 2026 Ali Asiri
