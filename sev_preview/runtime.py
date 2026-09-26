"""Sev Arabic Preview v0.1 inference: one encoder pass, calibrated `choice` / `noul` answers.

The model is the frozen Track-E checkpoint (mmBERT-base + LoRA adapter + typed marker head). This wrapper keeps the
behaviour the development evidence was measured with: the open-jev serializer (vendored unchanged in `_jevlite/`),
a 384-token input where the state absorbs any cut, at most 8 questions per request, and per-type temperature scaling
with the frozen preview temperatures. It never generates text.

Differences from the research harness are API safety only: more than 8 questions, `score` questions, or malformed
criteria raise ValueError instead of being dropped or scored; a truncated state is reported in `meta`.
"""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path
from typing import Any

import torch

_VENDOR = Path(__file__).resolve().parent / "_jevlite"
if str(_VENDOR) not in sys.path:
    sys.path.insert(0, str(_VENDOR))  # the vendored modules import each other by top-level name

import td_data  # noqa: E402
from model import JevLite  # noqa: E402
from typed_schema import iter_labels, state_to_text  # noqa: E402

CONFIG_NAME = "sev_preview_config.json"
SUPPORTED_TYPES = ("choice", "noul")


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def validate_questions(questions: dict, max_questions: int) -> None:
    if not isinstance(questions, dict) or not questions:
        raise ValueError("questions must be a non-empty object {question_id: question}")
    if len(questions) > max_questions:
        raise ValueError(f"at most {max_questions} questions per request (got {len(questions)}); split the request")
    for qid, q in questions.items():
        if q.get("type") not in SUPPORTED_TYPES:
            raise ValueError(f"{qid}: type must be one of {SUPPORTED_TYPES} in this preview (got {q.get('type')!r})")
        if not isinstance(q.get("instructions"), str) or not q["instructions"].strip():
            raise ValueError(f"{qid}: instructions must be a non-empty string")
        c = q.get("criteria")
        if q["type"] == "choice" and (not isinstance(c, dict) or len(c) < 2):
            raise ValueError(f"{qid}: a choice question needs criteria with at least 2 options")
        if q["type"] == "noul" and c is not None and set(c) != {"false", "true"}:
            raise ValueError(f"{qid}: noul criteria, when given, must have exactly the keys 'false' and 'true'")
        if c is not None and not all(isinstance(v, str) for v in c.values()):
            raise ValueError(f"{qid}: criteria descriptions must be strings")


class SevPreview:
    """Load once, then call `decide(state, questions)`."""

    def __init__(self, model_dir: str | Path, device: str | None = None, verify_hashes: bool = True):
        self.dir = Path(model_dir)
        self.cfg = json.loads((self.dir / CONFIG_NAME).read_text(encoding="utf-8"))
        if verify_hashes:
            for rel, expected in self.cfg["files_sha256"].items():
                got = sha256(self.dir / rel)
                if got != expected:
                    raise RuntimeError(f"hash mismatch for {rel}: {got} != {expected}; refusing to load")
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.temps = {k: float(v) for k, v in self.cfg["temperatures"].items()}
        self.max_len = int(self.cfg["max_input_tokens"])
        self.max_questions = int(self.cfg["max_questions"])

        from huggingface_hub import snapshot_download
        from peft import PeftModel
        from transformers import AutoTokenizer

        base = self.cfg["base_model"]
        need = ["config.json", base["weights_file"]]
        try:  # MIT base weights at the exact revision: local cache first, then the Hub
            base_dir = snapshot_download(base["repo_id"], revision=base["revision"], allow_patterns=need, local_files_only=True)
            if not all((Path(base_dir) / f).is_file() for f in need):
                raise FileNotFoundError("incomplete local snapshot")
        except Exception:
            base_dir = snapshot_download(base["repo_id"], revision=base["revision"], allow_patterns=need)
        if verify_hashes and sha256(Path(base_dir) / base["weights_file"]) != base["weights_sha256"]:
            raise RuntimeError(f"base model weights hash mismatch in {base_dir}; refusing to load")
        tok = AutoTokenizer.from_pretrained(self.dir / "tokenizer", local_files_only=True)
        model = JevLite(base_dir, tokenizer=tok)
        with torch.no_grad():  # marker embeddings: the fixed init the checkpoint was trained and evaluated with
            emb = model.encoder.get_input_embeddings().weight
            cls = model.tok.cls_token_id if model.tok.cls_token_id is not None else model.tok.pad_token_id
            sep = model.tok.sep_token_id if model.tok.sep_token_id is not None else cls
            emb[model.qid].copy_(emb[cls])
            emb[model.lid].copy_(emb[sep])
        model.encoder = PeftModel.from_pretrained(model.encoder, self.dir / "adapter")
        head = torch.load(self.dir / "head.pt", map_location="cpu", weights_only=True)
        model.score.load_state_dict(head["score"])
        self.model = model.to(self.device).eval()

    def _encode(self, state: Any, questions: dict) -> tuple[dict, bool]:
        row = {"state": state, "questions": questions}
        tok, qid, lid = self.model.tok, self.model.qid, self.model.lid
        enc = td_data.encode(row, tok, self.max_len, qid, lid, with_gold=False)
        n_state = len(tok.encode(state_to_text(state), add_special_tokens=False))
        # room for every state token plus the serializer's 32-token minimum state budget: nothing is cut
        full = td_data.encode(row, tok, len(enc["input_ids"]) + n_state + 34, qid, lid, with_gold=False)
        return enc, len(full["input_ids"]) > len(enc["input_ids"])

    @torch.no_grad()
    def decide(self, state: Any, questions: dict, return_logits: bool = False) -> dict:
        validate_questions(questions, self.max_questions)
        enc, truncated = self._encode(state, questions)
        batch = {k: v.to(self.device) for k, v in td_data.collate([enc], self.model.tok.pad_token_id).items()}
        logits, _ = self.model(batch, apply_temperature=False)
        logits = logits[0][: len(enc["group"])].double().cpu()
        answers, raw = {}, {}
        for gi, qname in enumerate(enc["qnames"]):
            q = questions[qname]
            keys = enc["labels"][gi]
            z = logits[[i for i, g in enumerate(enc["group"]) if g == gi]]
            p = torch.softmax(z / self.temps[q["type"]], -1)
            probs = dict(zip(keys, (float(x) for x in p)))
            if q["type"] == "noul":
                answers[qname] = {"type": "noul", "noul": probs["true"], "probabilities": probs}
            else:
                best = keys[int(torch.argmax(p))]  # ties: first key in the serializer's (sorted) order
                answers[qname] = {"type": "choice", "choice": best, "probabilities": {k: probs[k] for k in q["criteria"]}}
            raw[qname] = dict(zip(keys, (float(x) for x in z)))
        meta = {"input_tokens": len(enc["input_ids"]), "state_truncated": truncated}
        if return_logits:
            meta["logits"] = raw
        return {"model": self.cfg["name"], "version": self.cfg["version"], "answers": answers, "meta": meta}


def labels_for(question: dict) -> list[str]:
    """The option keys the model scores for a question, in its own order (exposed for tests)."""
    return [k for k, _ in iter_labels(question)]
