"""Packaged smoke test for Sev Arabic Preview v0.1. Run from the package directory:

    python smoke_test.py [--device cpu|cuda] [--out smoke-result.json]

Loads the model from this directory only (hash-verified), runs fixed Arabic cases, checks the output contract and
temperature scaling, and records the environment, memory and latency. Exit code 0 = every PASS check held.
The low-confidence, standalone-noul and free-form-noul cases are recorded as observations, not pass/fail
(the last two are documented v0.1 limitations).
"""
from __future__ import annotations

import argparse
import json
import math
import platform
import statistics
import sys
import time
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from sev_preview import SevPreview, __version__  # noqa: E402

INTENT = {"type": "choice", "instructions": "ما نية المستخدم؟",
          "criteria": {"balance": "الاستعلام عن الرصيد", "card_lost": "الإبلاغ عن فقدان بطاقة", "transfer": "تحويل أموال",
                       "alarm": "ضبط منبّه", "weather": "السؤال عن الطقس", "other": "طلب آخر"}}
VERIFY_CRITERIA = {"false": "هذا التصنيف لا يطابق الطلب.", "true": "هذا التصنيف يطابق الطلب."}


def verify(desc):
    return {"type": "noul", "instructions": f"هل التصنيف الصحيح لهذا الطلب هو «{desc}»؟", "criteria": VERIFY_CRITERIA}


# (id, kind, state, questions, expectation) — expectation {qid: label} for choice, {qid: bool} for noul, None = observe
CASES = [
    ("msa_choice", "Arabic MSA choice", "أرغب في معرفة الرصيد المتاح في حسابي الجاري.", {"intent": INTENT}, {"intent": "balance"}),
    ("gulf_choice_1", "Saudi/Gulf choice", "ابغى احول ٥٠٠ ريال لأخوي", {"intent": INTENT}, {"intent": "transfer"}),
    ("gulf_choice_2", "Saudi/Gulf choice", "صحني الساعة ستة الصبح بكرة", {"intent": INTENT}, {"intent": "alarm"}),
    ("noul_true", "Arabic noul (verification paired with its choice question)", "ضاعت بطاقتي امس ولا اعرف وين",
     {"intent": INTENT, "verify": verify("الإبلاغ عن فقدان بطاقة")}, {"intent": "card_lost", "verify": True}),
    ("noul_false", "Arabic noul (verification paired with its choice question)", "ضاعت بطاقتي امس ولا اعرف وين",
     {"intent": INTENT, "verify": verify("السؤال عن الطقس")}, {"intent": "card_lost", "verify": False}),
    ("multi_question", "multiple options + multiple questions", "هل بتمطر اليوم في الرياض؟",
     {"intent": INTENT, "verify": verify("السؤال عن الطقس")}, {"intent": "weather", "verify": True}),
    ("unclear", "unclear / low-confidence input", "طيب", {"intent": INTENT}, None),
    ("noul_standalone", "verification noul without its choice question (known limitation)", "ضاعت بطاقتي امس ولا اعرف وين",
     {"verify": verify("الإبلاغ عن فقدان بطاقة")}, None),
    ("noul_freeform", "free-form noul without criteria (known limitation)", "كم رصيدي في الحساب الجاري؟",
     {"banking": {"type": "noul", "instructions": "هل الطلب متعلق بخدمة بنكية؟"}}, None),
]


def check_schema(out: dict, questions: dict) -> list[str]:
    errs = []
    if set(out) != {"model", "version", "answers", "meta"}:
        errs.append(f"top-level keys {sorted(out)}")
    if set(out["answers"]) != set(questions):
        errs.append("answer ids differ from question ids")
    for qid, a in out["answers"].items():
        q = questions[qid]
        if a["type"] != q["type"]:
            errs.append(f"{qid}: type")
        probs = a["probabilities"]
        if not all(0.0 <= p <= 1.0 and math.isfinite(p) for p in probs.values()):
            errs.append(f"{qid}: probability outside [0,1]")
        if abs(sum(probs.values()) - 1.0) > 1e-9:
            errs.append(f"{qid}: probabilities sum to {sum(probs.values())}")
        if q["type"] == "choice":
            if list(probs) != list(q["criteria"]) or a["choice"] != max(probs, key=probs.get):
                errs.append(f"{qid}: choice keys/argmax")
        else:
            if set(probs) != {"false", "true"} or a["noul"] != probs["true"]:
                errs.append(f"{qid}: noul value")
    return errs


def check_temperature(out: dict, questions: dict, temps: dict) -> list[str]:
    errs = []
    for qid, z in out["meta"]["logits"].items():
        t = temps[questions[qid]["type"]]
        zs = torch.tensor(list(z.values()), dtype=torch.float64)
        expect = dict(zip(z, torch.softmax(zs / t, -1).tolist()))
        got = out["answers"][qid]["probabilities"]
        if any(abs(expect[k] - got[k]) > 1e-12 for k in got):
            errs.append(f"{qid}: probabilities are not softmax(logits / {t})")
        if t != 1.0 and max(abs(a - b) for a, b in zip(torch.softmax(zs, -1).tolist(), expect.values())) < 1e-9:
            errs.append(f"{qid}: temperature had no effect")
    return errs


def environment(device: str) -> dict:
    import peft
    import transformers

    env = {"python": sys.version.split()[0], "torch": torch.__version__, "transformers": transformers.__version__,
           "peft": peft.__version__, "os": platform.platform(), "machine": platform.machine(), "device": device}
    try:
        import psutil
        env["ram_total_gb"] = round(psutil.virtual_memory().total / 2**30, 1)
    except ImportError:
        pass
    if device.startswith("cuda"):
        env["gpu"] = torch.cuda.get_device_name(0)
        env["vram_total_gb"] = round(torch.cuda.get_device_properties(0).total_memory / 2**30, 2)
    return env


def peak_rss_gb() -> float | None:
    try:
        import psutil
        info = psutil.Process().memory_info()
        return round(getattr(info, "peak_wset", info.rss) / 2**30, 3)
    except ImportError:
        try:
            import resource
            return round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**20, 3)  # Linux: KiB
        except ImportError:
            return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default=None)
    ap.add_argument("--out", type=Path, default=Path("smoke-result.json"))
    ap.add_argument("--latency-rounds", type=int, default=20)
    a = ap.parse_args()

    t0 = time.perf_counter()
    sev = SevPreview(HERE, device=a.device)
    load_s = time.perf_counter() - t0
    failures, results = [], []

    base_cls = type(sev.model.encoder.get_base_model()).__name__
    if hasattr(sev.model, "generate") or "ForCausalLM" in base_cls or "LMHead" in base_cls:
        failures.append(f"core path is not encoder-only ({base_cls})")

    for cid, kind, state, qs, expect in CASES:
        out = sev.decide(state, qs, return_logits=True)
        errs = check_schema(out, qs) + check_temperature(out, qs, sev.temps)
        verdict = {}
        for qid, want in (expect or {}).items():
            ans = out["answers"][qid]
            ok = ans["choice"] == want if isinstance(want, str) else (ans["noul"] >= 0.5) == want
            verdict[qid] = ok
            if not ok:
                errs.append(f"{qid}: expected {want}")
        summary = {qid: ({"choice": x["choice"], "p": round(x["probabilities"][x["choice"]], 4)} if x["type"] == "choice"
                         else {"noul": round(x["noul"], 4)}) for qid, x in out["answers"].items()}
        results.append({"id": cid, "kind": kind, "state": state, "answers": summary, "expected": expect,
                        "status": "OBSERVED" if expect is None and not errs else ("PASS" if not errs else "FAIL"),
                        "errors": errs, "input_tokens": out["meta"]["input_tokens"],
                        "state_truncated": out["meta"]["state_truncated"]})
        failures += [f"{cid}: {e}" for e in errs]

    for bad, label in [({"s": {"type": "score", "instructions": "قيّم", "criteria": ["1", "2"]}}, "score rejected"),
                       ({f"q{i}": verify("x") for i in range(9)}, "more than 8 questions rejected")]:
        try:
            sev.decide("نص", bad)
            failures.append(f"{label}: no error raised")
        except ValueError:
            results.append({"id": label.replace(" ", "_"), "status": "PASS"})

    if sev.device.startswith("cuda"):
        torch.cuda.synchronize()
    lat = []
    for _ in range(a.latency_rounds):
        for _, _, state, qs, _ in CASES:
            t = time.perf_counter()
            sev.decide(state, qs)
            if sev.device.startswith("cuda"):
                torch.cuda.synchronize()
            lat.append((time.perf_counter() - t) * 1000)

    report = {"package": sev.cfg["name"], "package_version": __version__, "environment": environment(sev.device),
              "load_seconds": round(load_s, 2), "temperatures": sev.temps, "base_encoder_class": base_cls,
              "cases": results,
              "latency_ms": {"requests": len(lat), "p50": round(statistics.median(lat), 1),
                             "p90": round(sorted(lat)[int(0.9 * (len(lat) - 1))], 1)},
              "peak_ram_gb": peak_rss_gb(), "failures": failures, "result": "SMOKE_PASS" if not failures else "SMOKE_FAIL"}
    if sev.device.startswith("cuda"):
        report["peak_vram_allocated_gb"] = round(torch.cuda.max_memory_allocated() / 2**30, 3)
        report["peak_vram_reserved_gb"] = round(torch.cuda.max_memory_reserved() / 2**30, 3)
    a.out.write_text(json.dumps(report, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps({k: report[k] for k in ("result", "latency_ms", "peak_ram_gb", "failures")}, ensure_ascii=False))
    sys.exit(0 if not failures else 1)


if __name__ == "__main__":
    main()
