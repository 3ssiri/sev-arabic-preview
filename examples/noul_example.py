"""noul: a yes/no judgement returned as P(true).

This preview's noul was trained only as a verification question ("is this label correct for the request?") with
explicit Arabic false/true criteria, always asked together with the choice question it verifies. Use it that way:
a standalone verification or a free-form yes/no question is not reliable in v0.1.
"""
import json
import sys

from sev_preview import SevPreview

sev = SevPreview(sys.argv[1] if len(sys.argv) > 1 else ".")
out = sev.decide(
    "ضاعت بطاقتي امس ولا اعرف وين",
    {"intent": {"type": "choice", "instructions": "ما نية المستخدم؟",
                "criteria": {"balance": "الاستعلام عن الرصيد", "card_lost": "الإبلاغ عن فقدان بطاقة", "other": "طلب آخر"}},
     "verify": {"type": "noul", "instructions": "هل التصنيف الصحيح لهذا الطلب هو «الإبلاغ عن فقدان بطاقة»؟",
                "criteria": {"false": "هذا التصنيف لا يطابق الطلب.", "true": "هذا التصنيف يطابق الطلب."}}},
)
print(json.dumps(out, ensure_ascii=False, indent=1))
