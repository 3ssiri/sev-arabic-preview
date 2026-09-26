"""choice: pick one option and get a calibrated probability for every option."""
import json
import sys

from sev_preview import SevPreview

sev = SevPreview(sys.argv[1] if len(sys.argv) > 1 else ".")
out = sev.decide(
    "ابغى احول ٥٠٠ ريال لأخوي",
    {"intent": {"type": "choice", "instructions": "ما نية المستخدم؟",
                "criteria": {"balance": "الاستعلام عن الرصيد", "transfer": "تحويل أموال",
                             "card_lost": "الإبلاغ عن فقدان بطاقة", "other": "طلب آخر"}}},
)
print(json.dumps(out, ensure_ascii=False, indent=1))
