"""Medicare-specific query patterns, synonyms, and expansion data."""
from __future__ import annotations

import re

# LCD/coverage-determination query detection
LCD_QUERY_PATTERNS: list[re.Pattern[str]] = [
    re.compile(p, re.IGNORECASE)
    for p in [
        r"\blcds?\b",
        r"\blocal coverage determination\b",
        r"\bcoverage determination\b",
        r"\bncd\b",
        r"\bnational coverage determination\b",
        r"\bmcd\b",
        r"\bcontractor\b",
        r"\bjurisdiction\b",
        r"\bnovitas\b",
        r"\bfirst coast\b",
        r"\bcgs\b",
        r"\bngs\b",
        r"\bwps\b",
        r"\bpalmetto\b",
        r"\bnoridian\b",
        r"\b[jJ][a-l]\b",
        r"\bcover(?:ed)?\b.{0,40}\b(?:wound|hyperbaric|oxygen therapy|infusion|"
        r"imaging|MRI|CT scan|ultrasound|physical therapy|"
        r"cardiac rehab|chiropractic|acupuncture)\b",
        r"\bcoverage\b.{0,30}\b(?:wound|hyperbaric|oxygen|infusion|"
        r"imaging|MRI|CT|physical therapy|cardiac|"
        r"chiropractic|acupuncture|prosthetic|orthotic)\b",
        r"\b(?:wound|hyperbaric|oxygen therapy|infusion|"
        r"imaging|MRI|CT scan|physical therapy|cardiac rehab)"
        r"\b.{0,40}\bcover(?:ed)?\b",
    ]
]

LCD_TOPIC_PATTERNS: list[tuple[re.Pattern[str], str]] = [
    (re.compile(p, re.IGNORECASE), expansion)
    for p, expansion in [
        (r"\bcardiac\s*rehab", "cardiac rehabilitation program coverage criteria"),
        (r"\bhyperbaric\s*oxygen", "hyperbaric oxygen therapy wound healing coverage indications"),
        (r"\bphysical therapy", "outpatient physical therapy rehabilitation coverage"),
        (r"\b(?:wound\s*care|wound\s*vac)", "wound care negative pressure therapy coverage"),
        (r"\b(?:imaging|MRI|CT\s*scan)", "advanced diagnostic imaging coverage medical necessity"),
    ]
]

STRIP_LCD_NOISE = re.compile(
    r"\b(?:lcd|lcds|ncd|mcd|local coverage determination|"
    r"national coverage determination|coverage determination|"
    r"novitas|first coast|cgs|ngs|wps|palmetto|noridian|"
    r"contractor|jurisdiction|"
    r"[jJ][a-lA-L])\b",
    re.IGNORECASE,
)

STRIP_FILLER = re.compile(
    r"\b(?:does|have|has|an|the|for|is|are|what|which|apply to)\b",
    re.IGNORECASE,
)

# Cross-source query detection patterns
IOM_PATTERNS: list[re.Pattern[str]] = [
    re.compile(p, re.IGNORECASE)
    for p in [
        r"\bpart\s+[a-d]\b",
        r"\biom\b",
        r"\binternet\s+only\s+manual\b",
        r"\bcms\s+manual\b",
        r"\bclaim(?:s)?\s*(?:processing|submission|filing)\b",
        r"\bbenefit(?:s)?\s*(?:policy|period)\b",
        r"\benrollment\b",
        r"\beligibility\b",
        r"\bmedicare\b.*\b(?:policy|guideline|manual|chapter|rule)\b",
        r"\bgeneral\s+billing\b",
        r"\bmsn\b",
        r"\bmedicare\s+summary\s+notice\b",
        r"\bappeal(?:s)?\b",
        r"\bredetermination\b",
    ]
]

MCD_PATTERNS: list[re.Pattern[str]] = [
    re.compile(p, re.IGNORECASE)
    for p in [
        r"\blcds?\b",
        r"\bncds?\b",
        r"\bcoverage\s+determination\b",
        r"\bmedical\s+necessity\b",
        r"\bcoverage\s+criteria\b",
        r"\bindication(?:s)?\b",
        r"\blimitation(?:s)?\b",
        r"\bcontractor\b",
        r"\bjurisdiction\b",
        r"\bmcd\b",
        r"\bnovitas\b",
        r"\bfirst\s+coast\b",
        r"\bpalmetto\b",
        r"\bnoridian\b",
        r"\bcovered?\b.{0,30}\bservice",
    ]
]

CODES_PATTERNS: list[re.Pattern[str]] = [
    re.compile(p, re.IGNORECASE)
    for p in [
        r"\bhcpcs\b",
        r"\bcpt\b",
        r"\bicd[- ]?10\b",
        r"\bprocedure\s+code\b",
        r"\bdiagnosis\s+code\b",
        r"\bbilling\s+code\b",
        r"\bcode(?:s)?\s+for\b",
        r"\bmodifier\b",
        r"\bdrg\b",
        r"\brevenue\s+code\b",
        r"\b[A-V]\d{4}\b",
    ]
]

SOURCE_PATTERNS: dict[str, list[re.Pattern[str]]] = {
    "iom": IOM_PATTERNS,
    "mcd": MCD_PATTERNS,
    "codes": CODES_PATTERNS,
}

SOURCE_EXPANSIONS: dict[str, str] = {
    "iom": "Medicare policy guidelines manual chapter benefit rules",
    "mcd": "coverage determination LCD NCD criteria medical necessity indications limitations",
    "codes": "HCPCS CPT ICD-10 procedure diagnosis billing codes",
}

SYNONYM_MAP: list[tuple[re.Pattern[str], str]] = [
    (re.compile(p, re.IGNORECASE), expansion)
    for p, expansion in [
        (r"\bcoverage\b", "covered services benefits policy"),
        (r"\bbilling\b", "claims reimbursement payment"),
        (r"\brehabilitation\b", "rehab therapy treatment program"),
        (r"\bwound\s*care\b", "wound management debridement negative pressure therapy"),
        (r"\bimaging\b", "diagnostic imaging MRI CT scan X-ray ultrasound"),
        (r"\bdurable\s+medical\s+equipment\b", "DME prosthetic orthotic supplies"),
        (r"\bhome\s+health\b", "home health agency HHA skilled nursing"),
        (r"\bhospice\b", "hospice palliative end-of-life terminal care"),
        (r"\bambulance\b", "ambulance transport emergency non-emergency"),
        (r"\binfusion\b", "infusion injection drug administration"),
        (r"\bphysical\s+therapy\b", "physical therapy PT outpatient rehabilitation"),
        (r"\boccupational\s+therapy\b", "occupational therapy OT rehabilitation"),
        (r"\bspeech\s+therapy\b", "speech-language pathology SLP therapy"),
        (r"\bmental\s+health\b", "behavioral health psychiatric psychological services"),
        (r"\bdialysis\b", "dialysis ESRD end-stage renal disease"),
        (r"\bchemotherapy\b", "chemotherapy oncology cancer treatment"),
    ]
]

SYSTEM_PROMPT = (
    "You are a Medicare Revenue Cycle Management assistant. "
    "Answer the user's question using ONLY the provided context. "
    "Cite sources using [1], [2], etc. corresponding to the numbered context items. "
    "If the context is insufficient to answer, say so explicitly. "
    "This is not legal or medical advice."
)

DEFAULT_SOURCE_RELEVANCE: dict[str, float] = {"iom": 0.4, "mcd": 0.3, "codes": 0.3}

QUICK_QUESTIONS: list[str] = [
    "What is Medicare timely filing?",
    "How does LCD coverage determination work?",
    "Explain modifier 59 usage",
    "What are HCPCS Level II codes?",
    "ICD-10-CM coding guidelines overview",
    "Medicare claims appeal process",
    "What is a National Coverage Determination?",
    "Outpatient prospective payment system basics",
]
