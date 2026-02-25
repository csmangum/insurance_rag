"""Auto insurance query patterns, synonyms, and expansion data."""
from __future__ import annotations

import re

# Specialized query patterns for coverage-specific auto insurance queries
COVERAGE_QUERY_PATTERNS: list[re.Pattern[str]] = [
    re.compile(p, re.IGNORECASE)
    for p in [
        r"\bliability\s+(?:limit|coverage|minimum)\b",
        r"\bmin(?:imum)?\s+(?:coverage|liability|limits?)\b",
        r"\bbodily\s+injury\b",
        r"\bproperty\s+damage\b",
        r"\buninsured\s+motorist\b",
        r"\bunderinsured\s+motorist\b",
        r"\b(?:UM|UIM)\b",
        r"\bpersonal\s+injury\s+protection\b",
        r"\bPIP\b",
        r"\bno[- ]fault\b",
        r"\btort\s+(?:system|state|threshold)\b",
        r"\bcollision\s+coverage\b",
        r"\bcomprehensive\s+coverage\b",
        r"\bMedPay\b",
        r"\bmedical\s+payments?\b",
        r"\bgap\s+insurance\b",
        r"\brental\s+(?:car|reimbursement)\b",
        r"\btowing\b",
        r"\broadside\s+assistance\b",
    ]
]

COVERAGE_TOPIC_PATTERNS: list[tuple[re.Pattern[str], str]] = [
    (re.compile(p, re.IGNORECASE), expansion)
    for p, expansion in [
        (r"\bliability\b", "bodily injury property damage liability coverage limits minimum"),
        (r"\bPIP|personal injury protection\b", "PIP no-fault medical expenses lost wages"),
        (r"\bcollision\b", "collision coverage deductible accident damage repair"),
        (r"\bcomprehensive\b", "comprehensive coverage theft vandalism weather hail flood"),
        (r"\buninsured|underinsured|UM|UIM\b", "uninsured underinsured motorist coverage gap"),
        (r"\bsubrogation\b", "subrogation recovery third-party claim reimbursement"),
    ]
]

STRIP_COVERAGE_NOISE = re.compile(
    r"\b(?:auto insurance|car insurance|vehicle insurance|"
    r"motor vehicle|automobile|policy)\b",
    re.IGNORECASE,
)

STRIP_FILLER = re.compile(
    r"\b(?:does|have|has|an|the|for|is|are|what|which|apply to|do i need)\b",
    re.IGNORECASE,
)

# Source detection patterns
REGULATIONS_PATTERNS: list[re.Pattern[str]] = [
    re.compile(p, re.IGNORECASE)
    for p in [
        r"\bregulat(?:ion|ory|e)\b",
        r"\bstatute\b",
        r"\binsurance\s+code\b",
        r"\bDOI\b",
        r"\bdepartment\s+of\s+insurance\b",
        r"\binsurance\s+commissioner\b",
        r"\bstate\s+law\b",
        r"\bstate\s+require(?:ment|d)?\b",
        r"\bmandatory\b",
        r"\bcompulsory\b",
        r"\bfinancial\s+responsibility\b",
        r"\bNAIC\b",
        r"\bmodel\s+(?:law|regulation|act)\b",
    ]
]

FORMS_PATTERNS: list[re.Pattern[str]] = [
    re.compile(p, re.IGNORECASE)
    for p in [
        r"\bpolicy\s+form\b",
        r"\bendorsement\b",
        r"\bISO\b",
        r"\bdeclarations?\s+page\b",
        r"\bpersonal\s+auto\s+policy\b",
        r"\bPAP\b",
        r"\bcommercial\s+auto\b",
        r"\bBAP\b",
        r"\bcoverage\s+(?:part|form)\b",
        r"\bexclusion\b",
        r"\bconditions?\s+(?:section|clause)\b",
    ]
]

CLAIMS_PATTERNS: list[re.Pattern[str]] = [
    re.compile(p, re.IGNORECASE)
    for p in [
        r"\bclaim(?:s)?\s*(?:process|handling|settlement|adjustment)\b",
        r"\badjuster\b",
        r"\btotal\s+loss\b",
        r"\bsalvage\b",
        r"\bsubrogation\b",
        r"\bfraud\b",
        r"\bSIU\b",
        r"\bspecial\s+investigation\b",
        r"\bappraisal\b",
        r"\barbitration\b",
        r"\bdiminished\s+value\b",
    ]
]

RATES_PATTERNS: list[re.Pattern[str]] = [
    re.compile(p, re.IGNORECASE)
    for p in [
        r"\brate\s+(?:filing|increase|change|factor)\b",
        r"\bpremium\b",
        r"\bunderwriting\b",
        r"\brisk\s+(?:factor|classification|assessment)\b",
        r"\bactuarial\b",
        r"\bloss\s+ratio\b",
        r"\bcredit\s+(?:score|based|factor)\b",
        r"\btelematics\b",
        r"\busage[- ]based\b",
        r"\bdiscount\b",
        r"\bsurcharge\b",
    ]
]

SOURCE_PATTERNS: dict[str, list[re.Pattern[str]]] = {
    "regulations": REGULATIONS_PATTERNS,
    "forms": FORMS_PATTERNS,
    "claims": CLAIMS_PATTERNS,
    "rates": RATES_PATTERNS,
}

SOURCE_EXPANSIONS: dict[str, str] = {
    "regulations": (
        "state insurance regulation statute DOI requirement financial responsibility law"
    ),
    "forms": "policy form endorsement ISO PAP coverage declarations exclusion conditions",
    "claims": "claims handling adjustment settlement subrogation total loss appraisal arbitration",
    "rates": "premium rate filing underwriting risk factor actuarial loss ratio discount surcharge",
}

SYNONYM_MAP: list[tuple[re.Pattern[str], str]] = [
    (re.compile(p, re.IGNORECASE), expansion)
    for p, expansion in [
        (r"\bliability\b", "bodily injury property damage third-party coverage"),
        (r"\bcollision\b", "collision accident damage repair deductible"),
        (r"\bcomprehensive\b", "comprehensive theft vandalism weather hail flood fire"),
        (r"\bPIP\b", "personal injury protection no-fault medical expenses lost wages"),
        (r"\bUM\b", "uninsured motorist coverage gap protection"),
        (r"\bUIM\b", "underinsured motorist coverage additional protection"),
        (r"\bpremium\b", "premium rate cost price payment installment"),
        (r"\bdeductible\b", "deductible out-of-pocket self-insured retention"),
        (r"\btotal\s+loss\b", "total loss salvage actual cash value replacement"),
        (r"\bsubrogation\b", "subrogation recovery reimbursement third-party"),
        (r"\bfraud\b", "fraud staged accident investigation SIU"),
        (r"\bsurcharge\b", "surcharge points violation accident penalty"),
        (r"\bdiscount\b", "discount safe driver multi-policy bundling good student"),
        (r"\bgap\s+insurance\b", "gap insurance loan payoff depreciation difference"),
    ]
]

SYSTEM_PROMPT = (
    "You are a US auto insurance specialist. "
    "Answer the user's question using ONLY the provided context. "
    "When relevant, note state-specific requirements and variations. "
    "Cite sources using [1], [2], etc. corresponding to the numbered context items. "
    "If the context is insufficient to answer, say so explicitly. "
    "This is not legal or financial advice."
)

DEFAULT_SOURCE_RELEVANCE: dict[str, float] = {
    "regulations": 0.3,
    "forms": 0.25,
    "claims": 0.25,
    "rates": 0.2,
}

QUICK_QUESTIONS: list[str] = [
    "What are California's minimum auto liability limits?",
    "How does no-fault insurance work in Florida?",
    "What is PIP coverage and which states require it?",
    "Explain the difference between collision and comprehensive coverage",
    "What are uninsured/underinsured motorist requirements by state?",
    "How does the subrogation process work in auto claims?",
    "What factors affect auto insurance premiums?",
    "What is the tort vs no-fault system for auto insurance?",
]
