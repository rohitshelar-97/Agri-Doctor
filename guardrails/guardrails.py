"""
Agri Doctor – Comprehensive Guardrails Module
Implements all 9 guardrail categories:
  1. Input Guardrails
  2. Output Guardrails
  3. Tool Execution Guardrails
  4. Data Guardrails
  5. Security Guardrails
  6. Operational Guardrails
  7. Human-in-the-Loop Guardrails
  8. Multi-Agent Guardrails
  9. Ethical & Compliance Guardrails
"""

from __future__ import annotations

import hashlib
import re
import time
from dataclasses import dataclass, field
from typing import Any

# ─────────────────────────────────────────────────────────────────────────────
# Shared result type
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class GuardrailResult:
    passed: bool
    category: str
    check: str
    detail: str = ""
    blocked_value: str = ""


# ─────────────────────────────────────────────────────────────────────────────
# 1. Input Guardrails
# ─────────────────────────────────────────────────────────────────────────────

PROMPT_INJECTION_PATTERNS = [
    r"ignore\s+(previous|all|above)\s+(instructions?|prompts?|context)",
    r"you\s+are\s+now\s+(?:a|an)\s+\w+",
    r"act\s+as\s+(?:a|an)\s+\w+",
    r"jailbreak",
    r"DAN\s+mode",
    r"bypass\s+(safety|guardrail|filter)",
    r"<\s*script\s*>",
    r"system\s*:\s*(you|ignore)",
]

MALICIOUS_PATTERNS = [
    r"(rm\s+-rf|del\s+/[a-zA-Z]|format\s+[cC]:)",
    r"(exec|eval|__import__|subprocess)\s*\(",
    r"(DROP\s+TABLE|DELETE\s+FROM|INSERT\s+INTO)\s+\w+",
]

PII_PATTERNS = {
    "email": r"[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+",
    "phone": r"\+?[\d\s\-().]{10,}",
    "aadhar": r"\b\d{4}[\s\-]?\d{4}[\s\-]?\d{4}\b",
    "credit_card": r"\b(?:\d[ -]?){13,16}\b",
}

INAPPROPRIATE_KEYWORDS = [
    "violence", "weapon", "murder", "terrorism", "extremist",
    "porn", "explicit", "drug trafficking",
]

ALLOWED_IMAGE_TYPES = {"image/jpeg", "image/jpg", "image/png", "image/webp"}
MAX_IMAGE_SIZE_BYTES = 10 * 1024 * 1024  # 10 MB


def check_input_guardrails(
    text_input: str = "",
    image_mime: str | None = None,
    image_size: int = 0,
) -> list[GuardrailResult]:
    results: list[GuardrailResult] = []

    # Prompt injection detection
    for pat in PROMPT_INJECTION_PATTERNS:
        if re.search(pat, text_input, re.IGNORECASE):
            results.append(GuardrailResult(
                passed=False, category="Input", check="PromptInjection",
                detail=f"Potential prompt injection detected.", blocked_value=text_input[:80],
            ))
            break
    else:
        results.append(GuardrailResult(passed=True, category="Input", check="PromptInjection"))

    # Malicious input detection
    for pat in MALICIOUS_PATTERNS:
        if re.search(pat, text_input, re.IGNORECASE):
            results.append(GuardrailResult(
                passed=False, category="Input", check="MaliciousInput",
                detail="Potentially malicious command detected.", blocked_value=text_input[:80],
            ))
            break
    else:
        results.append(GuardrailResult(passed=True, category="Input", check="MaliciousInput"))

    # PII detection (mask, don't block)
    detected_pii = []
    for pii_type, pattern in PII_PATTERNS.items():
        if re.search(pattern, text_input):
            detected_pii.append(pii_type)
    results.append(GuardrailResult(
        passed=len(detected_pii) == 0,
        category="Input", check="PIIDetection",
        detail=f"PII types detected: {detected_pii}" if detected_pii else "",
    ))

    # Inappropriate content filtering
    low_text = text_input.lower()
    hits = [kw for kw in INAPPROPRIATE_KEYWORDS if kw in low_text]
    results.append(GuardrailResult(
        passed=len(hits) == 0,
        category="Input", check="InappropriateContent",
        detail=f"Flagged keywords: {hits}" if hits else "",
    ))

    # Image validation
    if image_mime is not None:
        results.append(GuardrailResult(
            passed=image_mime in ALLOWED_IMAGE_TYPES,
            category="Input", check="ImageTypeValidation",
            detail=f"Received MIME: {image_mime}" if image_mime not in ALLOWED_IMAGE_TYPES else "",
        ))
        results.append(GuardrailResult(
            passed=image_size <= MAX_IMAGE_SIZE_BYTES,
            category="Input", check="ImageSizeValidation",
            detail=f"Image size {image_size / 1024 / 1024:.2f} MB exceeds 10 MB limit" if image_size > MAX_IMAGE_SIZE_BYTES else "",
        ))

    return results


# ─────────────────────────────────────────────────────────────────────────────
# 2. Output Guardrails
# ─────────────────────────────────────────────────────────────────────────────

HALLUCINATION_PATTERNS = [
    r"as\s+of\s+my\s+knowledge\s+cutoff",
    r"I\s+don'?t\s+have\s+(access|information)",
    r"I\s+cannot\s+browse",
    r"\d{1,4}\s*mg\s*/\s*(?:litre|liter|l|ha|acre)",   # invented dosage
]

TOXICITY_KEYWORDS = [
    "idiot", "stupid", "worthless", "you should die", "kill yourself",
    "hate", "loathe", "despise",
]

SENSITIVE_OUTPUT_PATTERNS = {
    "api_key": r"[A-Za-z0-9_\-]{20,}",
    "password": r"password\s*[:=]\s*\S+",
}


def check_output_guardrails(response_text: str) -> list[GuardrailResult]:
    results: list[GuardrailResult] = []

    # Hallucination detection (heuristic)
    hall_hits = [p for p in HALLUCINATION_PATTERNS if re.search(p, response_text, re.IGNORECASE)]
    results.append(GuardrailResult(
        passed=len(hall_hits) == 0,
        category="Output", check="HallucinationDetection",
        detail=f"Possible hallucination markers: {hall_hits}" if hall_hits else "",
    ))

    # Toxicity filtering
    tox_hits = [kw for kw in TOXICITY_KEYWORDS if kw.lower() in response_text.lower()]
    results.append(GuardrailResult(
        passed=len(tox_hits) == 0,
        category="Output", check="ToxicityFilter",
        detail=f"Toxic keywords: {tox_hits}" if tox_hits else "",
    ))

    # Sensitive information filtering
    for label, pattern in SENSITIVE_OUTPUT_PATTERNS.items():
        match = re.search(pattern, response_text, re.IGNORECASE)
        if match:
            results.append(GuardrailResult(
                passed=False, category="Output", check="SensitiveInfoFilter",
                detail=f"Possible {label} leak detected.", blocked_value=match.group()[:20] + "...",
            ))
        else:
            results.append(GuardrailResult(passed=True, category="Output", check=f"SensitiveInfoFilter:{label}"))

    # Response length sanity check
    results.append(GuardrailResult(
        passed=len(response_text) > 50,
        category="Output", check="ResponseLengthValidation",
        detail="Response too short – may indicate an empty or failed LLM response." if len(response_text) <= 50 else "",
    ))

    return results


# ─────────────────────────────────────────────────────────────────────────────
# 3. Tool Execution Guardrails
# ─────────────────────────────────────────────────────────────────────────────

ALLOWED_AGENTS = {"pathologist_agent", "extension_officer_agent"}
ALLOWED_TOOLS = {"watsonx_vision", "watsonx_text"}


def check_tool_execution_guardrails(agent_name: str, tool_name: str) -> list[GuardrailResult]:
    results = [
        GuardrailResult(
            passed=agent_name in ALLOWED_AGENTS,
            category="ToolExecution", check="AgentAccessPermission",
            detail=f"Unknown agent: {agent_name}" if agent_name not in ALLOWED_AGENTS else "",
        ),
        GuardrailResult(
            passed=tool_name in ALLOWED_TOOLS,
            category="ToolExecution", check="ToolAccessPermission",
            detail=f"Unauthorized tool: {tool_name}" if tool_name not in ALLOWED_TOOLS else "",
        ),
    ]
    return results


# ─────────────────────────────────────────────────────────────────────────────
# 4. Data Guardrails
# ─────────────────────────────────────────────────────────────────────────────

def mask_pii(text: str) -> str:
    """Redact PII from text before logging or storing."""
    for pii_type, pattern in PII_PATTERNS.items():
        text = re.sub(pattern, f"[REDACTED_{pii_type.upper()}]", text)
    return text


def check_data_guardrails(data: dict[str, Any]) -> list[GuardrailResult]:
    results: list[GuardrailResult] = []
    # Check no raw API keys are stored in data payload
    for key, val in data.items():
        if isinstance(val, str) and re.search(r"[A-Za-z0-9_\-]{30,}", val):
            results.append(GuardrailResult(
                passed=False, category="Data", check="CredentialLeakPrevention",
                detail=f"Possible credential in field '{key}'",
            ))
        else:
            results.append(GuardrailResult(passed=True, category="Data", check="CredentialLeakPrevention"))
    return results if results else [GuardrailResult(passed=True, category="Data", check="CredentialLeakPrevention")]


# ─────────────────────────────────────────────────────────────────────────────
# 5. Security Guardrails
# ─────────────────────────────────────────────────────────────────────────────

SQL_INJECTION_PATTERNS = [
    r"(SELECT|INSERT|UPDATE|DELETE|DROP|UNION|ALTER)\s+",
    r"--\s*$",
    r";.*(DROP|DELETE|INSERT)",
]

JAILBREAK_PATTERNS = [
    r"ignore\s+all\s+(ethical|moral|safety)\s+guidelines",
    r"pretend\s+you\s+have\s+no\s+restrictions",
    r"developer\s+mode",
]


def check_security_guardrails(text: str) -> list[GuardrailResult]:
    results: list[GuardrailResult] = []

    # SQL injection
    sql_hits = [p for p in SQL_INJECTION_PATTERNS if re.search(p, text, re.IGNORECASE)]
    results.append(GuardrailResult(
        passed=len(sql_hits) == 0,
        category="Security", check="SQLInjectionPrevention",
        detail=f"SQL pattern detected" if sql_hits else "",
    ))

    # Jailbreak detection
    jb_hits = [p for p in JAILBREAK_PATTERNS if re.search(p, text, re.IGNORECASE)]
    results.append(GuardrailResult(
        passed=len(jb_hits) == 0,
        category="Security", check="JailbreakDetection",
        detail="Jailbreak attempt detected" if jb_hits else "",
    ))

    # Prompt injection (security layer, same patterns)
    pi_hits = [p for p in PROMPT_INJECTION_PATTERNS if re.search(p, text, re.IGNORECASE)]
    results.append(GuardrailResult(
        passed=len(pi_hits) == 0,
        category="Security", check="PromptInjectionProtection",
        detail="Prompt injection attempt" if pi_hits else "",
    ))

    return results


# ─────────────────────────────────────────────────────────────────────────────
# 6. Operational Guardrails
# ─────────────────────────────────────────────────────────────────────────────

MAX_EXECUTION_SECONDS = 120
MAX_INPUT_TOKENS = 4000
MAX_OUTPUT_TOKENS = 2000
MAX_ITERATIONS = 5
MAX_COST_USD = 0.50  # per session

_session_state: dict[str, Any] = {
    "iteration_count": 0,
    "total_cost_usd": 0.0,
    "request_count": 0,
    "last_request_time": 0.0,
}

RATE_LIMIT_SECONDS = 5  # minimum seconds between requests


def check_operational_guardrails(
    execution_seconds: float = 0.0,
    input_tokens: int = 0,
    output_tokens: int = 0,
    cost_usd: float = 0.0,
) -> list[GuardrailResult]:
    results: list[GuardrailResult] = []

    results.append(GuardrailResult(
        passed=execution_seconds <= MAX_EXECUTION_SECONDS,
        category="Operational", check="MaxExecutionTime",
        detail=f"Execution {execution_seconds:.1f}s > {MAX_EXECUTION_SECONDS}s limit" if execution_seconds > MAX_EXECUTION_SECONDS else "",
    ))
    results.append(GuardrailResult(
        passed=input_tokens <= MAX_INPUT_TOKENS,
        category="Operational", check="TokenInputLimit",
        detail=f"Input tokens {input_tokens} > {MAX_INPUT_TOKENS}" if input_tokens > MAX_INPUT_TOKENS else "",
    ))
    results.append(GuardrailResult(
        passed=output_tokens <= MAX_OUTPUT_TOKENS,
        category="Operational", check="TokenOutputLimit",
        detail=f"Output tokens {output_tokens} > {MAX_OUTPUT_TOKENS}" if output_tokens > MAX_OUTPUT_TOKENS else "",
    ))

    # Rate limiting
    now = time.time()
    time_since_last = now - _session_state["last_request_time"]
    _session_state["last_request_time"] = now
    _session_state["request_count"] += 1
    results.append(GuardrailResult(
        passed=_session_state["request_count"] <= 1 or time_since_last >= RATE_LIMIT_SECONDS,
        category="Operational", check="APIRateLimiting",
        detail=f"Rate limit: {time_since_last:.1f}s between requests (min {RATE_LIMIT_SECONDS}s)" if time_since_last < RATE_LIMIT_SECONDS else "",
    ))

    # Cost monitoring
    _session_state["total_cost_usd"] += cost_usd
    results.append(GuardrailResult(
        passed=_session_state["total_cost_usd"] <= MAX_COST_USD,
        category="Operational", check="CostBudgetLimit",
        detail=f"Session cost ${_session_state['total_cost_usd']:.4f} exceeds ${MAX_COST_USD} budget" if _session_state["total_cost_usd"] > MAX_COST_USD else "",
    ))

    # Iteration limit
    _session_state["iteration_count"] += 1
    results.append(GuardrailResult(
        passed=_session_state["iteration_count"] <= MAX_ITERATIONS,
        category="Operational", check="MaxAgentIterations",
        detail=f"Iteration {_session_state['iteration_count']} > {MAX_ITERATIONS}" if _session_state["iteration_count"] > MAX_ITERATIONS else "",
    ))

    return results


def reset_session_state() -> None:
    _session_state.update({
        "iteration_count": 0,
        "total_cost_usd": 0.0,
        "request_count": 0,
        "last_request_time": 0.0,
    })


# ─────────────────────────────────────────────────────────────────────────────
# 7. Human-in-the-Loop Guardrails
# ─────────────────────────────────────────────────────────────────────────────

HIGH_RISK_CHEMICALS = [
    "paraquat", "chlorpyrifos", "methyl bromide", "carbofuran",
    "endosulfan", "lindane", "aldicarb",
]


def check_hitl_guardrails(advisory_text: str) -> list[GuardrailResult]:
    """Flag advisories that mention high-risk or banned chemicals for human review."""
    hits = [chem for chem in HIGH_RISK_CHEMICALS if chem.lower() in advisory_text.lower()]
    return [GuardrailResult(
        passed=len(hits) == 0,
        category="HumanInTheLoop", check="HighRiskChemicalApproval",
        detail=f"Advisory mentions high-risk chemicals requiring human review: {hits}" if hits else "",
    )]


# ─────────────────────────────────────────────────────────────────────────────
# 8. Multi-Agent Guardrails
# ─────────────────────────────────────────────────────────────────────────────

AGENT_SEQUENCE = ["pathologist_agent", "extension_officer_agent"]


def check_multi_agent_guardrails(
    completed_agents: list[str],
    current_agent: str,
    handoff_data: dict[str, Any] | None = None,
) -> list[GuardrailResult]:
    results: list[GuardrailResult] = []

    # Task assignment validation – agents must run in sequence
    expected_idx = AGENT_SEQUENCE.index(current_agent) if current_agent in AGENT_SEQUENCE else -1
    if expected_idx > 0:
        required_predecessor = AGENT_SEQUENCE[expected_idx - 1]
        results.append(GuardrailResult(
            passed=required_predecessor in completed_agents,
            category="MultiAgent", check="TaskSequenceValidation",
            detail=f"{current_agent} requires {required_predecessor} to have completed first." if required_predecessor not in completed_agents else "",
        ))
    else:
        results.append(GuardrailResult(passed=True, category="MultiAgent", check="TaskSequenceValidation"))

    # Handoff data integrity
    if handoff_data is not None:
        required_keys = {"suspected_disease", "confidence", "symptoms"}
        missing = required_keys - set(handoff_data.keys())
        results.append(GuardrailResult(
            passed=len(missing) == 0,
            category="MultiAgent", check="HandoffDataIntegrity",
            detail=f"Missing handoff fields: {missing}" if missing else "",
        ))
    else:
        results.append(GuardrailResult(passed=True, category="MultiAgent", check="HandoffDataIntegrity"))

    # Loop detection
    agent_counts: dict[str, int] = {}
    for a in completed_agents:
        agent_counts[a] = agent_counts.get(a, 0) + 1
    loops = {a: c for a, c in agent_counts.items() if c > 1}
    results.append(GuardrailResult(
        passed=len(loops) == 0,
        category="MultiAgent", check="InfiniteLoopDetection",
        detail=f"Agents called more than once: {loops}" if loops else "",
    ))

    return results


# ─────────────────────────────────────────────────────────────────────────────
# 9. Ethical & Compliance Guardrails
# ─────────────────────────────────────────────────────────────────────────────

BIAS_INDICATORS = [
    r"(all\s+)?(farmers|peasants)\s+are\s+\w+",
    r"(only|always)\s+use\s+(brand\s+)?\w+",
]

DISCLAIMER_REQUIRED = "AI-generated preliminary assessment"


def check_ethical_guardrails(response_text: str) -> list[GuardrailResult]:
    results: list[GuardrailResult] = []

    # Bias detection (heuristic)
    bias_hits = [p for p in BIAS_INDICATORS if re.search(p, response_text, re.IGNORECASE)]
    results.append(GuardrailResult(
        passed=len(bias_hits) == 0,
        category="Ethical", check="BiasDetection",
        detail=f"Potential bias patterns: {bias_hits}" if bias_hits else "",
    ))

    # Explainability – disclaimer present
    results.append(GuardrailResult(
        passed=DISCLAIMER_REQUIRED.lower() in response_text.lower(),
        category="Ethical", check="DisclaimerPresence",
        detail="Response missing required AI disclaimer." if DISCLAIMER_REQUIRED.lower() not in response_text.lower() else "",
    ))

    # Audit trail hash
    audit_hash = hashlib.sha256(response_text.encode()).hexdigest()[:16]
    results.append(GuardrailResult(
        passed=True, category="Ethical", check="AuditTrail",
        detail=f"Response hash: {audit_hash}",
    ))

    return results


# ─────────────────────────────────────────────────────────────────────────────
# Master runner
# ─────────────────────────────────────────────────────────────────────────────

def run_all_guardrails(
    *,
    input_text: str = "",
    image_mime: str | None = None,
    image_size: int = 0,
    output_text: str = "",
    agent_name: str = "",
    tool_name: str = "",
    execution_seconds: float = 0.0,
    input_tokens: int = 0,
    output_tokens: int = 0,
    cost_usd: float = 0.0,
    completed_agents: list[str] | None = None,
    handoff_data: dict[str, Any] | None = None,
) -> dict[str, list[GuardrailResult]]:
    """Run every guardrail category and return a grouped dict."""
    completed_agents = completed_agents or []
    all_results: dict[str, list[GuardrailResult]] = {}

    all_results["Input"] = check_input_guardrails(input_text, image_mime, image_size)
    all_results["Security"] = check_security_guardrails(input_text)

    if output_text:
        all_results["Output"] = check_output_guardrails(output_text)
        all_results["HumanInTheLoop"] = check_hitl_guardrails(output_text)
        all_results["Ethical"] = check_ethical_guardrails(output_text)

    if agent_name and tool_name:
        all_results["ToolExecution"] = check_tool_execution_guardrails(agent_name, tool_name)
        all_results["MultiAgent"] = check_multi_agent_guardrails(completed_agents, agent_name, handoff_data)

    all_results["Operational"] = check_operational_guardrails(
        execution_seconds, input_tokens, output_tokens, cost_usd
    )

    return all_results


def summarise_guardrails(results: dict[str, list[GuardrailResult]]) -> dict[str, Any]:
    """Return a flat summary: total checks, passed, failed, blocked."""
    all_checks = [r for lst in results.values() for r in lst]
    passed = [r for r in all_checks if r.passed]
    failed = [r for r in all_checks if not r.passed]
    return {
        "total": len(all_checks),
        "passed": len(passed),
        "failed": len(failed),
        "blocked": any(not r.passed and r.category in ("Input", "Security") for r in all_checks),
        "failed_checks": [{"category": r.category, "check": r.check, "detail": r.detail} for r in failed],
    }
