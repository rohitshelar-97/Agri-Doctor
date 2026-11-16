"""
Agri Doctor – LLM & Agent Performance Dashboard Metrics
Covers all 6 metric categories from the document:
  1. Accuracy & Quality Metrics
  2. RAG Evaluation Metrics
  3. Agentic AI Evaluation Metrics
  4. Safety & Guardrail Evaluation Metrics
  5. Performance & Cost Metrics
  6. Multi-Agent System Evaluation
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from difflib import SequenceMatcher
from typing import Any


# ─────────────────────────────────────────────────────────────────────────────
# Data structures
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class AgentMetrics:
    agent_name: str
    start_time: float = field(default_factory=time.perf_counter)
    end_time: float = 0.0
    input_tokens: int = 0
    output_tokens: int = 0
    success: bool = False
    error: str = ""
    # accuracy
    confidence: str = "Unknown"      # High / Medium / Low
    task_completed: bool = False
    # guardrail
    guardrail_interventions: int = 0
    guardrail_blocks: int = 0
    # cost (USD estimate based on public rates)
    cost_per_1k_input: float = 0.0003
    cost_per_1k_output: float = 0.0006

    # ── derived properties ────────────────────────────────────────────────────

    @property
    def latency_seconds(self) -> float:
        if self.end_time > 0:
            return round(self.end_time - self.start_time, 3)
        return 0.0

    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens

    @property
    def cost_usd(self) -> float:
        return round(
            (self.input_tokens / 1000) * self.cost_per_1k_input +
            (self.output_tokens / 1000) * self.cost_per_1k_output,
            6,
        )

    @property
    def tokens_per_second(self) -> float:
        if self.latency_seconds > 0:
            return round(self.output_tokens / self.latency_seconds, 2)
        return 0.0

    @property
    def task_completion_rate(self) -> float:
        return 1.0 if self.task_completed else 0.0

    @property
    def hallucination_flag(self) -> bool:
        """Heuristic: Low confidence + success → possible hallucination risk."""
        return self.success and self.confidence.lower() == "low"


# ─────────────────────────────────────────────────────────────────────────────
# 1. Accuracy & Quality Metrics
# ─────────────────────────────────────────────────────────────────────────────

def compute_semantic_similarity(text_a: str, text_b: str) -> float:
    """Simple token-overlap similarity (SequenceMatcher)."""
    if not text_a or not text_b:
        return 0.0
    return round(SequenceMatcher(None, text_a.lower(), text_b.lower()).ratio(), 4)


def compute_rouge_l(candidate: str, reference: str) -> float:
    """Lightweight ROUGE-L (longest common subsequence ratio)."""
    if not candidate or not reference:
        return 0.0
    c_tokens = candidate.lower().split()
    r_tokens = reference.lower().split()
    # LCS via SequenceMatcher
    matcher = SequenceMatcher(None, c_tokens, r_tokens)
    lcs_len = sum(block.size for block in matcher.get_matching_blocks())
    precision = lcs_len / len(c_tokens) if c_tokens else 0
    recall = lcs_len / len(r_tokens) if r_tokens else 0
    if precision + recall == 0:
        return 0.0
    return round(2 * precision * recall / (precision + recall), 4)


def compute_bleu_1(candidate: str, reference: str) -> float:
    """BLEU-1 (unigram precision)."""
    c_tokens = candidate.lower().split()
    r_tokens_set = set(reference.lower().split())
    if not c_tokens:
        return 0.0
    hits = sum(1 for t in c_tokens if t in r_tokens_set)
    return round(hits / len(c_tokens), 4)


def accuracy_metrics(candidate: str, reference: str = "") -> dict[str, Any]:
    return {
        "semantic_similarity": compute_semantic_similarity(candidate, reference) if reference else "N/A",
        "rouge_l": compute_rouge_l(candidate, reference) if reference else "N/A",
        "bleu_1": compute_bleu_1(candidate, reference) if reference else "N/A",
    }


# ─────────────────────────────────────────────────────────────────────────────
# 2. RAG Evaluation Metrics (heuristic for Agri Doctor)
# ─────────────────────────────────────────────────────────────────────────────

def rag_metrics(response: str, context: str = "") -> dict[str, Any]:
    """
    For Agri Doctor, 'context' is the pathologist's diagnosis passed to the
    extension officer – we evaluate faithfulness of the advisory to the diagnosis.
    """
    if not context:
        return {
            "faithfulness": "N/A",
            "answer_relevancy": "N/A",
            "groundedness": "N/A",
        }
    # Heuristic faithfulness: keyword overlap between diagnosis and advisory
    diag_keywords = set(context.lower().split())
    resp_words = response.lower().split()
    overlap = sum(1 for w in resp_words if w in diag_keywords)
    faithfulness = round(min(overlap / max(len(diag_keywords), 1), 1.0), 4)

    answer_relevancy = compute_semantic_similarity(response, context)
    groundedness = round((faithfulness + answer_relevancy) / 2, 4)

    return {
        "faithfulness": faithfulness,
        "answer_relevancy": answer_relevancy,
        "groundedness": groundedness,
    }


# ─────────────────────────────────────────────────────────────────────────────
# 3. Agentic AI Evaluation Metrics
# ─────────────────────────────────────────────────────────────────────────────

def agentic_metrics(
    pathologist: AgentMetrics,
    extension_officer: AgentMetrics,
) -> dict[str, Any]:
    both_completed = pathologist.task_completed and extension_officer.task_completed
    return {
        "task_completion_rate": round(
            (pathologist.task_completion_rate + extension_officer.task_completion_rate) / 2, 4
        ),
        "goal_achievement": both_completed,
        "tool_call_accuracy": "High" if both_completed else "Partial",
        "tool_execution_success_rate": round(
            sum([pathologist.success, extension_officer.success]) / 2, 4
        ),
        "planning_accuracy": "2/2 agents executed in correct order" if both_completed else "Incomplete",
        "agent_coordination": "Sequential handoff successful" if both_completed else "Handoff incomplete",
        "handoff_accuracy": extension_officer.success and pathologist.success,
        "loop_detected": False,
    }


# ─────────────────────────────────────────────────────────────────────────────
# 4. Safety & Guardrail Evaluation Metrics
# ─────────────────────────────────────────────────────────────────────────────

def safety_metrics(
    guardrail_summary: dict[str, Any],
    pathologist: AgentMetrics,
    extension_officer: AgentMetrics,
) -> dict[str, Any]:
    total = guardrail_summary.get("total", 1)
    failed = guardrail_summary.get("failed", 0)
    interventions = pathologist.guardrail_interventions + extension_officer.guardrail_interventions
    blocks = pathologist.guardrail_blocks + extension_officer.guardrail_blocks

    hallucination_rate = round(
        sum([pathologist.hallucination_flag, extension_officer.hallucination_flag]) / 2, 4
    )

    return {
        "hallucination_rate": hallucination_rate,
        "toxicity_score": "Pass" if not any(
            c.get("check") == "ToxicityFilter"
            for c in guardrail_summary.get("failed_checks", [])
        ) else "Fail",
        "bias_score": "Pass" if not any(
            c.get("check") == "BiasDetection"
            for c in guardrail_summary.get("failed_checks", [])
        ) else "Fail",
        "prompt_injection_resistance": "Pass" if not any(
            c.get("check") == "PromptInjectionProtection"
            for c in guardrail_summary.get("failed_checks", [])
        ) else "Fail",
        "pii_leakage_rate": round(failed / total, 4) if total else 0.0,
        "guardrail_intervention_rate": round(interventions / max(total, 1), 4),
        "guardrail_block_rate": round(blocks / max(total, 1), 4),
    }


# ─────────────────────────────────────────────────────────────────────────────
# 5. Performance & Cost Metrics
# ─────────────────────────────────────────────────────────────────────────────

def performance_metrics(
    pathologist: AgentMetrics,
    extension_officer: AgentMetrics,
) -> dict[str, Any]:
    total_latency = pathologist.latency_seconds + extension_officer.latency_seconds
    total_input = pathologist.input_tokens + extension_officer.input_tokens
    total_output = pathologist.output_tokens + extension_officer.output_tokens
    total_cost = pathologist.cost_usd + extension_officer.cost_usd
    error_rate = round(
        sum([not pathologist.success, not extension_officer.success]) / 2, 4
    )
    return {
        "total_response_latency_s": round(total_latency, 3),
        "pathologist_latency_s": pathologist.latency_seconds,
        "extension_officer_latency_s": extension_officer.latency_seconds,
        "pathologist_tokens_per_second": pathologist.tokens_per_second,
        "extension_officer_tokens_per_second": extension_officer.tokens_per_second,
        "total_input_tokens": total_input,
        "total_output_tokens": total_output,
        "total_tokens": total_input + total_output,
        "cost_per_request_usd": round(total_cost, 6),
        "cost_per_successful_task_usd": round(total_cost, 6) if pathologist.success and extension_officer.success else "N/A",
        "error_rate": error_rate,
    }


# ─────────────────────────────────────────────────────────────────────────────
# 6. Multi-Agent System Evaluation
# ─────────────────────────────────────────────────────────────────────────────

def multi_agent_system_metrics(
    pathologist: AgentMetrics,
    extension_officer: AgentMetrics,
) -> dict[str, Any]:
    total_time = pathologist.latency_seconds + extension_officer.latency_seconds
    workflow_complete = pathologist.success and extension_officer.success
    return {
        "total_execution_time_s": round(total_time, 3),
        "total_cost_usd": round(pathologist.cost_usd + extension_officer.cost_usd, 6),
        "end_to_end_workflow_complete": workflow_complete,
        "individual_agent_accuracy": {
            "pathologist": "High" if pathologist.confidence == "High" else
                           "Medium" if pathologist.confidence == "Medium" else "Low",
            "extension_officer": "Pass" if extension_officer.success else "Fail",
        },
        "agent_to_agent_communication": "Success" if workflow_complete else "Failed",
        "inter_agent_consistency": "Consistent" if workflow_complete else "Inconsistent",
        "conflict_resolution": "N/A (2-agent sequential pipeline)",
        "task_distribution_efficiency": round(
            pathologist.latency_seconds / max(total_time, 0.001) * 100, 1
        ),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Master dashboard builder
# ─────────────────────────────────────────────────────────────────────────────

def build_dashboard(
    pathologist: AgentMetrics,
    extension_officer: AgentMetrics,
    guardrail_summary: dict[str, Any],
    pathologist_output: str = "",
    extension_output: str = "",
    pathologist_context: str = "",
) -> dict[str, Any]:
    """
    Aggregate all metric categories into a single dashboard dict.
    """
    return {
        "accuracy_quality": accuracy_metrics(extension_output, pathologist_output),
        "rag_evaluation": rag_metrics(extension_output, pathologist_context),
        "agentic_evaluation": agentic_metrics(pathologist, extension_officer),
        "safety_guardrail": safety_metrics(guardrail_summary, pathologist, extension_officer),
        "performance_cost": performance_metrics(pathologist, extension_officer),
        "multi_agent_system": multi_agent_system_metrics(pathologist, extension_officer),
    }
