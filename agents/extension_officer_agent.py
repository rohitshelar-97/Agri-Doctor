"""
Extension Officer Agent – Remedial Action Report
Receives the Pathologist Agent's diagnosis and uses IBM watsonx.ai Granite 4-H-Small
to generate a farmer-friendly, structured advisory report.
"""

from __future__ import annotations

import os
import time
from typing import Any


# ── watsonx text client ───────────────────────────────────────────────────────

def _build_text_client(api_key: str, project_id: str, url: str) -> Any:
    from ibm_watsonx_ai import APIClient, Credentials
    from ibm_watsonx_ai.foundation_models import ModelInference

    creds = Credentials(url=url, api_key=api_key)
    client = APIClient(credentials=creds, project_id=project_id)
    text_model = os.getenv("WATSONX_MODEL_ID", "ibm/granite-4-h-small")
    return ModelInference(model_id=text_model, api_client=client)


# ── agent prompt ──────────────────────────────────────────────────────────────

EXTENSION_SYSTEM_PROMPT = """You are a knowledgeable agricultural extension officer advising
smallholder farmers. You receive a crop-disease diagnostic report from a plant pathologist
and produce a clear, practical, and safe remedial action plan.

Guidelines:
• Write in simple language that a non-expert farmer can understand.
• Provide IMMEDIATE ACTIONS the farmer should take within 24–48 hours.
• Provide PREVENTIVE MEASURES to stop recurrence.
• Provide CROP-CARE RECOMMENDATIONS for ongoing plant health.
• If chemical treatment is needed, recommend commonly available, registered products
  only – state the active ingredient and suggest following the product label for dosage.
  NEVER invent or fabricate dosage figures.
• Include SAFETY PRECAUTIONS: PPE, re-entry intervals, environmental considerations.
• If the diagnosis is uncertain (Low confidence), recommend contacting a local
  agricultural extension officer or plant pathology laboratory before applying any
  treatment.
• End with a brief DISCLAIMER that this is AI-generated advice, not a certified
  agronomic opinion.

Format your response with clear section headings:
1. DIAGNOSIS SUMMARY
2. IMMEDIATE ACTIONS
3. PREVENTIVE MEASURES
4. CROP-CARE RECOMMENDATIONS
5. SAFE TREATMENT OPTIONS
6. SAFETY PRECAUTIONS
7. DISCLAIMER"""


def run_extension_officer_agent(
    diagnosis: dict[str, Any],
    api_key: str,
    project_id: str,
    watsonx_url: str,
) -> dict[str, Any]:
    """
    Generate a farmer-friendly advisory from the pathologist's diagnosis.

    Parameters
    ----------
    diagnosis : dict returned by pathologist_agent.run_pathologist_agent()

    Returns
    -------
    dict with keys: advisory_text, latency_seconds, input_tokens, output_tokens
    """
    start = time.perf_counter()

    # Build the hand-off message
    disease = diagnosis.get("suspected_disease", "Unknown")
    confidence = diagnosis.get("confidence", "Unknown")
    symptoms_list = diagnosis.get("symptoms", [])
    symptoms_str = "; ".join(symptoms_list) if symptoms_list else "Not specified"
    alternatives = diagnosis.get("alternatives", [])
    alt_str = ", ".join(alternatives) if alternatives else "None"
    notes = diagnosis.get("notes", "")

    user_message = f"""The crop pathologist has provided the following diagnostic report:

SUSPECTED DISEASE: {disease}
CONFIDENCE LEVEL: {confidence}
OBSERVED SYMPTOMS: {symptoms_str}
ALTERNATIVE DIAGNOSES: {alt_str}
PATHOLOGIST NOTES: {notes}

Please prepare a complete remedial action report following your guidelines."""

    try:
        model = _build_text_client(api_key, project_id, watsonx_url)

        messages = [
            {"role": "system", "content": EXTENSION_SYSTEM_PROMPT},
            {"role": "user", "content": user_message},
        ]

        response = model.chat(
            messages=messages,
            params={
                "max_new_tokens": 1200,
                "temperature": 0.3,
            },
        )

        advisory_text: str = response["choices"][0]["message"]["content"]
        usage = response.get("usage", {})
        input_tokens = usage.get("prompt_tokens", 0)
        output_tokens = usage.get("completion_tokens", 0)

    except Exception as exc:
        latency = time.perf_counter() - start
        return {
            "error": str(exc),
            "advisory_text": f"⚠️ Extension Officer Agent error: {exc}",
            "latency_seconds": round(latency, 3),
            "input_tokens": 0,
            "output_tokens": 0,
        }

    latency = time.perf_counter() - start
    return {
        "advisory_text": advisory_text,
        "latency_seconds": round(latency, 3),
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
    }
