"""
Pathologist Agent – Crop Disease Diagnosis
Uses IBM watsonx.ai vision model (mistralai/mistral-small-3-1-24b-instruct-2503)
to analyse an uploaded crop/plant-leaf image and return a structured diagnosis.
"""

from __future__ import annotations

import base64
import os
import time
from io import BytesIO
from typing import Any

from PIL import Image


# ── helpers ──────────────────────────────────────────────────────────────────

def _image_to_base64(image: Image.Image, fmt: str = "JPEG") -> str:
    buf = BytesIO()
    image.save(buf, format=fmt)
    return base64.b64encode(buf.getvalue()).decode("utf-8")


def _build_vision_client(api_key: str, project_id: str, url: str) -> Any:
    """Return an ibm-watsonx-ai ModelInference client for vision tasks."""
    from ibm_watsonx_ai import APIClient, Credentials
    from ibm_watsonx_ai.foundation_models import ModelInference

    creds = Credentials(url=url, api_key=api_key)
    client = APIClient(credentials=creds, project_id=project_id)
    vision_model = os.getenv("WATSONX_VISION_MODEL", "mistralai/mistral-small-3-1-24b-instruct-2503")
    return ModelInference(model_id=vision_model, api_client=client)


# ── main agent function ───────────────────────────────────────────────────────

PATHOLOGIST_SYSTEM_PROMPT = """You are an expert agricultural pathologist specialising in crop
disease diagnosis. When given an image of a crop or plant leaf, you:
1. Identify the most likely disease, pest infestation, or nutrient deficiency visible.
2. List the observed symptoms (colour changes, lesions, spots, wilting, etc.).
3. Provide a confidence level (High / Medium / Low) for your primary diagnosis.
4. Suggest up to two alternative diagnoses if applicable.
5. If the image is too blurry, too dark, or otherwise insufficient, clearly state that
   a clearer image is required rather than guessing.

IMPORTANT: This is an AI-generated preliminary assessment, NOT a confirmed laboratory
diagnosis. Always recommend consulting a local agricultural extension officer for
confirmation.

Respond in the following structured format:
---
SUSPECTED DISEASE: <disease name>
OBSERVED SYMPTOMS: <bullet list>
CONFIDENCE LEVEL: <High | Medium | Low>
ALTERNATIVE DIAGNOSES: <up to two, or "None">
NOTES: <any additional observations or caveats>
---"""


def run_pathologist_agent(
    image: Image.Image,
    api_key: str,
    project_id: str,
    watsonx_url: str,
) -> dict[str, Any]:
    """
    Analyse a crop image and return a diagnosis dict.

    Returns
    -------
    dict with keys: diagnosis_text, suspected_disease, symptoms, confidence,
                    alternatives, notes, latency_seconds, input_tokens, output_tokens
    """
    start = time.perf_counter()

    # Encode image
    img_b64 = _image_to_base64(image, fmt="JPEG")
    mime = "image/jpeg"

    # Build prompt
    user_message = (
        "Please examine this crop/plant-leaf image and provide a full disease diagnosis "
        "following the structured format in your instructions."
    )

    try:
        model = _build_vision_client(api_key, project_id, watsonx_url)

        # watsonx-ai vision inference – messages API
        messages = [
            {
                "role": "system",
                "content": PATHOLOGIST_SYSTEM_PROMPT,
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:{mime};base64,{img_b64}"
                        },
                    },
                    {"type": "text", "text": user_message},
                ],
            },
        ]

        response = model.chat(
            messages=messages,
            params={
                "max_new_tokens": 800,
                "temperature": 0.2,
            },
        )

        raw_text: str = response["choices"][0]["message"]["content"]
        usage = response.get("usage", {})
        input_tokens = usage.get("prompt_tokens", 0)
        output_tokens = usage.get("completion_tokens", 0)

    except Exception as exc:
        latency = time.perf_counter() - start
        return {
            "error": str(exc),
            "diagnosis_text": f"⚠️ Pathologist Agent error: {exc}",
            "suspected_disease": "Unknown",
            "symptoms": [],
            "confidence": "N/A",
            "alternatives": [],
            "notes": "API call failed.",
            "latency_seconds": latency,
            "input_tokens": 0,
            "output_tokens": 0,
        }

    latency = time.perf_counter() - start

    # ── parse structured response ─────────────────────────────────────────────
    parsed = _parse_diagnosis(raw_text)
    parsed.update(
        {
            "diagnosis_text": raw_text,
            "latency_seconds": round(latency, 3),
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
        }
    )
    return parsed


def _parse_diagnosis(text: str) -> dict[str, Any]:
    """Extract structured fields from the model's text response."""
    fields: dict[str, Any] = {
        "suspected_disease": "Unknown",
        "symptoms": [],
        "confidence": "Unknown",
        "alternatives": [],
        "notes": "",
    }
    for line in text.splitlines():
        line = line.strip()
        if line.upper().startswith("SUSPECTED DISEASE:"):
            fields["suspected_disease"] = line.split(":", 1)[-1].strip()
        elif line.upper().startswith("OBSERVED SYMPTOMS:"):
            sym = line.split(":", 1)[-1].strip()
            if sym:
                fields["symptoms"] = [s.strip("•- ") for s in sym.split(",")]
        elif line.startswith("•") or line.startswith("-"):
            # continuation of bullet-list symptoms
            if fields["symptoms"] is not None:
                fields["symptoms"].append(line.strip("•- "))
        elif line.upper().startswith("CONFIDENCE LEVEL:"):
            fields["confidence"] = line.split(":", 1)[-1].strip()
        elif line.upper().startswith("ALTERNATIVE DIAGNOSES:"):
            alt = line.split(":", 1)[-1].strip()
            fields["alternatives"] = [a.strip() for a in alt.split(",") if a.strip() and a.strip().lower() != "none"]
        elif line.upper().startswith("NOTES:"):
            fields["notes"] = line.split(":", 1)[-1].strip()
    return fields
