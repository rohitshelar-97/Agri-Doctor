"""
make_demo_video.py
==================
Generates  agri_doctor_demo.mp4  – a ~90-second narrated slide demo video
for the Agri Doctor project.

Dependencies (all already in requirements.txt or standard):
  pip install Pillow opencv-python

Run:
  python make_demo_video.py
"""

from __future__ import annotations

import os
import textwrap
from pathlib import Path
from typing import Sequence

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

# ─────────────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────────────

W, H = 1280, 720
FPS = 30
OUT_PATH = Path(__file__).parent / "agri_doctor_demo.mp4"

# Colour palette
BG_DARK    = (15,  27,  46)   # deep navy
BG_CARD    = (22,  40,  66)   # slightly lighter navy
ACCENT     = (52, 211, 153)   # emerald green
ACCENT2    = (99, 179, 237)   # sky blue
WHITE      = (255, 255, 255)
MUTED      = (148, 163, 184)
YELLOW     = (251, 191,  36)
RED        = (248, 113, 113)
ORANGE     = (251, 146,  60)

# ─────────────────────────────────────────────────────────────────────────────
# Font helpers
# ─────────────────────────────────────────────────────────────────────────────

def _load_font(size: int, bold: bool = False) -> ImageFont.ImageFont:
    """Try system fonts; fall back to PIL default."""
    candidates = []
    if bold:
        candidates = [
            "C:/Windows/Fonts/arialbd.ttf",
            "C:/Windows/Fonts/calibrib.ttf",
            "C:/Windows/Fonts/seguibl.ttf",
            "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        ]
    else:
        candidates = [
            "C:/Windows/Fonts/arial.ttf",
            "C:/Windows/Fonts/calibri.ttf",
            "C:/Windows/Fonts/segoeui.ttf",
            "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        ]
    for path in candidates:
        if os.path.exists(path):
            return ImageFont.truetype(path, size)
    return ImageFont.load_default()


FONT_HUGE   = _load_font(72, bold=True)
FONT_TITLE  = _load_font(52, bold=True)
FONT_H2     = _load_font(36, bold=True)
FONT_BODY   = _load_font(26)
FONT_SMALL  = _load_font(20)
FONT_TINY   = _load_font(16)

# ─────────────────────────────────────────────────────────────────────────────
# Drawing helpers
# ─────────────────────────────────────────────────────────────────────────────

def new_frame(bg: tuple = BG_DARK) -> Image.Image:
    img = Image.new("RGB", (W, H), bg)
    return img


def draw_gradient_bar(draw: ImageDraw.ImageDraw, y: int, height: int = 4):
    """Thin emerald gradient accent bar across top."""
    for x in range(W):
        t = x / W
        r = int(ACCENT[0] * (1 - t) + ACCENT2[0] * t)
        g = int(ACCENT[1] * (1 - t) + ACCENT2[1] * t)
        b = int(ACCENT[2] * (1 - t) + ACCENT2[2] * t)
        draw.line([(x, y), (x, y + height - 1)], fill=(r, g, b))


def draw_text_wrapped(
    draw: ImageDraw.ImageDraw,
    text: str,
    x: int,
    y: int,
    font: ImageFont.ImageFont,
    fill: tuple,
    max_width: int,
    line_spacing: int = 8,
) -> int:
    """Draw word-wrapped text; return bottom y."""
    lines = textwrap.wrap(text, width=max(10, int(max_width / (font.size * 0.55))))
    cy = y
    for line in lines:
        draw.text((x, cy), line, font=font, fill=fill)
        bbox = font.getbbox(line)
        cy += (bbox[3] - bbox[1]) + line_spacing
    return cy


def draw_pill(draw: ImageDraw.ImageDraw, x: int, y: int, w: int, h: int, color: tuple):
    r = h // 2
    draw.rounded_rectangle([x, y, x + w, y + h], radius=r, fill=color)


def centered_text(draw: ImageDraw.ImageDraw, text: str, cy: int, font, fill: tuple):
    bbox = font.getbbox(text)
    tw = bbox[2] - bbox[0]
    draw.text(((W - tw) // 2, cy), text, font=font, fill=fill)


def to_cv(img: Image.Image) -> np.ndarray:
    return cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)


def frames_from_pil(img: Image.Image, n: int) -> list[np.ndarray]:
    f = to_cv(img)
    return [f] * n


# ─────────────────────────────────────────────────────────────────────────────
# Transition: cross-fade
# ─────────────────────────────────────────────────────────────────────────────

def crossfade(a: np.ndarray, b: np.ndarray, n: int = 20) -> list[np.ndarray]:
    frames = []
    for i in range(n):
        alpha = i / n
        blended = cv2.addWeighted(a, 1 - alpha, b, alpha, 0)
        frames.append(blended)
    return frames


# ─────────────────────────────────────────────────────────────────────────────
# Individual slide renderers
# ─────────────────────────────────────────────────────────────────────────────

def slide_title() -> Image.Image:
    img = new_frame()
    draw = ImageDraw.Draw(img)
    draw_gradient_bar(draw, 0, 6)

    # Green leaf icon area (simple circle)
    draw.ellipse([W // 2 - 60, 90, W // 2 + 60, 210], fill=(30, 80, 50))
    centered_text(draw, "🌾", 120, FONT_HUGE, WHITE)

    centered_text(draw, "Agri Doctor", 240, FONT_TITLE, WHITE)
    centered_text(draw, "AI Crop Disease Advisor", 310, FONT_H2, ACCENT)

    # Tag line
    tag = "Powered by IBM watsonx.ai  ·  Granite 4-H-Small  +  Mistral Vision"
    centered_text(draw, tag, 390, FONT_BODY, MUTED)

    # Hackathon badge
    draw_pill(draw, W // 2 - 220, 460, 440, 48, (40, 70, 110))
    centered_text(draw, "🏆  Inter-Institute IBM Bob Hackathon 2026", 471, FONT_SMALL, YELLOW)

    draw_gradient_bar(draw, H - 6, 6)
    return img


def slide_problem() -> Image.Image:
    img = new_frame()
    draw = ImageDraw.Draw(img)
    draw_gradient_bar(draw, 0, 6)

    centered_text(draw, "The Problem", 50, FONT_H2, ACCENT)

    items = [
        ("😰", "Farmers see symptoms — yellow leaves, black spots, wilting —"),
        ("",   "but can't tell if it's fungal, bacterial, or nutrient deficiency."),
        ("💸", "Wrong diagnosis → wrong chemicals → wasted money + soil harm."),
        ("🌍", "Smallholder farmers have no easy access to plant pathologists."),
        ("⏱️", "Every hour of delay means more crop loss and financial stress."),
    ]

    y = 160
    for icon, text in items:
        prefix = f"{icon}  " if icon else "      "
        draw.text((100, y), prefix + text, font=FONT_BODY, fill=WHITE if icon else MUTED)
        y += 60

    # Stat box
    draw_pill(draw, 100, 530, 1080, 70, BG_CARD)
    centered_text(draw, "Up to 40% of crop yield is lost annually due to disease misdiagnosis.", 549, FONT_SMALL, YELLOW)

    draw_gradient_bar(draw, H - 6, 6)
    return img


def slide_solution() -> Image.Image:
    img = new_frame()
    draw = ImageDraw.Draw(img)
    draw_gradient_bar(draw, 0, 6)

    centered_text(draw, "Our Solution", 50, FONT_H2, ACCENT)
    centered_text(draw, "A Two-Agent AI Pipeline", 100, FONT_BODY, MUTED)

    # Agent 1 box
    draw.rounded_rectangle([80, 180, 580, 420], radius=16, fill=BG_CARD, outline=ACCENT, width=2)
    draw.text((110, 200), "Agent 1", font=FONT_H2, fill=ACCENT)
    draw.text((110, 250), "🔬  Pathologist Agent", font=FONT_BODY, fill=WHITE)
    draw.text((110, 295), "• Vision model analyses crop image", font=FONT_SMALL, fill=MUTED)
    draw.text((110, 325), "• Identifies disease, symptoms,", font=FONT_SMALL, fill=MUTED)
    draw.text((110, 350), "  confidence level & alternatives", font=FONT_SMALL, fill=MUTED)
    draw.text((110, 385), "Model: Mistral Vision (watsonx.ai)", font=FONT_TINY, fill=ACCENT2)

    # Arrow
    draw.text((600, 290), "→", font=FONT_TITLE, fill=ACCENT)

    # Agent 2 box
    draw.rounded_rectangle([700, 180, 1200, 420], radius=16, fill=BG_CARD, outline=ACCENT2, width=2)
    draw.text((730, 200), "Agent 2", font=FONT_H2, fill=ACCENT2)
    draw.text((730, 250), "📋  Extension Officer Agent", font=FONT_BODY, fill=WHITE)
    draw.text((730, 295), "• Converts diagnosis to plain-language", font=FONT_SMALL, fill=MUTED)
    draw.text((730, 325), "  farmer advisory", font=FONT_SMALL, fill=MUTED)
    draw.text((730, 355), "• Immediate actions + safe treatments", font=FONT_SMALL, fill=MUTED)
    draw.text((730, 385), "Model: IBM Granite 4-H-Small", font=FONT_TINY, fill=ACCENT2)

    # Bottom
    draw_pill(draw, 80, 450, 1120, 50, (20, 50, 30))
    centered_text(draw, "🛡️  9-Category Guardrails protect every step of the pipeline", 463, FONT_SMALL, ACCENT)

    draw_gradient_bar(draw, H - 6, 6)
    return img


def slide_architecture() -> Image.Image:
    img = new_frame()
    draw = ImageDraw.Draw(img)
    draw_gradient_bar(draw, 0, 6)

    centered_text(draw, "Architecture", 30, FONT_H2, ACCENT)

    steps = [
        ("📷", "User uploads crop image",        ACCENT2),
        ("🛡️", "Input Guardrails (9 checks)",    YELLOW),
        ("🔬", "Agent 1: Pathologist → Diagnosis JSON", ACCENT),
        ("🔄", "Agent Handoff + Output Guardrails",     ORANGE),
        ("📋", "Agent 2: Extension Officer → Advisory", ACCENT2),
        ("📊", "Performance Dashboard + Download",      MUTED),
    ]

    bx, by, bw, bh, gap = 160, 100, 960, 64, 10
    for i, (icon, label, color) in enumerate(steps):
        y = by + i * (bh + gap)
        alpha = 180 + int(75 * i / len(steps))
        draw.rounded_rectangle([bx, y, bx + bw, y + bh], radius=12,
                                fill=BG_CARD, outline=color, width=2)
        draw.text((bx + 20, y + 14), f"{icon}  {label}", font=FONT_BODY, fill=color)

        # connector line
        if i < len(steps) - 1:
            mid_x = bx + bw // 2
            draw.line([(mid_x, y + bh), (mid_x, y + bh + gap)], fill=MUTED, width=2)

    draw_gradient_bar(draw, H - 6, 6)
    return img


def slide_guardrails() -> Image.Image:
    img = new_frame()
    draw = ImageDraw.Draw(img)
    draw_gradient_bar(draw, 0, 6)

    centered_text(draw, "🛡️  9-Category Guardrails Framework", 30, FONT_H2, ACCENT)

    cats = [
        ("1", "Input",          "Prompt injection · PII · image validation"),
        ("2", "Output",         "Hallucination · toxicity · disclaimer enforcement"),
        ("3", "Tool Execution", "Agent access control · tool authorisation"),
        ("4", "Data",           "PII masking · credential leak prevention"),
        ("5", "Security",       "SQL injection · jailbreak resistance"),
        ("6", "Operational",    "Rate limiting · token budget · cost monitoring"),
        ("7", "Human-in-Loop",  "High-risk chemical flagging for review"),
        ("8", "Multi-Agent",    "Sequence validation · handoff integrity"),
        ("9", "Ethical",        "Bias detection · audit trail"),
    ]

    cols = 3
    col_w = W // cols
    for i, (num, name, desc) in enumerate(cats):
        col = i % cols
        row = i // cols
        x = col * col_w + 40
        y = 110 + row * 170
        draw.rounded_rectangle([x, y, x + col_w - 50, y + 148], radius=10, fill=BG_CARD)
        draw.text((x + 12, y + 10), f"#{num}  {name}", font=FONT_SMALL, fill=ACCENT)
        draw_text_wrapped(draw, desc, x + 12, y + 44, FONT_TINY, MUTED, col_w - 70, 6)

    draw_gradient_bar(draw, H - 6, 6)
    return img


def slide_dashboard() -> Image.Image:
    img = new_frame()
    draw = ImageDraw.Draw(img)
    draw_gradient_bar(draw, 0, 6)

    centered_text(draw, "📊  Performance Dashboard", 30, FONT_H2, ACCENT)
    centered_text(draw, "6 Metric Categories — real-time after every diagnosis", 90, FONT_SMALL, MUTED)

    metrics = [
        ("⚡ Performance & Cost",   "Latency · Tokens · Cost/Request · Error Rate",  ACCENT),
        ("🤖 Agentic AI Eval",      "Task Completion · Goal Achievement · Handoff",  ACCENT2),
        ("📚 RAG Evaluation",       "Faithfulness · Groundedness · Answer Relevancy", YELLOW),
        ("🎯 Accuracy & Quality",   "Semantic Similarity · ROUGE-L · BLEU-1",        ORANGE),
        ("🔒 Safety & Guardrails",  "Hallucination Rate · Toxicity · Bias Score",    RED),
        ("🔗 Multi-Agent System",   "Total Exec Time · Cost · Workflow Completion",   MUTED),
    ]

    for i, (title, detail, color) in enumerate(metrics):
        col = i % 2
        row = i // 2
        x = 60 + col * 620
        y = 150 + row * 155
        draw.rounded_rectangle([x, y, x + 570, y + 130], radius=12, fill=BG_CARD, outline=color, width=1)
        draw.text((x + 18, y + 16), title, font=FONT_BODY, fill=color)
        draw.text((x + 18, y + 60), detail, font=FONT_SMALL, fill=MUTED)

    draw_gradient_bar(draw, H - 6, 6)
    return img


def slide_demo_walkthrough() -> list[Image.Image]:
    """Returns 3 sequential slides showing the UI walkthrough."""
    slides = []

    # Step 1 – Upload
    img = new_frame()
    draw = ImageDraw.Draw(img)
    draw_gradient_bar(draw, 0, 6)
    centered_text(draw, "Live Demo  —  Step 1", 40, FONT_H2, ACCENT2)

    draw.rounded_rectangle([80, 120, 740, 600], radius=14, fill=BG_CARD, outline=ACCENT2, width=2)
    draw.text((110, 150), "📷  Upload Crop Image", font=FONT_H2, fill=WHITE)
    draw.rounded_rectangle([110, 220, 710, 420], radius=10, fill=(30, 50, 80), outline=MUTED, width=1)
    centered_text(draw, "[ Drop crop image here ]", 300, FONT_BODY, MUTED)
    draw.text((110, 440), "Supported: JPG · JPEG · PNG  (max 10 MB)", font=FONT_SMALL, fill=MUTED)

    draw.rounded_rectangle([110, 490, 710, 550], radius=10, fill=ACCENT)
    centered_text(draw, "🩺  Diagnose Crop", 507, FONT_BODY, BG_DARK)

    draw.rounded_rectangle([780, 120, 1200, 600], radius=14, fill=BG_CARD, outline=ACCENT, width=2)
    draw.text((810, 150), "⚙️ Configuration", font=FONT_H2, fill=WHITE)
    for label, val in [("API Key", "••••••••••••"), ("Project ID", "abc123..."), ("Endpoint URL", "us-south.ml.cloud.ibm.com")]:
        draw.text((810, 240 + [0, 90, 180][["API Key", "Project ID", "Endpoint URL"].index(label)]),
                  f"{label}:", font=FONT_SMALL, fill=MUTED)
        draw.rounded_rectangle([810, 260 + [0, 90, 180][["API Key", "Project ID", "Endpoint URL"].index(label)],
                                 1170, 290 + [0, 90, 180][["API Key", "Project ID", "Endpoint URL"].index(label)]],
                                radius=6, fill=(30, 50, 80))
        draw.text((820, 264 + [0, 90, 180][["API Key", "Project ID", "Endpoint URL"].index(label)]),
                  val, font=FONT_SMALL, fill=ACCENT2)

    draw_gradient_bar(draw, H - 6, 6)
    slides.append(img)

    # Step 2 – Diagnosis result
    img2 = new_frame()
    draw2 = ImageDraw.Draw(img2)
    draw_gradient_bar(draw2, 0, 6)
    centered_text(draw2, "Live Demo  —  Step 2", 40, FONT_H2, ACCENT2)

    draw2.rounded_rectangle([80, 120, 1200, 600], radius=14, fill=BG_CARD, outline=ACCENT, width=2)
    draw2.text((110, 140), "✅  Agent 1 – Pathologist diagnosis complete", font=FONT_BODY, fill=ACCENT)
    draw2.text((110, 190), "🌿  Suspected Disease:", font=FONT_SMALL, fill=MUTED)
    draw2.text((110, 218), "Late Blight (Phytophthora infestans)", font=FONT_H2, fill=WHITE)
    draw2.text((110, 270), "Confidence:  🟢  High", font=FONT_BODY, fill=ACCENT)
    draw2.text((110, 315), "Observed Symptoms:", font=FONT_SMALL, fill=MUTED)
    syms = ["• Dark water-soaked lesions on lower leaves", "• White powdery sporulation on leaf undersides",
            "• Brown necrotic patches spreading inward", "• Affected stems show brown discolouration"]
    for j, s in enumerate(syms):
        draw2.text((130, 345 + j * 32), s, font=FONT_SMALL, fill=MUTED)

    draw2.text((110, 490), "Alternative Diagnoses:  Early Blight,  Cercospora Leaf Spot", font=FONT_SMALL, fill=ORANGE)
    draw2.rounded_rectangle([110, 530, 1170, 575], radius=8, fill=(30, 50, 20))
    draw2.text((120, 544), "⚠️ AI-generated preliminary assessment. Consult a certified agronomist for confirmation.", font=FONT_TINY, fill=YELLOW)

    draw_gradient_bar(draw2, H - 6, 6)
    slides.append(img2)

    # Step 3 – Advisory + Download
    img3 = new_frame()
    draw3 = ImageDraw.Draw(img3)
    draw_gradient_bar(draw3, 0, 6)
    centered_text(draw3, "Live Demo  —  Step 3", 40, FONT_H2, ACCENT2)

    draw3.rounded_rectangle([80, 120, 1200, 580], radius=14, fill=BG_CARD, outline=ACCENT2, width=2)
    draw3.text((110, 140), "✅  Agent 2 – Extension Officer advisory complete", font=FONT_BODY, fill=ACCENT2)
    sections = [
        ("1. DIAGNOSIS SUMMARY",    "Late Blight confirmed with High confidence."),
        ("2. IMMEDIATE ACTIONS",    "Remove & destroy infected plant material within 24 h."),
        ("3. PREVENTIVE MEASURES",  "Increase plant spacing; avoid overhead irrigation."),
        ("4. SAFE TREATMENT",       "Apply copper-based fungicide (active: copper hydroxide)."),
        ("5. SAFETY PRECAUTIONS",   "Wear gloves, mask, and goggles. Observe re-entry interval."),
    ]
    for j, (sec, detail) in enumerate(sections):
        draw3.text((110, 200 + j * 68), sec, font=FONT_SMALL, fill=ACCENT)
        draw3.text((110, 226 + j * 68), detail, font=FONT_TINY, fill=MUTED)

    draw3.rounded_rectangle([110, 560, 500, 610], radius=10, fill=ACCENT)
    draw3.text((130, 575), "⬇️  Download Advisory Report (.txt)", font=FONT_SMALL, fill=BG_DARK)

    draw_gradient_bar(draw3, H - 6, 6)
    slides.append(img3)

    return slides


def slide_tech_stack() -> Image.Image:
    img = new_frame()
    draw = ImageDraw.Draw(img)
    draw_gradient_bar(draw, 0, 6)

    centered_text(draw, "🛠️  Technology Stack", 40, FONT_H2, ACCENT)

    stack = [
        ("Language",       "Python 3.10+",         ACCENT2),
        ("UI Framework",   "Streamlit",             ACCENT),
        ("LLM Platform",   "IBM watsonx.ai",        YELLOW),
        ("Text Model",     "ibm/granite-4-h-small", ORANGE),
        ("Vision Model",   "Mistral Small 3.1",     ACCENT2),
        ("Image",          "Pillow (PIL)",           MUTED),
        ("Visualisation",  "Pandas + Streamlit Charts", MUTED),
        ("Config",         "python-dotenv",         MUTED),
    ]

    cols = 2
    col_w = W // cols
    for i, (comp, tech, color) in enumerate(stack):
        col = i % cols
        row = i // cols
        x = 60 + col * 600
        y = 140 + row * 118
        draw.rounded_rectangle([x, y, x + 550, y + 96], radius=10, fill=BG_CARD, outline=color, width=1)
        draw.text((x + 18, y + 14), comp, font=FONT_SMALL, fill=MUTED)
        draw.text((x + 18, y + 48), tech, font=FONT_BODY, fill=color)

    draw_gradient_bar(draw, H - 6, 6)
    return img


def slide_cta() -> Image.Image:
    img = new_frame()
    draw = ImageDraw.Draw(img)
    draw_gradient_bar(draw, 0, 8)

    centered_text(draw, "🌾", 110, FONT_HUGE, WHITE)
    centered_text(draw, "Agri Doctor", 220, FONT_TITLE, WHITE)
    centered_text(draw, "AI Crop Disease Advisor", 295, FONT_H2, ACCENT)

    centered_text(draw, "Built with  IBM Bob  ·  IBM watsonx.ai  ·  Granite 4-H-Small  +  Mistral Vision", 380, FONT_SMALL, MUTED)

    draw_pill(draw, W // 2 - 300, 440, 600, 54, (20, 60, 40))
    centered_text(draw, "▶  streamlit run app.py  →  localhost:8501", 455, FONT_BODY, ACCENT)

    draw_pill(draw, W // 2 - 220, 520, 440, 48, (40, 70, 110))
    centered_text(draw, "🏆  Inter-Institute IBM Bob Hackathon 2026", 532, FONT_SMALL, YELLOW)

    centered_text(draw, "Made with ❤️ using IBM Bob and IBM watsonx.ai", 630, FONT_TINY, MUTED)

    draw_gradient_bar(draw, H - 8, 8)
    return img


# ─────────────────────────────────────────────────────────────────────────────
# Title card helper with subtitle (for section dividers)
# ─────────────────────────────────────────────────────────────────────────────

def section_divider(icon: str, title: str, subtitle: str = "") -> Image.Image:
    img = new_frame(BG_DARK)
    draw = ImageDraw.Draw(img)
    draw_gradient_bar(draw, 0, 6)
    draw_gradient_bar(draw, H - 6, 6)

    centered_text(draw, icon, H // 2 - 110, FONT_HUGE, WHITE)
    centered_text(draw, title, H // 2 - 20, FONT_TITLE, WHITE)
    if subtitle:
        centered_text(draw, subtitle, H // 2 + 70, FONT_H2, ACCENT)

    return img


# ─────────────────────────────────────────────────────────────────────────────
# Main builder
# ─────────────────────────────────────────────────────────────────────────────

def build_video():
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(OUT_PATH), fourcc, FPS, (W, H))

    def hold(img: Image.Image, seconds: float):
        f = to_cv(img)
        for _ in range(int(seconds * FPS)):
            writer.write(f)
        return f

    def transition(prev_cv, next_pil: Image.Image, hold_secs: float):
        next_cv = to_cv(next_pil)
        for f in crossfade(prev_cv, next_cv, n=int(0.5 * FPS)):
            writer.write(f)
        for _ in range(int(hold_secs * FPS)):
            writer.write(next_cv)
        return next_cv

    # ── Slide sequence ────────────────────────────────────────────────────────
    print("Rendering slides ...")

    # 1. Title  (3 s)
    t = slide_title()
    prev = hold(t, 3.5)

    # 2. Problem
    s = section_divider("❓", "The Problem", "Why do farmers need Agri Doctor?")
    prev = transition(prev, s, 1.0)
    prev = transition(prev, slide_problem(), 4.0)

    # 3. Solution
    s = section_divider("💡", "Our Solution", "Two AI agents working together")
    prev = transition(prev, s, 1.0)
    prev = transition(prev, slide_solution(), 4.5)

    # 4. Architecture
    s = section_divider("🏗️", "Architecture", "End-to-end agentic pipeline")
    prev = transition(prev, s, 1.0)
    prev = transition(prev, slide_architecture(), 4.5)

    # 5. Guardrails
    s = section_divider("🛡️", "Guardrails", "9 categories protecting every agent call")
    prev = transition(prev, s, 1.0)
    prev = transition(prev, slide_guardrails(), 5.0)

    # 6. Dashboard
    s = section_divider("📊", "Dashboard", "LLM & Agent performance metrics")
    prev = transition(prev, s, 1.0)
    prev = transition(prev, slide_dashboard(), 4.5)

    # 7. Demo walkthrough (3 steps)
    s = section_divider("▶️", "Live Demo", "Let's see it in action")
    prev = transition(prev, s, 1.0)
    for step in slide_demo_walkthrough():
        prev = transition(prev, step, 4.5)

    # 8. Tech stack
    s = section_divider("🛠️", "Tech Stack")
    prev = transition(prev, s, 0.8)
    prev = transition(prev, slide_tech_stack(), 4.0)

    # 9. CTA / Credits
    cta = slide_cta()
    prev = transition(prev, cta, 4.0)

    writer.release()
    size_mb = OUT_PATH.stat().st_size / 1_048_576
    print(f"\nDone! Video saved -> {OUT_PATH}  ({size_mb:.1f} MB)")
    print(f"   Resolution : {W}x{H}  |  FPS : {FPS}")


if __name__ == "__main__":
    build_video()
