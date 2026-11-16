"""
Agri Doctor – Main Streamlit Application
Agentic AI application for crop disease diagnosis and farmer advisory.

Architecture:
  Agent 1 → Pathologist Agent  : Analyses uploaded crop image → diagnosis
  Agent 2 → Extension Officer  : Converts diagnosis → farmer-friendly advisory

Features:
  • IBM watsonx.ai (Granite 4-H-Small text + Mistral vision)
  • 9-category guardrails framework
  • LLM & Agent Performance Dashboard (6 metric categories)
  • Downloadable advisory report
"""

import os
import time
from io import BytesIO

import streamlit as st
from dotenv import load_dotenv
from PIL import Image

# Load local env_config.txt when running locally (no-op on Streamlit Cloud)
load_dotenv(dotenv_path="env_config.txt")


def _secret(key: str, default: str = "") -> str:
    """Read a config value: st.secrets first (Streamlit Cloud), then os.environ."""
    try:
        return st.secrets[key]
    except (KeyError, FileNotFoundError):
        return os.environ.get(key, default)

from agents.pathologist_agent import run_pathologist_agent
from agents.extension_officer_agent import run_extension_officer_agent
from guardrails.guardrails import (
    run_all_guardrails,
    summarise_guardrails,
    reset_session_state,
)
from dashboard.metrics import (
    AgentMetrics,
    build_dashboard,
)

# ─────────────────────────────────────────────────────────────────────────────
# Page config
# ─────────────────────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="🌱 Agri Doctor",
    page_icon="🌾",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────────────────────────────────
# Sidebar – Configuration
# ─────────────────────────────────────────────────────────────────────────────

with st.sidebar:
    st.image("https://upload.wikimedia.org/wikipedia/commons/5/51/IBM_logo.svg", width=80)
    st.title("⚙️ Configuration")

    st.markdown("### IBM watsonx.ai Credentials")
    api_key = st.text_input(
        "API Key",
        value=_secret("WATSONX_API_KEY"),
        type="password",
        help="Your IBM Cloud API key – never shared or logged.",
    )
    project_id = st.text_input(
        "Project ID",
        value=_secret("WATSONX_PROJECT_ID"),
        help="Your IBM watsonx.ai Project ID.",
    )
    watsonx_url = st.text_input(
        "Endpoint URL",
        value=_secret("WATSONX_URL", "https://us-south.ml.cloud.ibm.com"),
    )

    st.markdown("---")
    st.markdown("### Model Configuration")
    text_model = st.text_input(
        "Text Model",
        value=_secret("WATSONX_MODEL_ID", "ibm/granite-4-h-small"),
    )
    vision_model = st.text_input(
        "Vision Model",
        value=_secret("WATSONX_VISION_MODEL", "mistralai/mistral-small-3-1-24b-instruct-2503"),
    )

    # Push model choices back into env for agents to pick up
    os.environ["WATSONX_MODEL_ID"] = text_model
    os.environ["WATSONX_VISION_MODEL"] = vision_model

    st.markdown("---")
    st.info(
        "🔒 **Security**: Your API key is used only during this session and "
        "is never stored or logged.",
        icon="🔒",
    )

    st.markdown("---")
    st.caption("Agri Doctor v1.0 · IBM Hackathon 2026")

# ─────────────────────────────────────────────────────────────────────────────
# Main layout
# ─────────────────────────────────────────────────────────────────────────────

st.title("🌾 Agri Doctor – AI Crop Disease Advisor")
st.caption(
    "An Agentic AI application powered by **IBM watsonx.ai** (Granite 4-H-Small + Mistral Vision). "
    "Upload a crop image to receive an instant diagnosis and farmer-friendly advisory."
)

# Tabs
tab_main, tab_guardrails, tab_dashboard = st.tabs([
    "🔍 Diagnosis & Advisory",
    "🛡️ Guardrails Report",
    "📊 Performance Dashboard",
])

# ─────────────────────────────────────────────────────────────────────────────
# Tab 1 – Diagnosis & Advisory
# ─────────────────────────────────────────────────────────────────────────────

with tab_main:
    col_upload, col_result = st.columns([1, 2])

    with col_upload:
        st.subheader("📷 Upload Crop Image")
        uploaded_file = st.file_uploader(
            "Choose a crop or plant-leaf image",
            type=["jpg", "jpeg", "png"],
            help="Supported formats: JPG, JPEG, PNG. Maximum 10 MB.",
        )

        if uploaded_file:
            image = Image.open(uploaded_file)
            st.image(image, caption=f"Uploaded: {uploaded_file.name}", use_container_width=True)
            st.caption(
                f"Size: {uploaded_file.size / 1024:.1f} KB · "
                f"Format: {image.format or uploaded_file.type}"
            )

        diagnose_btn = st.button(
            "🩺 Diagnose Crop",
            type="primary",
            disabled=(uploaded_file is None or not api_key or not project_id),
            use_container_width=True,
        )

        if not api_key or not project_id:
            st.warning("⚠️ Enter your IBM API Key and Project ID in the sidebar to proceed.")

    with col_result:
        if diagnose_btn and uploaded_file:
            # Reset operational counters for a fresh session
            reset_session_state()

            # ── Input Guardrails ──────────────────────────────────────────────
            mime_type = uploaded_file.type or "image/jpeg"
            guardrail_input_results = run_all_guardrails(
                image_mime=mime_type,
                image_size=uploaded_file.size,
                agent_name="pathologist_agent",
                tool_name="watsonx_vision",
                completed_agents=[],
            )
            g_summary_input = summarise_guardrails(guardrail_input_results)

            if g_summary_input["blocked"]:
                st.error(
                    "🚫 **Input blocked by guardrails.** Please upload a valid crop image.\n\n"
                    + "\n".join(f"• {c['detail']}" for c in g_summary_input["failed_checks"])
                )
                st.stop()

            # ── Agent 1 – Pathologist ─────────────────────────────────────────
            with st.spinner("🔬 Agent 1 – Pathologist analysing crop image…"):
                p_metrics = AgentMetrics(agent_name="pathologist_agent")
                p_metrics.start_time = time.perf_counter()

                image = Image.open(uploaded_file)
                diagnosis = run_pathologist_agent(image, api_key, project_id, watsonx_url)

                p_metrics.end_time = time.perf_counter()
                p_metrics.input_tokens = diagnosis.get("input_tokens", 0)
                p_metrics.output_tokens = diagnosis.get("output_tokens", 0)
                p_metrics.confidence = diagnosis.get("confidence", "Unknown")
                p_metrics.success = "error" not in diagnosis
                p_metrics.task_completed = p_metrics.success

            if "error" in diagnosis:
                st.error(f"❌ Pathologist Agent failed: {diagnosis['error']}")
            else:
                st.success("✅ Agent 1 – Pathologist diagnosis complete")

                # Display diagnosis card
                with st.expander("🔬 Pathologist Agent – Diagnosis Report", expanded=True):
                    st.markdown(f"### 🌿 Suspected Disease")
                    st.markdown(f"**{diagnosis.get('suspected_disease', 'Unknown')}**")

                    conf = diagnosis.get("confidence", "Unknown")
                    conf_color = {"High": "🟢", "Medium": "🟡", "Low": "🔴"}.get(conf, "⚪")
                    st.markdown(f"**Confidence:** {conf_color} {conf}")

                    symptoms = diagnosis.get("symptoms", [])
                    if symptoms:
                        st.markdown("**Observed Symptoms:**")
                        for s in symptoms:
                            if s.strip():
                                st.markdown(f"  • {s}")

                    alts = diagnosis.get("alternatives", [])
                    if alts:
                        st.markdown(f"**Alternative Diagnoses:** {', '.join(alts)}")

                    notes = diagnosis.get("notes", "")
                    if notes:
                        st.info(f"ℹ️ {notes}")

                    st.caption(
                        "⚠️ **This is an AI-generated preliminary assessment, NOT a confirmed "
                        "laboratory diagnosis.** Consult a certified agronomist for confirmation."
                    )

                # ── Output guardrails on pathologist output ───────────────────
                guardrail_path_output = run_all_guardrails(
                    output_text=diagnosis.get("diagnosis_text", ""),
                    agent_name="extension_officer_agent",
                    tool_name="watsonx_text",
                    completed_agents=["pathologist_agent"],
                    handoff_data=diagnosis,
                    execution_seconds=p_metrics.latency_seconds,
                    input_tokens=p_metrics.input_tokens,
                    output_tokens=p_metrics.output_tokens,
                )
                g_summary_path = summarise_guardrails(guardrail_path_output)
                p_metrics.guardrail_interventions = g_summary_path["total"]
                p_metrics.guardrail_blocks = g_summary_path["failed"]

                # ── Agent 2 – Extension Officer ───────────────────────────────
                with st.spinner("📋 Agent 2 – Extension Officer generating advisory…"):
                    e_metrics = AgentMetrics(agent_name="extension_officer_agent")
                    e_metrics.start_time = time.perf_counter()

                    advisory = run_extension_officer_agent(
                        diagnosis, api_key, project_id, watsonx_url
                    )

                    e_metrics.end_time = time.perf_counter()
                    e_metrics.input_tokens = advisory.get("input_tokens", 0)
                    e_metrics.output_tokens = advisory.get("output_tokens", 0)
                    e_metrics.success = "error" not in advisory
                    e_metrics.task_completed = e_metrics.success

                if "error" in advisory:
                    st.error(f"❌ Extension Officer Agent failed: {advisory['error']}")
                else:
                    st.success("✅ Agent 2 – Extension Officer advisory complete")

                    # Output guardrails on advisory
                    guardrail_ext_output = run_all_guardrails(
                        output_text=advisory.get("advisory_text", ""),
                        execution_seconds=e_metrics.latency_seconds,
                        input_tokens=e_metrics.input_tokens,
                        output_tokens=e_metrics.output_tokens,
                    )
                    g_summary_ext = summarise_guardrails(guardrail_ext_output)
                    e_metrics.guardrail_interventions = g_summary_ext["total"]
                    e_metrics.guardrail_blocks = g_summary_ext["failed"]

                    # Combine guardrail results for display
                    combined_guardrail_results = {
                        **guardrail_input_results,
                        **guardrail_path_output,
                        **guardrail_ext_output,
                    }
                    combined_summary = summarise_guardrails(combined_guardrail_results)

                    with st.expander("📋 Extension Officer Agent – Remedial Advisory", expanded=True):
                        advisory_text = advisory.get("advisory_text", "")
                        st.markdown(advisory_text)
                        st.caption(
                            "⚠️ Chemical treatments should follow local regulations and product labels. "
                            "Always wear appropriate PPE when handling pesticides."
                        )

                    # ── Combined final report ─────────────────────────────────
                    st.markdown("---")
                    st.subheader("📄 Combined Final Report")

                    final_report = (
                        f"AGRI DOCTOR – CROP DISEASE ADVISORY REPORT\n"
                        f"{'=' * 60}\n\n"
                        f"SECTION 1: PATHOLOGIST DIAGNOSIS\n"
                        f"{'-' * 40}\n"
                        f"{diagnosis.get('diagnosis_text', '')}\n\n"
                        f"SECTION 2: REMEDIAL ACTION ADVISORY\n"
                        f"{'-' * 40}\n"
                        f"{advisory_text}\n\n"
                        f"{'=' * 60}\n"
                        f"Generated by Agri Doctor (IBM watsonx.ai)\n"
                        f"DISCLAIMER: This report is an AI-generated preliminary assessment.\n"
                        f"It is NOT a certified agronomic opinion. Consult a qualified\n"
                        f"agricultural professional before acting on this advice.\n"
                    )

                    st.download_button(
                        label="⬇️ Download Advisory Report (.txt)",
                        data=final_report,
                        file_name="agri_doctor_advisory.txt",
                        mime="text/plain",
                        use_container_width=True,
                    )

                    # Store in session for other tabs
                    st.session_state["p_metrics"] = p_metrics
                    st.session_state["e_metrics"] = e_metrics
                    st.session_state["guardrail_results"] = combined_guardrail_results
                    st.session_state["guardrail_summary"] = combined_summary
                    st.session_state["diagnosis"] = diagnosis
                    st.session_state["advisory"] = advisory
                    st.session_state["dashboard"] = build_dashboard(
                        pathologist=p_metrics,
                        extension_officer=e_metrics,
                        guardrail_summary=combined_summary,
                        pathologist_output=diagnosis.get("diagnosis_text", ""),
                        extension_output=advisory_text,
                        pathologist_context=diagnosis.get("diagnosis_text", ""),
                    )

        elif not uploaded_file:
            st.info("👆 Upload a crop image on the left to begin diagnosis.")

# ─────────────────────────────────────────────────────────────────────────────
# Tab 2 – Guardrails Report
# ─────────────────────────────────────────────────────────────────────────────

with tab_guardrails:
    st.subheader("🛡️ Guardrails Framework Report")

    if "guardrail_results" not in st.session_state:
        st.info("Run a diagnosis first to see guardrail results.")
    else:
        summary = st.session_state["guardrail_summary"]
        results = st.session_state["guardrail_results"]

        # Summary metrics
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Total Checks", summary["total"])
        c2.metric("Passed ✅", summary["passed"])
        c3.metric("Failed ⚠️", summary["failed"])
        c4.metric("Blocked 🚫", "Yes" if summary["blocked"] else "No")

        st.markdown("---")

        # Category-wise breakdown
        CATEGORY_ICONS = {
            "Input": "1️⃣ Input",
            "Output": "2️⃣ Output",
            "ToolExecution": "3️⃣ Tool Execution",
            "Data": "4️⃣ Data",
            "Security": "5️⃣ Security",
            "Operational": "6️⃣ Operational",
            "HumanInTheLoop": "7️⃣ Human-in-the-Loop",
            "MultiAgent": "8️⃣ Multi-Agent",
            "Ethical": "9️⃣ Ethical & Compliance",
        }

        for cat, checks in results.items():
            label = CATEGORY_ICONS.get(cat, cat)
            all_pass = all(c.passed for c in checks)
            icon = "✅" if all_pass else "⚠️"
            with st.expander(f"{icon} {label} Guardrails ({len(checks)} checks)", expanded=not all_pass):
                for c in checks:
                    status = "✅ PASS" if c.passed else "❌ FAIL"
                    st.markdown(f"**{status}** · `{c.check}`")
                    if c.detail:
                        st.caption(f"   ↳ {c.detail}")

# ─────────────────────────────────────────────────────────────────────────────
# Tab 3 – Performance Dashboard
# ─────────────────────────────────────────────────────────────────────────────

with tab_dashboard:
    st.subheader("📊 LLM & Agent Performance Dashboard")

    if "dashboard" not in st.session_state:
        st.info("Run a diagnosis first to see performance metrics.")
    else:
        dash = st.session_state["dashboard"]

        # ── Performance & Cost ────────────────────────────────────────────────
        st.markdown("#### ⚡ Performance & Cost Metrics")
        pc = dash["performance_cost"]
        cols = st.columns(4)
        cols[0].metric("Total Latency", f"{pc['total_response_latency_s']}s")
        cols[1].metric("Total Tokens", pc["total_tokens"])
        cols[2].metric("Cost / Request", f"${pc['cost_per_request_usd']}")
        cols[3].metric("Error Rate", f"{pc['error_rate'] * 100:.0f}%")

        col_a, col_b = st.columns(2)
        with col_a:
            st.markdown("**Agent Latency Breakdown**")
            import pandas as pd
            latency_df = pd.DataFrame({
                "Agent": ["Pathologist", "Extension Officer"],
                "Latency (s)": [pc["pathologist_latency_s"], pc["extension_officer_latency_s"]],
            })
            st.bar_chart(latency_df.set_index("Agent"))

        with col_b:
            st.markdown("**Token Usage**")
            p_m = st.session_state["p_metrics"]
            e_m = st.session_state["e_metrics"]
            token_df = pd.DataFrame({
                "Agent": ["Pathologist", "Extension Officer"],
                "Input Tokens": [p_m.input_tokens, e_m.input_tokens],
                "Output Tokens": [p_m.output_tokens, e_m.output_tokens],
            })
            st.bar_chart(token_df.set_index("Agent"))

        st.markdown("---")

        # ── Agentic Metrics ───────────────────────────────────────────────────
        st.markdown("#### 🤖 Agentic AI Evaluation Metrics")
        ag = dash["agentic_evaluation"]
        a1, a2, a3, a4 = st.columns(4)
        a1.metric("Task Completion Rate", f"{ag['task_completion_rate'] * 100:.0f}%")
        a2.metric("Goal Achieved", "✅ Yes" if ag["goal_achievement"] else "❌ No")
        a3.metric("Tool Exec Success", f"{ag['tool_execution_success_rate'] * 100:.0f}%")
        a4.metric("Handoff Accurate", "✅ Yes" if ag["handoff_accuracy"] else "❌ No")

        with st.expander("Full Agentic Metrics", expanded=False):
            for k, v in ag.items():
                st.markdown(f"**{k.replace('_', ' ').title()}:** `{v}`")

        st.markdown("---")

        # ── RAG Evaluation ────────────────────────────────────────────────────
        st.markdown("#### 📚 RAG Evaluation Metrics")
        rag = dash["rag_evaluation"]
        r1, r2, r3 = st.columns(3)
        r1.metric("Faithfulness", rag.get("faithfulness", "N/A"))
        r2.metric("Answer Relevancy", rag.get("answer_relevancy", "N/A"))
        r3.metric("Groundedness", rag.get("groundedness", "N/A"))

        st.markdown("---")

        # ── Accuracy & Quality ────────────────────────────────────────────────
        st.markdown("#### 🎯 Accuracy & Quality Metrics")
        aq = dash["accuracy_quality"]
        q1, q2, q3 = st.columns(3)
        q1.metric("Semantic Similarity", aq.get("semantic_similarity", "N/A"))
        q2.metric("ROUGE-L", aq.get("rouge_l", "N/A"))
        q3.metric("BLEU-1", aq.get("bleu_1", "N/A"))

        st.markdown("---")

        # ── Safety & Guardrail Metrics ────────────────────────────────────────
        st.markdown("#### 🔒 Safety & Guardrail Evaluation Metrics")
        sg = dash["safety_guardrail"]
        s1, s2, s3, s4 = st.columns(4)
        s1.metric("Hallucination Rate", f"{sg['hallucination_rate'] * 100:.0f}%")
        s2.metric("Toxicity", sg["toxicity_score"])
        s3.metric("Bias Score", sg["bias_score"])
        s4.metric("Prompt Injection", sg["prompt_injection_resistance"])

        st.markdown("---")

        # ── Multi-Agent System ────────────────────────────────────────────────
        st.markdown("#### 🔗 Multi-Agent System Evaluation")
        ma = dash["multi_agent_system"]
        m1, m2, m3 = st.columns(3)
        m1.metric("Total Exec Time", f"{ma['total_execution_time_s']}s")
        m2.metric("Total Cost", f"${ma['total_cost_usd']}")
        m3.metric("Workflow Complete", "✅ Yes" if ma["end_to_end_workflow_complete"] else "❌ No")

        with st.expander("Full Multi-Agent Metrics", expanded=False):
            for k, v in ma.items():
                st.markdown(f"**{k.replace('_', ' ').title()}:** `{v}`")

        st.markdown("---")
        st.caption(
            "📊 Dashboard powered by Agri Doctor Metrics Engine · "
            "IBM watsonx.ai Granite 4-H-Small"
        )
