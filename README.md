# 🌾 Agri Doctor – AI Crop Disease Advisor

> An Agentic AI application built for the **Inter-Institute IBM Bob Hackathon**  
> powered by **IBM watsonx.ai** (Granite 4-H-Small + Mistral Vision)

---

## 📋 Problem Statement

Farmers often spot symptoms on their crops (e.g., yellow leaves, black spots) but struggle to distinguish between fungal infections, nutrient deficiencies, or pests. Misdiagnosing leads to spraying the wrong chemicals, wasting money and harming the soil.

**Agri Doctor** solves this with a two-agent AI pipeline that:
1. **Diagnoses** the crop disease from an uploaded image (Pathologist Agent)
2. **Generates** a safe, actionable, farmer-friendly advisory (Extension Officer Agent)

---

## 🏗️ Architecture

```
User Upload (image)
       │
       ▼
┌──────────────────────────────────────────────────────────┐
│  Input Guardrails (prompt injection, PII, image validation) │
└──────────────────────────────────────────────────────────┘
       │
       ▼
┌──────────────────────────────────────────────────────────┐
│  Agent 1: Pathologist Agent                              │
│  Model: mistralai/mistral-small-3-1-24b-instruct-2503    │
│  → Disease detection, symptoms, confidence level         │
└──────────────────────────────────────────────────────────┘
       │  (Structured diagnosis JSON)
       ▼
┌──────────────────────────────────────────────────────────┐
│  Agent 2: Extension Officer Agent                        │
│  Model: ibm/granite-4-h-small                            │
│  → Remedial actions, prevention, safe treatments         │
└──────────────────────────────────────────────────────────┘
       │
       ▼
┌──────────────────────────────────────────────────────────┐
│  Output Guardrails + Performance Dashboard               │
└──────────────────────────────────────────────────────────┘
       │
       ▼
  Streamlit UI (Diagnosis + Advisory + Dashboard + Download)
```

---

## 🚀 Features

### 🤖 Multi-Agent Workflow
- **Pathologist Agent** – Uses IBM watsonx.ai vision model to identify crop diseases from images
- **Extension Officer Agent** – Uses IBM Granite 4-H-Small to generate farmer-friendly advisories
- Genuine agent-to-agent handoff with structured data passing

### 🛡️ 9-Category Guardrails Framework
| # | Category | Description |
|---|----------|-------------|
| 1 | Input Guardrails | Prompt injection, PII detection, image validation |
| 2 | Output Guardrails | Hallucination detection, toxicity filtering |
| 3 | Tool Execution | Agent access control, tool authorization |
| 4 | Data Guardrails | PII masking, credential leak prevention |
| 5 | Security | SQL injection, jailbreak detection |
| 6 | Operational | Rate limiting, token limits, cost monitoring |
| 7 | Human-in-the-Loop | High-risk chemical flagging for review |
| 8 | Multi-Agent | Sequence validation, handoff integrity, loop detection |
| 9 | Ethical & Compliance | Bias detection, disclaimer enforcement, audit trail |

### 📊 LLM & Agent Performance Dashboard
- Accuracy & Quality Metrics (ROUGE-L, BLEU-1, Semantic Similarity)
- RAG Evaluation (Faithfulness, Groundedness, Answer Relevancy)
- Agentic AI Metrics (Task Completion Rate, Handoff Accuracy)
- Safety & Guardrail Metrics (Hallucination Rate, Toxicity Score)
- Performance & Cost Metrics (Latency, Tokens/sec, Cost/Request)
- Multi-Agent System Evaluation (Total Execution, Workflow Completion)

---

## 🛠️ Technology Stack

| Component | Technology |
|-----------|-----------|
| Language | Python 3.10+ |
| UI Framework | Streamlit |
| LLM Platform | IBM watsonx.ai |
| Text Model | `ibm/granite-4-h-small` |
| Vision Model | `mistralai/mistral-small-3-1-24b-instruct-2503` |
| Image Processing | Pillow |
| Data Visualisation | Pandas + Streamlit charts |
| Config | python-dotenv |

---

## 📁 Project Structure

```
AgriDoctor/
├── app.py                          # Main Streamlit application
├── requirements.txt                # Python dependencies
├── env_config.txt                  # Environment variables (rename to .env)
│
├── agents/
│   ├── __init__.py
│   ├── pathologist_agent.py        # Agent 1: Crop disease diagnosis
│   └── extension_officer_agent.py  # Agent 2: Remedial advisory
│
├── guardrails/
│   ├── __init__.py
│   └── guardrails.py               # All 9 guardrail categories
│
└── dashboard/
    ├── __init__.py
    └── metrics.py                  # LLM & Agent performance metrics
```

---

## ⚙️ Setup & Installation

### Prerequisites
- Python 3.10 or higher
- IBM Cloud account with Watson Studio service
- IBM watsonx.ai Project ID and API Key

### 1. Clone the repository
```bash
git clone <your-repo-url>
cd AgriDoctor
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Configure credentials
Rename `env_config.txt` to `.env` and fill in your credentials:
```env
WATSONX_API_KEY=your_ibm_api_key
WATSONX_PROJECT_ID=your_project_id
WATSONX_URL=https://us-south.ml.cloud.ibm.com
WATSONX_MODEL_ID=ibm/granite-4-h-small
WATSONX_VISION_MODEL=mistralai/mistral-small-3-1-24b-instruct-2503
```

### 4. Run the application
```bash
streamlit run app.py
```

The application opens at `http://localhost:8501`

---

## 🌐 Deployment on Streamlit Cloud

1. Push your project to GitHub (without `.env` – use Streamlit Secrets instead)
2. Go to [streamlit.io](https://streamlit.io) → Sign In → Create App
3. Connect your GitHub repository, branch: `main`, file: `app.py`
4. In **Advanced Settings → Secrets**, paste your credentials:
   ```toml
   WATSONX_API_KEY = "your_key"
   WATSONX_PROJECT_ID = "your_project_id"
   WATSONX_URL = "https://us-south.ml.cloud.ibm.com"
   WATSONX_MODEL_ID = "ibm/granite-4-h-small"
   WATSONX_VISION_MODEL = "mistralai/mistral-small-3-1-24b-instruct-2503"
   ```
5. Deploy!

---

## 🔒 Security Notes

- API keys are entered via password-type fields and never hardcoded
- All credentials loaded from environment variables or Streamlit Secrets
- PII masking applied before any logging
- Input sanitisation on every user upload

---

## ⚠️ Disclaimer

> Agri Doctor provides AI-generated preliminary assessments only. These are **NOT** confirmed laboratory diagnoses or certified agronomic opinions. Always consult a qualified agricultural extension officer or plant pathology expert before applying any treatment. Follow local agricultural regulations and product labels for all chemical applications.

---

## 🏆 Hackathon

**Event:** Inter-Institute IBM Bob Hackathon with IBM SkillsBuild  
**Category:** Agentic AI with IBM Advanced Features  
**Platform:** IBM watsonx.ai · IBM Granite Models

---

*Made with ❤️ using IBM Bob and IBM watsonx.ai*
