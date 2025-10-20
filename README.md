# ICU LLM Arena

A four-way blinded comparison tool for ICU-focused language models. Each prompt you enter is evaluated by ChatGPT-5, Gemini 2.0, Claude Sonnet 4.5, and Grok 4. Raters score every answer on Accuracy, Helpfulness, Instruction-following, and Style using Likert scales, then select an overall preference. All prompts, outputs, scores, and timing data are stored in SQLite for export.

## Features
- 🔒 Optional shared-password gate before accessing the app.
- 👥 Sidebar workflow for raters to identify themselves (email + display name) for linkage.
- 📝 Capture of stakeholder/category metadata alongside prompts.
- 🤖 Automatic execution of the four configured LLMs with randomized display order.
- 📊 Likert scales per model (Accuracy, Helpfulness, Instruction-following, Style) and overall preference selection.
- 🗃️ SQLite persistence with WAL mode for low-friction Streamlit deployment.
- 📤 CSV export that flattens per-model ratings along with rater metadata and latency metrics.

## Project layout
```
.
├── streamlit_app.py        # Streamlit UI + registry + persistence helpers
├── requirements.txt        # Python dependencies
├── .streamlit/
│   └── secrets.toml        # Local template (configure secrets in Streamlit Cloud)
└── README.md
```

## Installation
1. Create and activate a Python 3.10+ virtual environment.
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Secrets configuration
Populate `.streamlit/secrets.toml` locally (do **not** commit actual keys) and copy the same values into the Streamlit Community Cloud **Secrets** UI when deploying:

```toml
[auth]
shared_password = "change-me"  # optional simple gate; leave empty to disable

[models]
OPENAI_API_KEY = "sk-..."
ANTHROPIC_API_KEY = "sk-ant-..."
GEMINI_API_KEY = "AIza..."
XAI_API_KEY = "xai-..."

[app]
db_path = "icu_llm_arena.db"
```

If any API key is missing, the corresponding model is disabled and the sidebar displays a warning.

## Running locally
Launch the Streamlit app:
```bash
streamlit run streamlit_app.py
```
Set the shared password (if enabled), fill in your rater identity via the sidebar, then enter prompts to compare the four models. Ratings persist between reloads thanks to SQLite.

## Data export
Use the “Download ratings.csv” button in the sidebar to obtain a tidy CSV. Each row represents a single model response within an assignment and includes:
- Rater information (email, display name).
- Prompt metadata (stakeholder, category, prompt text, version).
- Model code/display name and randomized slot.
- Response text and latency (ms).
- Likert scores for Accuracy, Helpfulness, Instruction-following, and Style.
- Assignment-level preference and reviewer comment.

## Deployment tips
1. Push this repository to GitHub.
2. In Streamlit Community Cloud, select **New app** and point to the repo/branch.
3. Paste the secrets (auth/models/app) into the app’s **Settings → Secrets**.
4. Deploy. Streamlit will automatically install dependencies and initialize the SQLite database.

Happy evaluating—and remember: no PHI in prompts or responses.
