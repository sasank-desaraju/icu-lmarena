"""ICU LLM Arena - Streamlit app with SQLite database

This file provides a Streamlit app similar to code_gen.py but uses an SQLite database for persistence,
suitable for Streamlit Cloud deployment. It includes:
- Optional password gate via st.secrets['auth']['shared_password'].
- Sidebar for rater name input.
- Prompt input with stakeholder/category dropdowns.
- Blinded, randomized display of 4 LLM responses.
- 4-column layout for responses and ratings (Accuracy, Instruction-following, Style, Helpfulness).
- Overall preference and comment.
- Database storage for evaluations, with CSV export.
- Reset button to clear session for multiple rounds.
"""

from __future__ import annotations

import csv
import io
import os
import random
import time
import uuid
from datetime import datetime
from typing import Dict, List, Tuple

import streamlit as st
from openai import OpenAI
import anthropic
from google import genai
from google.genai import types
from sqlalchemy import create_engine
from sqlalchemy.sql import text as sql_text


SYSTEM_PROMPT = """
You are a helpful and accurate medical assistant in the ICU.
Patients and providers will speak to you to ask you questions.
Respond to providers using standard medical terminology and respond to patients and their families using the appropriate layperson language.
No emojis or special formatting. Be professional.
Be concise. Limit your answer to 3-4 sentences unless you are explicitly asked for more detail.
"""

# Constants
MAX_TOKENS = 600            # Max tokens for LLM responses
MODEL_ORDER = [
    ("chatgpt_5", "ChatGPT-5"),
    ("claude_sonnet_4_5", "Claude Sonnet 4.5"),
    ("gemini_2_0", "Gemini 2.0"),
    ("grok_4", "Grok 4"),
]

# Database setup
APP_DB_PATH = st.secrets.get("app", {}).get("db_path", "icu_llm_arena.db")
DB_URL = f"sqlite:///{APP_DB_PATH}"
engine = create_engine(DB_URL, future=True, pool_pre_ping=True)

DDL = """
PRAGMA journal_mode=WAL;

CREATE TABLE IF NOT EXISTS raters(
  id TEXT PRIMARY KEY,
  name TEXT,
  created_at TEXT
);

CREATE TABLE IF NOT EXISTS tasks(
  id TEXT PRIMARY KEY,
  stakeholder TEXT,
  category TEXT,
  system_prompt TEXT,
  user_prompt TEXT,
  seed INTEGER,
  created_at TEXT
);

CREATE TABLE IF NOT EXISTS assignments(
  id TEXT PRIMARY KEY,
  rater_id TEXT,
  task_id TEXT,
  preferred_option TEXT,
  overall_comment TEXT,
  created_at TEXT
);

CREATE TABLE IF NOT EXISTS assignment_models(
  id TEXT PRIMARY KEY,
  assignment_id TEXT,
  model_code TEXT,
  model_display_name TEXT,
  option_label TEXT,
  response_text TEXT,
  latency_ms INTEGER,
  created_at TEXT
);

CREATE TABLE IF NOT EXISTS ratings(
  id TEXT PRIMARY KEY,
  assignment_model_id TEXT,
  accuracy INTEGER,
  instruction_following INTEGER,
  style INTEGER,
  helpfulness INTEGER,
  created_at TEXT
);
"""

with engine.begin() as con:
    for stmt in DDL.strip().split(";\n\n"):
        if stmt.strip():
            con.exec_driver_sql(stmt)

# Helpers
def now_iso() -> str:
    return datetime.now().isoformat(timespec="seconds") + "Z"

def ensure_rater(name: str) -> str:
    with engine.begin() as con:
        row = con.execute(sql_text("SELECT id FROM raters WHERE name=:name"), {"name": name}).fetchone()
        if row:
            return row[0]
        rid = str(uuid.uuid4())
        con.execute(sql_text("INSERT INTO raters(id, name, created_at) VALUES (:id, :name, :created)"),
                    {"id": rid, "name": name, "created": now_iso()})
        return rid

def insert_task(stakeholder: str, category: str, system_prompt: str, user_prompt: str, seed: int) -> str:
    tid = str(uuid.uuid4())
    with engine.begin() as con:
        con.execute(sql_text("""
            INSERT INTO tasks(id, stakeholder, category, system_prompt, user_prompt, seed, created_at)
            VALUES(:id, :stakeholder, :category, :system, :user, :seed, :created)
        """), {
            "id": tid, "stakeholder": stakeholder, "category": category,
            "system": system_prompt, "user": user_prompt, "seed": seed, "created": now_iso()
        })
    return tid

def create_assignment(rater_id: str, task_id: str) -> str:
    aid = str(uuid.uuid4())
    with engine.begin() as con:
        con.execute(sql_text("INSERT INTO assignments(id, rater_id, task_id, created_at) VALUES (:id, :rater, :task, :created)"),
                    {"id": aid, "rater": rater_id, "task": task_id, "created": now_iso()})
    return aid

def insert_assignment_model(assignment_id: str, model_code: str, model_display_name: str, option_label: str, response_text: str, latency_ms: int) -> str:
    amid = str(uuid.uuid4())
    with engine.begin() as con:
        con.execute(sql_text("""
            INSERT INTO assignment_models(id, assignment_id, model_code, model_display_name, option_label, response_text, latency_ms, created_at)
            VALUES(:id, :assignment, :code, :display, :label, :response, :latency, :created)
        """), {
            "id": amid, "assignment": assignment_id, "code": model_code, "display": model_display_name,
            "label": option_label, "response": response_text, "latency": latency_ms, "created": now_iso()
        })
    return amid

def insert_rating(assignment_model_id: str, ratings: Dict[str, int]) -> None:
    with engine.begin() as con:
        con.execute(sql_text("""
            INSERT INTO ratings(id, assignment_model_id, accuracy, instruction_following, style, helpfulness, created_at)
            VALUES(:id, :amid, :acc, :instr, :style, :help, :created)
        """), {
            "id": str(uuid.uuid4()), "amid": assignment_model_id,
            "acc": ratings["accuracy"], "instr": ratings["instruction_following"],
            "style": ratings["style"], "help": ratings["helpfulness"], "created": now_iso()
        })

def finalize_assignment(assignment_id: str, preferred_option: str, comment: str) -> None:
    with engine.begin() as con:
        con.execute(sql_text("""
            UPDATE assignments SET preferred_option=:pref, overall_comment=:comment WHERE id=:id
        """), {"pref": preferred_option, "comment": comment, "id": assignment_id})

def export_csv() -> bytes:
    query = """
    SELECT
      assignments.id AS assignment_id,
      assignments.created_at,
      assignments.preferred_option,
      assignments.overall_comment,
      raters.name AS rater_name,
      tasks.stakeholder,
      tasks.category,
      tasks.system_prompt,
      tasks.user_prompt,
      tasks.seed,
      assignment_models.option_label,
      assignment_models.model_code,
      assignment_models.model_display_name,
      assignment_models.response_text,
      assignment_models.latency_ms,
      ratings.accuracy,
      ratings.instruction_following,
      ratings.style,
      ratings.helpfulness
    FROM assignments
    JOIN raters ON assignments.rater_id = raters.id
    JOIN tasks ON assignments.task_id = tasks.id
    JOIN assignment_models ON assignment_models.assignment_id = assignments.id
    LEFT JOIN ratings ON ratings.assignment_model_id = assignment_models.id
    ORDER BY assignments.created_at DESC
    """
    with engine.begin() as con:
        rows = con.execute(sql_text(query)).mappings().all()
    if not rows:
        return b""
    buffer = io.StringIO()
    writer = csv.DictWriter(buffer, fieldnames=list(rows[0].keys()) if rows else [])
    if rows:
        writer.writeheader()
        for row in rows:
            writer.writerow(dict(row))
    return buffer.getvalue().encode("utf-8")

# ---- LLM runners ----
def _chatgpt5_runner(system_prompt: str, user_prompt: str) -> Tuple[str, int]:
    key = st.secrets.get("models", {}).get("OPENAI_API_KEY")
    t0 = time.time()
    if not key:
        return ("[ChatGPT-5 skipped — missing API key]", -1)
    # return (str('test'), int((1) * 1000))   # Temporary return for testing
    try:
        client = OpenAI(api_key=key)
        response = client.responses.create(
            model="gpt-5",
            reasoning={"effort": "low"},
            max_output_tokens=MAX_TOKENS,
            input=[
                {"role": "developer", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ]
        )
        text = response.output_text
    except Exception as exc:
        text = f"[ChatGPT-5 error: {exc}]"
    return (str(text), int((time.time() - t0) * 1000))

def _claude_runner(system_prompt: str, user_prompt: str) -> Tuple[str, int]:
    key = st.secrets.get("models", {}).get("ANTHROPIC_API_KEY")
    t0 = time.time()
    if not key:
        return ("[Claude skipped — missing API key]", -1)
    # return (str('test'), int((1) * 1000))   # Temporary return for testing
    try:
        client = anthropic.Client(api_key=key)
        response = client.messages.create(
            model="claude-sonnet-4-5",
            max_tokens=MAX_TOKENS,
            system=system_prompt,
            messages=[{"role": "user", "content": user_prompt}]
        )
        text = response.content[0].text
    except Exception as exc:
        text = f"[Claude error: {exc}]"
    return (text, int((time.time() - t0) * 1000))

def _gemini_runner(system_prompt: str, user_prompt: str) -> Tuple[str, int]:
    key = st.secrets.get("models", {}).get("GEMINI_API_KEY")
    t0 = time.time()
    if not key:
        return ("[Gemini skipped — missing API key]", -1)
    # return (str('test'), int((1) * 1000))   # Temporary return for testing
    try:
        client = genai.Client(api_key=key)
        response = client.models.generate_content(
            model="gemini-2.5-flash",
            config=types.GenerateContentConfig(
                system_instruction=system_prompt,
                max_output_tokens=MAX_TOKENS
                ),
            contents=user_prompt
        )
        text = response.text
    except Exception as exc:
        text = f"[Gemini error: {exc}]"
    return (text, int((time.time() - t0) * 1000))

def _grok_runner(system_prompt: str, user_prompt: str) -> Tuple[str, int]:
    key = st.secrets.get("models", {}).get("XAI_API_KEY")
    t0 = time.time()
    if not key:
        return ("[Grok skipped — missing API key]", -1)
    # return (str('test'), int((1) * 1000))   # Temporary return for testing
    try:
        client = OpenAI(api_key=key, base_url="https://api.x.ai/v1")
        response = client.chat.completions.create(
            model="grok-4",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            max_tokens=MAX_TOKENS,
        )
        text = response.choices[0].message.content
    except Exception as exc:
        text = f"[Grok error: {exc}]"
    return (str(text), int((time.time() - t0) * 1000))

RUNNERS = {
    "chatgpt_5": ("ChatGPT-5", _chatgpt5_runner),
    "claude_sonnet_4_5": ("Claude Sonnet 4.5", _claude_runner),
    "gemini_2_0": ("Gemini 2.0", _gemini_runner),
    "grok_4": ("Grok 4", _grok_runner),
}

# ---- Streamlit UI ----
st.set_page_config(page_title="ICU LLM Arena (4-way blinded)", layout="wide")

# Password gate (optional)
SHARED_PW = st.secrets.get("auth", {}).get("shared_password", "")
if SHARED_PW:
    if not st.session_state.get("pw_ok"):
        pw = st.text_input("Access password", type="password")
        if st.button("Enter"):
            if pw == SHARED_PW:
                st.session_state["pw_ok"] = True
            else:
                st.error("Wrong password")
        if not st.session_state.get("pw_ok"):
            st.stop()

with st.sidebar:
    st.header("Rater")
    input_name = st.text_input("Your name / initials", value=st.session_state.get("user_name", ""))
    if st.button("Set name"):
        if not input_name.strip():
            st.error("Please provide a name")
        else:
            st.session_state["user_name"] = input_name.strip()
            st.success("Name set")

    st.divider()
    st.header("Admin")
    csv_data = export_csv()
    st.download_button(
        "Download evaluations.csv",
        data=csv_data if csv_data else None,
        file_name="evaluations.csv",
        mime="text/csv",
        disabled=not bool(csv_data),
    )
    if not csv_data:
        st.caption("No evaluations yet")

    if st.button("Reset session"):
        for k in list(st.session_state.keys()):
            if k.startswith("assignment_") or k.startswith("options") or k.startswith("user_prompt") or k.startswith("rating_"):
                st.session_state.pop(k, None)
        st.success("Session cleared")

st.title("ICU LLM Arena — Blinded 4-way comparison")
st.caption("Enter prompts without PHI. Results are recorded to database.")

if "user_name" not in st.session_state:
    st.info("Set your rater name in the sidebar to begin")
    st.stop()

st.markdown("### Enter a prompt")
user_prompt = st.text_area("Prompt", height=140, placeholder="Summarize initial management of suspected elevated ICP in an adult with severe TBI.")

col1, col2 = st.columns([1, 3])
with col1:
    stakeholder = st.selectbox("Stakeholder", ["Provider", "Patient/Family"])
with col2:
    category = st.selectbox("Category", ["Clinical Decision Support", "Protocol Clarification", "Patient Education", "Care Coordination", "Documentation", "Other"])

if st.button("Generate blinded responses", disabled=not user_prompt.strip()):
    seed = random.randint(1, 10**9)
    system_prompt = SYSTEM_PROMPT
    task_id = insert_task(stakeholder, category, system_prompt, user_prompt, seed)
    rater_id = ensure_rater(st.session_state["user_name"])
    assignment_id = create_assignment(rater_id, task_id)

    results = []
    for code, _ in MODEL_ORDER:
        display_name, runner = RUNNERS[code]
        text, latency = runner(system_prompt, user_prompt)
        results.append({"model_code": code, "model_display_name": display_name, "text": text, "latency": latency})

    rng = random.Random(seed)
    rng.shuffle(results)
    labels = ["A", "B", "C", "D"]
    options = []
    for i, r in enumerate(results):
        amid = insert_assignment_model(assignment_id, r["model_code"], r["model_display_name"], labels[i], r["text"], r["latency"])
        opt = {"label": labels[i], "assignment_model_id": amid, **r}
        options.append(opt)

    st.session_state["assignment_id"] = assignment_id
    st.session_state["options"] = options

if "options" in st.session_state and st.session_state.get("options"):
    options = st.session_state["options"]
    st.markdown("### Blinded responses (order randomized)")
    cols = st.columns(4)
    for i, opt in enumerate(options):
        with cols[i]:
            st.subheader(f"Option {opt['label']}")
            st.text_area("", value=opt["text"], height=300, disabled=True, key=f"option_text_{opt['label']}")
            st.caption(f"Latency: {opt['latency']} ms" if opt["latency"] >= 0 else "Latency unavailable")

    st.markdown("---")
    st.markdown("#### Rate each option (1 = low, 5 = high)")

    rating_cols = st.columns(4)
    rating_data = {}
    for i, opt in enumerate(options):
        with rating_cols[i]:
            st.markdown(f"**Option {opt['label']}**")
            acc = st.slider("Accuracy", 1, 5, 3, key=f"rating_accuracy_{opt['label']}")
            instr = st.slider("Instruction-following", 1, 5, 3, key=f"rating_instruction_{opt['label']}")
            style = st.slider("Style", 1, 5, 3, key=f"rating_style_{opt['label']}")
            helpf = st.slider("Helpfulness", 1, 5, 3, key=f"rating_helpfulness_{opt['label']}")
            rating_data[opt['label']] = {"accuracy": acc, "instruction_following": instr, "style": style, "helpfulness": helpf}

    preference = st.radio("Overall preference", [f"Option {o['label']}" for o in options] + ["Tie / No preference"], index=len(options), horizontal=True, key="overall_pref")
    comment = st.text_area("Optional comment / justification", height=100, key="overall_comment")

    if st.button("Submit ratings"):
        if any(not rating_data[lbl] for lbl in rating_data):
            st.error("Please rate every option before submitting.")
        else:
            for opt in options:
                insert_rating(opt["assignment_model_id"], rating_data[opt['label']])

            preferred = preference if preference != "Tie / No preference" else ""
            finalize_assignment(st.session_state["assignment_id"], preferred, comment)

            st.success("Saved ratings — thank you!")
            for k in list(st.session_state.keys()):
                if k.startswith("assignment_") or k.startswith("options") or k.startswith("user_prompt"):
                    st.session_state.pop(k, None)

else:
    st.info("Generate blinded responses to begin an evaluation")




