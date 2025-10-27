"""Classification Question Generator - Streamlit app

This app helps researchers generate classification questions and answers using guided LLM conversations.
It uses a 3-step process:
1. Generic prompt → LLM generates classification dimensions/options
2. Generate sample questions based on dimensions
3. Generate answers for each question
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
import math

import streamlit as st
from openai import OpenAI
import anthropic
from google import genai
from google.genai import types
from sqlalchemy import create_engine
from sqlalchemy.sql import text as sql_text

# Constants
MAX_TOKENS = 1000
DUMMY_RUNNERS = False  # Set to True for testing without API costs

DEFAULT_SYSTEM_PROMPT = """
You are a helpful medical research assistant specializing in classification tasks.
You help researchers create well-structured classification questions and provide accurate answers based on medical knowledge.
Be precise, evidence-based, and focus on clinically relevant classifications.
"""

DEFAULT_GENERIC_DIMENSION_PROMPT = """
Generate classification dimensions and options for the following medical classification task.
Provide a list of relevant dimensions and possible options for each dimension.
Be concise.

Make a classification question about the epidemiology of ARDS.
"""

DEFAULT_QUESTION_PROMPT = """
For each of the classification dimensions and options you just generated, please create a specific classification question.
Each question should be a concrete example that uses the dimensions you've defined.
Format each question with whitespace between them, and make sure they are medically accurate and clinically relevant.
"""

DEFAULT_ANSWER_PROMPT = """
For each classification question you just generated, please provide the correct answer based on current medical evidence and data.
Format your response with each question followed by its answer.
Be precise and cite general medical knowledge where relevant.
Do not repeat the questions; only provide the answers.
"""

# Available LLMs
LLM_OPTIONS = {
    "chatgpt_5": ("ChatGPT-5", "gpt-5"),
    "claude_sonnet_4_5": ("Claude Sonnet 4.5", "claude-sonnet-4-5"),
    "gemini_2_5": ("Gemini 2.5", "gemini-2.5-flash"),
    "grok_4": ("Grok 4", "grok-4"),
}

# Database setup
APP_DB_PATH = st.secrets.get("classification_app", {}).get("db_path", "classification.db")
DB_URL = f"sqlite:///{APP_DB_PATH}"
engine = create_engine(DB_URL, future=True, pool_pre_ping=True)

DDL = """
PRAGMA journal_mode=WAL;

CREATE TABLE IF NOT EXISTS raters(
  id TEXT PRIMARY KEY,
  name TEXT,
  created_at TEXT
);

CREATE TABLE IF NOT EXISTS conversations(
  id TEXT PRIMARY KEY,
  rater_id TEXT,
  llm_used TEXT,
  system_prompt TEXT,
  created_at TEXT
);

CREATE TABLE IF NOT EXISTS conversation_steps(
  id TEXT PRIMARY KEY,
  conversation_id TEXT,
  step_number INTEGER,
  user_prompt TEXT,
  assistant_response TEXT,
  latency_ms INTEGER,
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

def create_conversation(rater_id: str, llm_used: str, system_prompt: str) -> str:
    cid = str(uuid.uuid4())
    with engine.begin() as con:
        con.execute(sql_text("""
            INSERT INTO conversations(id, rater_id, llm_used, system_prompt, created_at)
            VALUES(:id, :rater, :llm, :system, :created)
        """), {
            "id": cid, "rater": rater_id, "llm": llm_used, "system": system_prompt, "created": now_iso()
        })
    return cid

def add_conversation_step(conversation_id: str, step_number: int, user_prompt: str, assistant_response: str, latency_ms: int) -> None:
    with engine.begin() as con:
        con.execute(sql_text("""
            INSERT INTO conversation_steps(id, conversation_id, step_number, user_prompt, assistant_response, latency_ms, created_at)
            VALUES(:id, :conv, :step, :user, :assistant, :latency, :created)
        """), {
            "id": str(uuid.uuid4()), "conv": conversation_id, "step": step_number,
            "user": user_prompt, "assistant": assistant_response, "latency": latency_ms, "created": now_iso()
        })

def get_conversation_steps(conversation_id: str) -> List[Dict]:
    with engine.begin() as con:
        rows = con.execute(sql_text("""
            SELECT step_number, user_prompt, assistant_response
            FROM conversation_steps
            WHERE conversation_id = :conv
            ORDER BY step_number
        """), {"conv": conversation_id}).mappings().all()
    return [dict(row) for row in rows]

def get_all_conversations_for_csv() -> str:
    with engine.begin() as con:
        rows = con.execute(sql_text("""
            SELECT c.id as conversation_id, r.name as rater_name, c.llm_used, c.system_prompt, cs.step_number, cs.user_prompt, cs.assistant_response, cs.latency_ms, cs.created_at
            FROM conversations c
            JOIN raters r ON c.rater_id = r.id
            JOIN conversation_steps cs ON c.id = cs.conversation_id
            ORDER BY c.created_at DESC, cs.step_number
        """)).mappings().all()
    output = io.StringIO()
    writer = csv.writer(output)
    writer.writerow(['Conversation ID', 'Rater Name', 'LLM Used', 'System Prompt', 'Step Number', 'User Prompt', 'Assistant Response', 'Latency (ms)', 'Created At'])
    for row in rows:
        writer.writerow([row['conversation_id'], row['rater_name'], row['llm_used'], row['system_prompt'], row['step_number'], row['user_prompt'], row['assistant_response'], row['latency_ms'], row['created_at']])
    return output.getvalue()

# LLM runners (reused from streamlit_app.py)
def _chatgpt5_runner(system_prompt: str, user_prompt: str, conversation_history: List[Dict] = None) -> Tuple[str, int]:
    key = st.secrets.get("models", {}).get("OPENAI_API_KEY")
    t0 = time.time()
    if not key:
        return ("[ChatGPT-5 skipped — missing API key]", -1)

    messages = [{"role": "system", "content": system_prompt}]
    if conversation_history:
        for step in conversation_history:
            messages.append({"role": "user", "content": step["user_prompt"]})
            messages.append({"role": "assistant", "content": step["assistant_response"]})
    messages.append({"role": "user", "content": user_prompt})

    try:
        client = OpenAI(api_key=key)
        response = client.responses.create(
            model="gpt-5",
            reasoning={"effort": "low"},
            max_output_tokens=MAX_TOKENS,
            input=messages
        )
        text = response.output_text
    except Exception as exc:
        text = f"[ChatGPT-5 error: {exc}]"
    return (str(text), int((time.time() - t0) * 1000))

def _claude_runner(system_prompt: str, user_prompt: str, conversation_history: List[Dict] = None) -> Tuple[str, int]:
    key = st.secrets.get("models", {}).get("ANTHROPIC_API_KEY")
    t0 = time.time()
    if not key:
        return ("[Claude skipped — missing API key]", -1)

    messages = []
    if conversation_history:
        for step in conversation_history:
            messages.append({"role": "user", "content": step["user_prompt"]})
            messages.append({"role": "assistant", "content": step["assistant_response"]})
    messages.append({"role": "user", "content": user_prompt})

    try:
        client = anthropic.Client(api_key=key)
        response = client.messages.create(
            model="claude-sonnet-4-5",
            max_tokens=MAX_TOKENS,
            system=system_prompt,
            messages=messages
        )
        text = response.content[0].text
    except Exception as exc:
        text = f"[Claude error: {exc}]"
    return (text, int((time.time() - t0) * 1000))

def _gemini_runner(system_prompt: str, user_prompt: str, conversation_history: List[Dict] = None) -> Tuple[str, int]:
    key = st.secrets.get("models", {}).get("GEMINI_API_KEY")
    t0 = time.time()
    if not key:
        return ("[Gemini skipped — missing API key]", -1)

    try:
        client = genai.Client(api_key=key)

        # Build conversation history for Gemini
        contents = []
        if conversation_history:
            # print(f"type of conversation_history: {type(conversation_history)}")
            # print(f"conversation history:\n", conversation_history)
            for step in conversation_history:
                contents.append(types.Content(role="user", parts=[types.Part(text=step['user_prompt'])]))
                contents.append(types.Content(role="model", parts=[types.Part(text=step['assistant_response'])]))

        contents.append(types.Content(role="user", parts=[types.Part(text=user_prompt)]))
        # print(f"contents are:\n", contents)

        response = client.models.generate_content(
            model="gemini-2.5-flash",
            config=types.GenerateContentConfig(
                system_instruction=system_prompt,
                max_output_tokens=MAX_TOKENS,
                thinking_config=types.ThinkingConfig(
                    thinking_budget=math.floor(MAX_TOKENS / 2),
                    include_thoughts=True)
            ),
            contents=contents
        )
        text = response.text
        # print(f"response: {response}")
    except Exception as exc:
        text = f"[Gemini error: {exc}]"
    return (text, int((time.time() - t0) * 1000))

def _grok_runner(system_prompt: str, user_prompt: str, conversation_history: List[Dict] = None) -> Tuple[str, int]:
    key = st.secrets.get("models", {}).get("XAI_API_KEY")
    t0 = time.time()
    if not key:
        return ("[Grok skipped — missing API key]", -1)

    messages = [{"role": "system", "content": system_prompt}]
    if conversation_history:
        for step in conversation_history:
            messages.append({"role": "user", "content": step["user_prompt"]})
            messages.append({"role": "assistant", "content": step["assistant_response"]})
    messages.append({"role": "user", "content": user_prompt})

    try:
        client = OpenAI(api_key=key, base_url="https://api.x.ai/v1")
        response = client.chat.completions.create(
            model="grok-4",
            messages=messages,
            max_tokens=MAX_TOKENS,
        )
        text = response.choices[0].message.content
    except Exception as exc:
        text = f"[Grok error: {exc}]"
    return (str(text), int((time.time() - t0) * 1000))

# Dummy runners for testing
def _dummy_chatgpt5_runner(system_prompt: str, user_prompt: str, conversation_history: List[Dict] = None) -> Tuple[str, int]:
    responses = [
        "[DUMMY] This is a dummy response from ChatGPT-5 for generating classification dimensions and options. For ARDS epidemiology, relevant dimensions include: Age groups (<18, 18-40, 40-60, >60), Sex (male, female), Comorbidities (none, diabetes, hypertension, obesity), Severity (mild, moderate, severe), and Setting (hospital, ICU, community).",
        "[DUMMY] This is a dummy response from ChatGPT-5 for creating sample classification questions. 1. 'Classify whether ARDS is most common in patients aged <18, 18-40, 40-60, or >60.' 2. 'Determine if ARDS occurs more frequently in males or females.' 3. 'Classify ARDS risk based on comorbidities: none, diabetes, hypertension, or obesity.'",
        "[DUMMY] This is a dummy response from ChatGPT-5 for generating answers. 1. ARDS is most common in patients aged 40-60. 2. ARDS occurs more frequently in males than females. 3. ARDS risk is highest in patients with obesity as a comorbidity."
    ]
    step = len(conversation_history) if conversation_history else 0
    return (responses[min(step, len(responses)-1)], -1)

def _dummy_claude_runner(system_prompt: str, user_prompt: str, conversation_history: List[Dict] = None) -> Tuple[str, int]:
    responses = [
        "[DUMMY] This is a dummy response from Claude for generating classification dimensions and options. For sepsis classification, dimensions could include: Source of infection (pulmonary, urinary, abdominal, bloodstream), Organ dysfunction (respiratory, cardiovascular, renal, hepatic), SOFA score (0-6, 7-12, 13-24), and Time to antibiotics (<1 hour, 1-3 hours, >3 hours).",
        "[DUMMY] This is a dummy response from Claude for creating sample classification questions. 1. 'Classify sepsis by primary source: pulmonary, urinary, abdominal, or bloodstream infection.' 2. 'Determine the number of organ dysfunctions in sepsis: single organ, two organs, or multi-organ failure.' 3. 'Classify sepsis severity by SOFA score: low (0-6), moderate (7-12), or high (13-24).'",
        "[DUMMY] This is a dummy response from Claude for generating answers. 1. The most common source of sepsis is pulmonary infection. 2. Most sepsis cases involve single organ dysfunction. 3. Moderate SOFA scores (7-12) are most common in sepsis cases."
    ]
    step = len(conversation_history) if conversation_history else 0
    return (responses[min(step, len(responses)-1)], -1)

def _dummy_gemini_runner(system_prompt: str, user_prompt: str, conversation_history: List[Dict] = None) -> Tuple[str, int]:
    responses = [
        "[DUMMY] This is a dummy response from Gemini for generating classification dimensions and options. For pneumonia classification, relevant dimensions include: Pathogen type (bacterial, viral, fungal), Location (community-acquired, hospital-acquired, ventilator-associated), CURB-65 score (0-1, 2-3, 4-5), and Patient age (<65, 65-80, >80).",
        "[DUMMY] This is a dummy response from Gemini for creating sample classification questions. 1. 'Classify pneumonia by pathogen: bacterial, viral, or fungal.' 2. 'Determine pneumonia acquisition: community-acquired, hospital-acquired, or ventilator-associated.' 3. 'Classify pneumonia severity using CURB-65 score: low risk (0-1), moderate risk (2-3), or high risk (4-5).'",
        "[DUMMY] This is a dummy response from Gemini for generating answers. 1. Bacterial pneumonia is the most common type. 2. Community-acquired pneumonia is most frequent. 3. Low risk CURB-65 scores (0-1) are most common in pneumonia cases."
    ]
    step = len(conversation_history) if conversation_history else 0
    return (responses[min(step, len(responses)-1)], -1)

def _dummy_grok_runner(system_prompt: str, user_prompt: str, conversation_history: List[Dict] = None) -> Tuple[str, int]:
    responses = [
        "[DUMMY] This is a dummy response from Grok for generating classification dimensions and options. For AKI classification, dimensions include: Cause (prerenal, intrinsic renal, postrenal), Severity (stage 1, stage 2, stage 3), Duration (transient, persistent, established), and Setting (hospital, ICU, outpatient).",
        "[DUMMY] This is a dummy response from Grok for creating sample classification questions. 1. 'Classify AKI by cause: prerenal, intrinsic renal, or postrenal.' 2. 'Determine AKI severity: stage 1, stage 2, or stage 3.' 3. 'Classify AKI duration: transient, persistent, or established.'",
        "[DUMMY] This is a dummy response from Grok for generating answers. 1. Prerenal AKI is the most common cause. 2. Stage 1 AKI is most frequent. 3. Transient AKI is the most common duration type."
    ]
    step = len(conversation_history) if conversation_history else 0
    return (responses[min(step, len(responses)-1)], -1)

# Runner selection
if DUMMY_RUNNERS:
    RUNNERS = {
        "chatgpt_5": _dummy_chatgpt5_runner,
        "claude_sonnet_4_5": _dummy_claude_runner,
        "gemini_2_5": _dummy_gemini_runner,
        "grok_4": _dummy_grok_runner,
    }
else:
    RUNNERS = {
        "chatgpt_5": _chatgpt5_runner,
        "claude_sonnet_4_5": _claude_runner,
        "gemini_2_5": _gemini_runner,
        "grok_4": _grok_runner,
    }



# Streamlit UI
st.set_page_config(page_title="Classification Question Generator", layout="wide")

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
    st.header("Researcher")
    input_name = st.text_input("Your name / initials", value=st.session_state.get("user_name", ""))
    if st.button("Set name"):
        if not input_name.strip():
            st.error("Please provide a name")
        else:
            st.session_state["user_name"] = input_name.strip()
            st.success("Name set")

    st.divider()
    st.header("LLM Selection")
    selected_llm = st.selectbox(
        "Choose LLM for this session",
        options=list(LLM_OPTIONS.keys()),
        format_func=lambda x: LLM_OPTIONS[x][0],
        help="Select which LLM to use for generating classification questions"
    )

    st.divider()
    st.header("Export Data")
    csv_data = get_all_conversations_for_csv()
    st.download_button(
        label="Download Conversations CSV",
        data=csv_data,
        file_name="classification_conversations.csv",
        mime="text/csv",
        key="download_csv"
    )

st.title("Classification Question Generator")
st.caption("Generate classification questions and answers using guided LLM conversations")

if "user_name" not in st.session_state:
    st.info("Set your researcher name in the sidebar to begin")
    st.stop()

# Initialize conversation state
if "conversation_id" not in st.session_state:
    st.session_state["conversation_id"] = None
if "completed_steps" not in st.session_state:
    st.session_state["completed_steps"] = set()
if "conversation_history" not in st.session_state:
    st.session_state["conversation_history"] = []

# Step 1: System Prompt
st.markdown("### System Prompt")
system_prompt = st.text_area(
    "System Prompt",
    value=DEFAULT_SYSTEM_PROMPT.strip(),
    height=150,
    disabled=1 in st.session_state.get("completed_steps", set()),
    key="system_prompt"
)

submit_disabled_1 = 1 in st.session_state.get("completed_steps", set())
if st.button("Submit System Prompt", disabled=submit_disabled_1):
    if not system_prompt.strip():
        st.error("Please enter a system prompt")
    else:
        # Create new conversation
        rater_id = ensure_rater(st.session_state["user_name"])
        conversation_id = create_conversation(rater_id, selected_llm, system_prompt)
        st.session_state["conversation_id"] = conversation_id
        st.session_state["completed_steps"].add(1)
        st.success("System prompt set!")
        st.rerun()

# Step 2: Generate Classification Dimensions and Options
st.markdown("### Generate Classification Dimensions and Options")
dimension_prompt = st.text_area(
    "Prompt for generating dimensions",
    value=DEFAULT_GENERIC_DIMENSION_PROMPT.strip(),
    height=200,
    disabled=2 in st.session_state.get("completed_steps", set()),
    key="dimension_prompt"
)

submit_disabled_2 = 2 in st.session_state.get("completed_steps", set()) or 1 not in st.session_state.get("completed_steps", set())
if st.button("Generate Dimensions", disabled=submit_disabled_2):
    runner = RUNNERS[selected_llm]
    response, latency = runner(system_prompt, dimension_prompt)
    add_conversation_step(st.session_state["conversation_id"], 2, dimension_prompt, response, latency)
    st.session_state["conversation_history"].append({
        "step": 2,
        "user_prompt": dimension_prompt,
        "assistant_response": response
    })
    st.session_state["completed_steps"].add(2)
    st.success("Dimensions generated!")
    st.rerun()

# Show response if completed
if 2 in st.session_state.get("completed_steps", set()):
    for step_data in st.session_state["conversation_history"]:
        if step_data["step"] == 2:
            st.markdown("**LLM Response:**")
            st.text_area("", value=step_data['assistant_response'], height=200, disabled=True, key="dimension_response")
            break

# Step 3: Generate Full Questions
st.markdown("### Generate Full Questions")
question_prompt = st.text_area(
    "Prompt for generating questions",
    value=DEFAULT_QUESTION_PROMPT.strip(),
    height=200,
    disabled=3 in st.session_state.get("completed_steps", set()),
    key="question_prompt"
)

submit_disabled_3 = 3 in st.session_state.get("completed_steps", set()) or 2 not in st.session_state.get("completed_steps", set())
if st.button("Generate Questions", disabled=submit_disabled_3):
    runner = RUNNERS[selected_llm]
    history = [step for step in st.session_state["conversation_history"] if step["step"] <= 2]
    response, latency = runner(system_prompt, question_prompt, history)
    add_conversation_step(st.session_state["conversation_id"], 3, question_prompt, response, latency)
    st.session_state["conversation_history"].append({
        "step": 3,
        "user_prompt": question_prompt,
        "assistant_response": response
    })
    st.session_state["completed_steps"].add(3)
    st.success("Questions generated!")
    st.rerun()

# Show response if completed
if 3 in st.session_state.get("completed_steps", set()):
    for step_data in st.session_state["conversation_history"]:
        if step_data["step"] == 3:
            st.markdown("**LLM Response:**")
            st.text_area("", value=step_data['assistant_response'], height=250, disabled=True, key="question_response")
            break

# Step 4: Generate Full Answers
st.markdown("### Generate Full Answers")
answer_prompt = st.text_area(
    "Prompt for generating answers",
    value=DEFAULT_ANSWER_PROMPT.strip(),
    height=200,
    disabled=4 in st.session_state.get("completed_steps", set()),
    key="answer_prompt"
)

submit_disabled_4 = 4 in st.session_state.get("completed_steps", set()) or 3 not in st.session_state.get("completed_steps", set())
if st.button("Generate Answers", disabled=submit_disabled_4):
    runner = RUNNERS[selected_llm]
    history = st.session_state["conversation_history"]
    response, latency = runner(system_prompt, answer_prompt, history)
    add_conversation_step(st.session_state["conversation_id"], 4, answer_prompt, response, latency)
    st.session_state["conversation_history"].append({
        "step": 4,
        "user_prompt": answer_prompt,
        "assistant_response": response
    })
    st.session_state["completed_steps"].add(4)
    st.success("Answers generated!")
    st.rerun()

# Show response if completed
if 4 in st.session_state.get("completed_steps", set()):
    for step_data in st.session_state["conversation_history"]:
        if step_data["step"] == 4:
            st.markdown("**LLM Response:**")
            st.text_area("", value=step_data['assistant_response'], height=300, disabled=True, key="answer_response")
            break

# Reset button
st.markdown("---")
if st.button("Reset Conversation"):
    for key in list(st.session_state.keys()):
        if key.startswith(("conversation_", "completed_steps", "conversation_history")):
            st.session_state.pop(key, None)
    st.success("Conversation reset")
    st.rerun()

# Initial state message
if not st.session_state.get("completed_steps"):
    st.info("Start by submitting a system prompt above")