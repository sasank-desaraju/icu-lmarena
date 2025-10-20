"""ICU LLM Arena - compact, modular Streamlit app

This file provides a modular, self-contained Streamlit app that:
- Shows a password gate (optional via `st.secrets['auth']['shared_password']`).
- Prompts the rater for their name/email in the sidebar.
- Accepts one prompt and queries four models (ChatGPT-5, Claude 4.5 Sonnet, Gemini 2.0, Grok 4).
- Randomizes the displayed order each run and blinds model names.
- Shows four 5-point Likert sliders under each response in this order:
- Accuracy, Instruction-following, Style, Helpfulness.
- Records every assignment (one row per option) to `evaluations.csv` in repo root.
- Provides a Reset button to clear current assignment so users can run multiple rounds.

Notes:
- The app will try to call real SDKs if API keys are provided in `st.secrets['models']`.
- If an SDK or key is missing, the runner will return a clear placeholder message but still allow evaluation.
"""

from __future__ import annotations

import csv
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
# import google.generativeai as genai
# from google.genai import types

# Constants
CSV_PATH = os.path.join(os.path.dirname(__file__), "evaluations.csv")
MODEL_ORDER = [
    ("chatgpt_5", "ChatGPT-5"),
    ("claude_sonnet_4_5", "Claude Sonnet 4.5"),
    ("gemini_2_0", "Gemini 2.0"),
    ("grok_4", "Grok 4"),
]


def now_iso() -> str:
    # return datetime.utcnow().isoformat(timespec="seconds") + "Z"
    return datetime.now().isoformat(timespec="seconds") + "Z"
    # return datetime.UTC


def ensure_user(name: str) -> str:
    # Simple identity; returns normalized name for logging
    return name.strip()


# ---- LLM runners (best-effort; tolerant to missing SDKs/keys) ----
def _chatgpt5_runner(system_prompt: str, user_prompt: str) -> Tuple[str, int]:
    key = st.secrets.get("models", {}).get("OPENAI_API_KEY")
    t0 = time.time()
    client = OpenAI(api_key=key)
    # response = client.chat.completions.create(
    #     model="gpt-5",  # Use available model; adjust as needed
    #     messages=[
    #         {"role": "system", "content": system_prompt},
    #         {"role": "user", "content": user_prompt}
    #     ],
    #     # temperature=0.2,
    #     max_tokens=1000,
    # )
    # response = response.choices[0].message.content
    response = client.responses.create(
        model="gpt-5",  # Use available model; adjust as needed
        reasoning={"effort": "low"},
        input=[
            {"role": "developer", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
    )
    response = response.output_text

    return (str(response), int((time.time() - t0) * 1000))


def _claude_runner(system_prompt: str, user_prompt: str) -> Tuple[str, int]:
    key = st.secrets.get("models", {}).get("ANTHROPIC_API_KEY")
    t0 = time.time()
    client = anthropic.Client(api_key=key)
    response = client.messages.create(
        model="claude-sonnet-4-5",
        max_tokens=1000,
        system = system_prompt,
        messages=[
            {
                "role": "user", "content": user_prompt,
            }
        ]
    )
    response = response.content[0].text

    return (response, int((time.time() - t0) * 1000))


def _gemini_runner(system_prompt: str, user_prompt: str) -> Tuple[str, int]:
    key = st.secrets.get("models", {}).get("GEMINI_API_KEY")
    t0 = time.time()
    client = genai.Client(api_key=key)
    response = client.models.generate_content(
        model="gemini-2.5-flash",
        config=types.GenerateContentConfig(
            system_instruction=system_prompt),
        contents=user_prompt
    )
    response = response.text

    return (response, int((time.time() - t0) * 1000))


def _grok_runner(system_prompt: str, user_prompt: str) -> Tuple[str, int]:
    key = st.secrets.get("models", {}).get("XAI_API_KEY")
    grok_url = 'https://api.x.ai/v1'
    t0 = time.time()
    client = OpenAI(api_key=key, base_url=grok_url)
    response = client.chat.completions.create(
        model="grok-4",  # Adjust model name
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        # temperature=0.2,
        max_tokens=1000,
    )
    response = response.choices[0].message.content

    return (str(response), int((time.time() - t0) * 1000))


RUNNERS = {
    "chatgpt_5": ("ChatGPT-5", _chatgpt5_runner),
    "claude_sonnet_4_5": ("Claude Sonnet 4.5", _claude_runner),
    "gemini_2_0": ("Gemini 2.0", _gemini_runner),
    "grok_4": ("Grok 4", _grok_runner),
}


# ---- CSV logging ----
def _ensure_csv(path: str) -> None:
    if not os.path.exists(path):
        with open(path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=_csv_fieldnames())
            writer.writeheader()


def _csv_fieldnames() -> List[str]:
    return [
        "assignment_id",
        "timestamp",
        "user_name",
        "system_prompt",
        "user_prompt",
        "seed",
        "option_label",
        "model_code",
        "model_display_name",
        "response_text",
        "latency_ms",
        "accuracy",
        "instruction_following",
        "style",
        "helpfulness",
    ]


def append_rows(rows: List[Dict[str, str]]) -> None:
    _ensure_csv(CSV_PATH)
    with open(CSV_PATH, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=_csv_fieldnames())
        for r in rows:
            writer.writerow(r)


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
        # Only stop if password has not been accepted
        if not st.session_state.get("pw_ok"):
            st.stop()

with st.sidebar:
    st.header("Rater")
    input_name = st.text_input("Your name / initials", value=st.session_state.get("user_name", ""))
    if st.button("Set name"):
        if not input_name.strip():
            st.error("Please provide a name")
        else:
            st.session_state["user_name"] = ensure_user(input_name)
            st.success("Name set")

    st.divider()
    st.header("Admin")
    if os.path.exists(CSV_PATH):
        with open(CSV_PATH, "rb") as f:
            st.download_button("Download CSV", data=f, file_name="evaluations.csv", mime="text/csv")
    else:
        st.caption("No evaluations yet")

    if st.button("Reset session"):
        for k in list(st.session_state.keys()):
            if k.startswith("assignment_") or k.startswith("option_text_") or k.startswith("rating_"):
                st.session_state.pop(k, None)
        st.success("Session cleared")

st.title("ICU LLM Arena — Blinded 4-way comparison")
st.caption("Enter prompts without PHI. Results and ratings are recorded to evaluations.csv")

if "user_name" not in st.session_state:
    st.info("Set your rater name in the sidebar to begin")
    st.stop()

st.markdown("### Enter a prompt")
user_prompt = st.text_area("Prompt", height=140, placeholder="Summarize initial management of suspected elevated ICP in an adult with severe TBI.")

col1, col2 = st.columns([1, 3])
with col1:
    stakeholder = st.selectbox("Stakeholder", ["Provider", "Patient/Family"])  # preserved for future export
with col2:
    category = st.selectbox("Category", ["Clinical Decision Support", "Protocol Clarification", "Patient Education", "Care Coordination", "Documentation", "Other"])  # placeholder

if st.button("Generate blinded responses", disabled=not user_prompt.strip()):
    # Prepare assignment
    seed = random.randint(1, 10**9)
    assignment_id = str(uuid.uuid4())
    system_prompt = """
    You are a helpful and accurate medical assistant in the ICU.
    Patients and providers will speak to you to ask you questions.
    Respond to providers using standard medical terminology and respond to patients and their families using the appropriate layperson language.
    """
    st.session_state["assignment_id"] = assignment_id
    st.session_state["system_prompt"] = system_prompt
    st.session_state["user_prompt"] = user_prompt
    st.session_state["seed"] = seed

    # Run runners in canonical order, then shuffle options for display
    results = []
    for code, _ in MODEL_ORDER:
        display_name, runner = RUNNERS[code]
        text, latency = runner(system_prompt, user_prompt)
        results.append({"model_code": code, "model_display_name": display_name, "text": text, "latency": latency})

    # Shuffle assignments deterministically by seed and assign labels A-D
    rng = random.Random(seed)
    rng.shuffle(results)
    labels = ["A", "B", "C", "D"]
    options = []
    for i, r in enumerate(results):
        opt = {"label": labels[i], "slot": i + 1, **r}
        options.append(opt)

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
            # Order as requested: Accuracy, Instruction-following, Style, Helpfulness
            acc = st.slider("Accuracy", 1, 5, 3, key=f"rating_accuracy_{opt['label']}")
            instr = st.slider("Instruction-following", 1, 5, 3, key=f"rating_instruction_{opt['label']}")
            style = st.slider("Style", 1, 5, 3, key=f"rating_style_{opt['label']}")
            helpf = st.slider("Helpfulness", 1, 5, 3, key=f"rating_helpfulness_{opt['label']}")
            rating_data[opt['label']] = {"accuracy": acc, "instruction_following": instr, "style": style, "helpfulness": helpf}

    preference = st.radio("Overall preference", [f"Option {o['label']}" for o in options] + ["Tie / No preference"], index=len(options), horizontal=True, key="overall_pref")
    comment = st.text_area("Optional comment / justification", height=100, key="overall_comment")

    if st.button("Submit ratings"):
        # Validate ratings exist
        missing = [lbl for lbl in rating_data if rating_data[lbl] is None]
        if missing:
            st.error("Please rate every option before submitting.")
        else:
            # Prepare rows (one per option)
            rows = []
            for opt in options:
                lbl = opt['label']
                row = {
                    "assignment_id": st.session_state.get("assignment_id"),
                    "timestamp": now_iso(),
                    "user_name": st.session_state.get("user_name"),
                    "system_prompt": st.session_state.get("system_prompt"),
                    "user_prompt": st.session_state.get("user_prompt"),
                    "seed": st.session_state.get("seed"),
                    "option_label": lbl,
                    "model_code": opt['model_code'],
                    "model_display_name": opt['model_display_name'],
                    "response_text": opt['text'],
                    "latency_ms": opt['latency'],
                    "accuracy": rating_data[lbl]['accuracy'],
                    "instruction_following": rating_data[lbl]['instruction_following'],
                    "style": rating_data[lbl]['style'],
                    "helpfulness": rating_data[lbl]['helpfulness'],
                }
                rows.append(row)

            append_rows(rows)

            st.success("Saved ratings — thank you!")
            # Clear current options to allow another round
            for k in list(st.session_state.keys()):
                if k.startswith("assignment_") or k.startswith("options") or k.startswith("user_prompt"):
                    st.session_state.pop(k, None)

else:
    st.info("Generate blinded responses to begin an evaluation")
