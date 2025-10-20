"""ICU LLM Arena - compact, modular Streamlit app

This file provides a modular, self-contained Streamlit app that:
- Shows a password gate (optional via `st.secrets['auth']['shared_password']`).
- Prompts the rater for their name/email in the sidebar.
- Accepts one prompt and queries four models (ChatGPT-5, Claude 4.5 Sonnet, Gemini 2.0, Grok 4).
- Randomizes the displayed order each run and blinds model names.
- Shows four 5-point Likert sliders under each response in this order:
  Accuracy, Instruction-following, Style, Helpfulness.
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

# Constants
CSV_PATH = os.path.join(os.path.dirname(__file__), "evaluations.csv")
MODEL_ORDER = [
    ("chatgpt_5", "ChatGPT-5"),
    ("claude_sonnet_4_5", "Claude Sonnet 4.5"),
    ("gemini_2_0", "Gemini 2.0"),
    ("grok_4", "Grok 4"),
]


def now_iso() -> str:
    return datetime.utcnow().isoformat(timespec="seconds") + "Z"


def ensure_user(name: str) -> str:
    # Simple identity; returns normalized name for logging
    return name.strip()


# ---- LLM runners (best-effort; tolerant to missing SDKs/keys) ----
def _chatgpt5_runner(prompt: str) -> Tuple[str, int]:
    key = st.secrets.get("models", {}).get("chatgpt5_api_key")
    t0 = time.time()
    if not key:
        return ("[ChatGPT-5 skipped — missing API key]", -1)
    try:
        # Try to call OpenAI package if present
        import openai

        openai.api_key = key
        resp = openai.ChatCompletion.create(
            model="gpt-5-chat",
            messages=[{"role": "system", "content": "You are an ICU assistant. Be concise and accurate."}, {"role": "user", "content": prompt}],
            temperature=0.2,
            max_tokens=700,
        )
        text = resp.choices[0].message.get("content") if resp.choices else resp.choices[0].text
    except Exception as exc:  # fallback message on error
        text = f"[ChatGPT-5 error: {exc}]"
    return (str(text), int((time.time() - t0) * 1000))


def _claude_runner(prompt: str) -> Tuple[str, int]:
    key = st.secrets.get("models", {}).get("claude_api_key")
    t0 = time.time()
    if not key:
        return ("[Claude skipped — missing API key]", -1)
    try:
        import anthropic

        client = anthropic.Client(api_key=key)
        resp = client.completions.create(
            model="claude-4.5-sonnet",
            prompt=f"{prompt}",
            max_tokens_to_sample=750,
            temperature=0.2,
        )
        text = getattr(resp, "completion", "") or str(resp)
    except Exception as exc:
        text = f"[Claude error: {exc}]"
    return (text, int((time.time() - t0) * 1000))


def _gemini_runner(prompt: str) -> Tuple[str, int]:
    key = st.secrets.get("models", {}).get("gemini_api_key")
    t0 = time.time()
    if not key:
        return ("[Gemini skipped — missing API key]", -1)
    try:
        import google.generativeai as genai

        genai.configure(api_key=key)
        resp = genai.generate(model="gemini-2.0-pro", input=["You are an ICU assistant. Be concise and accurate.", prompt], max_output_tokens=700, temperature=0.2)
        text = getattr(resp, "text", "") or str(resp)
    except Exception as exc:
        text = f"[Gemini error: {exc}]"
    return (text, int((time.time() - t0) * 1000))


def _grok_runner(prompt: str) -> Tuple[str, int]:
    key = st.secrets.get("models", {}).get("grok_api_key")
    t0 = time.time()
    if not key:
        return ("[Grok skipped — missing API key]", -1)
    try:
        import openai

        # Grok-compatible base URL may be provided; user can set in secrets as grok_base_url
        base = st.secrets.get("models", {}).get("grok_base_url")
        if base:
            client = openai.OpenAI(api_key=key, base_url=base)
            resp = client.chat.completions.create(model="grok-4", messages=[{"role": "system", "content": "You are an ICU assistant. Be concise and accurate."}, {"role": "user", "content": prompt}], temperature=0.2, max_tokens=700)
            text = resp.choices[0].message.get("content")
        else:
            # Try standard openai.ChatCompletion
            openai.api_key = key
            resp = openai.ChatCompletion.create(model="grok-4", messages=[{"role": "system", "content": "You are an ICU assistant. Be concise and accurate."}, {"role": "user", "content": prompt}], temperature=0.2, max_tokens=700)
            text = resp.choices[0].message.get("content")
    except Exception as exc:
        text = f"[Grok error: {exc}]"
    return (str(text), int((time.time() - t0) * 1000))


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
        "prompt",
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
                st.experimental_rerun()
            else:
                st.error("Wrong password")
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
prompt = st.text_area("Prompt", height=140, placeholder="Summarize initial management of suspected elevated ICP in an adult with severe TBI.")

col1, col2 = st.columns([1, 3])
with col1:
    stakeholder = st.selectbox("Stakeholder", ["Physician", "Nurse", "Patient/Family", "Pharmacist", "Therapist", "Other"])  # preserved for future export
with col2:
    category = st.selectbox("Category", ["Clinical Decision Support", "Protocol Clarification", "Patient Education", "Care Coordination", "Documentation", "Other"])  # placeholder

if st.button("Generate blinded responses", disabled=not prompt.strip()):
    # Prepare assignment
    seed = random.randint(1, 10**9)
    assignment_id = str(uuid.uuid4())
    st.session_state["assignment_id"] = assignment_id
    st.session_state["prompt"] = prompt
    st.session_state["seed"] = seed

    # Run runners in canonical order, then shuffle options for display
    results = []
    for code, _ in MODEL_ORDER:
        display_name, runner = RUNNERS[code]
        text, latency = runner(prompt)
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

    rating_data = {}
    for opt in options:
        st.markdown(f"**Option {opt['label']}** — (hidden model)")
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
                    "prompt": st.session_state.get("prompt"),
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
                if k.startswith("assignment_") or k.startswith("options") or k.startswith("prompt"):
                    st.session_state.pop(k, None)

else:
    st.info("Generate blinded responses to begin an evaluation")
