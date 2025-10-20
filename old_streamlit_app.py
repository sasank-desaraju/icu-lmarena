import csv
import datetime as dt
import io
import random
import time
import uuid
from dataclasses import dataclass
from typing import Callable, Dict, List, Sequence, Tuple

import anthropic
import google.generativeai as genai
import streamlit as st
from openai import OpenAI
from sqlalchemy import create_engine, text

# -----------------------------
# Config / Secrets
# -----------------------------
APP_DB_PATH = st.secrets.get("app", {}).get("db_path", "icu_llm_arena.db")
DB_URL = f"sqlite:///{APP_DB_PATH}"
CHATGPT5_KEY = st.secrets.get("models", {}).get("chatgpt5_api_key", "")
GEMINI_KEY = st.secrets.get("models", {}).get("gemini_api_key", "")
CLAUDE_KEY = st.secrets.get("models", {}).get("claude_api_key", "")
GROK_KEY = st.secrets.get("models", {}).get("grok_api_key", "")
SHARED_PASSWORD = st.secrets.get("auth", {}).get("shared_password", "")

# -----------------------------
# Database bootstrap (SQLite)
# -----------------------------
engine = create_engine(DB_URL, future=True, pool_pre_ping=True)

DDL = """
PRAGMA journal_mode=WAL;

CREATE TABLE IF NOT EXISTS raters(
  id TEXT PRIMARY KEY,
  email TEXT,
  display_name TEXT,
  created_at TEXT
);

CREATE TABLE IF NOT EXISTS models(
  id TEXT PRIMARY KEY,
  code TEXT UNIQUE,
  display_name TEXT
);

CREATE TABLE IF NOT EXISTS tasks(
  id TEXT PRIMARY KEY,
  stakeholder TEXT,
  category TEXT,
  prompt_text TEXT,
  version INTEGER,
  created_at TEXT
);

CREATE TABLE IF NOT EXISTS assignments(
  id TEXT PRIMARY KEY,
  rater_id TEXT,
  task_id TEXT,
  seed INTEGER,
  preferred_slot INTEGER,
  preferred_model_code TEXT,
  overall_comment TEXT,
  created_at TEXT,
  completed_at TEXT
);

CREATE TABLE IF NOT EXISTS assignment_models(
  id TEXT PRIMARY KEY,
  assignment_id TEXT,
  model_code TEXT,
  slot INTEGER,
  response_text TEXT,
  latency_ms INTEGER,
  created_at TEXT
);

CREATE TABLE IF NOT EXISTS ratings(
  id TEXT PRIMARY KEY,
  assignment_model_id TEXT,
  accuracy INTEGER,
  helpfulness INTEGER,
  instruction_following INTEGER,
  style INTEGER,
  created_at TEXT
);
"""

with engine.begin() as con:
    for stmt in DDL.strip().split(";\n\n"):
        if stmt.strip():
            con.exec_driver_sql(stmt)

# -----------------------------
# Model Registry
# -----------------------------
@dataclass
class ModelSpec:
    code: str
    display_name: str
    runner: Callable[[str], Tuple[str, int]]
    enabled: bool = True


class ModelRegistry:
    def __init__(self) -> None:
        self._models: Dict[str, ModelSpec] = {}

    def register(self, code: str, display_name: str, enabled: bool = True) -> Callable[[Callable[[str], Tuple[str, int]]], Callable[[str], Tuple[str, int]]]:
        def decorator(fn: Callable[[str], Tuple[str, int]]) -> Callable[[str], Tuple[str, int]]:
            self._models[code] = ModelSpec(code=code, display_name=display_name, runner=fn, enabled=enabled)
            return fn

        return decorator

    def specs(self) -> Sequence[ModelSpec]:
        return tuple(self._models.values())

    def enabled(self) -> Sequence[ModelSpec]:
        return tuple(spec for spec in self._models.values() if spec.enabled)

    def by_code(self, code: str) -> ModelSpec:
        return self._models[code]

    def as_rows(self) -> List[Tuple[str, str]]:
        return [(spec.code, spec.display_name) for spec in self._models.values()]


registry = ModelRegistry()


# ---- Model runners ----
def _ensure_clients() -> Dict[str, object]:
    clients: Dict[str, object] = {}

    if CHATGPT5_KEY:
        clients["chatgpt5"] = OpenAI(api_key=CHATGPT5_KEY)

    if GEMINI_KEY:
        genai.configure(api_key=GEMINI_KEY)
        clients["gemini"] = genai.GenerativeModel("gemini-2.0-pro")

    if CLAUDE_KEY:
        clients["claude"] = anthropic.Anthropic(api_key=CLAUDE_KEY)

    if GROK_KEY:
        clients["grok"] = OpenAI(api_key=GROK_KEY, base_url="https://api.x.ai/v1")

    return clients


clients_cache = _ensure_clients()


def _run_with_latency(fn: Callable[[], str]) -> Tuple[str, int]:
    t0 = time.time()
    text = fn()
    return text, int((time.time() - t0) * 1000)


@registry.register("chatgpt_5", "ChatGPT-5", enabled=bool(CHATGPT5_KEY))
def chatgpt5_runner(prompt: str) -> Tuple[str, int]:
    def _call() -> str:
        client = clients_cache.get("chatgpt5")
        if client is None:
            raise RuntimeError("ChatGPT-5 client unavailable. Check API key.")
        response = client.chat.completions.create(
            model="gpt-5-chat",
            messages=[
                {"role": "system", "content": "You are an ICU information assistant. Be accurate, concise, and safe."},
                {"role": "user", "content": prompt},
            ],
            temperature=0.2,
            max_tokens=700,
        )
        return response.choices[0].message.content

    return _run_with_latency(_call)


@registry.register("gemini_2_0", "Gemini 2.0", enabled=bool(GEMINI_KEY))
def gemini_runner(prompt: str) -> Tuple[str, int]:
    def _call() -> str:
        model = clients_cache.get("gemini")
        if model is None:
            raise RuntimeError("Gemini client unavailable. Check API key.")
        result = model.generate_content([
            "You are an ICU information assistant. Be accurate, concise, and safe.",
            prompt,
        ], generation_config={"temperature": 0.2, "max_output_tokens": 700})
        text_content = getattr(result, "text", "") or ""
        if text_content:
            return text_content
        candidates = getattr(result, "candidates", []) or []
        for candidate in candidates:
            parts = getattr(candidate, "content", None)
            if not parts:
                continue
            segments = []
            for part in getattr(parts, "parts", []) or []:
                segment_text = getattr(part, "text", None)
                if segment_text:
                    segments.append(segment_text)
            if segments:
                return "\n".join(segments)
        return ""

    return _run_with_latency(_call)


@registry.register("claude_sonnet_4_5", "Claude Sonnet 4.5", enabled=bool(CLAUDE_KEY))
def claude_runner(prompt: str) -> Tuple[str, int]:
    def _call() -> str:
        client = clients_cache.get("claude")
        if client is None:
            raise RuntimeError("Claude client unavailable. Check API key.")
        message = client.messages.create(
            model="claude-4.5-sonnet",
            max_tokens=750,
            temperature=0.2,
            system="You are an ICU information assistant. Be accurate, concise, and safe.",
            messages=[{"role": "user", "content": prompt}],
        )
        segments: List[str] = []
        for segment in message.content:
            if hasattr(segment, "text") and segment.text:
                segments.append(segment.text)
            elif isinstance(segment, dict) and segment.get("text"):
                segments.append(segment["text"])
        return "\n".join(segments)

    return _run_with_latency(_call)


@registry.register("grok_4", "Grok 4", enabled=bool(GROK_KEY))
def grok_runner(prompt: str) -> Tuple[str, int]:
    def _call() -> str:
        client = clients_cache.get("grok")
        if client is None:
            raise RuntimeError("Grok client unavailable. Check API key.")
        response = client.chat.completions.create(
            model="grok-4",
            messages=[
                {"role": "system", "content": "You are an ICU information assistant. Be accurate, concise, and safe."},
                {"role": "user", "content": prompt},
            ],
            temperature=0.2,
            max_tokens=700,
        )
        return response.choices[0].message.content

    return _run_with_latency(_call)


# Persist model catalog to DB
with engine.begin() as con:
    for code, display in registry.as_rows():
        con.execute(
            text(
                "INSERT OR IGNORE INTO models(id, code, display_name) VALUES (:id, :code, :display)"
            ),
            {"id": str(uuid.uuid4()), "code": code, "display": display},
        )


# -----------------------------
# Helpers
# -----------------------------
def now_iso() -> str:
    return dt.datetime.utcnow().isoformat(timespec="seconds") + "Z"


def ensure_rater(email: str, name: str) -> str:
    with engine.begin() as con:
        row = con.execute(
            text("SELECT id FROM raters WHERE email=:email"),
            {"email": email.strip().lower()},
        ).fetchone()
        if row:
            return row[0]
        rid = str(uuid.uuid4())
        con.execute(
            text(
                "INSERT INTO raters(id, email, display_name, created_at) VALUES (:id, :email, :name, :created)"
            ),
            {"id": rid, "email": email.strip().lower(), "name": name.strip(), "created": now_iso()},
        )
        return rid


def insert_task(stakeholder: str, category: str, prompt_text: str, version: int = 1) -> str:
    tid = str(uuid.uuid4())
    with engine.begin() as con:
        con.execute(
            text(
                """
                INSERT INTO tasks(id, stakeholder, category, prompt_text, version, created_at)
                VALUES(:id, :stakeholder, :category, :prompt, :version, :created)
                """
            ),
            {
                "id": tid,
                "stakeholder": stakeholder,
                "category": category,
                "prompt": prompt_text,
                "version": version,
                "created": now_iso(),
            },
        )
    return tid


def create_assignment(rater_id: str, task_id: str, seed: int) -> str:
    aid = str(uuid.uuid4())
    with engine.begin() as con:
        con.execute(
            text(
                """
                INSERT INTO assignments(id, rater_id, task_id, seed, created_at)
                VALUES(:id, :rater_id, :task_id, :seed, :created)
                """
            ),
            {
                "id": aid,
                "rater_id": rater_id,
                "task_id": task_id,
                "seed": seed,
                "created": now_iso(),
            },
        )
    return aid


def insert_assignment_model(assignment_id: str, model_code: str, slot: int, response_text: str, latency_ms: int) -> str:
    amid = str(uuid.uuid4())
    with engine.begin() as con:
        con.execute(
            text(
                """
                INSERT INTO assignment_models(id, assignment_id, model_code, slot, response_text, latency_ms, created_at)
                VALUES(:id, :assignment_id, :model_code, :slot, :response, :latency, :created)
                """
            ),
            {
                "id": amid,
                "assignment_id": assignment_id,
                "model_code": model_code,
                "slot": slot,
                "response": response_text,
                "latency": latency_ms,
                "created": now_iso(),
            },
        )
    return amid


def insert_rating(assignment_model_id: str, ratings_payload: Dict[str, int]) -> str:
    rid = str(uuid.uuid4())
    with engine.begin() as con:
        con.execute(
            text(
                """
                INSERT INTO ratings(id, assignment_model_id, accuracy, helpfulness, instruction_following, style, created_at)
                VALUES(:id, :amid, :accuracy, :helpfulness, :instruction, :style, :created)
                """
            ),
            {
                "id": rid,
                "amid": assignment_model_id,
                "accuracy": ratings_payload["accuracy"],
                "helpfulness": ratings_payload["helpfulness"],
                "instruction": ratings_payload["instruction_following"],
                "style": ratings_payload["style"],
                "created": now_iso(),
            },
        )
    return rid


def finalize_assignment(assignment_id: str, preferred_slot: int | None, preferred_model_code: str | None, comment: str | None) -> None:
    with engine.begin() as con:
        con.execute(
            text(
                """
                UPDATE assignments
                SET preferred_slot=:slot,
                    preferred_model_code=:model_code,
                    overall_comment=:comment,
                    completed_at=:completed
                WHERE id=:id
                """
            ),
            {
                "slot": preferred_slot,
                "model_code": preferred_model_code,
                "comment": comment or "",
                "completed": now_iso(),
                "id": assignment_id,
            },
        )


def export_csv() -> bytes:
    query = """
    SELECT
      assignments.id AS assignment_id,
      assignments.created_at,
      assignments.completed_at,
      assignments.preferred_slot,
      assignments.preferred_model_code,
      assignments.overall_comment,
      raters.email AS rater_email,
      raters.display_name AS rater_name,
      tasks.stakeholder,
      tasks.category,
      tasks.prompt_text,
      tasks.version,
      assignment_models.slot,
      assignment_models.model_code,
      models.display_name AS model_display_name,
      assignment_models.response_text,
      assignment_models.latency_ms,
      ratings.accuracy,
      ratings.helpfulness,
      ratings.instruction_following,
      ratings.style
    FROM assignments
    JOIN raters ON assignments.rater_id = raters.id
    JOIN tasks ON assignments.task_id = tasks.id
    JOIN assignment_models ON assignment_models.assignment_id = assignments.id
    LEFT JOIN models ON models.code = assignment_models.model_code
    LEFT JOIN ratings ON ratings.assignment_model_id = assignment_models.id
    ORDER BY assignments.created_at DESC, assignment_models.slot ASC
    """
    with engine.begin() as con:
        rows = con.execute(text(query)).mappings().all()
    if not rows:
        return b""
    buffer = io.StringIO()
    writer = csv.DictWriter(buffer, fieldnames=list(rows[0].keys()))
    writer.writeheader()
    for row in rows:
        writer.writerow(dict(row))
    return buffer.getvalue().encode("utf-8")


# -----------------------------
# Streamlit UI
# -----------------------------
st.set_page_config(page_title="ICU LLM Arena (4-way Blinded)", layout="wide")

if SHARED_PASSWORD:
    if not st.session_state.get("pw_ok"):
        password = st.text_input("Access password", type="password")
        if st.button("Enter"):
            if password == SHARED_PASSWORD:
                st.session_state["pw_ok"] = True
                st.experimental_rerun()
            else:
                st.error("Wrong password.")
        st.stop()

st.title("ICU LLM Arena (Blinded Four-Way Comparison)")

with st.sidebar:
    st.markdown("### Rater identity")
    r_email = st.text_input("Email (for linkage/export)", value=st.session_state.get("r_email", ""))
    r_name = st.text_input("Display name/initials", value=st.session_state.get("r_name", ""))
    if st.button("Set rater"):
        if not r_email or not r_name:
            st.error("Please enter both email and display name.")
        else:
            st.session_state["rater_id"] = ensure_rater(r_email, r_name)
            st.session_state["r_email"] = r_email
            st.session_state["r_name"] = r_name
            st.success("Rater set.")
    st.divider()
    st.markdown("### Models enabled")
    enabled_specs = registry.enabled()
    disabled_specs = [spec for spec in registry.specs() if not spec.enabled]
    if not enabled_specs:
        st.warning("No models enabled.")
    else:
        st.code("\n".join(f"{spec.code}: {spec.display_name}" for spec in enabled_specs), language="text")
    if disabled_specs:
        st.warning(
            "Missing API keys for: "
            + ", ".join(spec.display_name for spec in disabled_specs)
        )
    st.divider()
    st.markdown("### Admin")
    csv_data = export_csv()
    st.download_button(
        "Download ratings.csv",
        data=csv_data if csv_data else None,
        file_name="icu_llm_arena_ratings.csv",
        mime="text/csv",
        disabled=not bool(csv_data),
    )
    if not csv_data:
        st.caption("No ratings captured yet.")


if "rater_id" not in st.session_state:
    st.info("Set your rater identity in the left sidebar to begin.")
    st.stop()

st.markdown("#### Enter a question / task")
col_a, col_b, col_c = st.columns([1, 1, 2])
with col_a:
    stakeholder = st.selectbox(
        "Stakeholder",
        [
            "Physician",
            "Nurse",
            "Patient/Family",
            "Pharmacist",
            "Therapist",
            "Other",
        ],
    )
with col_b:
    category = st.selectbox(
        "Category",
        [
            "Clinical Decision Support",
            "Protocol Clarification",
            "Patient Education",
            "Care Coordination",
            "Documentation",
            "Other",
        ],
    )
with col_c:
    prompt_text = st.text_area(
        "Prompt",
        height=120,
        placeholder="e.g., Summarize initial management of suspected elevated ICP in an adult with severe TBI.",
    )

st.caption("No PHI. Prompts and outputs are stored in SQLite for analysis.")


def _clear_rating_state() -> None:
    for key in list(st.session_state.keys()):
        if key.startswith("rating_") or key.startswith("option_text_"):
            st.session_state.pop(key)
    for key in ["overall_preference", "overall_comment"]:
        st.session_state.pop(key, None)


def _generate_letters(n: int) -> List[str]:
    alphabet = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ")
    return alphabet[:n]


def run_generation() -> None:
    enabled = registry.enabled()
    if len(enabled) < 4:
        st.error("Enable all four models to proceed.")
        return

    tid = insert_task(stakeholder, category, prompt_text, version=1)
    seed = random.randint(1, 1_000_000)
    assignment_id = create_assignment(st.session_state["rater_id"], tid, seed)

    specs = list(enabled)
    random.Random(seed).shuffle(specs)

    letters = _generate_letters(len(specs))

    options = []
    for idx, spec in enumerate(specs):
        slot = idx + 1
        label = letters[idx]

        try:
            response_text, latency_ms = spec.runner(prompt_text)
        except Exception as exc:  # noqa: BLE001
            response_text = f"Error generating response: {exc}"
            latency_ms = -1

        amid = insert_assignment_model(
            assignment_id=assignment_id,
            model_code=spec.code,
            slot=slot,
            response_text=response_text,
            latency_ms=latency_ms,
        )
        options.append(
            {
                "slot": slot,
                "label": label,
                "assignment_model_id": amid,
                "latency": latency_ms,
                "text": response_text,
                "model_code": spec.code,
            }
        )

    st.session_state["current_assignment_id"] = assignment_id
    st.session_state["current_options"] = options
    _clear_rating_state()


if st.button("Generate blinded responses", type="primary", disabled=not prompt_text.strip()):
    run_generation()


if "current_assignment_id" in st.session_state and st.session_state.get("current_options"):
    options = st.session_state["current_options"]

    st.markdown("### Blinded Responses")
    cols = st.columns(2)
    for idx, option in enumerate(options):
        col = cols[idx % 2]
        with col:
            st.markdown(f"**Option {option['label']}**")
            st.text_area(
                "",
                value=option["text"],
                height=240,
                disabled=True,
                key=f"option_text_{option['slot']}",
            )
            latency_note = (
                f"Latency: {option['latency']} ms" if option["latency"] >= 0 else "Latency unavailable"
            )
            st.caption(latency_note)
        if idx % 2 == 1 and idx < len(options) - 1:
            cols = st.columns(2)

    st.markdown("#### Rate each option")
    rating_results: Dict[int, Dict[str, int]] = {}
    for option in options:
        with st.expander(f"Ratings for Option {option['label']}", expanded=True):
            accuracy = st.slider(
                "Accuracy", 1, 5, 3, key=f"rating_accuracy_{option['slot']}"
            )
            helpfulness = st.slider(
                "Helpfulness", 1, 5, 3, key=f"rating_helpfulness_{option['slot']}"
            )
            instruction = st.slider(
                "Instruction-following", 1, 5, 3, key=f"rating_instruction_{option['slot']}"
            )
            style = st.slider(
                "Style", 1, 5, 3, key=f"rating_style_{option['slot']}"
            )
            rating_results[option["slot"]] = {
                "accuracy": accuracy,
                "helpfulness": helpfulness,
                "instruction_following": instruction,
                "style": style,
            }

    preference_labels = [f"Option {option['label']}" for option in options]
    preference_choice = st.radio(
        "Overall preference",
        preference_labels + ["Tie / No preference"],
        index=len(preference_labels),
        horizontal=True,
        key="overall_preference",
    )

    overall_comment = st.text_area(
        "Brief justification (optional)",
        height=120,
        key="overall_comment",
    )

    if st.button("Submit rating", type="primary"):
        missing = [slot for slot in rating_results if rating_results[slot] is None]
        if len(rating_results) != len(options):
            st.error("Please rate every option before submitting.")
        else:
            for option in options:
                insert_rating(option["assignment_model_id"], rating_results[option["slot"]])

            if preference_choice == "Tie / No preference":
                preferred_slot = None
                preferred_code = None
            else:
                label = preference_choice.replace("Option ", "")
                preferred_option = next(opt for opt in options if opt["label"] == label)
                preferred_slot = preferred_option["slot"]
                preferred_code = preferred_option["model_code"]

            finalize_assignment(
                st.session_state["current_assignment_id"],
                preferred_slot,
                preferred_code,
                overall_comment,
            )

            st.success("Saved. You can enter another prompt above.")
            st.session_state.pop("current_assignment_id", None)
            st.session_state.pop("current_options", None)
            _clear_rating_state()

st.caption("© ICU LLM Arena — SQLite backend. Do not enter PHI.")
