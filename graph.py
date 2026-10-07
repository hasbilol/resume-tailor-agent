"""Resume Matchmaker & Tailor Agent — LangGraph state machine.

This module contains the complete agentic pipeline:

    START -> analyze_gaps -> draft_tailored_resume -> evaluate_ats_match
                                                              |
                                            +-----------------+-----------------+
                                            | ats_score >= 80 or iteration >= 3  |
                                            v                 v                 |
                                        [finish]          [refine] --------------+
                                          END        back to draft_tailored_resume

Key building blocks:

* ``TailorState``         - TypedDict graph state shared by every node.
* ``ATSScoringOutput``    - Pydantic schema used for structured ATS evaluation.
* Node functions          - ``analyze_gaps``, ``draft_tailored_resume``,
                            ``evaluate_ats_match``.
* ``should_continue``     - Conditional edge router (self-correction loop).
* ``stream_tailor_agent`` - Streaming runner that yields UI friendly events.
* ``run_tailor_agent``    - Convenience wrapper returning the final state.

The Gemini LLM is constructed lazily (only after an API key is resolved) so this
module can be imported by the Streamlit app without any credentials configured.
"""

from __future__ import annotations

import json
import os
import random
import re
import time
from dataclasses import dataclass, field as dataclass_field
from typing import Any, Callable, Dict, Generator, List, Optional, TypedDict

from langchain_core.exceptions import OutputParserException
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage
from langchain_core.runnables import RunnableConfig
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.graph import END, START, StateGraph
from pydantic import BaseModel, Field, ValidationError, field_validator

__all__ = [
    "ATSScoringOutput",
    "AgentRuntime",
    "AgentEvent",
    "DEFAULT_MODEL",
    "DEFAULT_TEMPERATURE",
    "MAX_ITERATIONS",
    "MissingAPIKeyError",
    "MODEL_CHOICES",
    "TailorAgentError",
    "TailorState",
    "TARGET_ATS_SCORE",
    "analyze_gaps",
    "build_graph",
    "create_initial_state",
    "draft_tailored_resume",
    "evaluate_ats_match",
    "graph",
    "run_tailor_agent",
    "should_continue",
    "stream_tailor_agent",
    "validate_inputs",
]

# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #

DEFAULT_MODEL: str = (os.getenv("GEMINI_MODEL") or "gemini-flash-lite-latest").strip()
DEFAULT_TEMPERATURE: float = 0.3

#: Free-tier friendly models offered in the UI sidebar.
#: Every id below was smoke-tested against the Gemini API (`models/{id}:generateContent`).
#: Gemini 2.x models (gemini-2.0/2.5-*) now return 404 for newly issued keys, and the
#: lite alias is the default because it is dramatically more reliable under free-tier
#: demand spikes (full-size flash models intermittently return 503 "high demand").
MODEL_CHOICES: tuple[str, ...] = (
    "gemini-flash-lite-latest",
    "gemini-flash-latest",
    "gemini-3.5-flash-lite",
    "gemini-3.8-flash",
    "gemini-3.7-flash",
    "gemini-3.6-flash",
    "gemini-3.5-flash",
)

TARGET_ATS_SCORE: int = 80
MAX_ITERATIONS: int = 3

#: Total LLM attempts per logical call (first try + retries on 429/5xx).
#: Gemini free tier intermittently returns 503 "high demand" spikes; the schedule
#: below (2s, 4s, 8s, 16s, 30s) rides out typical spike windows (~1 min).
MAX_ATTEMPTS: int = 6
INITIAL_RETRY_DELAY_SECONDS: float = 2.0
MAX_RETRY_DELAY_SECONDS: float = 30.0
LLM_TIMEOUT_SECONDS: float = 120.0
RECURSION_LIMIT: int = 40

#: Rough input floor so we never spend quota on an accidental empty paste.
MIN_INPUT_CHARS: int = 40


# --------------------------------------------------------------------------- #
# Errors
# --------------------------------------------------------------------------- #


class TailorAgentError(RuntimeError):
    """User-facing agent failure with an actionable message."""


class MissingAPIKeyError(TailorAgentError):
    """Raised when no Google/Gemini API key could be resolved."""


class StructuredParseError(TailorAgentError):
    """Structured output could not be parsed (triggers the JSON fallback path)."""


# --------------------------------------------------------------------------- #
# State & structured output schema
# --------------------------------------------------------------------------- #


class TailorState(TypedDict, total=False):
    """Shared state flowing through the LangGraph execution."""

    raw_resume: str
    target_jd: str
    gap_analysis: Optional[str]
    tailored_resume: Optional[str]
    ats_score: int
    missing_skills: List[str]
    critique_feedback: Optional[str]
    iteration_count: int
    logs: List[str]
    # Extras surfaced by the UI (gap section history & metric cards).
    critique_history: List[str]
    score_history: List[int]
    target_match: bool


class AgentEvent(TypedDict, total=False):
    """Event yielded by :func:`stream_tailor_agent` for live UI rendering."""

    node: str
    state: TailorState
    logs: List[str]
    delta_logs: List[str]


class ATSScoringOutput(BaseModel):
    """Strict Pydantic schema returned by the ATS evaluation node."""

    score: int = Field(
        description="ATS alignment score between 0 and 100",
        ge=0,
        le=100,
    )
    missing_critical_skills: List[str] = Field(
        default_factory=list,
        description="Key skills from the JD missing or under-emphasized",
    )
    critique_feedback: str = Field(
        description="Actionable instructions for the next draft refinement",
    )

    @field_validator("score", mode="before")
    @classmethod
    def _normalize_score(cls, value: Any) -> int:
        """Coerce noisy LLM values (``"87/100"``, ``87.0``, ``0.87``) safely."""
        if isinstance(value, bool):
            return 100 if value else 0
        if isinstance(value, str):
            match = re.search(r"-?\d+(?:\.\d+)?", value)
            if not match:
                return 0
            value = float(match.group())
        if value is None:
            return 0
        if isinstance(value, (int, float)):
            numeric = float(value)
            if 0 < numeric <= 1:  # model answered on a 0-1 scale
                numeric *= 100
            return int(max(0, min(100, round(numeric))))
        return 0

    @field_validator("missing_critical_skills", mode="before")
    @classmethod
    def _normalize_skills(cls, value: Any) -> List[str]:
        if value is None:
            return []
        if isinstance(value, str):
            return [s.strip() for s in re.split(r"[,;\n]+", value) if s.strip()]
        if isinstance(value, (list, tuple, set)):
            seen: set[str] = set()
            skills: List[str] = []
            for item in value:
                text = str(item).strip()
                if text and text.lower() not in seen:
                    seen.add(text.lower())
                    skills.append(text)
            return skills
        return [str(value)]

    @field_validator("critique_feedback", mode="before")
    @classmethod
    def _normalize_critique(cls, value: Any) -> str:
        if value is None:
            return ""
        return str(value).strip()


# --------------------------------------------------------------------------- #
# Input validation
# --------------------------------------------------------------------------- #


def validate_inputs(raw_resume: str, target_jd: str) -> tuple[str, str]:
    """Validate and normalise the two user inputs.

    Raises:
        TailorAgentError: if either input is empty or unreasonably short.
    """
    resume = (raw_resume or "").strip()
    jd = (target_jd or "").strip()

    if not resume:
        raise TailorAgentError("Raw resume is empty. Paste your resume text first.")
    if not jd:
        raise TailorAgentError("Target job description is empty. Paste a JD first.")
    if len(resume) < MIN_INPUT_CHARS:
        raise TailorAgentError(
            f"Raw resume looks too short ({len(resume)} chars). "
            f"Provide at least {MIN_INPUT_CHARS} characters of resume content."
        )
    if len(jd) < MIN_INPUT_CHARS:
        raise TailorAgentError(
            f"Job description looks too short ({len(jd)} chars). "
            f"Provide at least {MIN_INPUT_CHARS} characters of JD content."
        )
    return resume, jd


def create_initial_state(raw_resume: str, target_jd: str) -> TailorState:
    """Return a fully populated initial graph state."""
    resume, jd = validate_inputs(raw_resume, target_jd)
    return {
        "raw_resume": resume,
        "target_jd": jd,
        "gap_analysis": None,
        "tailored_resume": None,
        "ats_score": 0,
        "missing_skills": [],
        "critique_feedback": None,
        "iteration_count": 0,
        "logs": [],
        "critique_history": [],
        "score_history": [],
        "target_match": False,
    }


# --------------------------------------------------------------------------- #
# Runtime: lazy Gemini clients + resilient retry/backoff
# --------------------------------------------------------------------------- #

LogCallback = Callable[[str], None]
LLMFactory = Callable[..., Any]


def _exception_chain_text(exc: BaseException) -> str:
    """Flatten ``__cause__``/``__context__`` chain into one searchable string."""
    parts: List[str] = []
    seen: set[int] = set()
    current: Optional[BaseException] = exc
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        parts.append(f"{type(current).__name__}: {current}")
        current = current.__cause__ or current.__context__
    return " | ".join(parts)


def _status_code(exc: BaseException) -> Optional[int]:
    """Best-effort extraction of an HTTP/gRPC status code from an exception."""
    current: Optional[BaseException] = exc
    seen: set[int] = set()
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        for attr in ("code", "status_code", "http_status", "status"):
            value = getattr(current, attr, None)
            if isinstance(value, int):
                return value
            if value is not None:
                inner = getattr(value, "value", None)
                if isinstance(inner, int):
                    return inner
                match = re.search(r"\b([1-5]\d{2})\b", str(value))
                if match:
                    return int(match.group(1))
        response = getattr(current, "response", None)
        inner_status = getattr(response, "status_code", None)
        if isinstance(inner_status, int):
            return inner_status
        current = current.__cause__ or current.__context__
    return None


_RETRYABLE_MARKERS = (
    "429",
    "500",
    "502",
    "503",
    "504",
    "resource_exhausted",
    "resource exhausted",
    "rate limit",
    "ratelimit",
    "too many requests",
    "quota exceeded",
    "overloaded",
    "service unavailable",
    "deadline exceeded",
    "deadline_exceeded",
    "temporarily unavailable",
    "connection reset",
    "connection aborted",
    "connection error",
    "try again",
    "internal error",
    "internal server error",
    "server error",
    "timed out",
    "timeout",
    "unavailable",
)

_DAILY_QUOTA_MARKERS = (
    "per day",
    "per 24 hour",
    "per 24h",
    "24 hour",
    "24h limit",
    "daily quota",
    "day limit",
    "per project per day",
    "lifetime",
)


def is_daily_quota_exhausted(exc: BaseException) -> bool:
    """True when retrying cannot help (free-tier daily/24h quota spent)."""
    text = _exception_chain_text(exc).lower()
    return any(marker in text for marker in _DAILY_QUOTA_MARKERS)


def is_retryable_error(exc: BaseException) -> bool:
    """True for rate limits, transient 5xx, and network hiccups."""
    status = _status_code(exc)
    if status in (429, 500, 502, 503, 504):
        return True
    if status in (400, 401, 403, 404, 422):
        return False
    text = _exception_chain_text(exc).lower()
    return any(marker in text for marker in _RETRYABLE_MARKERS)


def is_structured_parse_error(exc: BaseException) -> bool:
    """Detect schema/JSON parsing failures (handled by the fallback path)."""
    if isinstance(exc, (StructuredParseError, OutputParserException, ValidationError)):
        return True
    text = _exception_chain_text(exc).lower()
    return any(
        marker in text
        for marker in (
            "could not parse",
            "invalid json",
            "json parse error",
            "validation error",
            "expected int",
            "invalid input",
            "structured output",
        )
    )


def _friendly_error(exc: BaseException, model: str, attempts: Optional[int] = None) -> str:
    """Translate library/transport exceptions into an actionable message."""
    text = _exception_chain_text(exc).lower()
    status = _status_code(exc)

    if is_daily_quota_exhausted(exc):
        return (
            "Gemini free-tier daily quota is exhausted. Wait for the 24-hour reset, "
            "or use a different Google AI Studio project/API key."
        )
    if status in (401, 403) or any(
        marker in text
        for marker in ("api key not valid", "api key not found", "unauthenticated", "permission_denied", "permission denied")
    ):
        return "Google rejected the API key. Check that it is valid and that the Gemini API is enabled for your project."
    if status == 404 or "is not found" in text or "not found for api key" in text or "model not found" in text:
        # Google often replies with the exact replacement model, e.g.
        # "Please update your code to use models/gemini-3.8-flash ..."
        hint = re.search(
            r"update your code to use (?:models/)?([a-z0-9][a-z0-9._-]*)",
            _exception_chain_text(exc),
            re.IGNORECASE,
        )
        suggestion = f" Google suggests '{hint.group(1)}'." if hint else ""
        return (
            f"Model '{model}' is not available for this API key/project.{suggestion} "
            "Pick a different model in the sidebar."
        )
    if is_retryable_error(exc):
        suffix = f" after {attempts} attempts" if attempts else ""
        status = _status_code(exc)
        if status in (500, 502, 503, 504):
            return (
                f"Gemini is temporarily overloaded (HTTP {status}){suffix}. "
                "Demand spikes usually clear within a minute — run the agent again."
            )
        reason = f"HTTP {status}" if status else "rate limit"
        return (
            f"Gemini API error ({reason}){suffix}. "
            "Free-tier limits are ~15 requests/minute — wait about a minute, then run again."
        )
    if isinstance(exc, (OutputParserException, ValidationError, StructuredParseError)):
        return (
            "Gemini returned a response that could not be parsed into the required "
            "schema. Run the agent again — this is usually transient."
        )
    first_line = str(exc).strip().splitlines()[0] if str(exc).strip() else "unknown error"
    return f"{type(exc).__name__}: {first_line[:300]}"


@dataclass
class AgentRuntime:
    """Runtime container for LLM clients, retry policy and log callbacks."""

    api_key: str
    model: str = DEFAULT_MODEL
    temperature: float = DEFAULT_TEMPERATURE
    max_attempts: int = MAX_ATTEMPTS
    on_log: Optional[LogCallback] = None
    llm_factory: Optional[LLMFactory] = None
    _chat_llm: Any = dataclass_field(default=None, init=False, repr=False, compare=False)
    _scoring_llm: Any = dataclass_field(default=None, init=False, repr=False, compare=False)

    def emit(self, message: str) -> None:
        """Push a transient progress message to the UI (never raises)."""
        if self.on_log is None:
            return
        try:
            self.on_log(message)
        except Exception:  # pragma: no cover - UI callbacks must not break the agent
            pass

    def chat(self) -> Any:
        """LLM used for gap analysis and drafting (uses configured temperature)."""
        if self._chat_llm is None:
            self._chat_llm = self._build(self.temperature)
        return self._chat_llm

    def scorer(self) -> Any:
        """Deterministic LLM used for ATS scoring (temperature 0)."""
        if self._scoring_llm is None:
            self._scoring_llm = self._build(0.0)
        return self._scoring_llm

    def _build(self, temperature: float) -> Any:
        if self.llm_factory is not None:
            return self.llm_factory(
                model=self.model,
                temperature=temperature,
                api_key=self.api_key,
            )
        return ChatGoogleGenerativeAI(
            model=self.model,
            api_key=self.api_key,
            temperature=temperature,
            # Our own backoff handles 429/5xx so the UI can log every retry.
            max_retries=0,
            timeout=LLM_TIMEOUT_SECONDS,
        )


def _with_retries(
    operation: Callable[[], Any],
    *,
    runtime: AgentRuntime,
    label: str,
) -> Any:
    """Run ``operation`` with exponential backoff on rate limits / transient errors."""
    attempt = 0
    while True:
        attempt += 1
        try:
            return operation()
        except Exception as exc:  # noqa: BLE001 - classified below
            if isinstance(exc, StructuredParseError):
                raise
            if is_daily_quota_exhausted(exc):
                raise TailorAgentError(_friendly_error(exc, runtime.model, attempt)) from exc
            if not is_retryable_error(exc) or attempt >= runtime.max_attempts:
                raise TailorAgentError(_friendly_error(exc, runtime.model, attempt)) from exc

            delay = min(
                INITIAL_RETRY_DELAY_SECONDS * (2 ** (attempt - 1)),
                MAX_RETRY_DELAY_SECONDS,
            ) + random.uniform(0.0, 0.75)
            status = _status_code(exc)
            reason = f"HTTP {status}" if status else "rate limit / transient error"
            runtime.emit(
                f"⏳ [{label}] {reason} from Gemini "
                f"(attempt {attempt}/{runtime.max_attempts}) — retrying in {delay:.0f}s..."
            )
            time.sleep(delay)


# --------------------------------------------------------------------------- #
# Prompt builders
# --------------------------------------------------------------------------- #


def _gap_analysis_messages(resume: str, jd: str) -> List[BaseMessage]:
    system = (
        "You are an expert technical recruiter and ATS keyword analyst. "
        "You compare a candidate's raw resume against one target job description "
        "and produce a precise, evidence-based gap analysis. You never invent facts."
    )
    user = f"""## TARGET JOB DESCRIPTION
{jd}

## CANDIDATE'S RAW RESUME
{resume}

## TASK
Write a gap analysis in Markdown with exactly these sections:

### 1. Must-Have Requirements
Bullets listing the JD's non-negotiable hard requirements (skills, years, tools, domains, certs).

### 2. Priority Keywords
The 10-15 exact keywords and phrases an ATS would extract from this JD, ranked by importance.

### 3. Candidate Evidence
For each priority keyword, quote or paraphrase the resume line that proves it, or write `NOT FOUND`.

### 4. Gaps & Missing Proficiencies
Every requirement the resume does not evidence. Explicit and specific.

### 5. ATS Risks
Formatting/phrasing problems that hurt ATS ranking: missing keywords, vague or unquantified bullets, generic titles.

RULES:
- Base the analysis only on the two texts above.
- Never invent experience, tools or credentials.
- Terse and specific. No preamble, no closing remarks."""
    return [SystemMessage(content=system), HumanMessage(content=user)]


def _draft_messages(state: TailorState) -> List[BaseMessage]:
    resume = state.get("raw_resume", "")
    jd = state.get("target_jd", "")
    gap_analysis = state.get("gap_analysis") or ""
    iteration = int(state.get("iteration_count") or 0)
    previous_draft = state.get("tailored_resume") or ""
    critique = (state.get("critique_feedback") or "").strip()
    score_history = state.get("score_history") or []
    previous_score = score_history[-1] if score_history else None
    is_refinement = iteration >= 1 and bool(critique) and bool(previous_draft)

    system = (
        "You are an elite resume writer and ATS optimization expert. "
        "You rewrite a candidate's resume so it ranks highly in applicant tracking "
        "systems for ONE specific job description while remaining 100% truthful to "
        "the candidate's actual history.\n"
        "HARD RULES — never break them:\n"
        "1. Never invent employers, titles, dates, degrees, certifications, metrics, "
        "tools or skills that are absent from the source resume.\n"
        "2. Never claim a JD skill the candidate has no evidence for. Rephrase "
        "adjacent, provable experience into the JD's vocabulary (for example "
        "'helped with CI pipelines' -> 'built CI/CD pipelines'), but do not fabricate.\n"
        "3. Keep every date, employer and title exactly as provided.\n"
        "4. Output ONLY the resume in Markdown: no commentary, no code fences, "
        "no explanations before or after."
    )

    if is_refinement:
        focus = f"""## REFINEMENT PASS — iteration {iteration + 1} of {MAX_ITERATIONS}
Revise the previous draft below. Do not start from scratch. Apply every feedback
item explicitly while preserving all truthful content and the word budget.

## PREVIOUS DRAFT (revise this)
{previous_draft}

## ATS EVALUATOR FEEDBACK (previous score: {previous_score}/100)
{critique}"""
    else:
        focus = "## FIRST DRAFT — produce the strongest ATS-aligned version of the source resume."

    user = f"""## TARGET JOB DESCRIPTION
{jd}

## SOURCE RESUME (single source of truth)
{resume}

## GAP ANALYSIS (uncover these keyword opportunities)
{gap_analysis}

{focus}

## REQUIRED OUTPUT STRUCTURE (clean Markdown)
1. `# <Candidate Name>` followed by the contact line from the source resume.
2. `## Professional Summary` — 3-4 lines densely packed with truthful JD keywords.
3. `## Experience` — reverse-chronological; each bullet starts with a strong verb,
   mirrors JD phrasing where truthful, and keeps every original metric.
4. `## Projects` — only if the source resume contains project material.
5. `## Skills` — grouped categories, ordered by what the JD weights most.

STYLE RULES:
- 450-750 words total (about one page).
- ATS-friendly plain Markdown: no tables, no images, no emoji, no graphics.
- Missing JD skills stay missing — never fabricate them; instead maximize the
  visibility of adjacent skills the candidate truly has.
- Output the Markdown resume only."""
    return [SystemMessage(content=system), HumanMessage(content=user)]


def _scoring_messages(jd: str, draft: str) -> List[BaseMessage]:
    system = (
        "You are a strict ATS (Applicant Tracking System) simulation engine. "
        "You score exactly how well a resume matches one job description. "
        "You are harsh, specific and never encouraging without evidence."
    )
    user = f"""## JOB DESCRIPTION
{jd}

## RESUME UNDER EVALUATION
{draft}

## SCORING RUBRIC (0-100)
- Keyword & hard-skill coverage: 40 pts — presence of the JD's priority keywords
  and required tools/technologies in the resume body.
- Role & seniority alignment: 20 pts — titles, scope and years match the JD.
- Impact & quantification: 15 pts — measurable outcomes, numbers, scale.
- ATS parseability & structure: 15 pts — clear sections, standard headings,
  keyword-rich bullets, scannable Markdown.
- Summary alignment: 10 pts — the summary sells the candidate for THIS role.

RULES:
- Score strictly: >= {TARGET_ATS_SCORE} means the resume would rank in the top tier
  for this JD; below that it needs work.
- Penalise keyword stuffing and any unsupported claim.
- critique_feedback must be a numbered, ordered list of concrete edits for the
  next revision (which sections, which keywords, which bullets to quantify).
- missing_critical_skills must list JD requirements that are missing or buried.

Return the result using the provided structured schema."""
    return [SystemMessage(content=system), HumanMessage(content=user)]


def _scoring_json_messages(jd: str, draft: str) -> List[BaseMessage]:
    schema_json = json.dumps(ATSScoringOutput.model_json_schema())
    system = (
        "You are a strict ATS simulation engine. Always respond with a single raw "
        "JSON object — no Markdown fences, no prose."
    )
    user = _scoring_messages(jd, draft)[1].content + f"""

## OUTPUT FORMAT
Return ONLY raw JSON conforming to this JSON Schema:
{schema_json}"""
    return [SystemMessage(content=system), HumanMessage(content=user)]


# --------------------------------------------------------------------------- #
# LLM helpers
# --------------------------------------------------------------------------- #


def _message_text(message: Any) -> str:
    """Extract plain text from a chat model response (str or content blocks)."""
    content = getattr(message, "content", message)
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        parts: List[str] = []
        for block in content:
            if isinstance(block, str):
                parts.append(block)
            elif isinstance(block, dict):
                text = block.get("text") or block.get("content") or ""
                if isinstance(text, str):
                    parts.append(text)
            else:
                text = getattr(block, "text", None)
                if isinstance(text, str):
                    parts.append(text)
        return "\n".join(parts).strip()
    return str(content).strip()


def _invoke_text(
    runtime: AgentRuntime,
    messages: List[BaseMessage],
    *,
    label: str,
) -> str:
    """Invoke a chat LLM and return its text, retrying transient failures."""

    def operation() -> str:
        text = _message_text(runtime.chat().invoke(messages))
        if not text:
            raise StructuredParseError(f"empty response from model during {label}")
        return text

    return _with_retries(operation, runtime=runtime, label=label)


def _parse_scoring_payload(raw: str, *, model: str) -> ATSScoringOutput:
    """Parse a JSON scoring payload out of a (possibly fenced) model response."""
    text = (raw or "").strip()
    fenced = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL | re.IGNORECASE)
    if fenced:
        text = fenced.group(1)
    else:
        match = re.search(r"\{.*\}", text, re.DOTALL)
        if match:
            text = match.group(0)
    try:
        payload = json.loads(text)
    except json.JSONDecodeError as exc:
        raise StructuredParseError(
            f"model '{model}' returned invalid JSON for ATS scoring: {exc}"
        ) from exc
    try:
        return ATSScoringOutput.model_validate(payload)
    except ValidationError as exc:
        raise StructuredParseError(f"scoring payload failed validation: {exc}") from exc


def _invoke_structured_scoring(
    runtime: AgentRuntime,
    jd: str,
    draft: str,
    *,
    label: str,
) -> ATSScoringOutput:
    """Score the draft via ``with_structured_output`` with a JSON fallback path."""
    messages = _scoring_messages(jd, draft)

    structured_runnable = None
    try:
        structured_runnable = runtime.scorer().with_structured_output(ATSScoringOutput)
    except Exception as exc:  # noqa: BLE001 - older models may not support schemas
        runtime.emit(
            f"⚠️ [{label}] structured output unavailable ({type(exc).__name__}) — "
            "falling back to JSON parsing."
        )

    if structured_runnable is not None:

        def operation() -> ATSScoringOutput:
            try:
                result = structured_runnable.invoke(messages)
            except Exception as exc:  # noqa: BLE001
                if is_structured_parse_error(exc):
                    raise StructuredParseError(str(exc)) from exc
                raise
            if isinstance(result, ATSScoringOutput):
                return result
            try:
                return ATSScoringOutput.model_validate(result)
            except ValidationError as exc:
                raise StructuredParseError(str(exc)) from exc

        try:
            return _with_retries(operation, runtime=runtime, label=label)
        except StructuredParseError as exc:
            runtime.emit(
                f"⚠️ [{label}] structured output parse failed — retrying with raw JSON prompt."
            )
            raw = _invoke_text(
                runtime,
                _scoring_json_messages(jd, draft),
                label=f"{label}:json-fallback",
            )
            return _parse_scoring_payload(raw, model=runtime.model)
        except TailorAgentError:
            raise

    raw = _invoke_text(
        runtime,
        _scoring_json_messages(jd, draft),
        label=f"{label}:json-fallback",
    )
    return _parse_scoring_payload(raw, model=runtime.model)


def _get_runtime(config: RunnableConfig) -> AgentRuntime:
    """Fetch the :class:`AgentRuntime` injected through ``configurable``."""
    try:
        runtime = config["configurable"]["runtime"]  # type: ignore[index]
    except (KeyError, TypeError, IndexError) as exc:
        raise TailorAgentError(
            "Agent runtime missing. Run the graph through stream_tailor_agent() "
            "or run_tailor_agent()."
        ) from exc
    if not isinstance(runtime, AgentRuntime):
        raise TailorAgentError("Invalid runtime configured for the agent graph.")
    return runtime


def _needs_refinement(ats_score: Any, iteration_count: Any) -> bool:
    """Single source of truth for the self-correction decision."""
    score = int(ats_score or 0)
    iterations = int(iteration_count or 0)
    return score < TARGET_ATS_SCORE and iterations < MAX_ITERATIONS


# --------------------------------------------------------------------------- #
# Graph nodes
# --------------------------------------------------------------------------- #


def analyze_gaps(state: TailorState, config: RunnableConfig) -> Dict[str, Any]:
    """Node 1 — compare the raw resume against the target JD."""
    runtime = _get_runtime(config)
    logs = list(state.get("logs") or [])
    runtime.emit(f"🔍 [analyze_gaps] comparing resume vs. JD with {runtime.model}...")

    started = time.perf_counter()
    gap_analysis = _invoke_text(
        runtime,
        _gap_analysis_messages(state.get("raw_resume", ""), state.get("target_jd", "")),
        label="analyze_gaps",
    )
    elapsed = time.perf_counter() - started
    bullets = len(re.findall(r"^\s*[-*]\s+", gap_analysis, re.MULTILINE))

    logs.append(
        f"✅ [analyze_gaps] gap analysis complete in {elapsed:.1f}s — "
        f"{bullets} findings, {len(gap_analysis.split())} words"
    )
    return {"gap_analysis": gap_analysis, "logs": logs}


def draft_tailored_resume(state: TailorState, config: RunnableConfig) -> Dict[str, Any]:
    """Node 2 — write/rewrite the ATS-aligned resume draft."""
    runtime = _get_runtime(config)
    logs = list(state.get("logs") or [])
    iteration = int(state.get("iteration_count") or 0) + 1
    is_refinement = bool(state.get("critique_feedback")) and iteration > 1
    mode = "refinement pass" if is_refinement else "first draft"
    runtime.emit(
        f"✍️ [draft_tailored_resume] drafting revision {iteration}/{MAX_ITERATIONS} "
        f"({mode})..."
    )

    started = time.perf_counter()
    draft = _invoke_text(runtime, _draft_messages(state), label="draft_tailored_resume")
    elapsed = time.perf_counter() - started
    draft = draft.strip()

    logs.append(
        f"✅ [draft_tailored_resume] revision {iteration}/{MAX_ITERATIONS} ready in "
        f"{elapsed:.1f}s — {len(draft.split())} words"
    )
    return {
        "tailored_resume": draft,
        "iteration_count": iteration,
        "logs": logs,
    }


def evaluate_ats_match(state: TailorState, config: RunnableConfig) -> Dict[str, Any]:
    """Node 3 — strictly score the draft with a Pydantic structured output."""
    runtime = _get_runtime(config)
    logs = list(state.get("logs") or [])
    iteration = int(state.get("iteration_count") or 0)
    runtime.emit(
        f"📊 [evaluate_ats_match] scoring revision {iteration}/{MAX_ITERATIONS} "
        "against the JD..."
    )

    started = time.perf_counter()
    evaluation = _invoke_structured_scoring(
        runtime,
        state.get("target_jd", ""),
        state.get("tailored_resume") or "",
        label="evaluate_ats_match",
    )
    elapsed = time.perf_counter() - started

    score = evaluation.score
    missing = evaluation.missing_critical_skills
    critique = evaluation.critique_feedback.strip() or (
        "No critique returned. Tighten keyword coverage in the summary, quantify "
        "more experience bullets, and mirror the JD's exact terminology."
    )
    critique_history = list(state.get("critique_history") or [])
    critique_history.append(
        f"### Iteration {iteration} — ATS Score {score}/100\n"
        f"**Missing skills:** {', '.join(missing) if missing else 'none'}\n\n"
        f"**Feedback:** {critique}"
    )

    if score >= TARGET_ATS_SCORE:
        status = f"🎯 target met ({score} >= {TARGET_ATS_SCORE})"
    elif iteration >= MAX_ITERATIONS:
        status = f"⛔ max iterations reached ({MAX_ITERATIONS}) — stopping"
    else:
        status = (
            f"🔁 score {score} < {TARGET_ATS_SCORE} → refinement round "
            f"{iteration + 1}/{MAX_ITERATIONS}"
        )

    missing_preview = ", ".join(missing[:5]) if missing else "none"
    logs.append(
        f"📊 [evaluate_ats_match] score {score}/100 in {elapsed:.1f}s — "
        f"missing: {missing_preview} | {status}"
    )

    return {
        "ats_score": score,
        "missing_skills": missing,
        "critique_feedback": critique,
        "critique_history": critique_history,
        "score_history": list(state.get("score_history") or []) + [score],
        "target_match": score >= TARGET_ATS_SCORE,
        "logs": logs,
    }


def should_continue(state: TailorState) -> str:
    """Conditional edge router.

    Returns ``"finish"`` when the score target or iteration budget is reached,
    otherwise ``"refine"`` to loop back into ``draft_tailored_resume``.
    """
    if _needs_refinement(state.get("ats_score"), state.get("iteration_count")):
        return "refine"
    return "finish"


# --------------------------------------------------------------------------- #
# Graph assembly
# --------------------------------------------------------------------------- #


def build_graph():
    """Compile the LangGraph state machine."""
    builder = StateGraph(TailorState)
    builder.add_node("analyze_gaps", analyze_gaps)
    builder.add_node("draft_tailored_resume", draft_tailored_resume)
    builder.add_node("evaluate_ats_match", evaluate_ats_match)

    builder.add_edge(START, "analyze_gaps")
    builder.add_edge("analyze_gaps", "draft_tailored_resume")
    builder.add_edge("draft_tailored_resume", "evaluate_ats_match")
    builder.add_conditional_edges(
        "evaluate_ats_match",
        should_continue,
        {"refine": "draft_tailored_resume", "finish": END},
    )
    return builder.compile()


#: Compiled graph (importable as ``graph.graph``; no API key needed at import).
graph = build_graph()


# --------------------------------------------------------------------------- #
# Runner
# --------------------------------------------------------------------------- #


def stream_tailor_agent(
    raw_resume: str,
    target_jd: str,
    *,
    api_key: Optional[str] = None,
    model: str = DEFAULT_MODEL,
    temperature: float = DEFAULT_TEMPERATURE,
    on_log: Optional[LogCallback] = None,
    max_attempts: int = MAX_ATTEMPTS,
    llm_factory: Optional[LLMFactory] = None,
) -> Generator[AgentEvent, None, Dict[str, Any]]:
    """Stream the agent execution as UI friendly events.

    Yields ``AgentEvent`` dicts with the node name, merged state snapshot and
    the accumulated log lines. The generator's return value (accessible via
    ``StopIteration.value`` or as the last event's ``state``) is the final state.

    Raises:
        MissingAPIKeyError / TailorAgentError: on configuration or runtime failure.
    """
    key = (api_key or "").strip() or (os.getenv("GOOGLE_API_KEY") or "").strip() or (
        os.getenv("GEMINI_API_KEY") or ""
    ).strip()
    if not key:
        raise MissingAPIKeyError(
            "No Google API key found. Add GOOGLE_API_KEY to your environment, "
            "store it in .streamlit/secrets.toml, or paste it in the sidebar."
        )

    resume, jd = validate_inputs(raw_resume, target_jd)
    resolved_model = (model or "").strip() or DEFAULT_MODEL

    display_logs: List[str] = []
    emitted_state_logs = 0

    def emit(message: str) -> None:
        display_logs.append(message)
        if on_log is not None:
            on_log(message)

    runtime = AgentRuntime(
        api_key=key,
        model=resolved_model,
        temperature=temperature,
        max_attempts=max(1, int(max_attempts)),
        on_log=emit,
        llm_factory=llm_factory,
    )

    initial_state = create_initial_state(resume, jd)
    merged: Dict[str, Any] = dict(initial_state)

    emit(f"🚀 [runner] LangGraph run started — model={resolved_model}, "
         f"target={TARGET_ATS_SCORE}/100, max iterations={MAX_ITERATIONS}")

    run_config: RunnableConfig = {
        "configurable": {"runtime": runtime},
        "recursion_limit": RECURSION_LIMIT,
    }

    try:
        for update in graph.stream(initial_state, config=run_config, stream_mode="updates"):
            if not isinstance(update, dict):
                continue
            for node_name, partial in update.items():
                if node_name in (START, END, "__start__", "__end__"):
                    continue
                if not isinstance(partial, dict):
                    continue

                node_logs = partial.get("logs") or []
                delta_logs = list(node_logs[emitted_state_logs:])
                emitted_state_logs = len(node_logs)
                for entry in delta_logs:
                    display_logs.append(entry)

                merged.update(partial)
                yield {
                    "node": node_name,
                    "state": dict(merged),
                    "logs": list(display_logs),
                    "delta_logs": delta_logs,
                }
    except TailorAgentError:
        raise
    except Exception as exc:  # noqa: BLE001 - normalise everything else
        raise TailorAgentError(_friendly_error(exc, runtime.model)) from exc

    final_score = int(merged.get("ats_score") or 0)
    merged["target_match"] = final_score >= TARGET_ATS_SCORE

    emit(
        f"🏁 [runner] finished — final score {final_score}/100 after "
        f"{merged.get('iteration_count', 0)} iteration(s): "
        + ("target met ✅" if final_score >= TARGET_ATS_SCORE else "below target ⚠️")
    )
    merged["logs"] = list(display_logs)

    yield {
        "node": "__end__",
        "state": dict(merged),
        "logs": list(display_logs),
        "delta_logs": [],
    }
    return dict(merged)


def run_tailor_agent(
    raw_resume: str,
    target_jd: str,
    **kwargs: Any,
) -> Dict[str, Any]:
    """Blocking convenience wrapper: run the full graph and return final state."""
    result: Dict[str, Any] = {}
    for event in stream_tailor_agent(raw_resume, target_jd, **kwargs):
        result = event.get("state") or result
    return result
