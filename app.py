"""Streamlit UI for the Resume Matchmaker & Tailor Agent.

Run with::

    streamlit run app.py

The app resolves the Gemini API key from ``st.secrets`` / environment variables
with a sidebar fallback, drives the LangGraph pipeline from ``graph.py`` with
live status updates, and renders metrics, comparisons, critique history and logs.
"""

from __future__ import annotations

import html
import os
from typing import Any, Dict, List, Optional, Tuple

import streamlit as st

from graph import (
    DEFAULT_MODEL,
    DEFAULT_TEMPERATURE,
    MAX_ITERATIONS,
    MIN_INPUT_CHARS,
    MODEL_CHOICES,
    TARGET_ATS_SCORE,
    MissingAPIKeyError,
    TailorAgentError,
    stream_tailor_agent,
)

# --------------------------------------------------------------------------- #
# Page config & styling
# --------------------------------------------------------------------------- #

st.set_page_config(
    page_title="Resume Matchmaker & Tailor Agent",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded",
)

CUSTOM_CSS = """
<style>
  .hero {
    background: linear-gradient(135deg, #5b5bd6 0%, #9333ea 55%, #c026d3 100%);
    border-radius: 18px;
    padding: 28px 32px;
    margin-bottom: 18px;
    color: #ffffff;
  }
  .hero h1 { color: #ffffff; margin-bottom: 4px; }
  .hero p { color: rgba(255,255,255,0.92); margin-bottom: 0; }
  .hero .badge {
    display: inline-block;
    background: rgba(255,255,255,0.18);
    border: 1px solid rgba(255,255,255,0.35);
    border-radius: 999px;
    padding: 3px 12px;
    margin-right: 8px;
    font-size: 0.78rem;
    font-weight: 600;
  }
  .step-pill {
    border: 1px solid rgba(128,128,128,0.35);
    border-radius: 14px;
    padding: 14px 16px;
    min-height: 132px;
    background: rgba(128,128,128,0.06);
  }
  .step-pill .step-n {
    font-size: 0.72rem;
    font-weight: 700;
    letter-spacing: 0.08em;
    opacity: 0.75;
  }
  .step-pill .step-t { font-size: 0.98rem; font-weight: 700; margin: 2px 0 6px; }
  .step-pill .step-d { font-size: 0.8rem; opacity: 0.85; }
  .log-stream {
    font-family: "JetBrains Mono", "Cascadia Code", Consolas, monospace;
    font-size: 0.76rem;
    line-height: 1.55;
    background: rgba(128,128,128,0.10);
    border: 1px solid rgba(128,128,128,0.30);
    border-radius: 10px;
    padding: 12px 14px;
    max-height: 360px;
    overflow-y: auto;
    white-space: pre-wrap;
    word-break: break-word;
  }
  .log-stream .entry::before { content: "› "; opacity: 0.6; }
  [data-testid="stMetric"] { padding: 6px 0; }
  footer { visibility: hidden; }
</style>
"""

st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

NODE_LABELS = {
    "analyze_gaps": "Gap analysis",
    "draft_tailored_resume": "Drafting tailored resume",
    "evaluate_ats_match": "ATS scoring",
}

NODE_PROGRESS = {
    "analyze_gaps": 0.15,
    "draft_tailored_resume": 0.40,
    "evaluate_ats_match": 0.65,
}

SAMPLE_RESUME = """\
Alex Rivera
San Francisco, CA | alex.rivera@email.com | linkedin.com/in/alexrivera | github.com/alexrivera

PROFESSIONAL SUMMARY
Software engineer with 4 years of experience building web applications and internal tools.

EXPERIENCE
Software Engineer — Northwind Labs (2022 - Present)
- Built REST APIs in Python (Flask) serving the internal analytics dashboard
- Migrated a legacy jQuery frontend to React, reducing page load time
- Worked with product managers to ship three customer-facing features
- Wrote SQL queries and PostgreSQL reports for the growth team

Junior Developer — Bluepeak Solutions (2020 - 2022)
- Maintained Django views and templates for a SaaS billing portal
- Fixed bugs and added unit tests with pytest
- Helped the team adopt Docker for local development

PROJECTS
- DevFind: a React app that indexes developer blogs; 500+ monthly users
- PyCLI: open-source command-line task runner (400 GitHub stars)

SKILLS
Python, Flask, Django, JavaScript, React, PostgreSQL, Git, Docker, pytest
"""

SAMPLE_JD = """\
Senior Backend Engineer — Meridian Health (Remote)

What you will do
- Design and operate high-throughput services in Python (FastAPI) on AWS
- Own PostgreSQL schema design, query optimization and data integrity
- Build Kubernetes-deployed microservices with CI/CD pipelines (GitHub Actions)
- Drive system design reviews, observability (Prometheus, Grafana) and on-call quality
- Mentor mid-level engineers and lead delivery of compliance-sensitive features (HIPAA)

Requirements
- 5+ years backend experience with Python in production
- Strong PostgreSQL and distributed systems fundamentals
- Experience running services on Kubernetes and AWS (ECS/EKS)
- Track record of improving reliability, latency and cost
- Excellent written communication
"""


# --------------------------------------------------------------------------- #
# Secrets / configuration helpers
# --------------------------------------------------------------------------- #


def _secret(key: str) -> Optional[str]:
    """Read a Streamlit secret without crashing when no secrets file exists."""
    try:
        value = st.secrets.get(key)  # type: ignore[arg-type]
    except Exception:  # noqa: BLE001 - missing secrets file, parse errors, cloud-less local runs
        return None
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def resolve_api_key() -> Tuple[str, str]:
    """Return ``(api_key, source_label)`` from secrets, then environment."""
    candidates = (
        ("st.secrets", _secret("GOOGLE_API_KEY")),
        ("$GOOGLE_API_KEY", os.getenv("GOOGLE_API_KEY")),
        ("$GEMINI_API_KEY", os.getenv("GEMINI_API_KEY")),
    )
    for source, value in candidates:
        if value:
            return value, source
    return "", ""


def default_model_id() -> str:
    return _secret("GEMINI_MODEL") or DEFAULT_MODEL


# --------------------------------------------------------------------------- #
# Rendering helpers
# --------------------------------------------------------------------------- #


def render_log_markdown(entries: List[str]) -> str:
    rows = "".join(f'<div class="entry">{html.escape(line)}</div>' for line in entries)
    return f'<div class="log-stream">{rows or "waiting for the agent…"}</div>'


def step_pill(number: str, title: str, description: str) -> str:
    return (
        f'<div class="step-pill"><div class="step-n">{number}</div>'
        f'<div class="step-t">{title}</div>'
        f'<div class="step-d">{description}</div></div>'
    )


def render_header() -> None:
    st.markdown(
        """
<div class="hero">
  <h1>Resume Matchmaker &amp; Tailor Agent</h1>
  <p>A self-correcting LangGraph agent that gap-analyzes your resume against a job
  description, rewrites it for ATS alignment, scores the match with a Pydantic
  validator, and refines until it hits the target score.</p>
  <p style="margin-top:12px">
    <span class="badge">LangGraph</span>
    <span class="badge">Google Gemini (free tier)</span>
    <span class="badge">Pydantic structured output</span>
    <span class="badge">Self-correction loop ≤ 3</span>
  </p>
</div>
""",
        unsafe_allow_html=True,
    )

    pills = st.columns(4, gap="small")
    content = [
        ("STEP 01", "Gap Analysis", "Extract JD keywords, map candidate evidence, flag missing proficiencies."),
        ("STEP 02", "Tailored Draft", "Rewrite summary, experience and projects — truthful and keyword-aligned."),
        ("STEP 03", "ATS Scoring", "Structured Pydantic scoring (0-100) with critique and missing skills."),
        ("STEP 04", "Self-Correction", f"Refine the draft until score ≥ {TARGET_ATS_SCORE} or 3 iterations."),
    ]
    for column, (num, title, desc) in zip(pills, content):
        with column:
            st.markdown(step_pill(num, title, desc), unsafe_allow_html=True)


def render_sidebar() -> Dict[str, Any]:
    stored_key, stored_source = resolve_api_key()
    model_default = default_model_id()

    with st.sidebar:
        st.markdown("## ⚙️ Configuration")

        entered_key = st.text_input(
            "Google API key",
            type="password",
            key="api_key_input",
            placeholder="AIza… (only needed if no secret/env key)",
            help="Priority: sidebar entry → st.secrets → GOOGLE_API_KEY/GEMINI_API_KEY. "
            "Get a free key at https://aistudio.google.com/apikey",
        )
        effective_key = (entered_key or "").strip() or stored_key
        if effective_key:
            origin = "sidebar entry" if (entered_key or "").strip() else stored_source
            st.success(f"API key active (source: {origin})", icon="🔑")
        else:
            st.warning("No API key yet — paste one above to run the agent.", icon="⚠️")

        known = list(MODEL_CHOICES) + ["Custom model ID…"]
        selection = st.selectbox(
            "Gemini model",
            known,
            index=known.index(model_default) if model_default in known else len(known) - 1,
            key="model_select",
        )
        if selection == "Custom model ID…":
            model_id = st.text_input("Model ID", value=model_default, key="model_custom").strip() or model_default
        else:
            model_id = selection

        temperature = st.slider(
            "Drafting temperature",
            min_value=0.0,
            max_value=1.0,
            value=float(DEFAULT_TEMPERATURE),
            step=0.05,
            key="temperature_slider",
            help="Scoring always runs at temperature 0. Higher values make drafts more varied.",
        )

        st.divider()
        st.markdown(
            """
**How the graph executes**

1. `analyze_gaps` — resume vs. JD gap report
2. `draft_tailored_resume` — ATS-aligned Markdown rewrite
3. `evaluate_ats_match` — `llm.with_structured_output(ATSScoringOutput)`
4. `should_continue` — loop back to step 2 until the score lands

Free-tier notes: Gemini enforces ~15 requests/minute and a daily quota.
The agent auto-retries 429s with exponential backoff and surfaces every
retry in the live log stream.
"""
        )

    return {
        "api_key": effective_key,
        "model": model_id,
        "temperature": float(temperature),
    }


# --------------------------------------------------------------------------- #
# Input panel
# --------------------------------------------------------------------------- #


def _load_sample() -> None:
    st.session_state["resume_input"] = SAMPLE_RESUME
    st.session_state["jd_input"] = SAMPLE_JD


def render_inputs() -> Tuple[str, str, bool]:
    st.markdown("### 📥 Inputs")
    left, right = st.columns(2, gap="large")

    with left:
        st.markdown("**Raw resume** (text or Markdown)")
        raw_resume = st.text_area(
            "Raw resume",
            key="resume_input",
            height=420,
            label_visibility="collapsed",
            placeholder="Paste your current resume exactly as it is today…",
        )
        resume_ok = len((raw_resume or "").strip()) >= MIN_INPUT_CHARS
        st.caption(
            f"{len((raw_resume or '').strip())} characters "
            + ("✓" if resume_ok else f"· need at least {MIN_INPUT_CHARS}")
        )

    with right:
        st.markdown("**Target job description**")
        target_jd = st.text_area(
            "Target job description",
            key="jd_input",
            height=420,
            label_visibility="collapsed",
            placeholder="Paste the full job description you are applying to…",
        )
        jd_ok = len((target_jd or "").strip()) >= MIN_INPUT_CHARS
        st.caption(
            f"{len((target_jd or '').strip())} characters "
            + ("✓" if jd_ok else f"· need at least {MIN_INPUT_CHARS}")
        )

    actions = st.columns([1, 5])
    with actions[0]:
        st.button("Load sample", on_click=_load_sample, use_container_width=True)
    with actions[1]:
        can_run = resume_ok and jd_ok
        run_clicked = st.button(
            "🚀 Optimize Resume with LangGraph",
            type="primary",
            use_container_width=True,
            disabled=not can_run,
        )

    if not can_run:
        st.caption("Provide a resume and a job description (≥ 40 characters each) to enable the agent.")
    return raw_resume or "", target_jd or "", run_clicked


# --------------------------------------------------------------------------- #
# Agent execution
# --------------------------------------------------------------------------- #


def run_agent(resume: str, jd: str, config: Dict[str, Any]) -> Tuple[Dict[str, Any], List[str]]:
    """Execute the graph inside an ``st.status`` with live logs and progress."""
    live_logs: List[str] = []
    # Publish the buffer immediately so failed runs can still show their logs.
    st.session_state["tailor_logs"] = live_logs
    final_state: Dict[str, Any] = {}

    def on_log(message: str) -> None:
        live_logs.append(message)
        log_box.markdown(render_log_markdown(live_logs), unsafe_allow_html=True)

    with st.status("🚀 Launching LangGraph agent…", expanded=True) as status:
        log_box = st.empty()
        progress = st.progress(0.02)
        status.update(label="Starting graph execution…", state="running")
        log_box.markdown(render_log_markdown(live_logs), unsafe_allow_html=True)

        try:
            events = stream_tailor_agent(
                resume,
                jd,
                api_key=config["api_key"],
                model=config["model"],
                temperature=config["temperature"],
                on_log=on_log,
            )
            for event in events:
                node = event.get("node", "")
                state = event.get("state") or {}
                live_logs.extend(event.get("delta_logs") or [])
                log_box.markdown(render_log_markdown(live_logs), unsafe_allow_html=True)
                final_state = state

                if node == "__end__":
                    progress.progress(1.0)
                else:
                    label = NODE_LABELS.get(node, node)
                    score = int(state.get("ats_score") or 0)
                    iteration = int(state.get("iteration_count") or 0)
                    status.update(
                        label=f"{label} — score {score}/100 · iteration {iteration}/{MAX_ITERATIONS}",
                        state="running",
                    )
                    base = NODE_PROGRESS.get(node, 0.1)
                    bump = min(0.3, 0.08 * max(0, iteration - 1))
                    progress.progress(min(0.95, base + bump))

            status.update(
                label=f"✅ Run complete — final ATS score {int(final_state.get('ats_score') or 0)}/100",
                state="complete",
                expanded=False,
            )
        except TailorAgentError:
            status.update(label="❌ Run failed — see details below", state="error")
            raise
        except Exception:
            status.update(label="❌ Unexpected failure — see details below", state="error")
            raise

    return final_state, live_logs


# --------------------------------------------------------------------------- #
# Results rendering
# --------------------------------------------------------------------------- #


def render_results(result: Dict[str, Any], logs: List[str]) -> None:
    score = int(result.get("ats_score") or 0)
    iterations = int(result.get("iteration_count") or 0)
    matched = bool(result.get("target_match"))
    score_history: List[int] = list(result.get("score_history") or [])
    missing_skills: List[str] = list(result.get("missing_skills") or [])
    critique_history: List[str] = list(result.get("critique_history") or [])

    st.divider()
    head_left, head_right = st.columns([4, 1])
    with head_left:
        st.markdown("### 📊 Results")
    with head_right:
        if st.button("🧹 Clear results", use_container_width=True):
            for key in ("tailor_result", "tailor_logs"):
                st.session_state.pop(key, None)
            st.rerun()

    cards = st.columns(3, gap="medium")
    with cards[0]:
        with st.container(border=True):
            delta = (
                f"{score - score_history[0]:+d} vs first draft"
                if len(score_history) > 1
                else None
            )
            st.metric("Final ATS Score", f"{score}/100", delta=delta, delta_color="normal")
    with cards[1]:
        with st.container(border=True):
            st.metric("Iterations Used", f"{iterations}/{MAX_ITERATIONS}")
    with cards[2]:
        with st.container(border=True):
            st.metric(
                f"Target Match (≥ {TARGET_ATS_SCORE})",
                "✅ Matched" if matched else "⚠️ Below target",
            )

    dl_left, dl_right = st.columns([4, 1])
    with dl_right:
        st.download_button(
            "⬇️ Download tailored_resume.md",
            data=result.get("tailored_resume") or "",
            file_name="tailored_resume.md",
            mime="text/markdown",
            type="primary",
            use_container_width=True,
        )

    tab_comparison, tab_gaps, tab_critique, tab_logs = st.tabs(
        ["🆚 Resume Comparison", "🧩 Gap Analysis", "🔁 Iteration Critique", "📜 Execution Logs"]
    )

    with tab_comparison:
        original_col, tailored_col = st.columns(2, gap="large")
        with original_col:
            st.markdown("**Original resume**")
            with st.container(height=520, border=True):
                st.markdown(result.get("raw_resume") or "_No resume in state._")
        with tailored_col:
            st.markdown("**Tailored resume (ATS-optimized)**")
            with st.container(height=520, border=True):
                st.markdown(result.get("tailored_resume") or "_No draft produced._")

    with tab_gaps:
        if missing_skills:
            st.markdown(
                "**Missing / under-emphasized skills:** "
                + " ".join(f"`{skill}`" for skill in missing_skills)
            )
        else:
            st.success("No critical skills flagged by the evaluator.")
        with st.container(border=True):
            st.markdown(result.get("gap_analysis") or "_No gap analysis produced._")

    with tab_critique:
        if len(score_history) > 1:
            st.caption("ATS score progression across iterations")
            st.line_chart({"ATS score": score_history})
        if critique_history:
            for index, entry in enumerate(critique_history, start=1):
                score_label = (
                    f" — {score_history[index - 1]}/100"
                    if index - 1 < len(score_history)
                    else ""
                )
                with st.expander(
                    f"Iteration {index}{score_label}",
                    expanded=index == len(critique_history),
                ):
                    st.markdown(entry)
        else:
            st.info("No iteration critique recorded for this run.")

    with tab_logs:
        st.markdown(
            '<div class="log-stream">'
            + "".join(f'<div class="entry">{html.escape(line)}</div>' for line in logs)
            + "</div>",
            unsafe_allow_html=True,
        )
        st.download_button(
            "⬇️ Download logs",
            data="\n".join(logs),
            file_name="execution_logs.txt",
            mime="text/plain",
        )


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #


def _show_run_failure(message: str) -> None:
    """Clear stale results and surface the error plus any logs captured pre-failure."""
    st.session_state.pop("tailor_result", None)
    st.error(message)
    partial_logs: List[str] = st.session_state.get("tailor_logs") or []
    if partial_logs:
        with st.expander(f"Execution log before failure ({len(partial_logs)} lines)"):
            st.markdown(render_log_markdown(partial_logs), unsafe_allow_html=True)


def main() -> None:
    render_header()
    config = render_sidebar()
    raw_resume, target_jd, run_clicked = render_inputs()

    if run_clicked:
        if not config["api_key"]:
            st.error(
                "No Google API key configured. Add one in the sidebar, or set "
                "`GOOGLE_API_KEY` in `.streamlit/secrets.toml` / your environment."
            )
        else:
            try:
                result, logs = run_agent(raw_resume, target_jd, config)
                st.session_state["tailor_result"] = result
                st.session_state["tailor_logs"] = logs
            except MissingAPIKeyError as exc:
                _show_run_failure(str(exc))
            except TailorAgentError as exc:
                _show_run_failure(f"**Agent run failed:** {exc}")
            except Exception as exc:  # noqa: BLE001 - surface unexpected bugs without a traceback dump
                _show_run_failure(f"**Unexpected error:** `{type(exc).__name__}`: {exc}")

    if st.session_state.get("tailor_result"):
        render_results(st.session_state["tailor_result"], st.session_state.get("tailor_logs", []))
    elif not run_clicked:
        st.info(
            "👋 Paste a resume and a target job description above, then hit "
            "**Optimize Resume with LangGraph** to start the agentic run."
        )

    st.divider()
    st.caption(
        "Resume Matchmaker & Tailor Agent · LangGraph state machine + Google Gemini "
        "free tier · Verify every claim in the tailored resume before sending it."
    )


if __name__ == "__main__":
    main()
