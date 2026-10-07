# 🎯 Resume Matchmaker & Tailor Agent

A production-grade, **self-correcting LangGraph agent** that turns a raw resume plus one target
job description into an ATS-optimized Markdown resume — then scores the draft with a strict
Pydantic validator and keeps refining it until the score reaches **≥ 80/100** (max 3 iterations).

**Stack:** Python · LangGraph · LangChain · Google Gemini (free tier) · Pydantic · Streamlit

---

## Highlights

- **Gap analysis first** — extracts JD must-haves, priority keywords, candidate evidence, gaps and ATS risks.
- **Truthful rewriting** — rewrites `Professional Summary`, `Experience` and `Projects` with JD keywords; never fabricates employers, dates, metrics or skills.
- **Structured ATS scoring** — `llm.with_structured_output(ATSScoringOutput)` returns `{score, missing_critical_skills, critique_feedback}`.
- **Self-correction loop** — `should_continue` routes back to the drafter while `score < 80` and `iteration_count < 3`.
- **Live execution UI** — `st.status` stream with node progress, retry warnings, log stream, metric cards, comparison tabs, critique history and one-click `.md` export.
- **Free-tier hardened** — exponential backoff with jitter on 429/5xx, daily-quota fail-fast, JSON fallback if structured output parsing fails, 120 s timeouts, input validation.

---

## How the graph works

```
START
  │
  ▼
analyze_gaps ──────────────► draft_tailored_resume ◄──────────┐
                                                              │
                                                              ▼
                                                   evaluate_ats_match
                                                   (ATSScoringOutput)
                                                              │
                                              ┌───────────────┴───────────────┐
                                              │  should_continue             │
                                              │  score ≥ 80  or  iter ≥ 3 ?   │
                                              └───────┬───────────────┬───────┘
                                                 no   ▼               ▼ yes
                                                      │             END (finish)
                                                      └── "refine"
```

| Node / edge | Responsibility |
| --- | --- |
| `analyze_gaps` | Compare `raw_resume` vs `target_jd`; write `gap_analysis` + log line. |
| `draft_tailored_resume` | Produce (or refine) the Markdown resume; applies `critique_feedback` from the previous round; increments `iteration_count`. |
| `evaluate_ats_match` | Score the draft via structured output; populates `ats_score`, `missing_skills`, `critique_feedback`, history. |
| `should_continue` | Returns `"finish"` when `ats_score >= 80` **or** `iteration_count >= 3`, else `"refine"`. |

State lives in `TailorState` (a `TypedDict`) — including `logs`, a step-by-step list rendered by the UI.

---

## Project structure

```
resume-tailor-agent/
├── app.py                       # Streamlit UI: config, inputs, live run, results, export
├── graph.py                     # State, Pydantic schema, nodes, router, compiled graph, runner
├── requirements.txt             # Pinned-compatible dependency ranges
├── README.md
├── .gitignore                   # Keeps .streamlit/secrets.toml out of git
└── .streamlit/
    └── secrets.toml.example     # Copy to secrets.toml (local) or paste into Cloud settings
```

---

## Local setup

### 1 · Prerequisites

- **Python 3.10 – 3.13** (3.12 recommended; the dependency set is tested on 3.12)
- A free **Google AI Studio** API key → <https://aistudio.google.com/apikey>
- Git

### 2 · Clone and install

```bash
git clone <your-repo-url>
cd resume-tailor-agent

python -m venv .venv

# Windows PowerShell
.venv\Scripts\Activate.ps1
# macOS / Linux
source .venv/bin/activate

pip install -r requirements.txt
```

### 3 · Configure the Gemini API key

The app resolves the key in this order: **sidebar entry → `st.secrets` → `GOOGLE_API_KEY` → `GEMINI_API_KEY`**.

**Option A — secrets file (recommended):**

```powershell
# Windows
Copy-Item .streamlit\secrets.toml.example .streamlit\secrets.toml
# macOS / Linux
cp .streamlit/secrets.toml.example .streamlit/secrets.toml
```

Then edit `.streamlit/secrets.toml`:

```toml
GOOGLE_API_KEY = "AIza...your-key..."
# GEMINI_MODEL = "gemini-flash-lite-latest"   # optional override
```

**Option B — environment variable:**

```powershell
# Windows PowerShell (current session)
$env:GOOGLE_API_KEY = "AIza...your-key..."
```

```bash
# macOS / Linux
export GOOGLE_API_KEY="AIza...your-key..."
```

**Option C — sidebar fallback:** run the app and paste the key into the sidebar
`Google API key` field (kept in the session only, never written to disk).

> `.streamlit/secrets.toml` is listed in `.gitignore` — never commit real keys.

### 4 · Run

```bash
streamlit run app.py
```

Open <http://localhost:8501>, paste your resume and a JD (or click **Load sample**), then hit
**🚀 Optimize Resume with LangGraph**.

---

## Using the app

1. **Inputs** — side-by-side panels for the raw resume (text/Markdown) and the target JD.
   The CTA stays disabled until both inputs have ≥ 40 characters.
2. **Live run** — an `st.status` panel streams node progress (`Gap analysis → Drafting → ATS scoring`),
   retry/backoff warnings and the full log stream, with a progress bar per stage.
3. **Results** — three metric cards (Final ATS Score, Iterations Used, Target Match status),
   plus tabs:
   - **Resume Comparison** — original vs. tailored, side by side
   - **Gap Analysis** — full gap report and missing-skill chips
   - **Iteration Critique** — score progression chart + per-iteration feedback expanders
   - **Execution Logs** — every graph event, downloadable as `.txt`
4. **Export** — one click downloads `tailored_resume.md`.

---

## Configuration reference

| Source | Key | Purpose |
| --- | --- | --- |
| Sidebar / secrets / env | `GOOGLE_API_KEY` (or `GEMINI_API_KEY`) | Gemini API credential (**required**) |
| Sidebar / secrets / env | `GEMINI_MODEL` | Model id override (default `gemini-flash-lite-latest`) |
| Sidebar | Gemini model picker | `gemini-flash-lite-latest` (default), `gemini-flash-latest`, `gemini-3.5-flash-lite`, `gemini-3.8-flash`, `gemini-3.7-flash`, `gemini-3.6-flash`, `gemini-3.5-flash`, or a custom id |
| Sidebar | Drafting temperature | 0.0–1.0 (default 0.3); scoring always runs at temperature 0 |

---

## Free-tier behavior (Gemini)

- **~15 requests/minute + demand spikes** — transient 429s **and 503 "high demand"**
  outages are retried up to **6 attempts** with exponential backoff (2 s → 30 s, jittered;
  ~60 s of backoff per call); every retry is written to the live log stream.
- **Daily quota** — detected and failed fast with a clear message instead of burning retries.
- **Structured output failure** — falls back to a raw-JSON prompt and re-parses with the same
  `ATSScoringOutput` Pydantic validators (score coercion, skill list cleanup).
- **Model availability (verified against the API)** — Gemini 2.x ids (`gemini-2.0-flash`,
  `gemini-2.5-flash`, `gemini-2.5-flash-lite`, `gemini-2.5-pro`) now return
  **404 “no longer available to new users”** for newly issued keys, and `*-pro-*` models
  return **429 (no free quota)**. The sidebar only lists flash models confirmed with
  `generateContent`, and errors surface Google’s suggested replacement when provided.
- **Bad key / unknown model** — mapped to actionable error messages surfaced in the UI.

---

## Deploying to Streamlit Community Cloud

1. **Push the project to GitHub** (only these files — `secrets.toml` must stay local):

   ```
   app.py  graph.py  requirements.txt  README.md  .gitignore  .streamlit/secrets.toml.example
   ```

   Verify your key is *not* staged:

   ```bash
   git status            # secrets.toml must not appear
   git check-ignore -v .streamlit/secrets.toml
   ```

2. **Create the app**
   - Go to <https://share.streamlit.io> and sign in with GitHub.
   - **Create app → Fill in the repo details**: pick the repository, branch `main`,
     and main file path `app.py`.
   - Click **Deploy**.

3. **Add the secret (Settings → Advanced settings → Secrets)**
   Paste the same TOML you use locally:

   ```toml
   GOOGLE_API_KEY = "AIza...your-key..."
   # GEMINI_MODEL = "gemini-flash-lite-latest"
   ```

   You can also do this later via **⚙️ Settings → Advanced settings → Secrets** in the
   deployed app menu. Streamlit injects it as `st.secrets["GOOGLE_API_KEY"]`, which the app
   reads on every run.

4. **Auto-deploy** — every push to the tracked branch rebuilds the app. Requirements are
   installed from `requirements.txt`; no system packages are needed (no `packages.txt`).

5. **Rotate keys if leaked** — revoke the key in Google AI Studio and update the Cloud secret.

---

## Troubleshooting

| Symptom | Fix |
| --- | --- |
| `No Google API key found` | Configure the key (secrets/env/sidebar) as shown in step 3. |
| `Google rejected the API key` | Check for typos; enable the **Generative Language API** for the key's project. |
| `rate limit` / HTTP 429 after retries | Wait ~1 minute and re-run; the free tier allows roughly 15 requests/minute. |
| `daily quota is exhausted` | Google AI Studio resets quotas every 24 h; use another key/project in the meantime. |
| `Model '…' is not available` (HTTP 404) | Gemini 2.x models are retired for new keys. Pick a verified flash model from the sidebar (`gemini-flash-lite-latest` by default), or use the replacement Google prints in the error (e.g. `gemini-3.8-flash`). |
| Inputs look stuck / CTA disabled | Both fields need ≥ 40 characters. |
| App starts but the page is blank | Hard-refresh the browser; check the terminal for tracebacks; confirm `streamlit run app.py` is used. |
| Cloud deploy fails during install | Keep the version ranges in `requirements.txt`; the app targets Python 3.10–3.13. |

---

## Sanity check

Confirm the graph compiles without any API key:

```bash
python -c "from graph import graph; print(sorted(graph.get_graph().nodes))"
```

Expected output includes `__start__`, `analyze_gaps`, `draft_tailored_resume`, `evaluate_ats_match`, `__end__`.
