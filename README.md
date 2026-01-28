# ToldYou - A recommendation compliance checker

An OSINT automation tool that operationalizes the "I told you so." It ingests regulatory post-mortems, extracts the "Must Do" directives, and deploys autonomous agents to verify if the entities actually did them.

I built this to check whether recommendations from a 2021 regulatory report were actually followed. The pipeline extracts recommendations from a PDF, hands each one to a browser agent that searches for evidence of compliance, then scores and ranks the findings by impact. The output is a filterable dashboard showing what got fixed and what just got a memo.

---

## Example: From Report to Verdict

Here's how a recommendation flows through the system, using real data from the 2021 Cold Weather Report:

### What the Report Said (Page 19)

> "Generator Owners and Operators should perform annual training on winterization plans"

In plain English: **"Power plant operators need to train their staff every year on how to keep equipment running in freezing weather."**

### What the Pipeline Extracted

```json
{
  "actor": "Generator Owners/Generator Operators",
  "action": "Perform annual training on winterization plans",
  "test_question": "Have Generator Owners/Operators performed annual training on winterization plans?"
}
```

The system broke it down into who's responsible, what they should do, and a yes/no question for the browser agent to investigate.

### What the Browser Agent Found

**Status: IMPLEMENTED**

The agent searched the web and found:
1. NERC Standard EOP-012-3 (effective October 2025) now **legally requires** annual training
2. There's an official audit checklist requiring plants to show training records, attendance logs, and course materials

*"Yes, this became law. Power plants must now do this training every year and prove it with paperwork during audits."*

### Final Output

```
Headline: "Annual training on winterization plans mandated by NERC Standard EOP-012-3"
Verdict: "Implemented - active audit structure confirms compliance"
Priority Score: 4.4/10 (lower priority because it's already done)
```

---

### A Recommendation That *Didn't* Fully Happen

**Report said:** "Generator Owners should develop Corrective Action Plans if they experience freeze-related outages"

**Browser agent found:** Status **PARTIALLY_IMPLEMENTED**
- The rule exists (NERC EOP-012-2 requires these plans)
- But during Winter Storm Elliott (Dec 2022), **90,500 MW of power failed anyway** — and 75% of those failures happened at temperatures the plants claimed they could handle

**Final output:**
```
Verdict: "Rule exists, but 90,500 MW failed during the next storm - inadequate implementation"
Priority Score: 6.9/10 (higher priority - needs attention)
Urgency: 9/10
```

The pipeline catches the difference between **"they said they'd do it"** vs **"it actually worked when tested."**

---

## The Pipeline

```
PDF → Project Folder → Decompose → Summarize → Verify → Enrich → Visualize Findings → Find Consequences → Visualize Consequences
```

1. **Decompose** (`01_decompose.py`) — Creates the project folder and extracts actionable recommendations.
2. **Summarize** (`02_summary.py`) — Scans for metadata, authors, and acronyms (e.g., "GO" = "Generator Owner").
3. **Verify** (`03_verify.py`) — Deploys browser agents to find Level 1 (Regulatory) and Level 2 (Physical) evidence.
4. **Enrich** (`04_enrich.py`) — Scores findings on 6 risk dimensions and generates headlines.
5. **Visualize** (`05_visualize.py`) — Generates the main compliance dashboard.
6. **Consequences** (`06_consequences.py`) — Matches unfollowed findings to recent news events using parallel web discovery and video transcription.
7. **Visualize Consequences** (`07_visualize.py`) — Generates an interactive report linking news to regulatory failures.

---

## Project Architecture (Refactored)

We recently refactored the pipeline to use a **holistic project-based workflow** managed by `pipeline_utils.py`. 

Previously, scripts required manual file chaining. Now, everything is organized into a single project directory. You only need to pass the path to that directory (or the original PDF) to any script, and it will automatically resolve the necessary inputs and outputs.

### Standardized Project Structure:
```
report-slug/
├── 01_findings.json
├── 02_context.json
├── 03_verified.jsonl
├── 04_enriched.jsonl
├── 05_report.html
├── 06_consequences.jsonl
├── 07_consequences_report.html
├── articles/          # Cached news content
└── debug/             # LLM prompts and raw responses
```

---

## Installation

Requires Python 3.8+ and API keys for OpenAI (extraction/enrichment) and Browser Use (verification).

```bash
pip install -r requirements.txt
```

Dependencies: `openai`, `pdfplumber`, `requests`, `pydantic`, `tqdm`, `nodriver`, `trafilatura`, `yt-dlp`, `webvtt-py`

Set your API keys:

```bash
# Linux/macOS
export OPENAI_API_KEY="sk-..."
export BROWSER_USE_API_KEY="bu_..."

# Windows (PowerShell)
$env:OPENAI_API_KEY="sk-..."
$env:BROWSER_USE_API_KEY="bu_..."
```

---

## Usage

The pipeline is project-based. You can pass either the original PDF or the resulting project directory to any script.

### 1. Extract findings

```bash
python 01_decompose.py "report.pdf"
```

Creates a folder named `report/` and outputs `01_findings.json`.

- `--test 10` — Only process the first 10 pages
- `--debug` — Save raw extracted text

### 2. Get context

```bash
python 02_summary.py "report/"
```

Outputs `02_context.json`.

### 3. Verify

```bash
python 03_verify.py "report/" --resume
```

Outputs `03_verified.jsonl`.

- `--resume` — Continue from where you left off.
- `--workers 3` — Parallel agents.
- `--try-hard` — Increase max steps and enable vision.

### 4. Enrich

```bash
python 04_enrich.py "report/"
```

Outputs `04_enriched.jsonl` with priority scores and headlines.

### 5. Visualize

```bash
python 05_visualize.py "report/"
```

Outputs `05_report.html`.

### 6. Find Consequences

```bash
python 06_consequences.py "report/" --event "texas winter storm january 2025"
```

Outputs `06_consequences.jsonl` matching findings to real-world news.

- `--days 14` — Look back window.
- `--debug` — Save raw matching prompts/responses.

### 7. Visualize Consequences

```bash
python 07_visualize.py "report/"
```

Outputs `07_consequences_report.html`. Uses `07_template.html`.

---

## The HTML report

The output file has filters, sorting, and search built in:

- Filter by status, confidence, change type, safety category, deadline
- Sort by priority score, safety impact, cost, urgency, scope, actor, or page
- Search across all text
- Min score slider to hide low-priority items
- Shareable URLs — filter state is written to the query string, so you can copy the URL and send someone your exact view

Each finding card shows a color-coded headline (green/yellow/red based on status), the recommended action, a verdict with confidence level, and classification tags. A collapsible section has the full verification summary, score breakdown, evidence links, and source page number.

---

## Scoring weights

The composite priority score (1-10) is calculated from:

| Dimension | Weight | Range |
|-----------|--------|-------|
| Implementation gap | 25% | Not implemented = high score |
| Safety impact | 25% | Compliance risk → life safety |
| Scope impact | 15% | Single entity → national |
| Cost estimate | 15% | <$100K → $100M+ |
| Change magnitude | 10% | Paperwork → infrastructure |
| Urgency | 10% | No deadline → overdue |

---

## File formats

### Findings JSON (from decompose)

```json
{
  "findings": [
    {
      "actor": "Generator Owners",
      "action": "Retrofit insulation packages...",
      "test_question": "Have Generator Owners retrofitted...",
      "citation_page": "45"
    }
  ]
}
```

### Verified JSONL (from verify)

```json
{
  "finding_id": "a1b2c3d4",
  "verification": {
    "status": "PARTIALLY_IMPLEMENTED",
    "confidence": "HIGH",
    "summary": "FERC approved the rule in 2022, but...",
    "evidence": [
      { "url": "https://ferc.gov/...", "title": "Order 896" }
    ]
  }
}
```

---

## Troubleshooting

**OutOfCreditsError (402)** — Browser Use ran out of credits. Add credits and run with `--resume` to pick up where you left off.

**Rate limiting (429)** — Reduce `--workers` to 1 or 2.

**"No text extracted"** — The PDF is probably a scanned image. This pipeline uses `pdfplumber`, which requires selectable text. OCR the PDF first.

**JSON decode errors** — Occasional LLM hiccups. The scripts log these and skip malformed records.

---

## Cost

The verification step is where the money goes. Each finding spawns a browser agent that runs 10-30+ steps of web searches. On a 300-finding report, expect real costs from Browser Use credits. The other steps use standard OpenAI API calls and cost a few dollars total.
