# Technical Architecture

This document provides a low-level overview of the internal logic for each script in the ToldYou pipeline.

## 01_decompose.py: Requirement Extraction

**Goal:** Turn a PDF into a structured JSON list of actionable findings.

**Process:**
1.  **Text Extraction (`pdfplumber`):**
    *   Iterates through the PDF page by page.
    *   Extracts raw text, preserving basic layout.
    *   Adds `--- PAGE X ---` markers to track citations.

2.  **Chunking (Sliding Window):**
    *   Since regulatory reports are long, text is split into overlapping chunks (default: 3 pages per chunk, sliding by 1 page).
    *   This ensures context isn't lost at page boundaries.

3.  **LLM Analysis (`gpt-4o`):**
    *   Each chunk is sent to the LLM with a specific system prompt.
    *   **Context Passing:** The prompt includes a summary of *already found* recommendations to prevent duplicates (deduplication at source).
    *   **Prompt Engineering:** Explicitly instructs the LLM to distinguish between *narrative* ("X failed") and *recommendations* ("Entity shall do Y").

4.  **Post-Processing:**
    *   Validates page numbers against the chunk's range.
    *   If a citation page is wildly incorrect (hallucination), it defaults to the middle page of the current chunk.

5.  **Output:** `01_findings.json` (List of actor/action pairs).

---

## 02_summary.py: Context Generation

**Goal:** Extract global metadata to ground the verification agent.

**Process:**
1.  **Targeted Extraction:**
    *   Reads only the first ~25 pages of the PDF (Executive Summary, Acronyms list).
    *   This saves tokens while capturing the most critical setup info.

2.  **LLM Extraction:**
    *   One-shot extraction of: Title, Date, Authors, Event Name, Regulatory Bodies, and **Key Terms**.
    *   **Critical:** The "Key Terms" dictionary is vital. It tells downstream agents that "GO" = "Generator Owner", preventing search hallucinations.

3.  **Output:** `02_context.json` (Metadata dictionary).

---

## 03_verify.py: Compliance Verification

**Goal:** Determine if a recommendation was implemented using live web searches.

**Process:**
1.  **Initialization:**
    *   Loads findings and context.
    *   Checks for a `.progress` file to support resumability.

2.  **Parallel Execution (`ThreadPoolExecutor`):**
    *   Spawns worker threads (default: 3) to process findings concurrently.
    *   Supports `--cost-tracking` mode which forces single-threaded execution for precise billing logs.

3.  **The Browser Agent (Browser Use API):**
    *   For each finding, constructs a specialized prompt.
    *   **Prompt Strategy:** Forces a "Level 1 vs. Level 2" distinction.
        *   *Level 1:* Is there a rule? (Paper compliance)
        *   *Level 2:* Did they actually do it? (Physical compliance/Audits)
    *   The agent autonomously searches Google, reads PDFs, and navigates relevant sites (NERC, FERC, news).

4.  **Status Determination:**
    *   The agent returns a status: `IMPLEMENTED`, `PARTIALLY_IMPLEMENTED`, `NOT_IMPLEMENTED`, or `UNABLE_TO_VERIFY`.
    *   It cites specific URLs as evidence.

5.  **Output:** `03_verified.jsonl` (Streaming JSON Lines).

---

## 04_enrich.py: Scoring & Classification

**Goal:** Turn raw verification data into a prioritized risk dashboard.

**Process:**
1.  **Input:** Reads the verified JSONL stream.

2.  **Scoring (`gpt-4o-mini`):**
    *   Uses a cheaper, faster model to classify the findings based on the verification evidence.
    *   **6 Dimensions:**
        1.  **Implementation Gap:** (Not done = 10, Done = 1)
        2.  **Safety Impact:** (Life safety = 10, Admin = 1)
        3.  **Scope:** (National = 10, Single plant = 1)
        4.  **Cost:** (>$100M = 10, <$10k = 1)
        5.  **Change Magnitude:** (Infrastructure = 10, Policy = 1)
        6.  **Urgency:** (Overdue = 10, No deadline = 1)
    *   Calculates a **Composite Score** (Weighted average).

3.  **Summarization:**
    *   Generates a "Headline" (e.g., "Generators failed to winterize") and "Verdict".

4.  **Output:** `04_enriched.jsonl`.

---

## 05_visualize.py: Dashboard Generation

**Goal:** Create a static, portable HTML interface.

**Process:**
1.  **Data Injection:**
    *   Reads the enriched JSONL.
    *   Reads `05_template.html`.
    *   Injects the JSON data directly into a `<script>` tag variable (`window.REPORT_DATA`).

2.  **Vue.js Frontend:**
    *   The template uses Vue 3 (CDN) for client-side reactivity.
    *   Filtering, sorting, and searching happen instantly in the browser memory. No backend required.

3.  **Output:** `05_report.html`.

---

## 06_consequences.py: Impact Monitoring

**Goal:** Find evidence that failure to implement these rules caused *new* problems.

**Process:**
1.  **Discovery (LLM + Search):**
    *   Filters findings to only those `NOT_IMPLEMENTED` or `PARTIALLY_IMPLEMENTED`.
    *   Generates 15+ diverse search queries targeting a specific new event (e.g., "Texas winter storm 2026").
    *   Uses OpenAI's `web_search` tool to find candidate URLs.

2.  **Fetch (Hybrid Engine):**
    *   **Parallel Fetching:** Uses `asyncio` and `nodriver` (Chrome via CDP) to fetch 4 pages at a time.
    *   **Rate Limiting:** Tracks domain access times to ensure politeness.
    *   **Video Support:** Detects YouTube/Vimeo URLs and uses `yt-dlp` + `webvtt-py` to download and parse transcripts/captions instead of HTML.
    *   **HTML Extraction:** Uses `trafilatura` to convert raw HTML into clean text.

3.  **Match (LLM Analysis):**
    *   **Strict Date Filtering:** Rejects articles outside the user-defined window (e.g., "last 14 days").
    *   Feeds the article text + list of unfollowed recommendations to the LLM.
    *   Asks: "Does this article show that *Failure X* caused *Problem Y*?"
    *   Returns specific excerpts and reasoning.

4.  **Output:** `06_consequences.jsonl` and a summary JSON.

---

## 07_visualize.py: Consequence Reporting

**Goal:** Visualize the link between old failures and new disasters.

**Process:**
1.  **Data Injection:** Similar to Step 05, injects consequence data into `07_template.html`.
2.  **Output:** `07_consequences_report.html`.

---

## pipeline_utils.py: Shared Infrastructure

**Goal:** Eliminate manual file management.

**Key Components:**
*   **`ProjectContext`:** A class that takes a single input path (PDF or Directory).
    *   If PDF: Creates a slugified directory name (e.g., `cold-weather-report`).
    *   If Directory: Validates existence.
    *   **Path Resolution:** Automatically generates standard paths for every step (`paths["findings"]`, `paths["context"]`, etc.).
*   **`setup_common_args`:** Provides a standardized `argparse` setup for all scripts.
