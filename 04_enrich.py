#!/usr/bin/env python3
"""
enrich.py - Enrich verified findings with LLM-generated summaries and priority scores.

Takes output from verify.py (*.verified.jsonl) and adds:
- headline: 1-line summary of the finding
- verdict: 1-line implementation status
- scores: 6-dimension scoring rubric + composite priority score
- classification: categorical labels for each dimension

Usage:
    python enrich.py "Report.verified.jsonl"
    python enrich.py "Report.verified.jsonl" --debug  # First 3 only
"""

import argparse
import asyncio
import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Optional

from openai import OpenAI
from pydantic import BaseModel, Field
from tqdm import tqdm

# Import pipeline utilities
from pipeline_utils import ProjectContext, setup_common_args

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("enrich")

# --- Configuration ---
MODEL = "gpt-4o-mini"  # Faster/cheaper for enrichment
MAX_RETRIES = 3

# ... (Keep scores and schema logic) ...
IMPLEMENTATION_GAP_SCORES = {
    "IMPLEMENTED": 1,
    "PARTIALLY_IMPLEMENTED": 5,
    "NOT_IMPLEMENTED": 9,
    "UNABLE_TO_VERIFY": 7,
}

COMPOSITE_WEIGHTS = {
    "implementation_gap": 0.25,
    "scope_impact": 0.15,
    "change_magnitude": 0.10,
    "cost_estimate": 0.15,
    "safety_impact": 0.25,
    "urgency": 0.10,
}

CLASSIFY_TOOL = {
    "type": "function",
    "function": {
        "name": "classify_finding",
        "description": "Classify and score a regulatory finding",
        "parameters": {
            "type": "object",
            "required": [
                "headline", "verdict",
                "scope_impact", "change_magnitude", "cost_estimate",
                "safety_impact", "urgency",
                "scope_level", "change_type", "cost_bracket",
                "safety_category", "deadline_status",
            ],
            "properties": {
                "headline": {
                    "type": "string",
                    "description": "Specific one-line summary: what must change, referencing concrete standards, thresholds, or components from the evidence (max 100 chars). Do NOT just restate the action.",
                },
                "verdict": {
                    "type": "string",
                    "description": "One-line status citing specific evidence: name rules/standards adopted, compliance percentages, audit findings, or enforcement dates from the verification data (max 150 chars).",
                },
                "scope_impact": {
                    "type": "integer", "minimum": 1, "maximum": 10,
                    "description": "1-3: single entity/operator, 4-6: state/regional, 7-10: national/industry-wide",
                },
                "change_magnitude": {
                    "type": "integer", "minimum": 1, "maximum": 10,
                    "description": "1-3: admin/policy, 4-6: training/procedure, 7-10: infrastructure/capital",
                },
                "cost_estimate": {
                    "type": "integer", "minimum": 1, "maximum": 10,
                    "description": "1-2: <$100K, 3-4: $100K-$1M, 5-6: $1M-$10M, 7-8: $10M-$100M, 9-10: $100M+",
                },
                "safety_impact": {
                    "type": "integer", "minimum": 1, "maximum": 10,
                    "description": "1-3: admin/compliance risk, 4-6: operational/reliability, 7-10: public safety/life",
                },
                "urgency": {
                    "type": "integer", "minimum": 1, "maximum": 10,
                    "description": "1-3: no deadline, 4-6: future 1-3yr, 7-10: overdue/imminent",
                },
                "scope_level": {
                    "type": "string",
                    "enum": [
                        "single_entity", "local", "state", "regional",
                        "national", "industry_wide", "international",
                    ],
                },
                "change_type": {
                    "type": "string",
                    "enum": [
                        "administrative", "policy", "training", "procedure",
                        "coordination", "infrastructure", "equipment", "capital",
                    ],
                },
                "cost_bracket": {
                    "type": "string",
                    "enum": ["<$100K", "$100K-$1M", "$1M-$10M", "$10M-$100M", "$100M+"],
                },
                "safety_category": {
                    "type": "string",
                    "enum": [
                        "compliance_risk", "operational_risk", "reliability_risk",
                        "public_safety_risk", "life_safety_risk",
                    ],
                },
                "deadline_status": {
                    "type": "string",
                    "enum": ["no_deadline", "long_term", "future_deadline", "imminent", "overdue"],
                },
            },
        },
    },
}

# --- I/O ---

def load_jsonl(path: Path) -> list[dict]:
    data = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                data.append(json.loads(line))
    return data


def append_jsonl(record: dict, path: Path):
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def load_completed_ids(progress_path: Path) -> set:
    if not progress_path.exists():
        return set()
    ids = set()
    with open(progress_path, "r") as f:
        for line in f:
            line = line.strip()
            if line:
                ids.add(line)
    return ids


def mark_complete(finding_id: str, progress_path: Path):
    with open(progress_path, "a") as f:
        f.write(finding_id + "\n")
        f.flush()
        os.fsync(f.fileno())


# --- Scoring ---

def get_implementation_gap(status: str) -> int:
    return IMPLEMENTATION_GAP_SCORES.get(status, 7)


def compute_composite(scores: dict) -> float:
    total = sum(scores[k] * COMPOSITE_WEIGHTS[k] for k in COMPOSITE_WEIGHTS)
    return round(total, 1)


# --- LLM Classification ---

def build_classification_prompt(record: dict) -> str:
    finding = record.get("finding", {})
    verification = record.get("verification", {})

    actor = finding.get("actor", "Unknown")
    action = finding.get("action", "Unknown")
    test_question = finding.get("test_question", "")
    v_status = verification.get("status", "UNKNOWN")
    v_summary = verification.get("summary", "")
    v_notes = verification.get("notes", "")

    evidence_summaries = ""
    for ev in verification.get("evidence", []):
        if ev.get("summary"):
            evidence_summaries += f"- {ev['summary']}\n"

    return f"""You are scoring a regulatory recommendation for priority triage.

=== FINDING ===
Actor: {actor}
Action Required: {action}
Test Question: {test_question}

=== VERIFICATION RESULT ===
Status: {v_status}
Summary: {v_summary}
Notes: {v_notes}

Evidence:
{evidence_summaries or '(none)'}

=== SCORING RUBRIC ===

Score each dimension from 1-10:

SCOPE IMPACT (breadth of who's affected):
  1-3: Single entity / individual operator
  4-6: State-level / regional group of entities
  7-10: National / industry-wide / multi-jurisdictional

CHANGE MAGNITUDE (what type of change):
  1-3: Administrative/policy/reporting changes
  4-6: Training/procedure/coordination changes
  7-10: Infrastructure/equipment/capital investment

COST ESTIMATE (rough order of magnitude to implement):
  1-2: <$100K (paperwork, minor process)
  3-4: $100K-$1M (training, software)
  5-6: $1M-$10M (equipment upgrades)
  7-8: $10M-$100M (major infrastructure)
  9-10: $100M+ (system-wide overhaul)

SAFETY IMPACT (if NOT implemented, what's at risk):
  1-3: Administrative/compliance risk only
  4-6: Operational/reliability risk
  7-10: Direct public safety/life risk

URGENCY (timeline pressure):
  1-3: No deadline / long-term aspirational
  4-6: Future deadline (1-3 years out)
  7-10: Overdue / imminent deadline / already past due

Also provide:
- headline: A SPECIFIC one-line summary. Reference concrete standards, component names, or thresholds from the evidence. Do NOT just restate the action text — add specificity from what the verification found.
- verdict: A SPECIFIC one-line implementation status. Cite evidence: name the standard/rule adopted, compliance rates, audit dates, enforcement actions, or gaps found. Avoid generic phrases like "partially implemented" without saying WHY.

Call the classify_finding function with your scores and classifications."""


def classify_finding(client: OpenAI, record: dict) -> dict:
    """Call LLM to classify a single finding. Returns enrichment dict."""
    prompt = build_classification_prompt(record)

    response = client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "user", "content": prompt}],
        tools=[CLASSIFY_TOOL],
        tool_choice={"type": "function", "function": {"name": "classify_finding"}},
        temperature=0.0,
    )

    tool_call = response.choices[0].message.tool_calls[0]
    return json.loads(tool_call.function.arguments)


# --- Main ---

def enrich_record(client: OpenAI, record: dict) -> dict:
    """Enrich a single verified record with scores and classification."""
    verification = record.get("verification") or {}
    status = verification.get("status", "UNABLE_TO_VERIFY")

    # Deterministic score
    implementation_gap = get_implementation_gap(status)

    # LLM classification
    llm_result = classify_finding(client, record)

    scores = {
        "implementation_gap": implementation_gap,
        "scope_impact": llm_result["scope_impact"],
        "change_magnitude": llm_result["change_magnitude"],
        "cost_estimate": llm_result["cost_estimate"],
        "safety_impact": llm_result["safety_impact"],
        "urgency": llm_result["urgency"],
    }
    scores["composite"] = compute_composite(scores)

    classification = {
        "scope_level": llm_result["scope_level"],
        "change_type": llm_result["change_type"],
        "cost_bracket": llm_result["cost_bracket"],
        "safety_category": llm_result["safety_category"],
        "deadline_status": llm_result["deadline_status"],
    }

    enrichment = {
        "headline": llm_result["headline"],
        "verdict": llm_result["verdict"],
        "scores": scores,
        "classification": classification,
    }

    # Return original record with enrichment added
    enriched = dict(record)
    enriched["enrichment"] = enrichment
    return enriched


def main():
    parser = setup_common_args("Enrich verified findings with LLM summaries and priority scores.")
    parser.add_argument("--resume", action="store_true", help="Resume previous run, skip already-enriched findings")
    parser.add_argument("--limit", type=int, help="Limit number of findings to process")
    parser.add_argument("--debug", action="store_true", help="Process only first 3 findings")
    args = parser.parse_args()

    ctx = ProjectContext(args.input)
    logger.info(f"Project: {ctx.project_name}")

    input_path = ctx.paths["verified"]
    output_path = ctx.paths["enriched"]

    if not input_path.exists():
        logger.error(f"Verified findings not found: {input_path}")
        logger.info("Did you run 03_verify.py?")
        sys.exit(1)

    # Load data
    records = load_jsonl(input_path)
    logger.info(f"Loaded {len(records)} findings")

    if args.debug:
        records = records[:3]
        logger.info(f"DEBUG MODE: Processing only {len(records)} findings")
    elif args.limit:
        records = records[:args.limit]
        logger.info(f"Limited to {len(records)} findings")

    if not records:
        logger.info("No findings to enrich.")
        sys.exit(0)

    # Progress tracking
    progress_path = output_path.with_suffix(".progress")

    completed_ids = set()
    if args.resume:
        completed_ids = load_completed_ids(progress_path)
        if completed_ids:
            logger.info(f"Resuming: {len(completed_ids)} already enriched")
    else:
        # Clear previous
        if output_path.exists():
            output_path.write_text("", encoding="utf-8")
        if progress_path.exists():
            progress_path.write_text("", encoding="utf-8")

    client = OpenAI()

    success = 0
    errors = 0
    skipped = 0

    for i, record in enumerate(records, 1):
        finding_id = record.get("finding_id", f"unknown-{i}")
        actor = record.get("finding", {}).get("actor", "Unknown")
        print(f"\n[{i}/{len(records)}] {finding_id}: {actor[:50]}")

        if finding_id in completed_ids:
            print(f"  Already enriched, skipping")
            skipped += 1
            continue

        if not record.get("verification"):
            print(f"  Skipping (no verification data)")
            enriched = dict(record)
            enriched["enrichment"] = None
            append_jsonl(enriched, output_path)
            continue

        try:
            enriched = enrich_record(client, record)
            append_jsonl(enriched, output_path)
            mark_complete(finding_id, progress_path)

            e = enriched["enrichment"]
            print(f"  Headline: {e['headline']}")
            print(f"  Verdict:  {e['verdict']}")
            print(f"  Score:    {e['scores']['composite']}/10")
            success += 1

        except Exception as e:
            print(f"  Error: {e}")
            errors += 1
            fallback = dict(record)
            fallback["enrichment"] = None
            append_jsonl(fallback, output_path)

    logger.info(f"Enrichment complete. Saved to {output_path}")

if __name__ == "__main__":
    main()

