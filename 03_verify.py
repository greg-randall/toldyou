#!/usr/bin/env python3
"""
verify.py - Use browser-use to verify whether recommendations were implemented.

Takes output from decompose.py (findings) and summary.py (context) and checks
whether each recommendation was actually implemented.

Usage:
    python verify.py findings.json --context context.json
    python verify.py findings.json --context context.json --resume
    python verify.py findings.json --context context.json --debug --workers 1
"""

import argparse
import csv
import hashlib
import json
import os
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from threading import Lock
from typing import Optional, List

import requests
from pydantic import BaseModel, Field

# --- Configuration ---
API_KEY = os.environ.get("BROWSER_USE_API_KEY")
DEFAULT_WORKERS = 3
MAX_STEPS_NORMAL = 60
MAX_STEPS_TRYHARD = 90
POLL_INTERVAL = 5
MAX_WAIT_TIME = 600  # 10 minutes


class OutOfCreditsError(Exception):
    """Raised when API returns 402 (out of credits)."""
    pass


# --- Pydantic Schema for Verification Output ---
class EvidenceItem(BaseModel):
    url: str
    title: Optional[str] = None
    date: Optional[str] = None
    summary: str


class VerificationResult(BaseModel):
    status: str = Field(
        description="One of: IMPLEMENTED, PARTIALLY_IMPLEMENTED, NOT_IMPLEMENTED, UNABLE_TO_VERIFY"
    )
    confidence: str = Field(
        description="One of: HIGH, MEDIUM, LOW"
    )
    evidence: List[EvidenceItem] = Field(default_factory=list)
    summary: str = Field(
        description="2-3 sentence summary of what was found"
    )
    notes: Optional[str] = Field(
        default=None,
        description="Additional context or caveats"
    )


def get_verification_schema() -> dict:
    """Get JSON schema from Pydantic model for API."""
    return VerificationResult.model_json_schema()


# --- Utility Functions ---

def generate_finding_id(finding: dict) -> str:
    """Generate a stable ID for a finding based on actor + action."""
    key = f"{finding.get('actor', '')}|{finding.get('action', '')}"
    return hashlib.md5(key.encode()).hexdigest()[:12]


def load_completed_ids(progress_file: Path) -> set:
    """Load set of already-completed finding IDs."""
    if not progress_file.exists():
        return set()
    with open(progress_file, 'r') as f:
        return {line.strip() for line in f if line.strip()}


def mark_complete(finding_id: str, progress_file: Path):
    """Mark a finding as complete."""
    with open(progress_file, 'a') as f:
        f.write(f"{finding_id}\n")
        f.flush()
        os.fsync(f.fileno())


def append_result(result: dict, output_file: Path):
    """Append a single result to JSONL file."""
    with open(output_file, 'a') as f:
        f.write(json.dumps(result) + "\n")
        f.flush()
        os.fsync(f.fileno())


# --- Cost Tracking Functions ---

def get_credit_balance() -> float:
    """Get current credit balance from Browser Use API."""
    url = "https://api.browser-use.com/api/v2/billing/account"
    headers = {"X-Browser-Use-API-Key": API_KEY}
    try:
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()
        data = response.json()
        return data.get("totalCreditsBalanceUsd", 0.0)
    except Exception as e:
        print(f"  Warning: Could not fetch credit balance: {e}")
        return -1.0  # Sentinel value indicating failure


def log_task_cost(log_path: Path, row: dict):
    """Append task cost data to CSV log."""
    fieldnames = [
        # Identifiers
        'timestamp', 'finding_id', 'task_id',
        # Config (what we requested)
        'flash_mode', 'vision_setting', 'max_steps_config',
        # Results
        'steps_actual', 'duration_sec', 'cost_usd', 'success', 'status',
        # API response metrics (if available)
        'session_id', 'model_used', 'input_tokens', 'output_tokens', 'image_tokens',
        # Extra info
        'actor', 'urls', 'stop_reason'
    ]
    file_exists = log_path.exists()
    with open(log_path, 'a', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction='ignore')
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)
        f.flush()
        os.fsync(f.fileno())


def save_raw_response(log_path: Path, finding_id: str, task_data: dict):
    """Save raw API response to JSONL for debugging."""
    raw_log_path = log_path.with_suffix(".raw.jsonl")
    entry = {
        "timestamp": datetime.now().isoformat(),
        "finding_id": finding_id,
        "task_data": task_data
    }
    with open(raw_log_path, 'a') as f:
        f.write(json.dumps(entry) + "\n")
        f.flush()


def extract_task_metrics(task_data: dict) -> dict:
    """Extract all available metrics from task API response."""
    metrics = {
        "session_id": task_data.get("sessionId", ""),
        "model_used": "",
        "input_tokens": 0,
        "output_tokens": 0,
        "image_tokens": 0,
        "stop_reason": task_data.get("stopReason", ""),
    }

    # Try to extract from steps array
    steps = task_data.get("steps", [])
    for step in steps:
        # Look for model info
        if step.get("model"):
            metrics["model_used"] = step.get("model")

        # Accumulate tokens if present
        if step.get("inputTokens"):
            metrics["input_tokens"] += step.get("inputTokens", 0)
        if step.get("outputTokens"):
            metrics["output_tokens"] += step.get("outputTokens", 0)
        if step.get("imageTokens"):
            metrics["image_tokens"] += step.get("imageTokens", 0)

        # Also check nested structures
        if step.get("llmCall"):
            llm = step["llmCall"]
            if llm.get("model"):
                metrics["model_used"] = llm.get("model")
            metrics["input_tokens"] += llm.get("inputTokens", 0)
            metrics["output_tokens"] += llm.get("outputTokens", 0)
            metrics["image_tokens"] += llm.get("imageTokens", 0)

    # Check top-level fields
    if task_data.get("totalInputTokens"):
        metrics["input_tokens"] = task_data.get("totalInputTokens", 0)
    if task_data.get("totalOutputTokens"):
        metrics["output_tokens"] = task_data.get("totalOutputTokens", 0)
    if task_data.get("totalImageTokens"):
        metrics["image_tokens"] = task_data.get("totalImageTokens", 0)
    if task_data.get("model"):
        metrics["model_used"] = task_data.get("model")

    return metrics


# --- Browser-Use API Functions ---

def create_task_api(
    task_prompt: str,
    schema_dict: dict,
    finding_id: str = None,
    max_retries: int = 5,
    try_hard: bool = False
) -> dict:
    """Create a browser-use task with retry logic."""
    url = "https://api.browser-use.com/api/v2/tasks"
    headers = {
        "X-Browser-Use-API-Key": API_KEY,
        "Content-Type": "application/json"
    }

    max_steps = MAX_STEPS_TRYHARD if try_hard else MAX_STEPS_NORMAL

    payload = {
        "task": task_prompt,
        "llm": "browser-use-llm",
        "structuredOutput": json.dumps(schema_dict),
        "maxSteps": max_steps,
        "flashMode": not try_hard,
        "vision": True if try_hard else "auto"
    }

    prefix = f"[{finding_id}] " if finding_id else ""

    for attempt in range(max_retries):
        try:
            response = requests.post(url, headers=headers, json=payload, timeout=30)

            if response.status_code == 429:
                wait_time = 30 * (attempt + 1)
                print(f"  {prefix}Rate limited, waiting {wait_time}s... (attempt {attempt + 1}/{max_retries})")
                time.sleep(wait_time)
                continue
            elif response.status_code in [502, 503, 504]:
                wait_time = 30 * (attempt + 1)
                print(f"  {prefix}Server error ({response.status_code}), waiting {wait_time}s...")
                time.sleep(wait_time)
                continue
            elif response.status_code == 402:
                raise OutOfCreditsError("API credits exhausted (402). Use --resume to continue later.")

            response.raise_for_status()
            return response.json()

        except requests.exceptions.Timeout:
            wait_time = 30 * (attempt + 1)
            print(f"  {prefix}Timeout, retrying in {wait_time}s...")
            if attempt < max_retries - 1:
                time.sleep(wait_time)
            else:
                raise

    raise Exception(f"{prefix}Max retries exceeded")


def wait_for_task_completion(
    task_id: str,
    finding_id: str = None,
    try_hard: bool = False
) -> dict:
    """Poll task until completion."""
    max_wait = int(MAX_WAIT_TIME * 1.5) if try_hard else MAX_WAIT_TIME
    url = f"https://api.browser-use.com/api/v2/tasks/{task_id}"
    headers = {"X-Browser-Use-API-Key": API_KEY}

    prefix = f"[{finding_id}] " if finding_id else ""
    start_time = time.time()

    while (time.time() - start_time) < max_wait:
        try:
            response = requests.get(url, headers=headers, timeout=30)
            response.raise_for_status()
            task_data = response.json()

            status = task_data.get("status")
            if status in ["finished", "stopped"]:
                return task_data
            elif status in ["created", "started", "paused"]:
                time.sleep(POLL_INTERVAL)
            else:
                raise Exception(f"Unexpected task status: {status}")

        except requests.exceptions.RequestException as e:
            print(f"  {prefix}Polling error: {e}, retrying...")
            time.sleep(POLL_INTERVAL)

    raise Exception(f"{prefix}Task {task_id} exceeded max wait time of {max_wait}s")


# --- Prompt Building ---

def build_verification_prompt(
    finding: dict,
    document_context: dict,
    verification_window_end: str = None
) -> str:
    """
    Build a prompt for browser-use to verify a single recommendation.
    """
    actor = finding.get("actor", "Unknown")
    action = finding.get("action", "Unknown")
    test_question = finding.get("test_question", f"Has {actor} done: {action}?")

    # Extract context
    title = document_context.get("title", "Unknown Report")
    pub_date = document_context.get("publication_date", "Unknown")
    event = document_context.get("event_investigated", {})
    event_name = event.get("name", "the event")
    geographic_scope = event.get("geographic_scope", "")

    reg_bodies = document_context.get("regulatory_bodies", {})
    jurisdiction = reg_bodies.get("jurisdiction", [])

    key_terms = document_context.get("key_terms", {})
    terms_str = "\n".join([f"  - {k}: {v}" for k, v in key_terms.items()]) if key_terms else "  (none provided)"

    # Default verification window: from report date to now
    if not verification_window_end:
        verification_window_end = datetime.now().strftime("%Y-%m-%d")

    today = datetime.now().strftime("%B %d, %Y")

    prompt = f"""You are verifying whether a regulatory recommendation was ACTUALLY implemented in practice—not just on paper.

=== CONTEXT ===
Report: {title}
Published: {pub_date}
Event Investigated: {event_name}
Geographic Scope: {geographic_scope}
Regulatory Bodies with Jurisdiction: {', '.join(jurisdiction) if jurisdiction else 'Not specified'}

Key Terms/Acronyms:
{terms_str}

=== RECOMMENDATION TO VERIFY ===
Actor: {actor}
Action Required: {action}
Test Question: {test_question}

=== CRITICAL DISTINCTION ===
You must distinguish between TWO levels of implementation:

LEVEL 1 - REGULATORY/PAPER ACTION:
- A rule, standard, or law was written
- A policy was adopted
- A requirement is "in effect" on paper
- An agency issued an order

LEVEL 2 - ACTUAL IMPLEMENTATION:
- Entities actually DID the required work
- Compliance has been audited/verified
- Enforcement actions for non-compliance exist (proves monitoring)
- Real-world performance data shows it worked
- Physical changes were made (equipment installed, procedures changed)

A standard being "approved" or "in effect" is NOT the same as it being implemented.
A rule requiring action is NOT evidence the action was taken.

=== SEARCH STRATEGY ===

1. First, find regulatory actions (Level 1):
   - Official rules, standards, orders, legislation
   - This tells you what SHOULD have happened

2. Then, critically, search for evidence of actual compliance (Level 2):
   - NERC or FERC compliance monitoring reports
   - Audit results or spot-check findings
   - Enforcement actions, violations, or penalties (these PROVE monitoring exists)
   - Industry surveys on compliance rates
   - Company-specific announcements of completed projects
   - Performance during SUBSEQUENT weather events (e.g., Winter Storm Elliott 2022, Winter Storm Heather 2024)
   - News coverage of compliance gaps or ongoing problems
   - Inspector general or GAO reports on implementation status

3. Be skeptical:
   - Search for "[topic] compliance problems" or "[topic] violations"
   - Search for "[topic] still not implemented" or "[topic] delays"
   - Check if there have been cold weather events AFTER the rules took effect—did the fixes work?

=== STATUS DEFINITIONS ===

IMPLEMENTED: 
- Level 1 AND Level 2 evidence exists
- Rules were created AND there's evidence of actual compliance/enforcement
- OR real-world performance shows the fix is working

PARTIALLY_IMPLEMENTED:
- Level 1 exists but Level 2 is incomplete or mixed
- Rules exist but compliance is incomplete, delayed, or unenforced
- Some entities complied but not all
- Rule passed but compliance deadline hasn't arrived yet

NOT_IMPLEMENTED:
- Little or no Level 1 action, OR
- Level 1 exists but clear evidence Level 2 hasn't happened
- Rules exist on paper but are widely ignored or unenforced

UNABLE_TO_VERIFY:
- Cannot find sufficient evidence either way for Level 2
- Note: Finding a rule exists is NOT enough—if you can't find compliance evidence, use this status

=== IMPORTANT ===
- Today is {today}. Look for actions taken AFTER {pub_date}.
- Be specific about what you find. Cite actual rules, dates, and sources.
- In your summary, clearly state what Level 1 and Level 2 evidence you found (or didn't find).
- If you only found that a rule exists but no evidence anyone followed it, say so explicitly.

=== OUTPUT ===
Provide your findings as JSON with:
- status: IMPLEMENTED | PARTIALLY_IMPLEMENTED | NOT_IMPLEMENTED | UNABLE_TO_VERIFY
- confidence: HIGH | MEDIUM | LOW  
- evidence: Array of sources found, each with url, title, date (if known), and summary
- summary: 2-3 sentence summary clearly stating what Level 1 (regulatory) and Level 2 (actual compliance) evidence was found
- notes: Any caveats, especially if you only found Level 1 evidence
"""
    return prompt


# --- Main Processing ---

def process_finding(
    finding: dict,
    finding_id: str,
    document_context: dict,
    try_hard: bool = False,
    cost_log_path: Path = None,
    cost_settle_delay: int = 5
) -> dict:
    """Process a single finding and return verification result."""
    print(f"\n{'='*60}")
    print(f"Verifying: [{finding_id}]")
    print(f"  Actor: {finding.get('actor', 'Unknown')}")
    print(f"  Action: {finding.get('action', 'Unknown')[:60]}...")
    print(f"{'='*60}")

    start_time = time.time()
    task_id = None
    task_data = None

    # Track config settings
    flash_mode = not try_hard
    vision_setting = True if try_hard else "auto"
    max_steps_config = MAX_STEPS_TRYHARD if try_hard else MAX_STEPS_NORMAL

    # Get credit balance before task
    balance_before = -1.0
    if cost_log_path:
        balance_before = get_credit_balance()
        if balance_before >= 0:
            print(f"  Credit balance: ${balance_before:.4f}")

    result = {
        "finding_id": finding_id,
        "finding": finding,
        "timestamp": int(time.time()),
        "duration_seconds": 0,
        "success": False,
        "verification": None,
        "error": None
    }

    try:
        prompt = build_verification_prompt(finding, document_context)
        schema = get_verification_schema()

        # Create task
        task_response = create_task_api(
            prompt, schema,
            finding_id=finding_id,
            try_hard=try_hard
        )
        task_id = task_response["id"]
        print(f"  Task ID: {task_id}")
        print(f"  Config: flash={flash_mode}, vision={vision_setting}, max_steps={max_steps_config}")
        print(f"  Waiting for completion...")

        # Wait for completion
        task_data = wait_for_task_completion(task_id, finding_id=finding_id, try_hard=try_hard)

        result["duration_seconds"] = time.time() - start_time
        result["steps"] = len(task_data.get("steps", []))
        result["success"] = task_data.get("isSuccess", False)
        result["task_id"] = task_id

        # Extract output
        if task_data.get("output"):
            try:
                verification = json.loads(task_data["output"])
                result["verification"] = verification
                print(f"  Status: {verification.get('status', 'Unknown')}")
                print(f"  Confidence: {verification.get('confidence', 'Unknown')}")
            except json.JSONDecodeError:
                # Try to extract JSON from text
                match = re.search(r'\{[\s\S]*\}', task_data["output"])
                if match:
                    try:
                        verification = json.loads(match.group())
                        result["verification"] = verification
                    except json.JSONDecodeError:
                        result["error"] = "Could not parse JSON output"
                        result["raw_output"] = task_data["output"]
                else:
                    result["error"] = "No JSON in output"
                    result["raw_output"] = task_data["output"]
        else:
            result["error"] = "No output from task"

        # Extract and print metrics from API response
        metrics = extract_task_metrics(task_data)
        if metrics.get("model_used"):
            print(f"  Model: {metrics['model_used']}")
        if metrics.get("input_tokens") or metrics.get("output_tokens"):
            print(f"  Tokens: in={metrics['input_tokens']}, out={metrics['output_tokens']}, img={metrics['image_tokens']}")
        if metrics.get("session_id"):
            print(f"  Session: {metrics['session_id']}")

        print(f"  ✓ Done ({result['duration_seconds']:.1f}s, {result.get('steps', 0)} steps)")

    except OutOfCreditsError:
        result["duration_seconds"] = time.time() - start_time
        result["error"] = "Out of API credits (402)"
        print(f"  ✗ OUT OF CREDITS")
        raise  # Re-raise to stop processing
    except Exception as e:
        result["duration_seconds"] = time.time() - start_time
        result["error"] = str(e)
        print(f"  ✗ Error: {e}")

    # Log cost to CSV
    if cost_log_path:
        # Save raw API response for debugging
        if task_data:
            save_raw_response(cost_log_path, finding_id, task_data)

        # Wait for billing to settle before checking balance
        if cost_settle_delay > 0:
            print(f"  Waiting {cost_settle_delay}s for billing to settle...")
            time.sleep(cost_settle_delay)

        balance_after = get_credit_balance()
        cost_usd = -1.0
        if balance_before >= 0 and balance_after >= 0:
            cost_usd = balance_before - balance_after
            print(f"  Cost: ${cost_usd:.4f}")

        # Extract URLs from evidence
        urls = []
        if result.get("verification") and result["verification"].get("evidence"):
            urls = [e.get("url", "") for e in result["verification"]["evidence"] if e.get("url")]

        # Extract metrics from API response
        metrics = extract_task_metrics(task_data) if task_data else {}

        cost_row = {
            "timestamp": datetime.now().isoformat(),
            "finding_id": finding_id,
            "task_id": task_id or "",
            # Config (what we requested)
            "flash_mode": flash_mode,
            "vision_setting": vision_setting,
            "max_steps_config": max_steps_config,
            # Results
            "steps_actual": result.get("steps", 0),
            "duration_sec": round(result["duration_seconds"], 1),
            "cost_usd": round(cost_usd, 4) if cost_usd >= 0 else "",
            "success": result.get("success", False),
            "status": result.get("verification", {}).get("status", "ERROR"),
            # API response metrics
            "session_id": metrics.get("session_id", ""),
            "model_used": metrics.get("model_used", ""),
            "input_tokens": metrics.get("input_tokens", 0),
            "output_tokens": metrics.get("output_tokens", 0),
            "image_tokens": metrics.get("image_tokens", 0),
            "stop_reason": metrics.get("stop_reason", ""),
            # Extra
            "actor": finding.get("actor", "")[:50],
            "urls": "; ".join(urls[:3])
        }
        log_task_cost(cost_log_path, cost_row)

    return result


def main():
    parser = argparse.ArgumentParser(
        description="Verify whether recommendations were implemented using browser-use"
    )
    parser.add_argument("findings_file", help="Path to findings JSON from decompose.py")
    parser.add_argument("--context", required=True, help="Path to context JSON from summary.py")
    parser.add_argument("--output", help="Output JSONL file (default: <findings>_verified.jsonl)")
    parser.add_argument("--resume", action="store_true", help="Resume from previous run")
    parser.add_argument("--workers", type=int, default=DEFAULT_WORKERS, help="Parallel workers (default: 3)")
    parser.add_argument("--try-hard", action="store_true", help="Use more steps and longer timeout")
    parser.add_argument("--debug", action="store_true", help="Process only first 3 findings")
    parser.add_argument("--limit", type=int, help="Limit to first N findings")
    parser.add_argument("--no-cost-tracking", action="store_true",
                        help="Disable per-task cost tracking (allows parallel workers)")
    parser.add_argument("--cost-settle-delay", type=int, default=5,
                        help="Seconds to wait after task for billing to settle (default: 5, try 30 if costs look wrong)")
    args = parser.parse_args()

    if not API_KEY:
        print("Error: BROWSER_USE_API_KEY environment variable not set")
        sys.exit(1)

    # Load files
    findings_path = Path(args.findings_file)
    context_path = Path(args.context)

    if not findings_path.exists():
        print(f"Error: Findings file not found: {findings_path}")
        sys.exit(1)
    if not context_path.exists():
        print(f"Error: Context file not found: {context_path}")
        sys.exit(1)

    with open(findings_path) as f:
        findings_data = json.load(f)
    with open(context_path) as f:
        context_data = json.load(f)

    findings = findings_data.get("findings", [])
    document_context = context_data.get("document_context", {})

    if not findings:
        print("No findings to verify.")
        sys.exit(0)

    print(f"Loaded {len(findings)} findings from {findings_path.name}")
    print(f"Document: {document_context.get('title', 'Unknown')}")

    # Setup output paths
    if args.output:
        output_path = Path(args.output)
    else:
        output_path = findings_path.with_suffix(".verified.jsonl")

    progress_path = output_path.with_suffix(".progress")
    cost_log_path = None
    if not args.no_cost_tracking:
        cost_log_path = output_path.with_name(output_path.stem + ".costs.csv")

    # Handle resume
    completed_ids = set()
    if args.resume:
        completed_ids = load_completed_ids(progress_path)
        if completed_ids:
            print(f"Resuming: {len(completed_ids)} already completed")

    # Generate IDs and filter
    findings_with_ids = [
        (finding, generate_finding_id(finding))
        for finding in findings
    ]

    # Filter out completed
    pending = [
        (f, fid) for f, fid in findings_with_ids
        if fid not in completed_ids
    ]

    # Apply limits
    if args.debug:
        pending = pending[:3]
        print(f"DEBUG MODE: Processing only {len(pending)} findings")
    elif args.limit:
        pending = pending[:args.limit]
        print(f"Limited to {len(pending)} findings")

    if not pending:
        print("Nothing to process.")
        sys.exit(0)

    print(f"\nProcessing {len(pending)} findings...")
    print(f"Output: {output_path}")
    if cost_log_path:
        print(f"Cost log: {cost_log_path}")
        print(f"Cost settle delay: {args.cost_settle_delay}s (use --cost-settle-delay 30 if costs look wrong)")
    if args.try_hard:
        print("TRY-HARD MODE: Extended steps and timeout")

    # Determine worker count
    max_workers = min(args.workers, 3)  # Cap at 3 for rate limits

    # Force single-threaded for accurate cost tracking
    # (parallel workers would cause balance checks to overlap)
    if cost_log_path and max_workers > 1:
        print(f"NOTE: Forcing workers=1 for accurate per-task cost tracking")
        print(f"      Use --no-cost-tracking to allow {args.workers} parallel workers")
        max_workers = 1
    all_results = []

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_finding = {
            executor.submit(
                process_finding,
                finding,
                finding_id,
                document_context,
                args.try_hard,
                cost_log_path,
                args.cost_settle_delay
            ): (finding, finding_id)
            for finding, finding_id in pending
        }

        for i, future in enumerate(as_completed(future_to_finding), 1):
            finding, finding_id = future_to_finding[future]

            try:
                result = future.result()
                all_results.append(result)

                # Write immediately
                append_result(result, output_path)
                # Only mark complete if verification succeeded
                if result.get('verification'):
                    mark_complete(finding_id, progress_path)

                status = result.get("verification", {}).get("status", "ERROR")
                print(f"\n[{i}/{len(pending)}] {finding_id}: {status}")

            except Exception as e:
                print(f"\n[{i}/{len(pending)}] {finding_id}: EXCEPTION - {e}")
                error_result = {
                    "finding_id": finding_id,
                    "finding": finding,
                    "timestamp": int(time.time()),
                    "success": False,
                    "error": str(e)
                }
                append_result(error_result, output_path)
                all_results.append(error_result)

                # Out-of-credits: exit gracefully so --resume works
                if isinstance(e, OutOfCreditsError):
                    print(f"\n*** OUT OF CREDITS - Stopping. Use --resume to continue later. ***")
                    # Cancel pending futures
                    for f in future_to_finding:
                        f.cancel()
                    break

    # Summary
    print(f"\n{'='*60}")
    print("VERIFICATION SUMMARY")
    print(f"{'='*60}")

    status_counts = {}
    for r in all_results:
        v = r.get("verification", {})
        status = v.get("status", "ERROR" if r.get("error") else "UNKNOWN")
        status_counts[status] = status_counts.get(status, 0) + 1

    for status, count in sorted(status_counts.items()):
        pct = count / len(all_results) * 100
        print(f"  {status}: {count} ({pct:.1f}%)")

    success_count = sum(1 for r in all_results if r.get("success"))
    print(f"\nSuccessful API calls: {success_count}/{len(all_results)}")
    print(f"Results saved to: {output_path}")

    # Cost summary from CSV
    if cost_log_path and cost_log_path.exists():
        try:
            total_cost = 0.0
            task_count = 0
            with open(cost_log_path, 'r') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    if row.get('cost_usd') and row['cost_usd'] != '':
                        total_cost += float(row['cost_usd'])
                        task_count += 1
            if task_count > 0:
                avg_cost = total_cost / task_count
                print(f"\n{'='*60}")
                print("COST SUMMARY")
                print(f"{'='*60}")
                print(f"  Tasks logged: {task_count}")
                print(f"  Total cost: ${total_cost:.4f}")
                print(f"  Average cost per task: ${avg_cost:.4f}")
                print(f"  Cost log: {cost_log_path}")
        except Exception as e:
            print(f"  Warning: Could not read cost log: {e}")


if __name__ == "__main__":
    main()