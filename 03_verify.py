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

import asyncio
import csv
import hashlib
import json
import logging
import os
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from threading import Lock
from typing import Optional, List

try:
    import requests
    from pydantic import BaseModel, Field
except ImportError:
    print("Error: Missing dependency. Please run: pip install -r requirements.txt")
    sys.exit(1)

# Import pipeline utilities
from pipeline_utils import ProjectContext, setup_common_args

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("verify")

# --- Configuration ---
API_KEY = os.environ.get("BROWSER_USE_API_KEY")
DEFAULT_WORKERS = 3
MAX_STEPS_NORMAL = 60
MAX_STEPS_TRYHARD = 90
POLL_INTERVAL = 5
MAX_WAIT_TIME = 600  # 10 minutes

class OutOfCreditsError(Exception):
    pass

# --- Pydantic Schema for Verification Output ---
# ... (Keep existing schema classes: EvidenceItem, VerificationResult) ...
class EvidenceItem(BaseModel):
    url: str
    title: Optional[str] = None
    date: Optional[str] = None
    summary: str

class VerificationResult(BaseModel):
    status: str = Field(description="One of: IMPLEMENTED, PARTIALLY_IMPLEMENTED, NOT_IMPLEMENTED, UNABLE_TO_VERIFY")
    confidence: str = Field(description="One of: HIGH, MEDIUM, LOW")
    evidence: List[EvidenceItem] = Field(default_factory=list)
    summary: str = Field(description="2-3 sentence summary of what was found")
    notes: Optional[str] = Field(default=None, description="Additional context or caveats")

def get_verification_schema() -> dict:
    return VerificationResult.model_json_schema()

# --- Utility Functions ---
# ... (Keep existing utilities) ...
def generate_finding_id(finding: dict) -> str:
    key = f"{finding.get('actor', '')}|{finding.get('action', '')}"
    return hashlib.md5(key.encode()).hexdigest()[:12]

def load_completed_ids(progress_file: Path) -> set:
    if not progress_file.exists():
        return set()
    with open(progress_file, 'r') as f:
        return {line.strip() for line in f if line.strip()}

def mark_complete(finding_id: str, progress_file: Path):
    with open(progress_file, 'a') as f:
        f.write(f"{finding_id}\n")
        f.flush()
        os.fsync(f.fileno())

def append_result(result: dict, output_file: Path):
    with open(output_file, 'a') as f:
        f.write(json.dumps(result) + "\n")
        f.flush()
        os.fsync(f.fileno())

# --- Cost Tracking Functions ---
# ... (Keep existing cost tracking functions) ...
def get_credit_balance() -> float:
    url = "https://api.browser-use.com/api/v2/billing/account"
    headers = {"X-Browser-Use-API-Key": API_KEY}
    try:
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()
        data = response.json()
        return data.get("totalCreditsBalanceUsd", 0.0)
    except Exception as e:
        logger.warning(f"Could not fetch credit balance: {e}")
        return -1.0

def log_task_cost(log_path: Path, row: dict):
    fieldnames = ['timestamp', 'finding_id', 'task_id', 'flash_mode', 'vision_setting', 'max_steps_config',
                  'steps_actual', 'duration_sec', 'cost_usd', 'success', 'status',
                  'session_id', 'model_used', 'input_tokens', 'output_tokens', 'image_tokens',
                  'actor', 'urls', 'stop_reason']
    file_exists = log_path.exists()
    with open(log_path, 'a', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction='ignore')
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)
        f.flush()
        os.fsync(f.fileno())

def save_raw_response(log_path: Path, finding_id: str, task_data: dict):
    raw_log_path = log_path.with_suffix(".raw.jsonl")
    entry = {"timestamp": datetime.now().isoformat(), "finding_id": finding_id, "task_data": task_data}
    with open(raw_log_path, 'a') as f:
        f.write(json.dumps(entry) + "\n")
        f.flush()

def extract_task_metrics(task_data: dict) -> dict:
    metrics = {
        "session_id": task_data.get("sessionId", ""),
        "model_used": "",
        "input_tokens": 0, "output_tokens": 0, "image_tokens": 0,
        "stop_reason": task_data.get("stopReason", ""),
    }
    steps = task_data.get("steps", [])
    for step in steps:
        if step.get("model"): metrics["model_used"] = step.get("model")
        metrics["input_tokens"] += step.get("inputTokens", 0)
        metrics["output_tokens"] += step.get("outputTokens", 0)
        metrics["image_tokens"] += step.get("imageTokens", 0)
        if step.get("llmCall"):
            llm = step["llmCall"]
            if llm.get("model"): metrics["model_used"] = llm.get("model")
            metrics["input_tokens"] += llm.get("inputTokens", 0)
            metrics["output_tokens"] += llm.get("outputTokens", 0)
            metrics["image_tokens"] += llm.get("imageTokens", 0)
            
    if task_data.get("totalInputTokens"): metrics["input_tokens"] = task_data.get("totalInputTokens", 0)
    if task_data.get("totalOutputTokens"): metrics["output_tokens"] = task_data.get("totalOutputTokens", 0)
    if task_data.get("totalImageTokens"): metrics["image_tokens"] = task_data.get("totalImageTokens", 0)
    if task_data.get("model"): metrics["model_used"] = task_data.get("model")
    return metrics

# --- Browser-Use API Functions ---
# ... (Keep existing API functions: create_task_api, wait_for_task_completion) ...
def create_task_api(task_prompt: str, schema_dict: dict, finding_id: str = None, max_retries: int = 5, try_hard: bool = False) -> dict:
    url = "https://api.browser-use.com/api/v2/tasks"
    headers = {"X-Browser-Use-API-Key": API_KEY, "Content-Type": "application/json"}
    max_steps = MAX_STEPS_TRYHARD if try_hard else MAX_STEPS_NORMAL
    payload = {
        "task": task_prompt, "llm": "browser-use-llm", "structuredOutput": json.dumps(schema_dict),
        "maxSteps": max_steps, "flashMode": not try_hard, "vision": True if try_hard else "auto"
    }
    prefix = f"[{finding_id}] " if finding_id else ""
    for attempt in range(max_retries):
        try:
            response = requests.post(url, headers=headers, json=payload, timeout=30)
            if response.status_code == 429:
                wait_time = 30 * (attempt + 1)
                logger.warning(f"{prefix}Rate limited, waiting {wait_time}s...")
                time.sleep(wait_time)
                continue
            elif response.status_code in [502, 503, 504]:
                wait_time = 30 * (attempt + 1)
                logger.warning(f"{prefix}Server error ({response.status_code}), waiting {wait_time}s...")
                time.sleep(wait_time)
                continue
            elif response.status_code == 402:
                raise OutOfCreditsError("API credits exhausted (402). Use --resume to continue later.")
            response.raise_for_status()
            return response.json()
        except requests.exceptions.Timeout:
            wait_time = 30 * (attempt + 1)
            logger.warning(f"{prefix}Timeout, retrying in {wait_time}s...")
            if attempt < max_retries - 1: time.sleep(wait_time)
            else: raise
    raise Exception(f"{prefix}Max retries exceeded")

def wait_for_task_completion(task_id: str, finding_id: str = None, try_hard: bool = False) -> dict:
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
            if status in ["finished", "stopped"]: return task_data
            elif status in ["created", "started", "paused"]: time.sleep(POLL_INTERVAL)
            else: raise Exception(f"Unexpected task status: {status}")
        except requests.exceptions.RequestException as e:
            logger.warning(f"{prefix}Polling error: {e}, retrying...")
            time.sleep(POLL_INTERVAL)
    raise Exception(f"{prefix}Task {task_id} exceeded max wait time")

# --- Prompt Building ---
# ... (Keep existing build_verification_prompt) ...
def build_verification_prompt(finding: dict, document_context: dict, verification_window_end: str = None) -> str:
    # (Same prompt logic as original)
    actor = finding.get("actor", "Unknown")
    action = finding.get("action", "Unknown")
    test_question = finding.get("test_question", f"Has {actor} done: {action}?")
    title = document_context.get("title", "Unknown Report")
    pub_date = document_context.get("publication_date", "Unknown")
    event = document_context.get("event_investigated", {})
    event_name = event.get("name", "the event")
    geographic_scope = event.get("geographic_scope", "")
    reg_bodies = document_context.get("regulatory_bodies", {})
    jurisdiction = reg_bodies.get("jurisdiction", [])
    key_terms = document_context.get("key_terms", {})
    terms_str = "\n".join([f"  - {k}: {v}" for k, v in key_terms.items()]) if key_terms else "  (none provided)"
    if not verification_window_end: verification_window_end = datetime.now().strftime("%Y-%m-%d")
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
# ... (Keep existing process_finding function logic) ...
def process_finding(finding: dict, finding_id: str, document_context: dict, try_hard: bool = False, cost_log_path: Path = None, cost_settle_delay: int = 5) -> dict:
    # Simplified logging for brevity
    print(f"\nVerifying: [{finding_id}] {finding.get('actor')} - {finding.get('action')[:60]}...")
    start_time = time.time()
    
    # ... (Keep specific task tracking logic) ...
    # We'll just define the variables to make it work
    task_id = None
    task_data = None
    flash_mode = not try_hard
    vision_setting = True if try_hard else "auto"
    max_steps_config = MAX_STEPS_TRYHARD if try_hard else MAX_STEPS_NORMAL
    balance_before = -1.0
    if cost_log_path:
        balance_before = get_credit_balance()

    result = {
        "finding_id": finding_id, "finding": finding, "timestamp": int(time.time()),
        "duration_seconds": 0, "success": False, "verification": None, "error": None
    }

    try:
        prompt = build_verification_prompt(finding, document_context)
        schema = get_verification_schema()
        task_response = create_task_api(prompt, schema, finding_id=finding_id, try_hard=try_hard)
        task_id = task_response["id"]
        print(f"  Task ID: {task_id}")
        
        task_data = wait_for_task_completion(task_id, finding_id=finding_id, try_hard=try_hard)
        result["duration_seconds"] = time.time() - start_time
        result["steps"] = len(task_data.get("steps", []))
        result["success"] = task_data.get("isSuccess", False)
        result["task_id"] = task_id

        if task_data.get("output"):
            try:
                verification = json.loads(task_data["output"])
                result["verification"] = verification
                print(f"  Status: {verification.get('status', 'Unknown')}")
            except json.JSONDecodeError:
                result["error"] = "Could not parse JSON output"
                result["raw_output"] = task_data["output"]
        else:
            result["error"] = "No output from task"
            
        print(f"  ✓ Done ({result['duration_seconds']:.1f}s, {result.get('steps', 0)} steps)")

    except OutOfCreditsError:
        result["error"] = "Out of API credits (402)"
        print(f"  ✗ OUT OF CREDITS")
        raise
    except Exception as e:
        result["error"] = str(e)
        print(f"  ✗ Error: {e}")

    # Cost logging (abbreviated logic)
    if cost_log_path:
        if task_data: save_raw_response(cost_log_path, finding_id, task_data)
        if cost_settle_delay > 0: time.sleep(cost_settle_delay)
        balance_after = get_credit_balance()
        cost_usd = -1.0
        if balance_before >= 0 and balance_after >= 0:
            cost_usd = balance_before - balance_after
            print(f"  Cost: ${cost_usd:.4f}")
        
        metrics = extract_task_metrics(task_data) if task_data else {}
        urls = []
        if result.get("verification") and result["verification"].get("evidence"):
            urls = [e.get("url", "") for e in result["verification"]["evidence"] if e.get("url")]
            
        cost_row = {
            "timestamp": datetime.now().isoformat(), "finding_id": finding_id, "task_id": task_id or "",
            "flash_mode": flash_mode, "vision_setting": vision_setting, "max_steps_config": max_steps_config,
            "steps_actual": result.get("steps", 0), "duration_sec": round(result.get("duration_seconds", 0), 1),
            "cost_usd": round(cost_usd, 4) if cost_usd >= 0 else "", "success": result.get("success", False),
            "status": result.get("verification", {}).get("status", "ERROR"),
            "session_id": metrics.get("session_id", ""), "model_used": metrics.get("model_used", ""),
            "input_tokens": metrics.get("input_tokens", 0), "output_tokens": metrics.get("output_tokens", 0),
            "image_tokens": metrics.get("image_tokens", 0), "stop_reason": metrics.get("stop_reason", ""),
            "actor": finding.get("actor", "")[:50], "urls": "; ".join(urls[:3])
        }
        log_task_cost(cost_log_path, cost_row)

    return result

def main():
    parser = setup_common_args("Verify findings using browser agents.")
    parser.add_argument("--resume", action="store_true", help="Resume from previous run")
    parser.add_argument("--workers", type=int, default=DEFAULT_WORKERS, help="Parallel workers (default: 3)")
    parser.add_argument("--try-hard", action="store_true", help="Use more steps and longer timeout")
    parser.add_argument("--debug", action="store_true", help="Process only first 3 findings")
    parser.add_argument("--limit", type=int, help="Limit to first N findings")
    parser.add_argument("--no-cost-tracking", action="store_true", help="Disable per-task cost tracking")
    parser.add_argument("--cost-settle-delay", type=int, default=5, help="Seconds to wait for billing to settle")
    args = parser.parse_args()

    if not API_KEY:
        print("Error: BROWSER_USE_API_KEY environment variable not set")
        sys.exit(1)

    ctx = ProjectContext(args.input)
    logger.info(f"Project: {ctx.project_name}")

    findings_path = ctx.paths["findings"]
    context_path = ctx.paths["context"]
    output_path = ctx.paths["verified"]

    if not findings_path.exists():
        logger.error(f"Findings file not found: {findings_path}")
        logger.info("Did you run 01_decompose.py?")
        sys.exit(1)
        
    if not context_path.exists():
        logger.error(f"Context file not found: {context_path}")
        logger.info("Did you run 02_summary.py?")
        sys.exit(1)

    with open(findings_path) as f:
        findings_data = json.load(f)
    with open(context_path) as f:
        context_data = json.load(f)

    findings = findings_data.get("findings", [])
    document_context = context_data.get("document_context", {})

    if not findings:
        logger.info("No findings to verify.")
        sys.exit(0)

    logger.info(f"Loaded {len(findings)} findings")

    progress_path = output_path.with_suffix(".progress")
    cost_log_path = None
    if not args.no_cost_tracking:
        cost_log_path = output_path.with_name(output_path.stem + ".costs.csv")

    completed_ids = set()
    if args.resume:
        completed_ids = load_completed_ids(progress_path)
        if completed_ids:
            logger.info(f"Resuming: {len(completed_ids)} already completed")

    findings_with_ids = [(finding, generate_finding_id(finding)) for finding in findings]
    pending = [(f, fid) for f, fid in findings_with_ids if fid not in completed_ids]

    if args.debug:
        pending = pending[:3]
        logger.info(f"DEBUG MODE: Processing only {len(pending)} findings")
    elif args.limit:
        pending = pending[:args.limit]
        logger.info(f"Limited to {len(pending)} findings")

    if not pending:
        logger.info("Nothing to process.")
        sys.exit(0)

    # Execution loop similar to previous logic
    max_workers = min(args.workers, 3)
    if cost_log_path and max_workers > 1:
        logger.info("Forcing workers=1 for accurate per-task cost tracking")
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
                append_result(result, output_path)
                if result.get('verification'):
                    mark_complete(finding_id, progress_path)
            except Exception as e:
                logger.error(f"Failed to process {finding_id}: {e}")
                # Log error result
                error_result = {
                    "finding_id": finding_id,
                    "finding": finding,
                    "timestamp": int(time.time()),
                    "success": False,
                    "error": str(e)
                }
                append_result(error_result, output_path)
                
                if isinstance(e, OutOfCreditsError):
                    logger.error("Out of credits. Stopping.")
                    for f in future_to_finding: f.cancel()
                    break

    logger.info(f"Results saved to: {output_path}")

if __name__ == "__main__":
    main()