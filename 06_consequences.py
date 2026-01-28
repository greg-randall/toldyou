#!/usr/bin/env python3
"""
consequences.py - Find evidence that unfollowed recommendations caused real-world problems.

Takes enriched findings and searches for news articles about a specific event,
then matches articles to findings that were NOT_IMPLEMENTED or PARTIALLY_IMPLEMENTED.
"""

import argparse
import asyncio
import hashlib
import json
import logging
import os
import re
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional
from urllib.parse import urlparse

import trafilatura
import webvtt
import yt_dlp
from openai import OpenAI
from pydantic import BaseModel, Field

from nodriver_helper import FetchConfig, NodriverBrowser
# Import pipeline utilities
from pipeline_utils import ProjectContext, setup_common_args

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("consequences")

# --- Configuration ---

MODEL = "gpt-4o-mini"
SEARCH_MODEL = "gpt-4o-mini"  # Model for web_search tool
FETCH_DELAY = 2.0  # Seconds between page fetches (politeness)
PAGE_WAIT = 3.0  # Seconds to wait for JS rendering
PAGE_TIMEOUT = 30.0  # Max seconds for page load

# Function calling schema for matching articles to findings
MATCH_TOOL = {
    "type": "function",
    "function": {
        "name": "match_findings",
        "description": "Match an article to relevant unfollowed findings",
        "parameters": {
            "type": "object",
            "required": ["matches"],
            "properties": {
                "matches": {
                    "type": "array",
                    "description": "List of findings that this article provides evidence for",
                    "items": {
                        "type": "object",
                        "required": ["finding_id", "relevance_score", "reasoning", "excerpt"],
                        "properties": {
                            "finding_id": {
                                "type": "string",
                                "description": "The finding_id from the input list",
                            },
                            "relevance_score": {
                                "type": "integer",
                                "minimum": 1,
                                "maximum": 10,
                                "description": "1-3: tangentially related, 4-6: moderately related, 7-10: directly demonstrates consequence of unfollowed recommendation",
                            },
                            "reasoning": {
                                "type": "string",
                                "description": "2-3 sentences explaining how this article shows consequences of not following the recommendation",
                            },
                            "excerpt": {
                                "type": "string",
                                "description": "Key quote or data point from article (max 200 chars)",
                            },
                        },
                    },
                },
            },
        },
    },
}


# --- Pydantic Schemas ---

class ArticleMatch(BaseModel):
    finding_id: str
    relevance_score: int = Field(ge=1, le=10)
    reasoning: str
    excerpt: str


class ConsequenceResult(BaseModel):
    finding_id: str
    finding_summary: dict
    article: dict
    match: dict
    timestamp: int


# --- I/O Utilities ---

def load_jsonl(path: Path) -> list[dict]:
    """Load JSONL file."""
    data = []
    if not path.exists():
        return data
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                data.append(json.loads(line))
    return data


def append_jsonl(record: dict, path: Path):
    """Append record to JSONL file."""
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")
        f.flush()
        os.fsync(f.fileno())


def load_completed_urls(progress_path: Path) -> set:
    """Load set of already-processed URLs."""
    if not progress_path.exists():
        return set()
    urls = set()
    with open(progress_path, "r") as f:
        for line in f:
            line = line.strip()
            if line:
                urls.add(line)
    return urls


def mark_url_complete(url: str, progress_path: Path):
    """Mark URL as processed."""
    with open(progress_path, "a") as f:
        f.write(url + "\n")
        f.flush()
        os.fsync(f.fileno())


def slugify_local(text: str, max_len: int = 50) -> str:
    """Convert text to filesystem-safe slug."""
    text = text.lower()
    text = re.sub(r'[^\w\s-]', '', text)
    text = re.sub(r'[\s_-]+', '-', text)
    text = text.strip('-')
    return text[:max_len]

def url_to_filename(url: str) -> str:
    """Convert URL to unique filename."""
    parsed = urlparse(url)
    domain = parsed.netloc.replace("www.", "")
    path_slug = slugify_local(parsed.path, 30)
    url_hash = hashlib.md5(url.encode()).hexdigest()[:8]
    return f"{domain}_{path_slug}_{url_hash}"


# --- Video Utilities ---

def is_video_url(url: str) -> bool:
    """Check if URL is a supported video platform."""
    domain = urlparse(url).netloc.lower()
    return any(v in domain for v in ["youtube.com", "youtu.be", "vimeo.com"])

def fetch_video_transcript(url: str) -> dict:
    """Fetch video metadata and transcript using yt-dlp."""
    temp_id = hashlib.md5(url.encode()).hexdigest()[:8]
    temp_base = Path(f"temp_vid_{temp_id}")
    
    ydl_opts = {
        'skip_download': True,
        'writesubtitles': True,
        'writeautomaticsub': True,
        'subtitleslangs': ['en'],
        'subtitlesformat': 'vtt',
        'quiet': True,
        'no_warnings': True,
        'outtmpl': str(temp_base),
    }
    
    transcript_text = ""
    
    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(url, download=True)
            
            title = info.get('title', 'Unknown Video')
            upload_date = info.get('upload_date', '')
            if upload_date and len(upload_date) == 8:
                upload_date = f"{upload_date[:4]}-{upload_date[4:6]}-{upload_date[6:]}"
            
            potential_files = list(Path('.').glob(f"temp_vid_{temp_id}*.vtt"))
            
            if potential_files:
                sub_file = potential_files[0]
                try:
                    vtt = webvtt.read(str(sub_file))
                    lines = []
                    seen_lines = set()
                    for cue in vtt:
                        text = cue.text.strip()
                        text = re.sub(r'<[^>]+>', '', text)
                        if text and text not in seen_lines:
                            lines.append(text)
                            seen_lines.add(text)
                    transcript_text = " ".join(lines)
                finally:
                    sub_file.unlink()
            
            for f in Path('.').glob(f"temp_vid_{temp_id}*"):
                try: f.unlink()
                except: pass

            text_parts = [f"Title: {title}", f"Description: {info.get('description', '')}"]
            if info.get('tags'):
                text_parts.append(f"Tags: {', '.join(info['tags'])}")
            if transcript_text:
                text_parts.append(f"\n--- Transcript ---\n{transcript_text}")
            else:
                text_parts.append("\n--- Transcript ---\n(No subtitles available)")
                
            return {
                "title": title,
                "date": upload_date,
                "text": "\n\n".join(text_parts),
                "source": "YouTube/Video",
                "error": None
            }
            
    except Exception as e:
        for f in Path('.').glob(f"temp_vid_{temp_id}*"):
            try: f.unlink()
            except: pass
        return {"error": str(e)}


# --- Debugging ---

def log_debug(debug_dir: Path, filename: str, content: str):
    """Write debug content to file."""
    if not debug_dir:
        return
    try:
        path = debug_dir / filename
        path.write_text(content, encoding="utf-8")
    except Exception as e:
        logger.warning(f"Debug log error: {e}")

# --- Phase A: Discovery ---

def generate_search_queries(
    client: OpenAI,
    event_description: str,
    context: dict,
    unfollowed_findings: list[dict],
    num_queries: int = 15,
    debug_dir: Path = None,
) -> list[str]:
    """Generate diverse search queries to find news about the event."""
    actors_actions = []
    for f in unfollowed_findings[:10]:
        finding = f.get("finding", {})
        actor = finding.get("actor", "")
        action = finding.get("action", "")[:100]
        status = f.get("verification", {}).get("status", "")
        if actor and action:
            actors_actions.append(f"- {actor}: {action} (Status: {status})")

    actors_actions_str = "\n".join(actors_actions) if actors_actions else "(none)"
    geographic = context.get("event_investigated", {}).get("geographic_scope", "")
    event_name = context.get("event_investigated", {}).get("name", "")

    prompt = f"""Generate {num_queries} diverse search queries to find recent news articles about this event.

EVENT: {event_description}

CONTEXT:
- Original event investigated: {event_name}
- Geographic scope: {geographic}

UNFOLLOWED RECOMMENDATIONS (these were NOT fully implemented):
{actors_actions_str}

QUERY GUIDELINES:
1. Include the event description + location variations
2. Include power outages, grid failures, blackouts
3. Include specific infrastructure (generators, winterization, gas supply)
4. Include regulatory responses, ERCOT, NERC, FERC
Return ONLY a JSON array of query strings."""

    if debug_dir:
        log_debug(debug_dir, "phase_a_query_prompt.txt", prompt)

    response = client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.7,
    )

    text = response.choices[0].message.content.strip()
    
    if debug_dir:
        log_debug(debug_dir, "phase_a_query_response.txt", text)

    try:
        if "```" in text:
            match = re.search(r'```(?:json)?\s*([\s\S]*?)```', text)
            if match:
                text = match.group(1).strip()
        queries = json.loads(text)
        if isinstance(queries, list):
            return queries[:num_queries]
    except json.JSONDecodeError:
        pass

    queries = re.findall(r'"([^"]+)"', text)
    return queries[:num_queries] if queries else [event_description]


def execute_search(
    client: OpenAI,
    query: str,
    days_back: int = 7,
) -> list[dict]:
    """Execute a single web search."""
    date_limit = (datetime.now() - timedelta(days=days_back)).strftime("%Y-%m-%d")
    enhanced_query = f"{query} after:{date_limit}"

    try:
        response = client.responses.create(
            model=SEARCH_MODEL,
            tools=[{"type": "web_search"}],
            input=enhanced_query,
            tool_choice="auto",
        )

        results = []
        seen_urls = set()

        for item in response.output:
            if hasattr(item, "content") and item.content:
                for content_block in item.content:
                    if hasattr(content_block, "annotations"):
                        for ann in content_block.annotations:
                            if hasattr(ann, "url") and ann.url:
                                url = ann.url
                                if url not in seen_urls:
                                    seen_urls.add(url)
                                    results.append({
                                        "url": url,
                                        "title": getattr(ann, "title", "") or "",
                                        "snippet": getattr(ann, "text", "") or "",
                                    })
        return results
    except Exception as e:
        logger.warning(f"Search error for '{query[:50]}...': {e}")
        return []


def discover_articles(
    client: OpenAI,
    event_description: str,
    context: dict,
    unfollowed_findings: list[dict],
    days_back: int = 7,
    max_articles: int = 30,
    debug_dir: Path = None,
) -> list[dict]:
    """Phase A: Discover candidate articles via web search."""
    print("\n" + "=" * 60 + "\nPHASE A: DISCOVERY\n" + "=" * 60)
    queries = generate_search_queries(client, event_description, context, unfollowed_findings, num_queries=15, debug_dir=debug_dir)
    print(f"Generated {len(queries)} queries")

    all_candidates = []
    seen_urls = set()

    for i, query in enumerate(queries, 1):
        print(f"  [{i}/{len(queries)}] Searching: {query[:60]}...")
        results = execute_search(client, query, days_back)
        for r in results:
            url = r["url"]
            if url not in seen_urls:
                seen_urls.add(url)
                all_candidates.append(r)
        if len(all_candidates) >= max_articles * 2:
            break

    candidates = all_candidates[:max_articles]
    print(f"\nDiscovered {len(candidates)} unique article candidates")
    return candidates


# --- Phase B: Fetch ---

async def fetch_articles(
    candidates: list[dict],
    articles_dir: Path,
    headless: bool = False,
    completed_urls: set = None,
) -> list[dict]:
    """Phase B: Fetch article content using nodriver + yt-dlp."""
    print("\n" + "=" * 60 + "\nPHASE B: FETCH\n" + "=" * 60)

    if completed_urls is None: completed_urls = set()
    to_fetch = [c for c in candidates if c["url"] not in completed_urls]
    print(f"  {len(to_fetch)} articles to fetch ({len(candidates) - len(to_fetch)} already cached)")

    if not to_fetch:
        return _load_cached_articles(candidates, articles_dir)

    raw_dir = articles_dir / "raw"
    text_dir = articles_dir / "text"
    raw_dir.mkdir(parents=True, exist_ok=True)
    text_dir.mkdir(parents=True, exist_ok=True)

    fetched = []
    video_candidates = []
    web_candidates = []
    
    for c in to_fetch:
        if is_video_url(c["url"]): video_candidates.append(c)
        else: web_candidates.append(c)
            
    # Process Videos
    for i, candidate in enumerate(video_candidates, 1):
        url = candidate["url"]
        print(f"  [Video {i}/{len(video_candidates)}] Processing: {url[:70]}...")
        vid_data = fetch_video_transcript(url)
        if vid_data.get("error"):
            print(f"           Video error: {vid_data['error']}")
            article = {"url": url, "title": candidate.get("title", ""), "source": "Video", "date": "", "text": candidate.get("snippet", ""), "text_path": "", "snippet": candidate.get("snippet", ""), "source_type": "snippet"}
        else:
            filename = url_to_filename(url)
            text_path = text_dir / f"{filename}.txt"
            text_path.write_text(vid_data["text"], encoding="utf-8")
            article = {"url": url, "title": vid_data["title"], "source": vid_data["source"], "date": vid_data["date"], "text": vid_data["text"], "text_path": str(text_path), "snippet": candidate.get("snippet", "")}
            print(f"           Extracted metadata & transcript")
        fetched.append(article)

    # Process Web
    if web_candidates:
        print(f"  Starting parallel browser fetch for {len(web_candidates)} web pages (headless={headless})...")
        config = FetchConfig(headless=headless, wait_time=PAGE_WAIT, selector_timeout=PAGE_TIMEOUT)
        sem = asyncio.Semaphore(4)
        domain_next_allowed = {}
        
        async def fetch_single_web(candidate, index, total, browser):
            url = candidate["url"]
            domain = urlparse(url).netloc
            filename = url_to_filename(url)
            
            now = time.time()
            allowed = domain_next_allowed.get(domain, 0)
            wait = max(0, allowed - now)
            domain_next_allowed[domain] = now + wait + FETCH_DELAY
            if wait > 0: await asyncio.sleep(wait)

            async with sem:
                print(f"  [Web {index}/{total}] Fetching: {url[:60]}...")
                html = None
                page = None
                try:
                    page = await browser.get(url, new_tab=True)
                    await page.sleep(PAGE_WAIT)
                    html = await page.get_content()
                except Exception as e:
                    logger.warning(f"Error fetching {url[:30]}...: {e}")
                finally:
                    if page:
                        try: await page.close()
                        except: pass

                if html:
                    raw_path = raw_dir / f"{filename}.html"
                    raw_path.write_text(html, encoding="utf-8")
                    text = await asyncio.to_thread(trafilatura.extract, html, include_comments=False, include_tables=True)
                    metadata = await asyncio.to_thread(trafilatura.extract_metadata, html)
                    if text:
                        text_path = text_dir / f"{filename}.txt"
                        text_path.write_text(text, encoding="utf-8")
                        return {"url": url, "title": (metadata.title if metadata else None) or candidate.get("title", ""), "source": (metadata.sitename if metadata else None) or domain, "date": (metadata.date if metadata else None) or "", "text": text, "text_path": str(text_path), "snippet": candidate.get("snippet", "")}
                return {"url": url, "title": candidate.get("title", ""), "source": domain, "date": "", "text": candidate.get("snippet", ""), "text_path": "", "snippet": candidate.get("snippet", ""), "source_type": "snippet"}

        async with NodriverBrowser(config) as browser:
            tasks = [fetch_single_web(c, i, len(web_candidates), browser) for i, c in enumerate(web_candidates, 1)]
            results = await asyncio.gather(*tasks)
            fetched.extend(results)

    cached_articles = _load_cached_articles([c for c in candidates if c["url"] in completed_urls], articles_dir)
    return fetched + cached_articles


def _load_cached_articles(candidates: list[dict], articles_dir: Path) -> list[dict]:
    """Load previously-fetched articles from cache."""
    text_dir = articles_dir / "text"
    cached = []
    for c in candidates:
        url = c["url"]
        filename = url_to_filename(url)
        text_path = text_dir / f"{filename}.txt"
        if text_path.exists():
            text = text_path.read_text(encoding="utf-8")
            cached.append({"url": url, "title": c.get("title", ""), "source": urlparse(url).netloc, "date": "", "text": text, "text_path": str(text_path), "snippet": c.get("snippet", "")})
    return cached


# --- Phase C: Match ---

def is_date_in_range(date_str: str, cutoff_date: datetime) -> bool:
    """Check if article date is within the allowed range."""
    if not date_str: return True
    try:
        match = re.search(r'(\d{4}-\d{2}-\d{2})', date_str)
        if match:
            d = datetime.strptime(match.group(1), "%Y-%m-%d")
            return d >= cutoff_date
    except Exception: pass
    return True


def build_match_prompt(
    article: dict,
    unfollowed_findings: list[dict],
    event_description: str,
    cutoff_date: str,
) -> str:
    """Build prompt for matching article to findings."""
    findings_ref = []
    for f in unfollowed_findings:
        finding = f.get("finding", {})
        enrichment = f.get("enrichment", {})
        verification = f.get("verification", {})
        findings_ref.append({
            "finding_id": f.get("finding_id", ""),
            "actor": finding.get("actor", ""),
            "action": finding.get("action", ""),
            "status": verification.get("status", ""),
            "headline": enrichment.get("headline", ""),
        })

    findings_json = json.dumps(findings_ref, indent=2)
    article_text = article.get("text", "")[:8000]

    return f"""Analyze news article for evidence that unfollowed regulatory recommendations caused problems.
CRITICAL: Only interested in consequences from the RECENT event occurring between {cutoff_date} and today. 
EVENT: {event_description}
ARTICLE:
Title: {article.get('title', 'Unknown')}
Source: {article.get('source', 'Unknown')}
Date: {article.get('date', 'Unknown')}
Content:
{article_text}
---
UNFOLLOWED RECOMMENDATIONS:
{findings_json}
---
TASK:
1. If article is BEFORE {cutoff_date}, return empty matches array.
2. Identify connections between problems and unfollowed recommendations.
Call match_findings function."""

def match_article_to_findings(
    client: OpenAI,
    article: dict,
    unfollowed_findings: list[dict],
    event_description: str,
    cutoff_date: str,
    min_relevance: int = 5,
    debug_dir: Path = None,
) -> list[dict]:
    """Match a single article to relevant findings."""
    prompt = build_match_prompt(article, unfollowed_findings, event_description, cutoff_date)
    if debug_dir:
        slug = url_to_filename(article.get("url", ""))[:50]
        log_debug(debug_dir, f"match_prompt_{slug}.txt", prompt)

    try:
        response = client.chat.completions.create(
            model=MODEL,
            messages=[{"role": "user", "content": prompt}],
            tools=[MATCH_TOOL],
            tool_choice={"type": "function", "function": {"name": "match_findings"}},
            temperature=0.0,
        )
        tool_call = response.choices[0].message.tool_calls[0]
        result = json.loads(tool_call.function.arguments)
        
        if debug_dir:
            slug = url_to_filename(article.get("url", ""))[:50]
            log_debug(debug_dir, f"match_response_{slug}.json", json.dumps(result, indent=2))

        matches = result.get("matches", [])
        return [m for m in matches if m.get("relevance_score", 0) >= min_relevance]
    except Exception as e:
        logger.warning(f"Match error: {e}")
        return []


def match_all_articles(
    client: OpenAI,
    articles: list[dict],
    unfollowed_findings: list[dict],
    event_description: str,
    cutoff_date_dt: datetime,
    min_relevance: int = 5,
    output_path: Path = None,
    progress_path: Path = None,
    debug_dir: Path = None,
) -> list[dict]:
    """Phase C: Match all articles to findings."""
    print("\n" + "=" * 60 + "\nPHASE C: MATCH\n" + "=" * 60)
    cutoff_str = cutoff_date_dt.strftime("%Y-%m-%d")
    print(f"  Strict date filter: >= {cutoff_str}")

    finding_lookup = {f.get("finding_id", ""): f for f in unfollowed_findings}
    all_consequences = []

    for i, article in enumerate(articles, 1):
        url = article.get("url", "")
        title = article.get("title", "Unknown")[:50]
        date_str = article.get("date", "")
        
        if not is_date_in_range(date_str, cutoff_date_dt):
            print(f"  [{i}/{len(articles)}] Skipping: {title} (Date {date_str} is out of range)")
            if progress_path: mark_url_complete(url, progress_path)
            continue

        print(f"  [{i}/{len(articles)}] Matching: {title} ({date_str or 'unknown date'})...")
        matches = match_article_to_findings(client, article, unfollowed_findings, event_description, cutoff_str, min_relevance, debug_dir)

        if matches:
            print(f"           Found {len(matches)} matches")
            for m in matches:
                finding_id = m.get("finding_id", "")
                finding = finding_lookup.get(finding_id, {})
                consequence = {
                    "finding_id": finding_id,
                    "finding_summary": {
                        "actor": finding.get("finding", {}).get("actor", ""),
                        "action": finding.get("finding", {}).get("action", ""),
                        "status": finding.get("verification", {}).get("status", ""),
                        "headline": finding.get("enrichment", {}).get("headline", ""),
                    },
                    "article": {"url": url, "title": article.get("title", ""), "source": article.get("source", ""), "date": article.get("date", ""), "text_path": article.get("text_path", "")},
                    "match": {"relevance_score": m.get("relevance_score", 0), "reasoning": m.get("reasoning", ""), "excerpt": m.get("excerpt", "")},
                    "timestamp": int(time.time()),
                }
                all_consequences.append(consequence)
                if output_path: append_jsonl(consequence, output_path)
        else:
            print(f"           No matches above threshold")

        if progress_path: mark_url_complete(url, progress_path)

    return all_consequences


def write_summary(
    consequences: list[dict],
    event_description: str,
    articles_found: int,
    articles_fetched: int,
    output_path: Path,
):
    """Write summary JSON file."""
    finding_counts = {}
    for c in consequences:
        fid = c.get("finding_id", "")
        if fid not in finding_counts:
            finding_counts[fid] = {"finding_id": fid, "headline": c.get("finding_summary", {}).get("headline", ""), "article_count": 0}
        finding_counts[fid]["article_count"] += 1

    top_findings = sorted(finding_counts.values(), key=lambda x: x["article_count"], reverse=True)[:10]
    summary = {
        "event_query": event_description,
        "generated_at": datetime.now().isoformat(),
        "articles_found": articles_found,
        "articles_fetched": articles_fetched,
        "total_matches": len(consequences),
        "findings_with_evidence": len(finding_counts),
        "top_findings": top_findings,
    }
    output_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


# --- Main ---

def main():
    parser = setup_common_args("Find evidence that unfollowed recommendations caused real-world problems")
    parser.add_argument("--event", required=True, help="Event description to search for")
    parser.add_argument("--days", type=int, default=7, help="Lookback window in days (default: 7)")
    parser.add_argument("--max-articles", type=int, default=30, help="Max articles to fetch (default: 30)")
    parser.add_argument("--min-relevance", type=int, default=5, help="Min relevance score 1-10 (default: 5)")
    parser.add_argument("--headless", action="store_true", help="Run browser headless")
    parser.add_argument("--resume", action="store_true", help="Skip already-fetched URLs")
    parser.add_argument("--debug", action="store_true", help="Process only 5 articles and log prompts")
    args = parser.parse_args()

    # Initialize Project Context
    ctx = ProjectContext(args.input)
    logger.info(f"Project: {ctx.project_name}")

    findings_path = ctx.paths["enriched"]
    context_path = ctx.paths["context"]
    output_path = ctx.paths["consequences"]
    progress_path = output_path.with_suffix(".progress")
    summary_path = ctx.paths["consequences_summary"]
    articles_dir = ctx.paths["articles_dir"]

    # Fallback to verified if enriched not found
    if not findings_path.exists() and ctx.paths["verified"].exists():
        logger.info(f"Enriched findings not found, falling back to verified: {ctx.paths['verified']}")
        findings_path = ctx.paths["verified"]

    if not findings_path.exists():
        logger.error(f"Input file not found: {findings_path}")
        return 1
    if not context_path.exists():
        logger.error(f"Context file not found: {context_path}")
        return 1

    cutoff_date = datetime.now() - timedelta(days=args.days)
    print(f"Timeframe: last {args.days} days (since {cutoff_date.strftime('%Y-%m-%d')})")

    # Debug setup
    debug_dir = ctx.paths["debug_dir"] if args.debug else None
    if debug_dir: ctx.ensure_debug_dir()

    # Load data
    print("Loading inputs...")
    findings = load_jsonl(findings_path)
    with open(context_path, "r", encoding="utf-8") as f:
        context_data = json.load(f)
    context = context_data.get("document_context", {})

    print(f"  Loaded {len(findings)} findings")
    unfollowed = [f for f in findings if (f.get("verification") or {}).get("status") in ("NOT_IMPLEMENTED", "PARTIALLY_IMPLEMENTED")]
    print(f"  {len(unfollowed)} unfollowed findings")

    if not unfollowed:
        print("No unfollowed findings to match against.")
        return 0

    ctx.ensure_articles_dir()

    # Handle resume
    completed_urls = set()
    if args.resume:
        completed_urls = load_completed_urls(progress_path)
        if completed_urls: print(f"Resuming: {len(completed_urls)} URLs already processed")
    else:
        if output_path.exists(): output_path.write_text("", encoding="utf-8")
        if progress_path.exists(): progress_path.write_text("", encoding="utf-8")

    client = OpenAI()

    # Phase A: Discovery
    candidates = discover_articles(client, args.event, context, unfollowed, days_back=args.days, max_articles=args.max_articles, debug_dir=debug_dir)
    if args.debug: candidates = candidates[:5]
    if not candidates:
        print("No articles found.")
        return 0

    # Phase B: Fetch
    articles = asyncio.run(fetch_articles(candidates, articles_dir, headless=args.headless, completed_urls=completed_urls if args.resume else None))
    if not articles:
        print("No articles to match.")
        return 0

    # Phase C: Match
    consequences = match_all_articles(client, articles, unfollowed, args.event, cutoff_date_dt=cutoff_date, min_relevance=args.min_relevance, output_path=output_path, progress_path=progress_path, debug_dir=debug_dir)

    # Write summary
    summary = write_summary(consequences, args.event, len(candidates), len(articles), summary_path)

    print("\n" + "=" * 60 + "\nCONSEQUENCES SUMMARY\n" + "=" * 60)
    print(f"  Event: {args.event}")
    print(f"  Articles found: {summary['articles_found']}")
    print(f"  Articles fetched: {summary['articles_fetched']}")
    print(f"  Total matches: {summary['total_matches']}")
    print(f"  Findings with evidence: {summary['findings_with_evidence']}")

    if summary["top_findings"]:
        print("\n  Top findings with evidence:")
        for tf in summary["top_findings"][:5]:
            print(f"    - {tf['headline'][:60]}... ({tf['article_count']} articles)")

    print(f"\n  Output: {output_path}")
    print(f"  Summary: {summary_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
