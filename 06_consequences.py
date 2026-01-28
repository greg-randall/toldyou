#!/usr/bin/env python3
"""
consequences.py - Find evidence that unfollowed recommendations caused real-world problems.

Takes enriched findings and searches for news articles about a specific event,
then matches articles to findings that were NOT_IMPLEMENTED or PARTIALLY_IMPLEMENTED.

Usage:
    python 06_consequences.py --findings "Report.enriched.jsonl" \\
        --context "Report_context.json" \\
        --event "texas winter storm january 2025"

    # Options
    python 06_consequences.py --findings "Report.enriched.jsonl" \\
        --context "Report_context.json" \\
        --event "texas winter storm january 2025" \\
        --days 7 \\
        --max-articles 30 \\
        --min-relevance 5 \\
        --headless \\
        --resume \\
        --debug
"""

import argparse
import asyncio
import hashlib
import json
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


def slugify(text: str, max_len: int = 50) -> str:
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
    path_slug = slugify(parsed.path, 30)
    url_hash = hashlib.md5(url.encode()).hexdigest()[:8]
    return f"{domain}_{path_slug}_{url_hash}"


# --- Video Utilities ---

def is_video_url(url: str) -> bool:
    """Check if URL is a supported video platform."""
    domain = urlparse(url).netloc.lower()
    return any(v in domain for v in ["youtube.com", "youtu.be", "vimeo.com"])

def fetch_video_transcript(url: str) -> dict:
    """Fetch video metadata and transcript using yt-dlp."""
    # Create a temp identifier for this download to avoid collisions
    temp_id = hashlib.md5(url.encode()).hexdigest()[:8]
    temp_base = Path(f"temp_vid_{temp_id}")
    
    ydl_opts = {
        'skip_download': True,      # Do not download the video file
        'writesubtitles': True,     # Write manual subtitles
        'writeautomaticsub': True,  # Write auto-generated subtitles
        'subtitleslangs': ['en'],   # English only
        'subtitlesformat': 'vtt',   # VTT format is easier to parse
        'quiet': True,
        'no_warnings': True,
        'outtmpl': str(temp_base),  # Output filename template
    }
    
    transcript_text = ""
    
    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            # download=True triggers subtitle download even if skip_download=True
            info = ydl.extract_info(url, download=True)
            
            # Extract basic metadata
            title = info.get('title', 'Unknown Video')
            upload_date = info.get('upload_date', '') # YYYYMMDD
            if upload_date and len(upload_date) == 8:
                upload_date = f"{upload_date[:4]}-{upload_date[4:6]}-{upload_date[6:]}"
            
            # Find the downloaded subtitle file (e.g. temp_vid_xyz.en.vtt)
            # yt-dlp might append lang code
            potential_files = list(Path('.').glob(f"temp_vid_{temp_id}*.vtt"))
            
            if potential_files:
                # Take the first one (manual or auto)
                sub_file = potential_files[0]
                try:
                    # Use webvtt-py to parse the file
                    # It handles deduplication of cues and rolling captions much better
                    vtt = webvtt.read(str(sub_file))
                    
                    # Extract text from cues
                    lines = []
                    seen_lines = set()
                    
                    for cue in vtt:
                        # Clean up text
                        text = cue.text.strip()
                        # Remove tags if any remain (webvtt usually handles this, but just in case)
                        text = re.sub(r'<[^>]+>', '', text)
                        
                        if text and text not in seen_lines:
                            lines.append(text)
                            seen_lines.add(text)
                            
                    transcript_text = " ".join(lines)
                finally:
                    # Cleanup
                    sub_file.unlink()
            
            # Cleanup any other potential artifacts
            for f in Path('.').glob(f"temp_vid_{temp_id}*"):
                try: f.unlink()
                except: pass

            # Construct final text
            text_parts = [f"Title: {title}", f"Description: {info.get('description', '')}"]
            
            if info.get('tags'):
                text_parts.append(f"Tags: {', '.join(info['tags'])}")
            
            if transcript_text:
                text_parts.append(f"\n--- Transcript ---\n{transcript_text}") # No more cap
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
        # Cleanup on error
        for f in Path('.').glob(f"temp_vid_{temp_id}*"):
            try: f.unlink()
            except: pass
        return {"error": str(e)}


# --- Phase A: Discovery ---
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


def slugify(text: str, max_len: int = 50) -> str:
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
    path_slug = slugify(parsed.path, 30)
    url_hash = hashlib.md5(url.encode()).hexdigest()[:8]
    return f"{domain}_{path_slug}_{url_hash}"


# --- Phase A: Discovery ---

def generate_search_queries(
    client: OpenAI,
    event_description: str,
    context: dict,
    unfollowed_findings: list[dict],
    num_queries: int = 15,
) -> list[str]:
    """
    Generate diverse search queries to find news about the event.

    Args:
        client: OpenAI client
        event_description: User-provided event description
        context: Document context with geographic scope, etc.
        unfollowed_findings: Findings with NOT_IMPLEMENTED or PARTIALLY_IMPLEMENTED status
        num_queries: Number of queries to generate

    Returns:
        List of search query strings
    """
    # Extract key actors and actions from findings
    actors_actions = []
    for f in unfollowed_findings[:10]:  # Limit to avoid huge prompts
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
2. Include queries about power outages, grid failures, blackouts
3. Include queries about specific infrastructure (generators, winterization, gas supply)
4. Include queries about regulatory responses, ERCOT, NERC, FERC
5. Include queries about deaths, impacts, costs
6. Use news-focused terms: "news", "report", specific news outlets
7. Vary query specificity from broad to narrow
8. Include date qualifiers where helpful (e.g., "January 2025")

Return ONLY a JSON array of query strings, no other text.
Example: ["texas winter storm power outage january 2025", "ercot grid failure winter 2025", ...]"""

    response = client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.7,
    )

    text = response.choices[0].message.content.strip()

    # Parse JSON array from response
    try:
        # Handle markdown code blocks
        if "```" in text:
            match = re.search(r'```(?:json)?\s*([\s\S]*?)```', text)
            if match:
                text = match.group(1).strip()
        queries = json.loads(text)
        if isinstance(queries, list):
            return queries[:num_queries]
    except json.JSONDecodeError:
        pass

    # Fallback: extract quoted strings
    queries = re.findall(r'"([^"]+)"', text)
    return queries[:num_queries] if queries else [event_description]


def execute_search(
    client: OpenAI,
    query: str,
    days_back: int = 7,
) -> list[dict]:
    """
    Execute a single web search using OpenAI Responses API.

    Args:
        client: OpenAI client
        query: Search query string
        days_back: Limit results to last N days

    Returns:
        List of dicts with url, title, snippet
    """
    # Add recency to query
    date_limit = (datetime.now() - timedelta(days=days_back)).strftime("%Y-%m-%d")
    enhanced_query = f"{query} after:{date_limit}"

    try:
        response = client.responses.create(
            model=SEARCH_MODEL,
            tools=[{"type": "web_search"}],
            input=enhanced_query,
            tool_choice="auto",
        )

        # Extract URLs from response output annotations
        # Based on pattern from 03_verify_openai.py
        results = []
        seen_urls = set()

        for item in response.output:
            # Check for annotations in content
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
        print(f"  Search error for '{query[:50]}...': {e}")
        return []


def discover_articles(
    client: OpenAI,
    event_description: str,
    context: dict,
    unfollowed_findings: list[dict],
    days_back: int = 7,
    max_articles: int = 30,
) -> list[dict]:
    """
    Phase A: Discover candidate articles via web search.

    Returns:
        List of unique article candidates with url, title, snippet
    """
    print("\n" + "=" * 60)
    print("PHASE A: DISCOVERY")
    print("=" * 60)

    # Generate search queries
    print("Generating search queries...")
    queries = generate_search_queries(
        client, event_description, context, unfollowed_findings
    )
    print(f"  Generated {len(queries)} queries")

    # Execute searches
    all_candidates = []
    seen_urls = set()

    for i, query in enumerate(queries, 1):
        print(f"  [{i}/{len(queries)}] Searching: {query[:60]}...")
        results = execute_search(client, query, days_back)
        print(f"           Found {len(results)} results")

        for r in results:
            url = r["url"]
            # Dedupe by URL
            if url not in seen_urls:
                seen_urls.add(url)
                all_candidates.append(r)

        # Stop if we have enough
        if len(all_candidates) >= max_articles * 2:
            break

    # Cap results
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
    """
    Phase B: Fetch article content using nodriver + trafilatura.

    Args:
        candidates: List of article candidates from Phase A
        articles_dir: Directory to save raw HTML and extracted text
        headless: Run browser without visible window
        completed_urls: URLs to skip (already fetched)

    Returns:
        List of articles with text content
    """
    print("\n" + "=" * 60)
    print("PHASE B: FETCH")
    print("=" * 60)

    if completed_urls is None:
        completed_urls = set()

    # Filter out already-fetched URLs
    to_fetch = [c for c in candidates if c["url"] not in completed_urls]
    print(f"  {len(to_fetch)} articles to fetch ({len(candidates) - len(to_fetch)} already cached)")

    if not to_fetch:
        # Load from cache
        return _load_cached_articles(candidates, articles_dir)

    # Create directories
    raw_dir = articles_dir / "raw"
    text_dir = articles_dir / "text"
    raw_dir.mkdir(parents=True, exist_ok=True)
    text_dir.mkdir(parents=True, exist_ok=True)

    fetched = []

    # Separate video and web candidates
    video_candidates = []
    web_candidates = []
    
    for c in to_fetch:
        if is_video_url(c["url"]):
            video_candidates.append(c)
        else:
            web_candidates.append(c)
            
    # 1. Process Video Candidates
    for i, candidate in enumerate(video_candidates, 1):
        url = candidate["url"]
        print(f"  [Video {i}/{len(video_candidates)}] Processing: {url[:70]}...")
        
        vid_data = fetch_video_transcript(url)
        
        if vid_data.get("error"):
            print(f"           Video error: {vid_data['error']}")
            # Fallback to snippet
            article = {
                "url": url,
                "title": candidate.get("title", ""),
                "source": "Video",
                "date": "",
                "text": candidate.get("snippet", ""),
                "text_path": "",
                "snippet": candidate.get("snippet", ""),
                "source_type": "snippet",
            }
        else:
            # Save transcript text
            filename = url_to_filename(url)
            text_path = text_dir / f"{filename}.txt"
            text_path.write_text(vid_data["text"], encoding="utf-8")
            
            article = {
                "url": url,
                "title": vid_data["title"],
                "source": vid_data["source"],
                "date": vid_data["date"],
                "text": vid_data["text"],
                "text_path": str(text_path),
                "snippet": candidate.get("snippet", ""),
            }
            print(f"           Extracted metadata & description")
            
        fetched.append(article)

    # Configure browser
    config = FetchConfig(
        headless=headless,
        wait_time=PAGE_WAIT,
        selector_timeout=PAGE_TIMEOUT
    )

    # 2. Process Web Candidates with Browser
    if web_candidates:
        print(f"  Starting parallel browser fetch for {len(web_candidates)} web pages (headless={headless})...")
        
        # Concurrency control
        sem = asyncio.Semaphore(4)  # Max 4 concurrent tabs
        domain_next_allowed = {}    # domain -> timestamp when next request is allowed
        
        async def fetch_single_web(candidate, index, total):
            url = candidate["url"]
            domain = urlparse(url).netloc
            filename = url_to_filename(url)
            
            # Per-domain rate limiting logic
            now = time.time()
            allowed = domain_next_allowed.get(domain, 0)
            if now < allowed:
                wait = allowed - now
                domain_next_allowed[domain] = allowed + FETCH_DELAY
            else:
                wait = 0
                domain_next_allowed[domain] = now + FETCH_DELAY
            
            if wait > 0:
                await asyncio.sleep(wait)

            async with sem:
                print(f"  [Web {index}/{total}] Fetching: {url[:60]}...")
                
                html = None
                page = None
                
                try:
                    page = await browser.get(url, new_tab=True)
                    await page.sleep(PAGE_WAIT)
                    html = await page.get_content()
                except Exception as e:
                    print(f"           Error fetching {url[:30]}...: {e}")
                finally:
                    if page:
                        try:
                            await page.close()
                        except Exception:
                            pass

                if html:
                    # Save raw HTML
                    raw_path = raw_dir / f"{filename}.html"
                    raw_path.write_text(html, encoding="utf-8")

                    # Extract text (run in thread to avoid blocking loop)
                    text = await asyncio.to_thread(
                        trafilatura.extract, 
                        html, 
                        include_comments=False, 
                        include_tables=True
                    )
                    
                    metadata = await asyncio.to_thread(trafilatura.extract_metadata, html)

                    if text:
                        # Save extracted text
                        text_path = text_dir / f"{filename}.txt"
                        text_path.write_text(text, encoding="utf-8")

                        return {
                            "url": url,
                            "title": (metadata.title if metadata else None) or candidate.get("title", ""),
                            "source": (metadata.sitename if metadata else None) or domain,
                            "date": (metadata.date if metadata else None) or "",
                            "text": text,
                            "text_path": str(text_path),
                            "snippet": candidate.get("snippet", ""),
                        }
                    else:
                        print(f"           Extraction failed for {url[:30]}... using snippet")
                else:
                    print(f"           Fetch failed for {url[:30]}... using snippet")

                # Fallback
                return {
                    "url": url,
                    "title": candidate.get("title", ""),
                    "source": domain,
                    "date": "",
                    "text": candidate.get("snippet", ""),
                    "text_path": "",
                    "snippet": candidate.get("snippet", ""),
                    "source_type": "snippet",
                }

        async with NodriverBrowser(config) as browser:
            tasks = [
                fetch_single_web(c, i, len(web_candidates)) 
                for i, c in enumerate(web_candidates, 1)
            ]
            results = await asyncio.gather(*tasks)
            fetched.extend(results)

    # Combine with cached articles
    all_articles = fetched + _load_cached_articles(
        [c for c in candidates if c["url"] in completed_urls],
        articles_dir
    )

    print(f"\nFetched {len(fetched)} articles, {len(all_articles)} total available")
    return all_articles


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
            cached.append({
                "url": url,
                "title": c.get("title", ""),
                "source": urlparse(url).netloc,
                "date": "",
                "text": text,
                "text_path": str(text_path),
                "snippet": c.get("snippet", ""),
            })

    return cached


# --- Phase C: Match ---

def is_date_in_range(date_str: str, cutoff_date: datetime) -> bool:
    """Check if article date is within the allowed range."""
    if not date_str:
        return True  # If unknown, let the LLM decide based on text
    try:
        # trafilatura usually returns YYYY-MM-DD or ISO strings
        # Extract just the date part
        match = re.search(r'(\d{4}-\d{2}-\d{2})', date_str)
        if match:
            d = datetime.strptime(match.group(1), "%Y-%m-%d")
            return d >= cutoff_date
    except Exception:
        pass
    return True


def build_match_prompt(
    article: dict,
    unfollowed_findings: list[dict],
    event_description: str,
    cutoff_date: str,
) -> str:
    """Build prompt for matching article to findings."""
    # Build findings reference
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

    # Truncate article text if too long
    article_text = article.get("text", "")[:8000]

    return f"""You are analyzing a news article to find evidence that unfollowed regulatory recommendations caused real-world problems.

CRITICAL: We are ONLY interested in consequences from the RECENT event occurring between {cutoff_date} and today. 
If this article is from a previous year or an older event, it is NOT relevant for this analysis.

EVENT BEING INVESTIGATED: {event_description}

ARTICLE:
Title: {article.get('title', 'Unknown')}
Source: {article.get('source', 'Unknown')}
Date: {article.get('date', 'Unknown')}
URL: {article.get('url', '')}

Content:
{article_text}

---

UNFOLLOWED RECOMMENDATIONS (these were NOT fully implemented):
{findings_json}

---

TASK:
1. Verify the article date. If it is BEFORE {cutoff_date}, return an empty matches array immediately.
2. Read the article carefully to find connections to the event problems.
3. For each relevant finding, assess how directly the article demonstrates consequences of not following that recommendation.

SCORING GUIDE:
- 1-3: Article mentions topic area but doesn't clearly show consequence of the specific unfollowed recommendation
- 4-6: Article shows problems that are moderately related to the unfollowed recommendation
- 7-10: Article directly demonstrates that not following the recommendation led to the problems described

IMPORTANT:
- Only include matches where the article provides meaningful evidence
- If the article is not relevant to any findings OR is from the wrong time period, return an empty matches array.

Call the match_findings function with your analysis."""


def match_article_to_findings(
    client: OpenAI,
    article: dict,
    unfollowed_findings: list[dict],
    event_description: str,
    cutoff_date: str,
    min_relevance: int = 5,
) -> list[dict]:
    """
    Match a single article to relevant findings.

    Returns:
        List of matches above min_relevance threshold
    """
    prompt = build_match_prompt(article, unfollowed_findings, event_description, cutoff_date)

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
        matches = result.get("matches", [])

        # Filter by relevance
        filtered = [m for m in matches if m.get("relevance_score", 0) >= min_relevance]
        return filtered

    except Exception as e:
        print(f"    Match error: {e}")
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
) -> list[dict]:
    """
    Phase C: Match all articles to findings.

    Returns:
        List of ConsequenceResult records
    """
    print("\n" + "=" * 60)
    print("PHASE C: MATCH")
    print("=" * 60)

    cutoff_str = cutoff_date_dt.strftime("%Y-%m-%d")
    print(f"  Strict date filter: >= {cutoff_str}")

    # Build finding lookup
    finding_lookup = {f.get("finding_id", ""): f for f in unfollowed_findings}

    all_consequences = []

    for i, article in enumerate(articles, 1):
        url = article.get("url", "")
        title = article.get("title", "Unknown")[:50]
        date_str = article.get("date", "")
        
        # Hard code-level filter
        if not is_date_in_range(date_str, cutoff_date_dt):
            print(f"  [{i}/{len(articles)}] Skipping: {title} (Date {date_str} is out of range)")
            if progress_path:
                mark_url_complete(url, progress_path)
            continue

        print(f"  [{i}/{len(articles)}] Matching: {title} ({date_str or 'unknown date'})...")

        matches = match_article_to_findings(
            client, article, unfollowed_findings, event_description, cutoff_str, min_relevance
        )

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
                    "article": {
                        "url": url,
                        "title": article.get("title", ""),
                        "source": article.get("source", ""),
                        "date": article.get("date", ""),
                        "text_path": article.get("text_path", ""),
                    },
                    "match": {
                        "relevance_score": m.get("relevance_score", 0),
                        "reasoning": m.get("reasoning", ""),
                        "excerpt": m.get("excerpt", ""),
                    },
                    "timestamp": int(time.time()),
                }
                all_consequences.append(consequence)

                # Write immediately
                if output_path:
                    append_jsonl(consequence, output_path)
        else:
            print(f"           No matches above threshold")

        # Mark URL processed
        if progress_path:
            mark_url_complete(url, progress_path)

    return all_consequences


# --- Summary ---

def write_summary(
    consequences: list[dict],
    event_description: str,
    articles_found: int,
    articles_fetched: int,
    output_path: Path,
):
    """Write summary JSON file."""
    # Count matches per finding
    finding_counts = {}
    for c in consequences:
        fid = c.get("finding_id", "")
        if fid not in finding_counts:
            finding_counts[fid] = {
                "finding_id": fid,
                "headline": c.get("finding_summary", {}).get("headline", ""),
                "article_count": 0,
            }
        finding_counts[fid]["article_count"] += 1

    # Sort by article count
    top_findings = sorted(
        finding_counts.values(),
        key=lambda x: x["article_count"],
        reverse=True
    )[:10]

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
    parser = argparse.ArgumentParser(
        description="Find evidence that unfollowed recommendations caused real-world problems"
    )
    parser.add_argument("--findings", required=True, help="Path to enriched JSONL")
    parser.add_argument("--context", required=True, help="Path to context JSON")
    parser.add_argument("--event", required=True, help="Event description to search for")
    parser.add_argument("--days", type=int, default=7, help="Lookback window in days (default: 7)")
    parser.add_argument("--max-articles", type=int, default=30, help="Max articles to fetch (default: 30)")
    parser.add_argument("--min-relevance", type=int, default=5, help="Min relevance score 1-10 (default: 5)")
    parser.add_argument("--headless", action="store_true", help="Run browser headless")
    parser.add_argument("--resume", action="store_true", help="Skip already-fetched URLs")
    parser.add_argument("--debug", action="store_true", help="Process only 5 articles")
    parser.add_argument("--output", help="Override output path for consequences.jsonl")
    args = parser.parse_args()

    # Validate inputs
    findings_path = Path(args.findings)
    context_path = Path(args.context)

    if not findings_path.exists():
        print(f"Error: Findings file not found: {findings_path}")
        sys.exit(1)
    if not context_path.exists():
        print(f"Error: Context file not found: {context_path}")
        sys.exit(1)

    # Calculate cutoff date for strict filtering
    cutoff_date = datetime.now() - timedelta(days=args.days)
    print(f"Timeframe: last {args.days} days (since {cutoff_date.strftime('%Y-%m-%d')})")

    # Load data
    print("Loading inputs...")
    findings = load_jsonl(findings_path)
    with open(context_path, "r", encoding="utf-8") as f:
        context_data = json.load(f)
    context = context_data.get("document_context", {})

    print(f"  Loaded {len(findings)} findings")
    print(f"  Event: {args.event}")

    # Filter to unfollowed findings
    unfollowed = [
        f for f in findings
        if (f.get("verification") or {}).get("status") in ("NOT_IMPLEMENTED", "PARTIALLY_IMPLEMENTED")
    ]
    print(f"  {len(unfollowed)} unfollowed findings (NOT_IMPLEMENTED or PARTIALLY_IMPLEMENTED)")

    if not unfollowed:
        print("No unfollowed findings to match against.")
        sys.exit(0)

    # Setup output paths
    if args.output:
        output_path = Path(args.output)
    else:
        event_slug = slugify(args.event, 30)
        output_path = findings_path.parent / f"{findings_path.stem}.{event_slug}.consequences.jsonl"

    progress_path = output_path.with_suffix(".progress")
    summary_path = output_path.with_name(output_path.stem + "_summary.json")
    articles_dir = output_path.parent / "articles"

    print(f"\nOutput: {output_path}")
    print(f"Articles cache: {articles_dir}")

    # Handle resume
    completed_urls = set()
    if args.resume:
        completed_urls = load_completed_urls(progress_path)
        if completed_urls:
            print(f"Resuming: {len(completed_urls)} URLs already processed")
    else:
        # Clear previous output
        if output_path.exists():
            output_path.write_text("", encoding="utf-8")
        if progress_path.exists():
            progress_path.write_text("", encoding="utf-8")

    # Initialize OpenAI client
    client = OpenAI()

    # Phase A: Discovery
    candidates = discover_articles(
        client,
        args.event,
        context,
        unfollowed,
        days_back=args.days,
        max_articles=args.max_articles,
    )

    if args.debug:
        candidates = candidates[:5]
        print(f"DEBUG MODE: Limited to {len(candidates)} articles")

    if not candidates:
        print("No articles found.")
        sys.exit(0)

    # Phase B: Fetch
    articles = asyncio.run(fetch_articles(
        candidates,
        articles_dir,
        headless=args.headless,
        completed_urls=completed_urls if args.resume else None,
    ))

    if not articles:
        print("No articles to match.")
        sys.exit(0)

    # Phase C: Match
    consequences = match_all_articles(
        client,
        articles,
        unfollowed,
        args.event,
        cutoff_date_dt=cutoff_date,
        min_relevance=args.min_relevance,
        output_path=output_path,
        progress_path=progress_path,
    )

    # Write summary
    summary = write_summary(
        consequences,
        args.event,
        len(candidates),
        len(articles),
        summary_path,
    )

    # Final report
    print("\n" + "=" * 60)
    print("CONSEQUENCES SUMMARY")
    print("=" * 60)
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


if __name__ == "__main__":
    main()
