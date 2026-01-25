#!/usr/bin/env python3
"""Generate a visual HTML report from verification results."""

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)s: %(message)s"
)
logger = logging.getLogger(__name__)


def load_jsonl(file_path: Path) -> list[dict[str, Any]]:
    """Load data from a JSONL file."""
    data = []
    with open(file_path, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if line:
                try:
                    data.append(json.loads(line))
                except json.JSONDecodeError as e:
                    logger.warning(f"Skipping invalid JSON on line {line_num}: {e}")
    return data


def load_json(file_path: Path) -> dict[str, Any]:
    """Load data from a JSON file."""
    with open(file_path, "r", encoding="utf-8") as f:
        return json.load(f)


def flatten_context(ctx: dict) -> dict | None:
    """Flatten nested context structure for template consumption."""
    if not ctx:
        return None

    doc = ctx.get("document_context", {})
    event = doc.get("event_investigated", {})

    return {
        "title": doc.get("title"),
        "authors": doc.get("authors"),
        "publication_date": doc.get("publication_date"),
        "event_name": event.get("name"),
        "geographic_scope": event.get("geographic_scope"),
        "event_date_range": event.get("date_range"),
        "acronyms": doc.get("key_terms"),
        "summary": doc.get("summary"),
        "regulatory_bodies": doc.get("regulatory_bodies"),
    }


def find_template(template_arg: str) -> Path:
    """Find the template file, checking script directory as fallback."""
    template_path = Path(template_arg)
    if template_path.exists():
        return template_path

    # Fallback: check script's directory
    script_dir_template = Path(__file__).parent / "05_template.html"
    if script_dir_template.exists():
        return script_dir_template

    raise FileNotFoundError(f"Template file not found: {template_arg}")


def inject_data(
    template: str,
    findings: list[dict[str, Any]],
    context: dict[str, Any] | None
) -> str:
    """Inject findings and context data into the template."""
    data_placeholder = "{{REPORT_DATA_PLACEHOLDER}}"
    context_placeholder = "{{REPORT_CONTEXT_PLACEHOLDER}}"

    if data_placeholder not in template:
        raise ValueError(f"Placeholder '{data_placeholder}' not found in template")

    # Serialize data
    findings_json = json.dumps(findings, ensure_ascii=False)
    flat_context = flatten_context(context)
    context_json = json.dumps(flat_context, ensure_ascii=False) if flat_context else "null"

    result = template.replace(data_placeholder, findings_json)
    result = result.replace(context_placeholder, context_json)

    return result


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Generate a visual HTML report from verification results."
    )
    parser.add_argument("input_file", help="Path to verified findings JSONL file")
    parser.add_argument(
        "--context",
        help="Path to context JSON file (from summary.py)"
    )
    parser.add_argument(
        "--template",
        default="05_template.html",
        help="Path to HTML template (default: 05_template.html)"
    )
    parser.add_argument(
        "--output",
        help="Output HTML file path (default: <input_filename>.html)"
    )
    args = parser.parse_args()

    input_path = Path(args.input_file)
    output_path = Path(args.output) if args.output else input_path.with_suffix(".html")

    # Validate input file
    if not input_path.exists():
        logger.error(f"Input file not found: {input_path}")
        return 1

    # Find template
    try:
        template_path = find_template(args.template)
    except FileNotFoundError as e:
        logger.error(str(e))
        return 1

    # Load findings data
    logger.info(f"Reading findings from {input_path}")
    try:
        findings = load_jsonl(input_path)
    except Exception as e:
        logger.error(f"Failed to read input file: {e}")
        return 1
    logger.info(f"Loaded {len(findings)} findings")

    # Load context data (optional)
    context = None
    if args.context:
        context_path = Path(args.context)
        if not context_path.exists():
            logger.error(f"Context file not found: {context_path}")
            return 1
        try:
            logger.info(f"Reading context from {context_path}")
            context = load_json(context_path)
        except Exception as e:
            logger.error(f"Failed to read context file: {e}")
            return 1

    # Load template
    logger.info(f"Reading template from {template_path}")
    try:
        template_content = template_path.read_text(encoding="utf-8")
    except Exception as e:
        logger.error(f"Failed to read template: {e}")
        return 1

    # Generate output
    try:
        final_html = inject_data(template_content, findings, context)
    except ValueError as e:
        logger.error(str(e))
        return 1

    # Write output
    logger.info(f"Writing report to {output_path}")
    try:
        output_path.write_text(final_html, encoding="utf-8")
    except Exception as e:
        logger.error(f"Failed to write output: {e}")
        return 1

    logger.info(f"Success! Open {output_path} in your browser to view the report.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
