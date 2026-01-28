#!/usr/bin/env python3
"""Generate a visual HTML report from consequence findings."""

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


def find_template(template_arg: str) -> Path:
    """Find the template file, checking script directory as fallback."""
    template_path = Path(template_arg)
    if template_path.exists():
        return template_path

    # Fallback: check script's directory
    script_dir_template = Path(__file__).parent / "07_template.html"
    if script_dir_template.exists():
        return script_dir_template

    raise FileNotFoundError(f"Template file not found: {template_arg}")


def inject_data(
    template: str,
    consequences: list[dict[str, Any]],
    summary: dict[str, Any]
) -> str:
    """Inject consequences and summary data into the template."""
    data_placeholder = "{{CONSEQUENCES_DATA_PLACEHOLDER}}"
    summary_placeholder = "{{CONSEQUENCES_SUMMARY_PLACEHOLDER}}"

    if data_placeholder not in template:
        raise ValueError(f"Placeholder '{data_placeholder}' not found in template")
    if summary_placeholder not in template:
        raise ValueError(f"Placeholder '{summary_placeholder}' not found in template")

    # Serialize data
    consequences_json = json.dumps(consequences, ensure_ascii=False)
    summary_json = json.dumps(summary, ensure_ascii=False)

    result = template.replace(data_placeholder, consequences_json)
    result = result.replace(summary_placeholder, summary_json)

    return result


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Generate a visual HTML report from consequence findings."
    )
    parser.add_argument("input_file", help="Path to consequences JSONL file")
    parser.add_argument(
        "--summary",
        help="Path to summary JSON file (default: <input_filename>_summary.json)"
    )
    parser.add_argument(
        "--template",
        default="07_template.html",
        help="Path to HTML template (default: 07_template.html)"
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

    # Load consequences data
    logger.info(f"Reading consequences from {input_path}")
    try:
        consequences = load_jsonl(input_path)
    except Exception as e:
        logger.error(f"Failed to read input file: {e}")
        return 1
    logger.info(f"Loaded {len(consequences)} consequence matches")

    # Load summary data
    summary_path = Path(args.summary) if args.summary else input_path.with_name(input_path.stem + "_summary.json")
    if not summary_path.exists():
        logger.error(f"Summary file not found: {summary_path}")
        return 1
    try:
        logger.info(f"Reading summary from {summary_path}")
        summary = load_json(summary_path)
    except Exception as e:
        logger.error(f"Failed to read summary file: {e}")
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
        final_html = inject_data(template_content, consequences, summary)
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
