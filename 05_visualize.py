import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any

# Import pipeline utilities
from pipeline_utils import ProjectContext, setup_common_args

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

# ... (Keep existing load functions and flatten_context) ...
def load_jsonl(file_path: Path) -> list[dict[str, Any]]:
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
    with open(file_path, "r", encoding="utf-8") as f:
        return json.load(f)

def flatten_context(ctx: dict) -> dict | None:
    if not ctx: return None
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
    template_path = Path(template_arg)
    if template_path.exists():
        return template_path
    script_dir_template = Path(__file__).parent / "05_template.html"
    if script_dir_template.exists():
        return script_dir_template
    raise FileNotFoundError(f"Template file not found: {template_arg}")

def inject_data(template: str, findings: list[dict[str, Any]], context: dict[str, Any] | None) -> str:
    data_placeholder = "{{REPORT_DATA_PLACEHOLDER}}"
    context_placeholder = "{{REPORT_CONTEXT_PLACEHOLDER}}"
    if data_placeholder not in template:
        raise ValueError(f"Placeholder '{data_placeholder}' not found in template")
    findings_json = json.dumps(findings, ensure_ascii=False)
    flat_context = flatten_context(context)
    context_json = json.dumps(flat_context, ensure_ascii=False) if flat_context else "null"
    result = template.replace(data_placeholder, findings_json)
    result = result.replace(context_placeholder, context_json)
    return result

def main() -> int:
    parser = setup_common_args("Generate a visual HTML report from verification results.")
    parser.add_argument("--template", default="05_template.html", help="Path to HTML template")
    args = parser.parse_args()

    ctx = ProjectContext(args.input)
    logger.info(f"Project: {ctx.project_name}")

    input_path = ctx.paths["enriched"]
    context_path = ctx.paths["context"]
    output_path = ctx.paths["report_html"]

    # Fallback to verified if enriched not found (allows visualization of partial progress)
    if not input_path.exists() and ctx.paths["verified"].exists():
        logger.info(f"Enriched findings not found, falling back to verified: {ctx.paths['verified']}")
        input_path = ctx.paths["verified"]

    if not input_path.exists():
        logger.error(f"Input file not found: {input_path}")
        return 1

    try:
        template_path = find_template(args.template)
    except FileNotFoundError as e:
        logger.error(str(e))
        return 1

    logger.info(f"Reading findings from {input_path}")
    try:
        findings = load_jsonl(input_path)
    except Exception as e:
        logger.error(f"Failed to read input file: {e}")
        return 1
    logger.info(f"Loaded {len(findings)} findings")

    context = None
    if context_path.exists():
        try:
            logger.info(f"Reading context from {context_path}")
            context = load_json(context_path)
        except Exception as e:
            logger.error(f"Failed to read context file: {e}")
            return 1
    else:
        logger.warning(f"Context file not found: {context_path}")

    logger.info(f"Reading template from {template_path}")
    try:
        template_content = template_path.read_text(encoding="utf-8")
    except Exception as e:
        logger.error(f"Failed to read template: {e}")
        return 1

    try:
        final_html = inject_data(template_content, findings, context)
    except ValueError as e:
        logger.error(str(e))
        return 1

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

