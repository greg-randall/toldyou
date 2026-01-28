import json
import logging
import re
from pathlib import Path
from typing import Optional, Dict, Any

try:
    import pdfplumber
    from openai import OpenAI
except ImportError:
    print("Error: Missing dependency. Please run: pip install -r requirements.txt")
    sys.exit(1)

# Import pipeline utilities
from pipeline_utils import ProjectContext, setup_common_args

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("summary")

# --- Configuration ---
MODEL = "gpt-4o"
MAX_TOKENS = 2000

SYSTEM_PROMPT = """You are an expert document analyst creating a context file for an autonomous verification agent.

YOUR TASK:
Analyze the provided introductory text of a regulatory report (title page, executive summary, acronyms list) and extract key metadata that will help a downstream agent verify the report's recommendations.

OUTPUT SCHEMA (JSON):
{
  "title": "Full report title",
  "authors": ["Authoring organizations"],
  "publication_date": "YYYY-MM-DD or YYYY-MM if day unknown",
  "event_investigated": {
    "name": "Name of the specific event (e.g. 'February 2021 Texas Winter Storm')",
    "date_range": "Dates of the event",
    "geographic_scope": "Region affected (e.g. 'ERCOT, SPP, MISO')"
  },
  "regulatory_bodies": {
    "primary": ["Bodies that authored or commissioned the report"],
    "jurisdiction": ["Bodies with authority to implement recommendations"]
  },
  "summary": "2-3 sentence summary of what happened and what the report recommends",
  "key_terms": {
    "ACRONYM": "Definition (extract important ones like GO, GOP, RC, BA, TOP)"
  }
}

IMPORTANT:
- The "key_terms" section is CRITICAL. The downstream agent needs to know that "GO" means "Generator Owner" to search effectively. Extract as many relevant domain-specific acronyms as found in the text.
- If the document is a "Joint Inquiry" or "Task Force" report, note all participating agencies in "authors".
"""

def extract_intro_text(pdf_path: Path, max_pages: int = 25) -> str:
    """Extracts text from the first N pages of the PDF."""
    logger.info(f"Extracting intro text from first {max_pages} pages of {pdf_path}...")
    full_text = []
    try:
        with pdfplumber.open(pdf_path) as pdf:
            limit = min(len(pdf.pages), max_pages)
            for i in range(limit):
                text = pdf.pages[i].extract_text()
                if text:
                    full_text.append(f"--- PAGE {i+1} ---\n{text}")
    except Exception as e:
        logger.error(f"Error reading PDF: {e}")
        return ""
    
    return "\n".join(full_text)

def generate_context(text: str, filename: str) -> Dict[str, Any]:
    """Generates the context JSON using OpenAI."""
    # ... (Keep existing prompt logic) ...
    user_prompt = f"""
DOCUMENT FILENAME: {filename}

--- BEGIN DOCUMENT TEXT (First {len(text)} chars) ---
{text[:50000]} 
--- END DOCUMENT TEXT ---
"""
    # ... (Rest of logic) ...
    try:
        client = OpenAI()
        response = client.chat.completions.create(
            model=MODEL,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.0,
            response_format={"type": "json_object"}
        )
        content = response.choices[0].message.content
        return json.loads(content)
    except Exception as e:
        logger.error(f"Error generating context: {e}")
        return {}

def main():
    parser = setup_common_args("Generate document context/metadata.")
    parser.add_argument("--pages", type=int, default=25, help="Number of pages to scan (default 25)")
    args = parser.parse_args()

    ctx = ProjectContext(args.input)
    logger.info(f"Project: {ctx.project_name}")

    input_pdf = ctx.paths["source"]
    output_path = ctx.paths["context"]

    if not input_pdf or not input_pdf.exists():
        logger.error(f"Input PDF not found: {input_pdf}")
        return

    # 1. Extract Text
    text = extract_intro_text(input_pdf, max_pages=args.pages)
    if not text:
        logger.error("No text extracted. Exiting.")
        sys.exit(1)

    # 2. Generate Context
    logger.info("Generating context with LLM...")
    context_data = generate_context(text, input_pdf.name)
    
    if not context_data:
        logger.error("Failed to generate context.")
        sys.exit(1)

    # 3. Save
    wrapper = {
        "source_file": str(input_pdf),
        "document_context": context_data
    }
    
    try:
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(wrapper, f, indent=2)
        logger.info(f"Context saved to {output_path}")
    except Exception as e:
        logger.error(f"Error saving output: {e}")

if __name__ == "__main__":
    main()

