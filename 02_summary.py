import argparse
import json
import sys
from pathlib import Path
from typing import Optional

try:
    import pdfplumber
    from openai import OpenAI
except ImportError as e:
    print(f"Error: Missing dependency. {e}")
    print("Please run: pip install pdfplumber openai")
    sys.exit(1)

client = OpenAI()

SYSTEM_PROMPT = """You are analyzing a regulatory/investigation report. Extract structured metadata from the opening pages.

OUTPUT SCHEMA (JSON):
{
  "title": "Full report title",
  "authors": ["Authoring organizations"],
  "publication_date": "YYYY-MM-DD or YYYY-MM if day unknown",
  "event_investigated": {
    "name": "Common name for the event (e.g., 'February 2021 Texas Winter Storm')",
    "date_range": "YYYY-MM-DD to YYYY-MM-DD",
    "geographic_scope": "Affected regions/states"
  },
  "regulatory_bodies": {
    "primary": ["Bodies that authored or commissioned the report"],
    "jurisdiction": ["Bodies with authority to implement recommendations"]
  },
  "summary": "2-3 sentence summary of what happened and what the report recommends",
  "key_terms": {
    "ACRONYM": "Definition"
  }
}

RULES:
1. Extract only what is explicitly stated. Do not infer or fabricate.
2. For key_terms, include acronyms and domain-specific terms that appear frequently 
   (e.g., "GOs" = "Generator Owners", "BA" = "Balancing Authority", "BES" = "Bulk Electric System")
3. If a field cannot be determined from the text, use null.
4. For date_range, use the actual event dates, not the report publication date.
5. geographic_scope should list specific regions, states, or grid operators affected.
6. In the summary, briefly describe: (a) what happened, (b) the main causes identified, 
   (c) the thrust of the recommendations."""


def extract_pages(pdf_path: Path, num_pages: int = 25) -> str:
    """
    Extract text from the first N pages of a PDF.
    Returns concatenated text with page markers.
    """
    pages_text = []
    print(f"Extracting first {num_pages} pages from {pdf_path}...")
    
    try:
        with pdfplumber.open(pdf_path) as pdf:
            total_pages = len(pdf.pages)
            pages_to_extract = min(num_pages, total_pages)
            
            for i in range(pages_to_extract):
                page = pdf.pages[i]
                text = page.extract_text() or ""
                formatted_text = f"\n--- PAGE {i + 1} ---\n{text}"
                pages_text.append(formatted_text)
                
    except Exception as e:
        print(f"Error reading PDF: {e}")
        sys.exit(1)
    
    return "\n".join(pages_text)


def generate_summary(text: str, model: str = "gpt-4o") -> Optional[dict]:
    """
    Send extracted text to LLM and get structured document context.
    """
    print(f"Generating document summary with {model}...")
    
    user_prompt = f"""Analyze the following pages from a report and extract the document metadata.

--- TEXT (Opening Pages) ---
{text}
"""

    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.0,
            response_format={"type": "json_object"}
        )
        
        content = response.choices[0].message.content
        if not content:
            print("Error: Empty response from API")
            return None
            
        return json.loads(content)
        
    except Exception as e:
        print(f"Error calling API: {e}")
        return None


def main():
    parser = argparse.ArgumentParser(
        description="Extract document context/metadata from a regulatory report PDF."
    )
    parser.add_argument("input_file", help="Path to PDF")
    parser.add_argument(
        "--pages", 
        type=int, 
        default=25, 
        help="Number of pages to extract (default: 25)"
    )
    parser.add_argument(
        "--model",
        default="gpt-4o",
        help="Model to use (default: gpt-4o)"
    )
    parser.add_argument(
        "--output",
        help="Output file path (default: <input>_context.json)"
    )
    args = parser.parse_args()

    input_path = Path(args.input_file)
    if not input_path.exists():
        print(f"Error: File {input_path} not found.")
        sys.exit(1)
    
    # Default output path
    if args.output:
        output_path = Path(args.output)
    else:
        output_path = input_path.with_name(f"{input_path.stem}_context.json")

    print(f"--- Extracting context from {input_path.name} ---")
    
    # 1. Extract text
    text = extract_pages(input_path, num_pages=args.pages)
    if not text.strip():
        print("No text extracted. Exiting.")
        sys.exit(1)
    
    # 2. Generate summary
    context = generate_summary(text, model=args.model)
    if not context:
        print("Failed to generate context. Exiting.")
        sys.exit(1)
    
    # 3. Add source metadata
    output = {
        "source_file": str(args.input_file),
        "pages_analyzed": args.pages,
        "document_context": context
    }
    
    # 4. Save
    try:
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(output, f, indent=2)
        print(f"Success! Saved document context to {output_path}")
    except Exception as e:
        print(f"Error saving output: {e}")
        sys.exit(1)
    
    # 5. Print summary to console
    print("\n--- Document Summary ---")
    print(f"Title: {context.get('title', 'Unknown')}")
    print(f"Authors: {', '.join(context.get('authors', []))}")
    print(f"Date: {context.get('publication_date', 'Unknown')}")
    if context.get('event_investigated'):
        event = context['event_investigated']
        print(f"Event: {event.get('name', 'Unknown')}")
        print(f"  Date Range: {event.get('date_range', 'Unknown')}")
        print(f"  Scope: {event.get('geographic_scope', 'Unknown')}")
    print(f"\nSummary: {context.get('summary', 'N/A')}")


if __name__ == "__main__":
    main()
