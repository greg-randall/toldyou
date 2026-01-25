import argparse
import json
import os
import re
import sys
from pathlib import Path
from typing import List, Dict, Any

try:
    import pdfplumber
    from tqdm import tqdm
    from openai import OpenAI
except ImportError as e:
    print(f"Error: Missing dependency. {e}")
    print("Please run: pip install -r requirements.txt")
    sys.exit(1)

# Initialize OpenAI Client
# Expects OPENAI_API_KEY environment variable to be set
try:
    client = OpenAI()
except Exception as e:
    # We'll handle this gracefully if the call fails later, 
    # but initializing the client without a key might warn or fail depending on version.
    # Usually it's fine until a request is made if the env var is missing.
    pass

def extract_pages(pdf_path: Path, limit: int = None) -> List[str]:
    """
    Opens PDF and returns list of strings with page markers.
    """
    pages_text = []
    print(f"Extracting text from {pdf_path}...")
    try:
        with pdfplumber.open(pdf_path) as pdf:
            # Determine how many pages to process
            total_pages = len(pdf.pages)
            if limit:
                num_pages_to_process = min(limit, total_pages)
            else:
                num_pages_to_process = total_pages
            
            for i in range(num_pages_to_process):
                page = pdf.pages[i]
                text = page.extract_text() or ""
                # Format: "\n--- PAGE {n} ---" (using 1-based index for display)
                formatted_text = f"\n--- PAGE {i + 1} ---\n{text}"
                pages_text.append(formatted_text)
                
    except Exception as e:
        print(f"Error reading PDF: {e}")
        sys.exit(1)
        
    return pages_text

def create_chunks(pages: List[str]) -> List[Dict[str, Any]]:
    """
    Generates overlapping windows of 3 pages, stride 1.
    If fewer than 3 pages, creates a single chunk.
    """
    chunks = []
    window_size = 3
    stride = 1
    
    if len(pages) == 0:
        return chunks
        
    if len(pages) < window_size:
        # If we have fewer pages than the window size, just take all of them as one chunk
        chunk_text = "\n".join(pages)
        chunks.append({
            "text": chunk_text,
            "page_range": f"1-{len(pages)}"
        })
        return chunks

    # Sliding window
    for i in range(0, len(pages) - window_size + 1, stride):
        window = pages[i : i + window_size]
        chunk_text = "\n".join(window)
        
        # Calculate page range for metadata (1-based)
        start_page = i + 1
        end_page = i + window_size
        
        chunks.append({
            "text": chunk_text,
            "page_range": f"{start_page}-{end_page}"
        })
        
    # Handle any remaining pages if the stride/window logic missed the tail?
    # With stride 1 and loop range `len(pages) - window_size + 1`, we stop when the window hits the end.
    # Example: 5 pages, window 3.
    # i=0: [0,1,2] (Pages 1-3)
    # i=1: [1,2,3] (Pages 2-4)
    # i=2: [2,3,4] (Pages 3-5)
    # i=3: Loop terminates (3 > 5-3). Correct.
    
    return chunks

def analyze_chunk(text: str, previous_context: str, page_range: str, debug_dir: Path = None) -> List[Dict[str, Any]]:
    """
    Calls GPT-4o-mini to extract findings.
    """
    
    system_prompt = """You are a policy analyst extracting actionable recommendations from official reports.

YOUR TASK:
Identify specific, forward-looking recommendations, mandates, or directives where an entity is told what they SHOULD DO in the future. These will later be used to verify whether the recommended actions were taken.

OUTPUT SCHEMA (JSON):
{
  "findings": [
    {
      "actor": "The entity responsible for taking action",
      "action": "The specific action they are directed to take",
      "test_question": "A yes/no question to verify compliance at a future date",
      "citation_page": "Page number from the --- PAGE X --- header, page number integer only"
    }
  ]
}

═══════════════════════════════════════════════════════════════════════════════
CRITICAL DISTINCTION: NARRATIVE vs. RECOMMENDATION
═══════════════════════════════════════════════════════════════════════════════

NARRATIVE (DO NOT EXTRACT):
- Describes what happened, what failed, what was observed
- Uses past tense: "did", "failed", "occurred", "was", "began", "resulted"
- Provides background, context, timeline of events, or root cause analysis
- Explains why something went wrong

RECOMMENDATION (EXTRACT):
- Prescribes what should happen going forward
- Uses directive language: "shall", "must", "should", "recommend", "direct", 
  "require", "is required to", "needs to", "will need to"
- Identifies a responsible party and a specific future action
- Can be verified months or years later by checking if the action was taken

═══════════════════════════════════════════════════════════════════════════════
EXAMPLES
═══════════════════════════════════════════════════════════════════════════════

EXAMPLE INPUT A:
"The agency failed to communicate effectively with local partners during the 
incident. Coordination broke down within the first six hours."

CORRECT OUTPUT: {"findings": []}
REASON: This is narrative describing what went wrong. No recommendation.

---

EXAMPLE INPUT B:
"The agency failed to communicate effectively with local partners during the 
incident. The Committee recommends that the agency establish a dedicated 
liaison office to coordinate with local partners during emergencies."

CORRECT OUTPUT:
{"findings": [
  {
    "actor": "The agency",
    "action": "Establish a dedicated liaison office to coordinate with local partners during emergencies",
    "test_question": "Has the agency established a dedicated liaison office for emergency coordination with local partners?",
    "citation_page": "5"
  }
]}
REASON: Second sentence is a forward-looking recommendation with a clear actor and action.

---

EXAMPLE INPUT C:
"Employees were not properly trained on the new procedures."

CORRECT OUTPUT: {"findings": []}
REASON: This identifies a problem but does not recommend a solution. If the 
document later says "Management should implement mandatory training," extract that.

---

EXAMPLE INPUT D:
"Section 4.2 - Training Requirements........42
 Section 4.3 - Certification Standards.....47"

CORRECT OUTPUT: {"findings": []}
REASON: This is a table of contents, not a recommendation.

═══════════════════════════════════════════════════════════════════════════════
ADDITIONAL RULES
═══════════════════════════════════════════════════════════════════════════════

1. ONE RECOMMENDATION PER FINDING. If a sentence contains multiple directives 
   for different actors, split them into separate findings.

2. BE SPECIFIC. "Improve coordination" is vague. Prefer the document's actual 
   language: "Establish a joint task force" or "Conduct quarterly reviews."

3. SKIP BOILERPLATE. Ignore acknowledgments, disclaimers, document metadata, 
   tables of contents, lists of figures, appendix headers.

4. SKIP EXISTING REQUIREMENTS. If the text merely restates current law or 
   regulation without recommending changes, do not extract it.

5. CONDITIONAL RECOMMENDATIONS still count. "If X occurs, the agency should Y" 
   is extractable — note the condition in the action field.

6. CITATION PAGE: Use the page number from the "--- PAGE X ---" marker that 
   appears immediately before the relevant text. Do NOT use page numbers that 
   appear within the text itself (e.g., from a table of contents).

7. DEDUPLICATION: If a finding appears in the "ALREADY FOUND" context provided, 
   do not extract it again even if it appears in this chunk.

8. EMPTY IS OK. If the text contains no recommendations, return {"findings": []}.
   Do not invent findings to fill the output.
"""

    user_prompt = f"""Analyze the following text from pages {page_range}.

--- CONTEXT: ALREADY FOUND IN PREVIOUS WINDOW ---
(The LLM must ignore these if they appear again in the current text)
{previous_context}
-------------------------------------------------

--- TEXT TO ANALYZE ---
{text}
"""

    if debug_dir:
        try:
            # Sanitize page_range for filename just in case, though usually "1-3" is fine
            safe_range = page_range.replace(" ", "_")
            prompt_file = debug_dir / f"prompt_chunk_{safe_range}.txt"
            with open(prompt_file, "w", encoding="utf-8") as f:
                f.write("--- SYSTEM PROMPT ---\n")
                f.write(system_prompt)
                f.write("\n\n--- USER PROMPT ---\n")
                f.write(user_prompt)
        except Exception as e:
            print(f"Warning: Could not save debug prompt: {e}")

    try:
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.0,
            response_format={"type": "json_object"}
        )
        
        content = response.choices[0].message.content
        if not content:
            return []
            
        if debug_dir:
            try:
                safe_range = page_range.replace(" ", "_")
                response_file = debug_dir / f"response_chunk_{safe_range}.json"
                with open(response_file, "w", encoding="utf-8") as f:
                    f.write(content)
            except Exception as e:
                print(f"Warning: Could not save debug response: {e}")

        data = json.loads(content)
        findings = data.get("findings", [])
        
        # Post-process: Validate and Correct Page Numbers
        try:
            # page_range format is expected to be "start-end" (e.g., "1-3")
            if "-" in page_range:
                start_s, end_s = page_range.split("-")
                start_p = int(start_s)
                end_p = int(end_s)
            else:
                start_p = int(page_range)
                end_p = int(page_range)
            
            # Calculate safe middle page
            safe_page = start_p + (end_p - start_p) // 2
            
            for f in findings:
                raw_page = str(f.get("citation_page", ""))
                # Extract first sequence of digits
                match = re.search(r"\d+", raw_page)
                if match:
                    page_num = int(match.group())
                    if not (start_p <= page_num <= end_p):
                        # Page is out of bounds for this chunk
                        # print(f"Debug: Correcting page {page_num} to {safe_page} (Range: {page_range})")
                        f["citation_page"] = str(safe_page)
                        f["original_citation_page"] = raw_page # Keep record just in case
                else:
                    # No number found, assign safe default
                    f["citation_page"] = str(safe_page)
                    
        except Exception as e:
            print(f"Warning: Error validating page numbers: {e}")
            
        return findings
        
    except Exception as e:
        print(f"\nError processing chunk {page_range}: {e}")
        return []

def main():
    parser = argparse.ArgumentParser(description="Decompose regulatory PDF into requirements.")
    parser.add_argument("input_file", help="Path to PDF")
    parser.add_argument("--test", type=int, help="Limit to N pages", default=None)
    parser.add_argument("--debug", action="store_true", help="Save extracted text pages to a debug folder")
    args = parser.parse_args()

    input_path = Path(args.input_file)
    if not input_path.exists():
        print(f"Error: File {input_path} not found.")
        sys.exit(1)
        
    output_path = input_path.with_suffix(".json")

    print(f"--- Processing {input_path.name} ---")
    
    # 1. Extract
    pages = extract_pages(input_path, limit=args.test)
    if not pages:
        print("No text extracted. Exiting.")
        sys.exit(0)
        
    debug_dir = None
    if args.debug:
        debug_dir = input_path.parent / input_path.stem
        try:
            debug_dir.mkdir(parents=True, exist_ok=True)
            print(f"Debug mode enabled. Saving pages to directory: {debug_dir}")
            for i, page_text in enumerate(pages):
                # page_text already includes "\n--- PAGE {n} ---\n" header
                # We'll strip the leading newline for the file if desired, or keep exactly as exported.
                # Keeping as exported is safer per instructions.
                page_num = i + 1
                page_file = debug_dir / f"page_{page_num:03d}.txt"
                with open(page_file, "w", encoding="utf-8") as f:
                    f.write(page_text)
        except Exception as e:
            print(f"Warning: Could not save debug files: {e}")
        
    # 2. Chunk
    chunks = create_chunks(pages)
    print(f"Created {len(chunks)} chunks from {len(pages)} pages.")
    
    # 3. Analyze Loop
    all_data = []
    previous_buffer = [] # State variable
    
    print("Analyzing chunks...")
    # Using tqdm for progress bar
    for chunk in tqdm(chunks):
        # Format the previous findings into a simple string for the prompt
        context_str = "NONE"
        if previous_buffer:
            # We use a set to dedup lines just in case, though the logic is strictly sequential
            # The prompt asks for "Actor: Action" to check for overlap
            formatted_findings = [f"- {f.get('actor', 'Unknown')}: {f.get('action', 'Unknown')}" for f in previous_buffer]
            context_str = "\n".join(formatted_findings)
        
        # Call LLM
        new_findings = analyze_chunk(chunk['text'], context_str, chunk['page_range'], debug_dir=debug_dir)
        
        # Store
        all_data.extend(new_findings)
        
        # Update Buffer (Overwrite buffer with LATEST results only)
        previous_buffer = new_findings

    # 4. Save
    try:
        with open(output_path, "w") as f:
            json.dump({"source": str(args.input_file), "findings": all_data}, f, indent=2)
        
        print(f"Success! Saved {len(all_data)} findings to {output_path}")
    except Exception as e:
        print(f"Error saving output: {e}")

if __name__ == "__main__":
    main()
