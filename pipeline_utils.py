import argparse
import logging
import os
import re
import sys
from pathlib import Path
from typing import Optional

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%H:%M:%S"
)
logger = logging.getLogger("pipeline")

def slugify(text: str) -> str:
    """Convert text to a filesystem-safe directory name."""
    text = text.lower()
    text = re.sub(r'[^\w\s-]', '', text)
    text = re.sub(r'[\s_-]+', '-', text)
    return text.strip('-')

class ProjectContext:
    def __init__(self, input_path_str: str):
        self.input_path = Path(input_path_str).resolve()
        
        # Determine if input is the PDF or the Project Directory
        if self.input_path.is_file():
            self.project_name = slugify(self.input_path.stem)
            self.project_dir = self.input_path.parent / self.project_name
            self.source_pdf = self.input_path
        elif self.input_path.is_dir():
            self.project_name = self.input_path.name
            self.project_dir = self.input_path
            # Try to find source PDF inside
            pdfs = list(self.project_dir.glob("*.pdf"))
            self.source_pdf = pdfs[0] if pdfs else None
        else:
            # Fallback for non-existent path (assuming it's a new project from a PDF path)
            self.project_name = slugify(self.input_path.stem)
            self.project_dir = self.input_path.parent / self.project_name
            self.source_pdf = self.input_path

        # Ensure project dir exists
        self.project_dir.mkdir(parents=True, exist_ok=True)

        # Define Standard Paths
        self.paths = {
            "source": self.source_pdf,
            "findings": self.project_dir / "01_findings.json",
            "context": self.project_dir / "02_context.json",
            "verified": self.project_dir / "03_verified.jsonl",
            "enriched": self.project_dir / "04_enriched.jsonl",
            "report_html": self.project_dir / "05_report.html",
            "consequences": self.project_dir / "06_consequences.jsonl",
            "consequences_summary": self.project_dir / "06_consequences_summary.json",
            "consequences_html": self.project_dir / "07_consequences_report.html",
            "debug_dir": self.project_dir / "debug",
            "articles_dir": self.project_dir / "articles",
        }

    def get_path(self, key: str) -> Path:
        return self.paths.get(key)

    def ensure_debug_dir(self):
        self.paths["debug_dir"].mkdir(exist_ok=True)
        
    def ensure_articles_dir(self):
        self.paths["articles_dir"].mkdir(exist_ok=True)

def setup_common_args(description: str) -> argparse.ArgumentParser:
    """Create a standard argument parser with the input file argument."""
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument(
        "input", 
        help="Path to the source PDF file or the project directory."
    )
    return parser
