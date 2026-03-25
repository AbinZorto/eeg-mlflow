#!/usr/bin/env python3
"""
BibTeX Formatter - Convert search results to BibTeX format.

Reads search results from JSON (produced by search_citations.py) and
converts them to properly formatted BibTeX entries. Appends to analysis.bib
without creating duplicates.

Usage:
    python scripts/format_bibtex.py
    python scripts/format_bibtex.py --input results.json --output analysis.bib
"""

from __future__ import annotations

import argparse
import json
import os
import re
import unicodedata
from dataclasses import dataclass
from typing import List, Dict, Set, Optional


@dataclass
class BibEntry:
    """Represents a BibTeX entry."""
    entry_type: str  # article, inproceedings, misc, etc.
    cite_key: str
    title: str
    authors: str
    year: Optional[int]
    journal: Optional[str] = None
    booktitle: Optional[str] = None
    volume: Optional[str] = None
    number: Optional[str] = None
    pages: Optional[str] = None
    doi: Optional[str] = None
    url: Optional[str] = None
    abstract: Optional[str] = None
    publisher: Optional[str] = None
    keywords: Optional[str] = None
    
    def to_bibtex(self) -> str:
        """Convert to BibTeX string format."""
        lines = [f"@{self.entry_type}{{{self.cite_key},"]
        
        # Required fields
        lines.append(f'  title = {{{self.title}}},')
        lines.append(f'  author = {{{self.authors}}},')
        
        if self.year:
            lines.append(f'  year = {{{self.year}}},')
        
        # Optional fields
        if self.journal:
            lines.append(f'  journal = {{{self.journal}}},')
        if self.booktitle:
            lines.append(f'  booktitle = {{{self.booktitle}}},')
        if self.volume:
            lines.append(f'  volume = {{{self.volume}}},')
        if self.number:
            lines.append(f'  number = {{{self.number}}},')
        if self.pages:
            lines.append(f'  pages = {{{self.pages}}},')
        if self.doi:
            lines.append(f'  doi = {{{self.doi}}},')
        if self.url:
            lines.append(f'  url = {{{self.url}}},')
        if self.publisher:
            lines.append(f'  publisher = {{{self.publisher}}},')
        if self.keywords:
            lines.append(f'  keywords = {{{self.keywords}}},')
        
        lines.append('}')
        return '\n'.join(lines)


def normalize_string(s: str) -> str:
    """Normalize string for comparison (lowercase, remove accents, punctuation)."""
    # Remove accents
    s = unicodedata.normalize('NFKD', s).encode('ASCII', 'ignore').decode('ASCII')
    # Lowercase and remove non-alphanumeric
    s = re.sub(r'[^a-z0-9\s]', '', s.lower())
    # Collapse whitespace
    s = re.sub(r'\s+', ' ', s).strip()
    return s


def generate_cite_key(authors: str, year: Optional[int], title: str, existing_keys: Set[str]) -> str:
    """Generate a unique citation key."""
    # Get first author's last name
    if authors:
        first_author = authors.split(',')[0].strip()
        # Handle "Last, First" format
        if ',' in first_author:
            last_name = first_author.split(',')[0].strip()
        else:
            # Handle "First Last" format
            parts = first_author.split()
            last_name = parts[-1] if parts else "Unknown"
    else:
        last_name = "Unknown"
    
    # Clean last name
    last_name = re.sub(r'[^a-zA-Z]', '', last_name)
    last_name = last_name.lower()
    
    # Get year
    year_str = str(year) if year else "nodate"
    
    # Get first significant word from title
    title_words = re.sub(r'[^a-zA-Z\s]', '', title).lower().split()
    stop_words = {'a', 'an', 'the', 'of', 'in', 'on', 'for', 'to', 'and', 'or', 'with'}
    title_word = ""
    for word in title_words:
        if word not in stop_words and len(word) > 2:
            title_word = word
            break
    
    # Base key
    base_key = f"{last_name}{year_str}"
    if title_word:
        base_key += f"_{title_word}"
    
    # Ensure uniqueness
    key = base_key
    counter = 1
    while key in existing_keys:
        key = f"{base_key}_{counter}"
        counter += 1
    
    return key


def escape_bibtex(text: str) -> str:
    """Escape special characters for BibTeX."""
    if not text:
        return ""
    
    # Replace problematic characters
    replacements = {
        '&': r'\&',
        '%': r'\%',
        '#': r'\#',
        '_': r'\_',
        '$': r'\$',
        '{': r'\{',
        '}': r'\}',
        '~': r'\textasciitilde{}',
        '^': r'\textasciicircum{}',
    }
    
    for char, replacement in replacements.items():
        # Don't double-escape
        if f'\\{char}' not in text:
            text = text.replace(char, replacement)
    
    return text


def format_authors_bibtex(authors_str: str) -> str:
    """Format authors string for BibTeX (Last, First and Last, First format)."""
    if not authors_str:
        return "Unknown"
    
    authors = [a.strip() for a in authors_str.split(',')]
    formatted = []
    
    for i, author in enumerate(authors):
        if not author:
            continue
        
        # Skip if already in "Last, First" format (has comma inside)
        # This is a simple heuristic - authors from APIs are usually "First Last"
        parts = author.split()
        if len(parts) >= 2:
            # Assume "First Last" format, convert to "Last, First"
            first_names = ' '.join(parts[:-1])
            last_name = parts[-1]
            formatted.append(f"{last_name}, {first_names}")
        else:
            formatted.append(author)
    
    return ' and '.join(formatted)


def determine_entry_type(result: Dict) -> str:
    """Determine BibTeX entry type based on venue information."""
    venue = result.get('venue', '') or ''
    venue_lower = venue.lower()
    
    if any(x in venue_lower for x in ['conference', 'proceedings', 'workshop', 'symposium']):
        return 'inproceedings'
    elif any(x in venue_lower for x in ['journal', 'transactions', 'review']):
        return 'article'
    elif result.get('doi'):
        return 'article'
    else:
        return 'misc'


def load_existing_bibtex(bib_path: str) -> tuple[Set[str], Set[str], str]:
    """Load existing BibTeX file and extract citation keys and DOIs."""
    existing_keys: Set[str] = set()
    existing_dois: Set[str] = set()
    existing_content = ""
    
    if os.path.exists(bib_path):
        with open(bib_path, 'r', encoding='utf-8') as f:
            existing_content = f.read()
        
        # Extract citation keys
        key_pattern = r'@\w+\{([^,]+),'
        existing_keys = set(re.findall(key_pattern, existing_content))
        
        # Extract DOIs
        doi_pattern = r'doi\s*=\s*\{([^}]+)\}'
        dois = re.findall(doi_pattern, existing_content, re.IGNORECASE)
        existing_dois = set(d.lower() for d in dois)
    
    return existing_keys, existing_dois, existing_content


def convert_result_to_bibtex(result: Dict, existing_keys: Set[str]) -> Optional[BibEntry]:
    """Convert a search result to a BibTeX entry."""
    title = result.get('title', '')
    if not title:
        return None
    
    authors = result.get('authors', '')
    year = result.get('year')
    
    # Generate citation key
    cite_key = generate_cite_key(authors, year, title, existing_keys)
    existing_keys.add(cite_key)  # Add to set to avoid duplicates in this batch
    
    # Determine entry type
    entry_type = determine_entry_type(result)
    
    # Format authors
    formatted_authors = format_authors_bibtex(authors)
    
    # Get venue
    venue = result.get('venue', '')
    
    # Create entry
    entry = BibEntry(
        entry_type=entry_type,
        cite_key=cite_key,
        title=escape_bibtex(title),
        authors=escape_bibtex(formatted_authors),
        year=year,
        journal=escape_bibtex(venue) if entry_type == 'article' else None,
        booktitle=escape_bibtex(venue) if entry_type == 'inproceedings' else None,
        doi=result.get('doi'),
        url=result.get('url') or result.get('download_url'),
        keywords=', '.join(result.get('keywords', [])[:5]) if result.get('keywords') else None,
    )
    
    return entry


def format_results_to_bibtex(
    results: List[Dict],
    output_path: str,
    skip_existing_dois: bool = True
) -> int:
    """Convert search results to BibTeX and append to file."""
    
    # Load existing entries
    existing_keys, existing_dois, existing_content = load_existing_bibtex(output_path)
    print(f"Loaded {len(existing_keys)} existing citation keys from {output_path}")
    print(f"Loaded {len(existing_dois)} existing DOIs")
    
    # Track statistics
    added = 0
    skipped_doi = 0
    skipped_no_title = 0
    
    new_entries = []
    
    for result in results:
        # Skip if no title
        if not result.get('title'):
            skipped_no_title += 1
            continue
        
        # Skip if DOI already exists
        doi = result.get('doi', '')
        if skip_existing_dois and doi and doi.lower() in existing_dois:
            skipped_doi += 1
            continue
        
        # Convert to BibTeX
        entry = convert_result_to_bibtex(result, existing_keys)
        if entry:
            new_entries.append(entry)
            if doi:
                existing_dois.add(doi.lower())
            added += 1
    
    # Append new entries to file
    if new_entries:
        with open(output_path, 'a', encoding='utf-8') as f:
            f.write("\n\n% ===== Auto-generated citations (DC offset removal paper) =====\n")
            f.write(f"% Added {added} new entries\n\n")
            
            for entry in new_entries:
                f.write(entry.to_bibtex())
                f.write('\n\n')
    
    # Print summary
    print(f"\nBibTeX Conversion Summary:")
    print(f"  Total results processed: {len(results)}")
    print(f"  New entries added: {added}")
    print(f"  Skipped (DOI exists): {skipped_doi}")
    print(f"  Skipped (no title): {skipped_no_title}")
    print(f"  Output file: {output_path}")
    
    if new_entries:
        print(f"\nNew citation keys:")
        for entry in new_entries[:10]:  # Show first 10
            print(f"  - {entry.cite_key}")
        if len(new_entries) > 10:
            print(f"  ... and {len(new_entries) - 10} more")
    
    return added


def main():
    parser = argparse.ArgumentParser(description="Convert search results to BibTeX format")
    parser.add_argument("--input", "-i", default="scripts/search_results.json",
                        help="Input JSON file from search_citations.py")
    parser.add_argument("--output", "-o", default="analysis.bib",
                        help="Output BibTeX file (will append)")
    parser.add_argument("--no-skip-existing", action="store_true",
                        help="Don't skip entries with existing DOIs")
    
    args = parser.parse_args()
    
    # Load search results
    if not os.path.exists(args.input):
        print(f"Error: Input file not found: {args.input}")
        print("Run search_citations.py first to generate search results.")
        return 1
    
    with open(args.input, 'r', encoding='utf-8') as f:
        results = json.load(f)
    
    print(f"Loaded {len(results)} search results from {args.input}")
    
    # Convert and append
    added = format_results_to_bibtex(
        results,
        args.output,
        skip_existing_dois=not args.no_skip_existing
    )
    
    if added > 0:
        print(f"\nSuccessfully added {added} new citations to {args.output}")
    else:
        print("\nNo new citations were added (all may already exist).")
    
    return 0


if __name__ == "__main__":
    exit(main())

