#!/usr/bin/env python3
"""
Citation Search Script - Search academic databases for relevant papers.

Searches CORE, OpenAlex, and Semantic Scholar APIs for papers matching
given queries. Results can be exported to BibTeX format.

Usage:
    python scripts/search_citations.py "DC offset removal EEG"
    python scripts/search_citations.py --topics  # Search predefined topics
    
Environment Variables:
    CORE_API_KEY - Required for CORE API
    SEMANTIC_SCHOLAR_API_KEY - Optional, recommended for higher rate limits
    OPENAI_API_KEY - Optional, for query expansion
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import List, Optional, Dict, Any
from urllib.parse import quote

import requests
from dotenv import load_dotenv


REPO_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(REPO_ROOT / ".env")
load_dotenv()

# API endpoints
CORE_API_URL = "https://api.core.ac.uk/v3/search/works"
OPENALEX_API_URL = "https://api.openalex.org/works"
SEMANTIC_SCHOLAR_API_URL = "https://api.semanticscholar.org/graph/v1"

# API keys (from environment variables)
CORE_API_KEY = os.getenv("CORE_API_KEY", "")
SEMANTIC_SCHOLAR_API_KEY = os.getenv("SEMANTIC_SCHOLAR_API_KEY", "")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
CONTACT_EMAIL = os.getenv("CONTACT_EMAIL", "info@writemine.com")

# Predefined search topics for the DC offset removal paper
SEARCH_TOPICS = [
    "DC offset removal EEG preprocessing",
    "EEG baseline correction signal processing",
    "preprocessing vs model complexity machine learning",
    "EEG signal quality deep learning classification",
    "per-channel normalization neural network",
    "EEG artifact removal depression",
    "signal centering time series classification",
    "data quality model architecture comparison",
]


@dataclass
class SearchResult:
    """Represents a single search result from any API."""
    source: str
    title: str
    authors: str
    abstract: str
    year: Optional[int] = None
    doi: Optional[str] = None
    url: Optional[str] = None
    download_url: Optional[str] = None
    venue: Optional[str] = None
    citation_count: Optional[int] = None
    fields_of_study: List[str] = field(default_factory=list)
    paper_id: Optional[str] = None
    keywords: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return asdict(self)


def search_core(query: str, limit: int = 10) -> List[SearchResult]:
    """Search CORE API for academic papers."""
    if not CORE_API_KEY:
        print("Warning: CORE_API_KEY not set, skipping CORE search")
        return []
    
    try:
        url = f"{CORE_API_URL}/?q={quote(query)}&apiKey={CORE_API_KEY}&limit={limit}"
        print(f"Searching CORE API: {query[:50]}...")
        
        response = requests.get(url, headers={"Accept": "application/json"}, timeout=30)
        
        if response.status_code != 200:
            print(f"CORE API error: {response.status_code} {response.text[:200]}")
            return []
        
        data = response.json()
        results = []
        
        for item in data.get("results", []):
            # Skip if no abstract
            abstract = item.get("abstract", "")
            if not abstract or not abstract.strip():
                continue
                
            result = SearchResult(
                source="CORE",
                title=item.get("title", ""),
                authors=", ".join([a.get("name", "") for a in item.get("authors", [])]),
                abstract=abstract,
                year=item.get("yearPublished"),
                doi=item.get("doi"),
                url=item.get("sourceFulltextUrls", [None])[0] if item.get("sourceFulltextUrls") else None,
                download_url=item.get("downloadUrl"),
                venue=item.get("journals", [{}])[0].get("title") if item.get("journals") else None,
                citation_count=item.get("citationCount", 0),
                paper_id=str(item.get("id", "")),
            )
            results.append(result)
        
        print(f"  CORE returned {len(results)} results with abstracts")
        return results
        
    except Exception as e:
        print(f"CORE API error: {e}")
        return []


def search_openalex(query: str, limit: int = 10) -> List[SearchResult]:
    """Search OpenAlex API for academic papers."""
    try:
        url = f"{OPENALEX_API_URL}?filter=has_abstract:true&search={quote(query)}&per_page={limit}"
        headers = {
            "Accept": "application/json",
            "User-Agent": f"mailto:{CONTACT_EMAIL}"
        }
        
        print(f"Searching OpenAlex API: {query[:50]}...")
        response = requests.get(url, headers=headers, timeout=30)
        
        if response.status_code != 200:
            print(f"OpenAlex API error: {response.status_code}")
            return []
        
        data = response.json()
        results = []
        
        for item in data.get("results", []):
            if not isinstance(item, dict):
                continue

            # Reconstruct abstract from inverted index if needed
            abstract = item.get("abstract", "")
            if not abstract and item.get("abstract_inverted_index"):
                abstract = reconstruct_abstract(item["abstract_inverted_index"])
            
            if not abstract or not abstract.strip():
                continue
            
            # Get authors
            authorships = item.get("authorships") or []
            authors = ", ".join([
                a.get("author", {}).get("display_name", "")
                for a in authorships
                if isinstance(a, dict)
            ])
            
            # Get DOI (remove https://doi.org/ prefix if present)
            doi = item.get("doi", "")
            if doi and doi.startswith("https://doi.org/"):
                doi = doi[16:]

            best_oa_location = item.get("best_oa_location") or {}
            primary_location = item.get("primary_location") or {}
            primary_source = primary_location.get("source") or {}
            topics = [
                topic.get("display_name", "")
                for topic in (item.get("topics") or [])
                if isinstance(topic, dict)
            ]
            keywords = [
                keyword.get("display_name", "")
                for keyword in (item.get("keywords") or [])
                if isinstance(keyword, dict)
            ]
            
            result = SearchResult(
                source="OpenAlex",
                title=item.get("title", ""),
                authors=authors,
                abstract=abstract,
                year=item.get("publication_year"),
                doi=doi if doi else None,
                url=(item.get("ids") or {}).get("doi") or best_oa_location.get("landing_page_url"),
                download_url=best_oa_location.get("pdf_url"),
                venue=primary_source.get("display_name"),
                citation_count=item.get("cited_by_count", 0),
                fields_of_study=topics[:5],
                paper_id=item.get("id", ""),
                keywords=keywords[:10],
            )
            results.append(result)
        
        print(f"  OpenAlex returned {len(results)} results with abstracts")
        return results
        
    except Exception as e:
        print(f"OpenAlex API error: {e}")
        return []


def reconstruct_abstract(inverted_index: Dict[str, List[int]]) -> str:
    """Reconstruct abstract from OpenAlex inverted index format."""
    if not inverted_index:
        return ""
    
    word_positions = []
    for word, positions in inverted_index.items():
        for pos in positions:
            word_positions.append((pos, word))
    
    word_positions.sort(key=lambda x: x[0])
    return " ".join([word for _, word in word_positions])


def search_semantic_scholar(query: str, limit: int = 10) -> List[SearchResult]:
    """Search Semantic Scholar API for academic papers."""
    try:
        fields = "paperId,title,authors,abstract,year,citationCount,venue,fieldsOfStudy,openAccessPdf,isOpenAccess,url,externalIds"
        url = f"{SEMANTIC_SCHOLAR_API_URL}/paper/search?query={quote(query)}&limit={limit}&fields={fields}"
        
        headers = {
            "Accept": "application/json",
            "User-Agent": f"research_app ({CONTACT_EMAIL})"
        }
        
        if SEMANTIC_SCHOLAR_API_KEY:
            headers["x-api-key"] = SEMANTIC_SCHOLAR_API_KEY
        
        print(f"Searching Semantic Scholar API: {query[:50]}...")
        response = _semantic_scholar_request(url, headers)
        if response is not None and response.status_code == 403 and "x-api-key" in headers:
            print("  Semantic Scholar returned 403 with API key; retrying once without key.")
            fallback_headers = dict(headers)
            fallback_headers.pop("x-api-key", None)
            response = _semantic_scholar_request(url, fallback_headers)
        
        if response is None:
            return []

        if response.status_code != 200:
            snippet = response.text[:200].replace("\n", " ")
            print(f"Semantic Scholar API error: {response.status_code} {snippet}")
            return []
        
        data = response.json()
        results = []
        
        for item in data.get("data", []):
            if not isinstance(item, dict):
                continue

            abstract = item.get("abstract", "")
            if not abstract or not abstract.strip():
                continue
            
            authors = ", ".join(
                a.get("name", "")
                for a in (item.get("authors") or [])
                if isinstance(a, dict)
            )
            
            # Get DOI from externalIds
            external_ids = item.get("externalIds") or {}
            doi = external_ids.get("DOI")
            open_access_pdf = item.get("openAccessPdf") or {}
            fields_of_study_raw = item.get("fieldsOfStudy") or []
            
            result = SearchResult(
                source="Semantic Scholar",
                title=item.get("title", ""),
                authors=authors,
                abstract=abstract,
                year=item.get("year"),
                doi=doi,
                url=item.get("url") or f"https://www.semanticscholar.org/paper/{item.get('paperId', '')}",
                download_url=open_access_pdf.get("url"),
                venue=item.get("venue"),
                citation_count=item.get("citationCount", 0),
                fields_of_study=[
                    value if isinstance(value, str) else value.get("category", "")
                    for value in fields_of_study_raw
                    if isinstance(value, (str, dict))
                ],
                paper_id=item.get("paperId", ""),
            )
            results.append(result)
        
        print(f"  Semantic Scholar returned {len(results)} results with abstracts")
        return results
        
    except Exception as e:
        print(f"Semantic Scholar API error: {e}")
        return []


def _semantic_scholar_request(url: str, headers: Dict[str, str]) -> Optional[requests.Response]:
    response = None
    for attempt in range(3):
        response = requests.get(url, headers=headers, timeout=30)
        if response.status_code != 429:
            return response
        wait_seconds = 3 * (attempt + 1)
        print(f"  Semantic Scholar rate limit hit, waiting {wait_seconds} seconds...")
        time.sleep(wait_seconds)
    return response


def search_crossref_for_doi(title: str, author: str = "") -> Optional[str]:
    """Try to find DOI using CrossRef API."""
    try:
        query = quote(title)
        if author:
            query += f"+{quote(author.split(',')[0].strip())}"
        
        url = f"https://api.crossref.org/works?query={query}&rows=1"
        headers = {
            "Accept": "application/json",
            "User-Agent": f"mailto:{CONTACT_EMAIL}"
        }
        
        response = requests.get(url, headers=headers, timeout=15)
        
        if response.status_code != 200:
            return None
        
        data = response.json()
        items = data.get("message", {}).get("items", [])
        
        if items:
            found_title = items[0].get("title", [""])[0].lower()
            original_title = title.lower()
            
            # Simple similarity check
            if calculate_similarity(original_title, found_title) > 0.7:
                return items[0].get("DOI")
        
        return None
        
    except Exception:
        return None


def calculate_similarity(s1: str, s2: str) -> float:
    """Calculate Jaccard similarity between two strings."""
    if not s1 or not s2:
        return 0.0
    
    words1 = set(s1.lower().split())
    words2 = set(s2.lower().split())
    
    intersection = words1 & words2
    union = words1 | words2
    
    return len(intersection) / len(union) if union else 0.0


def deduplicate_results(results: List[SearchResult]) -> List[SearchResult]:
    """Remove duplicate results based on DOI or title similarity."""
    seen_dois = set()
    seen_titles = set()
    unique_results = []
    
    for result in results:
        # Check DOI
        if result.doi:
            doi_lower = result.doi.lower()
            if doi_lower in seen_dois:
                continue
            seen_dois.add(doi_lower)
        
        # Check title similarity
        title_lower = result.title.lower().strip()
        is_duplicate = False
        
        for seen_title in seen_titles:
            if calculate_similarity(title_lower, seen_title) > 0.85:
                is_duplicate = True
                break
        
        if not is_duplicate:
            seen_titles.add(title_lower)
            unique_results.append(result)
    
    return unique_results


def search_all(query: str, limit_per_source: int = 10, find_missing_dois: bool = True) -> List[SearchResult]:
    """Search all available APIs and merge results."""
    all_results = []
    
    # Search each API
    all_results.extend(search_core(query, limit_per_source))
    all_results.extend(search_openalex(query, limit_per_source))
    all_results.extend(search_semantic_scholar(query, limit_per_source))
    
    # Deduplicate
    unique_results = deduplicate_results(all_results)
    
    # Try to find missing DOIs
    if find_missing_dois:
        print("Finding missing DOIs via CrossRef...")
        for result in unique_results:
            if not result.doi:
                doi = search_crossref_for_doi(result.title, result.authors)
                if doi:
                    result.doi = doi
                    print(f"  Found DOI for: {result.title[:50]}...")
    
    # Sort by citation count (descending)
    unique_results.sort(key=lambda x: x.citation_count or 0, reverse=True)
    
    return unique_results


def search_topics(topics: List[str], limit_per_topic: int = 5) -> List[SearchResult]:
    """Search multiple topics and combine results."""
    all_results = []
    
    for topic in topics:
        print(f"\n{'='*60}")
        print(f"Searching: {topic}")
        print('='*60)
        
        results = search_all(topic, limit_per_source=limit_per_topic, find_missing_dois=False)
        all_results.extend(results)
        
        # Rate limiting between topics
        time.sleep(1)
    
    # Final deduplication and DOI finding
    print(f"\n{'='*60}")
    print("Final deduplication and DOI lookup...")
    unique_results = deduplicate_results(all_results)
    
    print(f"Finding missing DOIs for {len([r for r in unique_results if not r.doi])} papers...")
    for result in unique_results:
        if not result.doi:
            doi = search_crossref_for_doi(result.title, result.authors)
            if doi:
                result.doi = doi
            time.sleep(0.5)  # Rate limit CrossRef
    
    return unique_results


def save_results(results: List[SearchResult], output_path: str):
    """Save results to JSON file."""
    data = [r.to_dict() for r in results]
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    
    print(f"Saved {len(results)} results to {output_path}")


def print_summary(results: List[SearchResult]):
    """Print summary of search results."""
    print(f"\n{'='*60}")
    print("SEARCH RESULTS SUMMARY")
    print('='*60)
    print(f"Total papers found: {len(results)}")
    print(f"With DOI: {len([r for r in results if r.doi])}")
    print(f"With download URL: {len([r for r in results if r.download_url])}")
    
    # Source breakdown
    sources = {}
    for r in results:
        sources[r.source] = sources.get(r.source, 0) + 1
    
    print("\nBy source:")
    for source, count in sorted(sources.items()):
        print(f"  {source}: {count}")
    
    # Year distribution
    years = [r.year for r in results if r.year]
    if years:
        print(f"\nYear range: {min(years)} - {max(years)}")
    
    print(f"\nTop 5 by citations:")
    for i, r in enumerate(results[:5], 1):
        print(f"  {i}. [{r.citation_count or 0} cites] {r.title[:60]}...")


def main():
    parser = argparse.ArgumentParser(description="Search academic databases for citations")
    parser.add_argument("query", nargs="?", help="Search query")
    parser.add_argument("--topics", action="store_true", help="Search predefined topics for DC offset paper")
    parser.add_argument("--output", "-o", default="scripts/search_results.json", help="Output JSON file")
    parser.add_argument("--limit", "-l", type=int, default=10, help="Results per source per query")
    
    args = parser.parse_args()
    
    if args.topics:
        results = search_topics(SEARCH_TOPICS, limit_per_topic=args.limit)
    elif args.query:
        results = search_all(args.query, limit_per_source=args.limit)
    else:
        parser.print_help()
        print("\nPredefined search topics:")
        for topic in SEARCH_TOPICS:
            print(f"  - {topic}")
        sys.exit(1)
    
    print_summary(results)
    save_results(results, args.output)
    
    print(f"\nNext step: Run format_bibtex.py to convert results to BibTeX")


if __name__ == "__main__":
    main()
