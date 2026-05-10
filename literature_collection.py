import html
import re
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional
from urllib.parse import quote_plus

import pandas as pd
import requests


BOILERPLATE_PREFIX_PATTERN = re.compile(
    r"(?i)^\s*(abstract|background|summary|significance statement)\s*[:\-]?\s*"
)


def clean_text(txt: Any) -> str:
    """Normalize external metadata text for robust embedding and CSV export."""
    out = "" if txt is None else str(txt)
    out = html.unescape(out)
    out = re.sub(r"<[^>]+>", " ", out)
    out = out.replace("\n", " ").replace("\r", " ")
    out = " ".join(out.split()).strip()
    out = BOILERPLATE_PREFIX_PATTERN.sub("", out).strip()
    return out


def sanitize_for_export(df: pd.DataFrame) -> pd.DataFrame:
    """Apply final defensive cleanup before writing the collection CSV."""
    out = df.copy()
    text_cols = [
        "title",
        "abstract",
        "text",
        "source",
        "query",
        "url",
        "doi",
        "doc_id",
        "api",
    ]
    for col in text_cols:
        if col not in out.columns:
            out[col] = ""
        out[col] = out[col].astype(str).map(clean_text)
    return out


def _get_json(url: str, timeout: int = 25) -> Optional[Dict[str, Any]]:
    try:
        response = requests.get(url, timeout=timeout)
        if response.status_code != 200:
            return None
        return response.json()
    except Exception:
        return None


def fetch_openalex(
    query: str,
    per_page: int = 25,
    max_pages: int = 2,
    mailto: Optional[str] = None,
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    cursor = "*"

    for _ in range(max_pages):
        filter_param = f"title_and_abstract.search:{query}"
        url = (
            "https://api.openalex.org/works"
            f"?filter={quote_plus(filter_param)}"
            f"&per-page={int(per_page)}"
            f"&cursor={quote_plus(cursor)}"
            "&select=id,display_name,publication_year,doi,primary_location,abstract_inverted_index,host_venue"
        )
        if mailto:
            url += f"&mailto={quote_plus(mailto)}"

        payload = _get_json(url)
        if not payload or "results" not in payload:
            break

        for work in payload.get("results", []):
            title = clean_text(work.get("display_name", ""))
            year = work.get("publication_year", None)
            doi = clean_text(work.get("doi", ""))
            work_id = clean_text(work.get("id", ""))

            abstract = ""
            inv = work.get("abstract_inverted_index")
            if isinstance(inv, dict) and inv:
                try:
                    max_pos = max(pos for positions in inv.values() for pos in positions)
                    tokens = [""] * (max_pos + 1)
                    for token, positions in inv.items():
                        for position in positions:
                            tokens[position] = token
                    abstract = clean_text(" ".join(tokens))
                except Exception:
                    abstract = ""

            source = ""
            primary_loc = work.get("primary_location") or {}
            source_obj = primary_loc.get("source") or {}
            source = clean_text(source_obj.get("display_name", ""))

            landing_page = clean_text(primary_loc.get("landing_page_url", ""))
            if not landing_page:
                landing_page = clean_text(work_id)

            if not title and not abstract:
                continue

            rows.append(
                {
                    "doc_id": work_id or doi or f"openalex_{len(rows)}",
                    "title": title,
                    "year": year,
                    "source": source or "openalex",
                    "abstract": abstract,
                    "text": abstract if abstract else title,
                    "doi": doi,
                    "url": landing_page,
                    "api": "openalex",
                    "query": query,
                }
            )

        meta = payload.get("meta", {})
        next_cursor = meta.get("next_cursor")
        if not next_cursor:
            break
        cursor = next_cursor
        time.sleep(0.15)

    return rows


def fetch_crossref(
    query: str,
    rows_per_query: int = 40,
    mailto: Optional[str] = None,
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    url = (
        "https://api.crossref.org/works"
        f"?query.bibliographic={quote_plus(query)}"
        f"&rows={int(rows_per_query)}"
        "&select=DOI,title,issued,container-title,abstract,URL"
    )
    if mailto:
        url += f"&mailto={quote_plus(mailto)}"

    payload = _get_json(url)
    if not payload:
        return rows

    items = ((payload.get("message") or {}).get("items") or [])
    for item in items:
        doi = clean_text(item.get("DOI", ""))

        title = ""
        if isinstance(item.get("title"), list) and item["title"]:
            title = clean_text(item["title"][0])

        source = ""
        if isinstance(item.get("container-title"), list) and item["container-title"]:
            source = clean_text(item["container-title"][0])

        year = None
        issued = item.get("issued", {})
        date_parts = issued.get("date-parts") if isinstance(issued, dict) else None
        if isinstance(date_parts, list) and date_parts and date_parts[0]:
            year = date_parts[0][0]

        abstract = clean_text(item.get("abstract", ""))
        item_url = clean_text(item.get("URL", ""))

        if not title and not abstract:
            continue

        rows.append(
            {
                "doc_id": doi or f"crossref_{len(rows)}",
                "title": title,
                "year": year,
                "source": source or "crossref",
                "abstract": abstract,
                "text": abstract if abstract else title,
                "doi": doi,
                "url": item_url,
                "api": "crossref",
                "query": query,
            }
        )

    return rows


def collect_literature_dataset(
    queries: Iterable[str],
    target_n: int = 600,
    per_query_openalex: int = 25,
    pages_openalex: int = 2,
    per_query_crossref: int = 30,
    min_text_chars: int = 200,
    mailto: Optional[str] = None,
) -> pd.DataFrame:
    collected: List[Dict[str, Any]] = []

    for query in queries:
        collected.extend(
            fetch_openalex(
                query,
                per_page=per_query_openalex,
                max_pages=pages_openalex,
                mailto=mailto,
            )
        )
        collected.extend(fetch_crossref(query, rows_per_query=per_query_crossref, mailto=mailto))

        if len(collected) >= target_n * 2:
            break

    df = pd.DataFrame(collected)
    if df.empty:
        return df

    text_cols = ["doc_id", "title", "source", "abstract", "text", "doi", "url", "api", "query"]
    for col in text_cols:
        if col not in df.columns:
            df[col] = ""
        df[col] = df[col].astype(str).map(clean_text)

    doi_nonempty = df["doi"].str.len() > 0
    df_doi = df[doi_nonempty].drop_duplicates(subset=["doi"], keep="first")
    df_no_doi = df[~doi_nonempty].drop_duplicates(subset=["title", "year"], keep="first")
    df = pd.concat([df_doi, df_no_doi], ignore_index=True)

    df = df[df["text"].str.len() >= int(min_text_chars)].copy()

    keywords = [
        "sensory",
        "perception",
        "multisensory",
        "modality",
        "interoception",
        "proprioception",
        "vestibular",
        "auditory",
        "visual",
        "olfactory",
        "gustatory",
        "tactile",
        "sensorimotor",
    ]
    text_lower = df["text"].str.lower()
    df["keyword_hits"] = sum(text_lower.str.contains(keyword, na=False).astype(int) for keyword in keywords)

    df["year_num"] = pd.to_numeric(df["year"], errors="coerce").fillna(0).astype(int)
    df = df.sort_values(["keyword_hits", "year_num"], ascending=[False, False], kind="mergesort")

    if target_n is not None and len(df) > int(target_n):
        df = df.head(int(target_n)).copy()

    return df.reset_index(drop=True)


def save_collected_corpus(df: pd.DataFrame, out_file: Path) -> pd.DataFrame:
    """Sanitize and persist the collection in the notebook-compatible schema."""
    cleaned = sanitize_for_export(df)
    keep_cols = [
        "doc_id",
        "title",
        "year",
        "source",
        "abstract",
        "text",
        "doi",
        "url",
        "api",
        "query",
        "keyword_hits",
    ]
    for col in keep_cols:
        if col not in cleaned.columns:
            cleaned[col] = ""

    out_file.parent.mkdir(parents=True, exist_ok=True)
    cleaned[keep_cols].to_csv(out_file, index=False)
    return cleaned
