"""
Tool implementations: fetch_notes (deep ingest), paper_search (simulated), get_current_time.

fetch_notes(file_path, mm, thread_id) reads file, chunks it (1500/200), writes chunks to SEMANTIC_MEMORY
with metadata including source, chunk_id, num_chunks, and returns full text.

paper_search(query, fixtures_dir) looks for markdown files in fixtures_dir/kb and returns a list
of candidate dicts (id, title, authors, published, abstract).
"""
from typing import List, Dict
from pathlib import Path
import math


def _chunk_text(text: str, chunk_size: int = 1500, overlap: int = 200) -> List[str]:
    if chunk_size <= 0:
        return [text]
    chunks = []
    start = 0
    n = len(text)
    while start < n:
        end = min(start + chunk_size, n)
        chunks.append(text[start:end])
        if end == n:
            break
        start = end - overlap
    return chunks


def fetch_notes(file_path: str, mm, thread_id: str) -> Dict:
    p = Path(file_path)
    text = p.read_text(encoding="utf-8")
    chunks = _chunk_text(text, chunk_size=1500, overlap=200)
    num_chunks = len(chunks)
    ids = []
    for i, c in enumerate(chunks):
        meta = {"source": p.name, "chunk_id": i, "num_chunks": num_chunks}
        cid = mm.write_semantic("SEMANTIC_MEMORY", c, metadata=meta)
        ids.append(cid)
    return {"file": str(p), "num_chunks": num_chunks, "chunk_ids": ids, "full_text": text}


def paper_search(query: str, fixtures_kb_dir: str) -> List[Dict]:
    # very simple: return each file's first line as title and author placeholder
    p = Path(fixtures_kb_dir)
    out = []
    for f in sorted(p.glob("*.md")):
        txt = f.read_text(encoding="utf-8")
        first = txt.strip().splitlines()[0] if txt else ""
        out.append({"arxiv_id": f.stem, "entry_id": f.stem, "title": first[:200], "authors": ["Author A"], "published": "2025-01-01", "abstract": txt[:2500]})
    return out


def get_current_time() -> str:
    from datetime import datetime
    return datetime.utcnow().isoformat() + "Z"
