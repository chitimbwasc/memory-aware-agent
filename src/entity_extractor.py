"""
Entity extractor harness: uses the LLM to extract entities from a given text (first 500 chars),
writes PERSON/PLACE/SYSTEM records to the entity store, and swallows any exceptions.

Call: extract_entities_and_store(mm, llm, thread_id, text)
Returns: list of written entity ids (may be empty)
"""
from typing import List
from .memory_manager import MemoryManager


def extract_entities_and_store(mm: MemoryManager, llm, thread_id: str, text: str) -> List[str]:
    try:
        snippet = text[:500]
        prompt = f"EXTRACT_ENTITIES\nExtract entities as bullets of the form 'TYPE: Name' where TYPE in PERSON|PLACE|SYSTEM.\nSource:\n{snippet}"
        resp = llm.call([{"role": "user", "content": prompt}])
        content = resp.get("content") if isinstance(resp, dict) else str(resp)
        lines = [l.strip().lstrip("- ") for l in content.splitlines() if l.strip()]
        ids = []
        for line in lines:
            if ":" in line:
                typ, name = [p.strip() for p in line.split(":", 1)]
                typ = typ.upper()
                if typ not in ("PERSON", "PLACE", "SYSTEM"):
                    typ = "UNKNOWN"
                eid = mm.write_entity(name=name, type=typ, description="extracted from conversation")
                ids.append(eid)
        return ids
    except Exception:
        # swallow failures per R12
        return []
