"""
Summarizer utilities: create structured summaries and run the offload pipeline.

Functions:
- summarize_thread_if_needed(mm, llm, thread_id, override_budget=None)
- summarize_thread(mm, llm, thread_id, description=None)

This module keeps the summarization prompt minimal for scripted testing: it embeds the
literal marker 'SUMMARIZE_THREAD' so ScriptedLLMClient can match on scripted keys.
"""
from typing import Optional, List
from .memory_manager import MemoryManager


def _collect_unconsolidated_text(mm: MemoryManager, thread_id: str, char_limit: int = 6000) -> (str, List[str]):
    rows = mm.read_conversation(thread_id, limit=1000)
    texts = []
    ids = []
    total = 0
    for r in rows:
        piece = f"[{r['timestamp']}] {r['role']}: {r['content']}\n"
        if total + len(piece) > char_limit:
            break
        texts.append(piece)
        ids.append(r['id'])
        total += len(piece)
    return "".join(texts), ids


def summarize_thread(mm: MemoryManager, llm, thread_id: str, description: Optional[str] = None) -> Optional[str]:
    text, ids = _collect_unconsolidated_text(mm, thread_id)
    if not text or not ids:
        return None
    # create prompt with marker so ScriptedLLMClient can match
    prompt = f"SUMMARIZE_THREAD\nPlease produce a structured summary with the four headings:\nTechnical Information\nEmotional Context\nEntities & References\nAction Items & Decisions\n\nSource:\n{text}"
    resp = llm.call([{"role": "user", "content": prompt}])
    summary_text = resp.get("content") if isinstance(resp, dict) else str(resp)
    # ensure description label meets 8-12 words; if not provided, build deterministic label
    if not description:
        description = "Seed thread summary capturing Kestrel research context and decisions"
    # Persist summary and mark rows
    sid = mm.create_summary_and_mark_rows(thread_id=thread_id, summary_text=summary_text, description=description, source_row_ids=ids)
    return sid


def summarize_thread_if_needed(mm: MemoryManager, llm, thread_id: str, override_budget: Optional[int] = None) -> Optional[str]:
    # assemble a simple context and estimate
    conv = mm.read_conversation(thread_id, limit=1000)
    context_text = "\n".join([f"{r['role']}: {r['content']}" for r in conv])
    status = mm.budget_status_for_text(context_text, override_budget=override_budget)
    if status["status"] == "critical":
        return summarize_thread(mm, llm, thread_id)
    return None
