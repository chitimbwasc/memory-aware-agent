"""
Context assembler: assemble the partitioned context per D9 and R2, with budget guard (R6) and offload (R7).

Function:
- assemble_partitioned_context(mm, llm, thread_id, question, override_budget=None) -> Dict with keys:
    - system_message (str)
    - user_message (str)  # begins with '# Question' and the partitioned segments in order
    - budget (dict)       # result of budget_status_for_text on the assembled context
    - summary_id (str|None) # summary created during offload, if any

Behavior notes:
- Conversation preload: mm.read_conversation(thread_id, limit=10)
- KB preload: mm.semantic_search('SEMANTIC_MEMORY', question, k=mm.get_config('kb_k'))
- Workflow preload: latest workflow_memory rows with num_steps>0 limit kb_k (fallback if similarity not available)
- Entity preload: most recent entities limit mm.get_config('entity_k') or 5
- Summary preload: summaries for the thread id (if any) limit 10
- If budget status is 'critical', call summarizer.summarize_thread_if_needed to produce a summary and replace conversation segment with a stub including the summary id and add the summary under Summary Memory segment.
- The system message names the segments and states the conflict priority (current question > latest conversation > knowledge-base evidence > older summaries/workflows).
"""
from typing import Dict, Any, List, Optional
from .memory_manager import MemoryManager
from . import summarizer


def assemble_partitioned_context(mm: MemoryManager, llm, thread_id: str, question: str, override_budget: Optional[int] = None) -> Dict[str, Any]:
    # preload conversation
    conv_rows = mm.read_conversation(thread_id, limit=10)
    conv_text_parts = []
    for r in conv_rows:
        conv_text_parts.append(f"[{r['timestamp']}] {r['role']}: {r['content']}")
    conversation_segment = "\n".join(conv_text_parts) if conv_text_parts else "(no recent conversation)"

    # KB preload
    kb_k = int(mm.get_config('kb_k') or 3)
    kb_hits = mm.semantic_search('SEMANTIC_MEMORY', question, kb_k)
    kb_segment_parts = []
    for h in kb_hits:
        md = h.get('metadata', {})
        src = md.get('source') or md.get('file') or md.get('source_file') or 'unknown'
        kb_segment_parts.append(f"- {h.get('text')[:300]} (source: {src})")
    kb_segment = "\n".join(kb_segment_parts) if kb_segment_parts else "(no knowledge-base hits)"

    # Workflow preload: latest runs with num_steps>0
    wf_k = int(mm.get_config('kb_k') or 3)
    cur = mm.conn.cursor()
    cur.execute("SELECT id, query, steps, answer_excerpt, num_steps, success, timestamp FROM workflow_memory WHERE num_steps > 0 ORDER BY timestamp DESC LIMIT ?", (wf_k,))
    wf_rows = cur.fetchall()
    wf_parts = []
    for r in wf_rows:
        wf_parts.append(f"- {r['query']} -> steps={r['num_steps']} success={bool(r['success'])} excerpt={r['answer_excerpt']}")
    workflow_segment = "\n".join(wf_parts) if wf_parts else "(no recent workflows)"

    # Entity preload: most recent entities (limit 5)
    entity_k = 5
    ents = mm.list_entities(limit=entity_k)
    ent_parts = [f"- {e['type']}: {e['name']} ({e.get('description','')})" for e in ents]
    entity_segment = "\n".join(ent_parts) if ent_parts else "(no entities)"

    # Summary preload: summaries for thread
    cur = mm.conn.cursor()
    cur.execute("SELECT id, summary, description, created_at FROM summary_memory WHERE thread_id = ? ORDER BY created_at DESC LIMIT 10", (thread_id,))
    summaries = cur.fetchall()
    summary_parts = []
    for s in summaries:
        summary_parts.append(f"- [Summary ID: {s['id']}] {s['description']}")
    summary_segment = "\n".join(summary_parts) if summary_parts else "(no summaries)"

    # assemble user_message tentatively to estimate budget
    user_message = f"# Question\n{question}\n\n## Conversation Memory\n{conversation_segment}\n\n## Knowledge Base Memory\n{kb_segment}\n\n## Workflow Memory\n{workflow_segment}\n\n## Entity Memory\n{entity_segment}\n\n## Summary Memory\n{summary_segment}\n"

    budget = mm.budget_status_for_text(user_message, override_budget=override_budget)
    summary_id = None
    if budget.get('status') == 'critical':
        # perform summarization offload
        sid = summarizer.summarize_thread_if_needed(mm, llm, thread_id, override_budget=override_budget)
        if sid:
            summary_id = sid
            # replace conversation segment with stub
            conversation_segment = f"(Conversation consolidated into summary: [Summary ID: {sid}])"
            # fetch the summary text to include under Summary Memory
            summ = mm.expand_summary(sid)
            added = f"- [Summary ID: {sid}] {summ.get('summary','(no summary text)')}"
            if summary_segment and summary_segment != "(no summaries)":
                summary_segment = added + "\n" + summary_segment
            else:
                summary_segment = added
            # rebuild user message
            user_message = f"# Question\n{question}\n\n## Conversation Memory\n{conversation_segment}\n\n## Knowledge Base Memory\n{kb_segment}\n\n## Workflow Memory\n{workflow_segment}\n\n## Entity Memory\n{entity_segment}\n\n## Summary Memory\n{summary_segment}\n"
            # recompute budget
            budget = mm.budget_status_for_text(user_message, override_budget=override_budget)

    # system message naming segments and conflict priority
    system_message = (
        "You are a memory-aware research assistant. The user query is provided first.\n"
        "Context segments follow, in this order: Conversation Memory, Knowledge Base Memory, Workflow Memory, Entity Memory, Summary Memory.\n"
        "Use Conversation Memory for the latest thread details; use Knowledge Base Memory for factual evidence; use Workflow Memory for prior procedural runs; use Entity Memory for known entities; use Summary Memory only after expanding the referenced summary.\n"
        "Conflict priority: current question > latest conversation > knowledge-base evidence > older summaries/workflows.\n"
        "If a summary reference is present, expand it before relying on its details. State uncertainty rather than inventing facts.\n"
    )

    return {"system_message": system_message, "user_message": user_message, "budget": budget, "summary_id": summary_id}
