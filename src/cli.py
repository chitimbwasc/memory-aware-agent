"""
Simple CLI for the memory-aware agent (prototype).

Runs a minimal REPL that persists conversation records and demonstrates deterministic
conversation persistence (R1). Uses ScriptedLLMClient when OPENAI_API_KEY is absent.
"""

import os
from src.memory_manager import MemoryManager
from src.llm_client import ScriptedLLMClient, OpenAIClient


def build_system_prompt():
    return (
        "# System\n" 
        "You are a memory-aware research assistant. Use the provided context segments as named."
    )


def assemble_partitioned_context(mm: MemoryManager, thread_id: str, question: str) -> str:
    parts = []
    parts.append("# Question\n" + question)
    conv = mm.read_conversation(thread_id, limit=10)
    parts.append("## Conversation Memory\n" + "\n".join([f"[{r['timestamp']}] {r['role']}: {r['content']}" for r in conv]) if conv else "## Conversation Memory\n(none)")
    # KB / workflow / entity / summary placeholders
    parts.append("## Knowledge Base Memory\n(none)")
    parts.append("## Workflow Memory\n(none)")
    parts.append("## Entity Memory\n(none)")
    parts.append("## Summary Memory\n(none)")
    return "\n\n".join(parts)


def main(db_path: str = None):
    mm = MemoryManager(db_path=db_path) if db_path else MemoryManager()
    # choose scripted client if no OPENAI_API_KEY
    try:
        llm = OpenAIClient()
    except Exception:
        llm = ScriptedLLMClient()

    thread_id = "default-thread"
    print("Memory-aware agent REPL (type 'exit' to quit)")
    while True:
        try:
            user = input("You: ")
        except EOFError:
            break
        if not user:
            continue
        if user.strip().lower() in ("exit", "quit"):
            break
        # persist user message
        mm.append_conversation_record(thread_id=thread_id, role="user", content=user)
        context = assemble_partitioned_context(mm, thread_id, user)
        # call LLM
        messages = [{"role": "system", "content": build_system_prompt()}, {"role": "user", "content": context}]
        resp = llm.call(messages)
        content = resp.get("content") if isinstance(resp, dict) else str(resp)
        # persist assistant reply
        mm.append_conversation_record(thread_id=thread_id, role="assistant", content=content)
        print("Assistant:", content)


if __name__ == "__main__":
    main()
