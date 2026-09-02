"""
Toolbox augmenter: enrich a tool's docstring via the LLM and register it in the toolbox store.

Usage:
  augment_and_register_tool(mm, llm, name, docstring, signature, metadata=None, augment=True, augment_text=None)

Behavior:
- If augment_text is provided, uses it directly (useful for scripted tests).
- Else, tries to call the llm with a short augmentation prompt. If the llm call fails or returns empty,
  falls back to a mechanically generated augmented text consisting of the docstring + 5 synthetic example queries.
- Registers idempotently via mm.register_tool, storing the augmented text as the stored description.
"""
from typing import Optional, Dict
from .memory_manager import MemoryManager


def _make_synthetic_queries(docstring: str, n: int = 5):
    # naive split into topics for placeholder queries
    lines = [l.strip() for l in docstring.splitlines() if l.strip()]
    topic = lines[0][:80] if lines else "tool"
    queries = [f"How to use {topic} for task example {i+1}?" for i in range(n)]
    return queries


def augment_and_register_tool(mm: MemoryManager, llm, name: str, docstring: str, signature: Dict = None, metadata: Dict = None, augment: bool = True, augment_text: Optional[str] = None) -> str:
    # Prepare augmented text
    if augment and augment_text:
        augmented = augment_text
    elif augment and hasattr(llm, "call"):
        try:
            prompt = (
                "AUGMENT_TOOL\nRewrite the following function docstring into a rich description suitable for embedding and retrieval. "
                "Also list 5 short example user queries that would indicate this tool should be used.\n\n"
                f"DOCSTRING:\n{docstring}\n"
            )
            resp = llm.call([{"role": "user", "content": prompt}])
            augmented = resp.get("content") if isinstance(resp, dict) else str(resp)
            if not augmented:
                raise ValueError("empty augmentation")
        except Exception:
            # fallback
            queries = _make_synthetic_queries(docstring)
            augmented = docstring + "\n\nExample queries:\n" + "\n".join([f"- {q}" for q in queries])
    else:
        # no augmentation requested
        queries = _make_synthetic_queries(docstring)
        augmented = docstring + "\n\nExample queries:\n" + "\n".join([f"- {q}" for q in queries])

    # include the 5 synthetic queries in metadata for searchability too
    meta = metadata.copy() if metadata else {}
    meta.setdefault("name", name)
    meta.setdefault("signature", signature or {})
    meta.setdefault("augmented_marker", True)
    meta.setdefault("example_queries", _make_synthetic_queries(docstring))

    # Idempotent register; mm.register_tool will avoid duplicates by name
    tid = mm.register_tool(name=name, description=augmented, signature=signature, metadata=meta, augment_text=augmented)
    return tid
