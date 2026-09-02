"""
LLM client abstractions and scripted behaviors for tests.

ScriptedLLMClient supports exact-match scripts and special markers:
- if the last user message starts with 'SUMMARIZE_THREAD' it returns a canned structured summary
  with the four headings required by R8.
- if the last user message starts with 'EXTRACT_ENTITIES' it returns a bullet list of entities.

OpenAIClient remains a thin wrapper.
"""
from typing import List, Dict, Any, Optional
import os

try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    pass

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")


class ScriptedLLMClient:
    """Deterministic scripted client for offline testing.

    Provide a `script` dict mapping input triggers to canned outputs. If no match is found
    it returns a simple default assistant reply. It recognizes SUMMARIZE_THREAD and
    EXTRACT_ENTITIES markers.
    """

    def __init__(self, script: Optional[Dict[str, str]] = None):
        self.script = script or {}

    def call(self, messages: List[Dict[str, str]], tools: Optional[List[Dict[str, Any]]] = None) -> Dict[str, Any]:
        # Use the last user message as the key
        user_msgs = [m for m in messages if m.get("role") == "user"]
        last = user_msgs[-1]["content"] if user_msgs else ""
        # simple exact-match scripted behaviour
        if last in self.script:
            return {"role": "assistant", "content": self.script[last]}
        # special markers
        if isinstance(last, str) and last.startswith("SUMMARIZE_THREAD"):
            # return a structured 4-heading summary and a deterministic label
            summary = (
                "Technical Information:\n- Kestrel streaming memory consolidation reduces resumption errors.\n\n"
                "Emotional Context:\n- User expressed curiosity and urgency about reproducibility.\n\n"
                "Entities & References:\n- Kestrel (KX-2025-011)\n- Heron (HX-2024-007)\n\n"
                "Action Items & Decisions:\n- Ingest Kestrel notes; summarize and mark rows for consolidation.\n"
            )
            return {"role": "assistant", "content": summary}
        if isinstance(last, str) and last.startswith("EXTRACT_ENTITIES"):
            # produce two entities as bullets
            ents = "- PERSON: R. Marlow\n- SYSTEM: Kestrel Simulation Framework\n"
            return {"role": "assistant", "content": ents}
        # default echo
        return {"role": "assistant", "content": f"I received: {last[:200]}"}


class OpenAIClient:
    """Thin wrapper for OpenAI Chat completions.
    Note: this requires OPENAI_API_KEY in the environment.
    """

    def __init__(self, model: str = "gpt-4o-mini"):
        if not OPENAI_API_KEY:
            raise RuntimeError("OPENAI_API_KEY is not set for OpenAIClient")
        try:
            import openai
        except Exception as e:
            raise
        openai.api_key = OPENAI_API_KEY
        self.model = model

    def call(self, messages: List[Dict[str, str]], tools: Optional[List[Dict[str, Any]]] = None) -> Dict[str, Any]:
        import openai
        # simple call using chat.completions
        resp = openai.ChatCompletion.create(model=self.model, messages=messages)
        content = resp.choices[0].message.content
        return {"role": "assistant", "content": content}
