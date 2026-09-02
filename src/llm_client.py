"""
LLM client abstractions.

Provides a ScriptedLLMClient for offline deterministic tests and a thin OpenAI wrapper
for live runs. The scripted client supports two simple behaviors used by tests:
- echo: returns a canned assistant response
- extract_entities / summarize could be added later

This file intentionally keeps the interface minimal: call(messages, tools=None) -> dict
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
    it returns a simple default assistant reply.
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
