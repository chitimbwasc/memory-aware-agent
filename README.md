Memory-Aware Research Assistant (prototype)

Overview
- CLI/REPL research assistant with persistent Agent Memory Core (SQLite).
- Seven typed stores (conversational, tool_log, summary, workflow, entity, KB, toolbox).
- Deterministic preload, budget guard, summarization offload, tool logging, search-and-store acquisition.
- Modes: offline (scripted LLM) and live (OpenAI keys required).

Quickstart (offline/scripted)
- Create virtualenv, install dependencies:
  python -m venv .venv && source .venv/bin/activate
  pip install -r requirements.txt
- Copy .env.example -> .env and set OPENAI_API_KEY only if you will run live tests
- Run CLI:
  python -m src.cli

Tests
- pytest -q -m "not live"

Notes
- This is the initial scaffolding. The code follows the Decision Ledger defaults from the spec.
