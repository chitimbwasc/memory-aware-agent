"""
Memory Manager extended with toolbox registration, toolbox retrieval, and budget monitoring.

Added functions:
- register_tool(name, description, signature, metadata, augment=False, augmented_text=None)
- toolbox_retrieve(query, k=None)
- count_toolbox_entries()
- estimate_token_usage(text) -> int
- budget_status_for_text(text, override_budget=None) -> (status, estimate)

Toolbox entries are stored in semantic_store_kv with store_name 'TOOLBOX_MEMORY' and metadata must include {'name': tool_name}
"""

import sqlite3
import json
import uuid
from datetime import datetime
from typing import List, Dict, Optional
from pathlib import Path
from math import sqrt

from . import embeddings

DEFAULT_DB = "data/agent_memory.db"


def _now_iso():
    return datetime.utcnow().isoformat() + "Z"


class MemoryManager:
    def __init__(self, db_path: str = DEFAULT_DB):
        self.db_path = db_path
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(db_path, check_same_thread=False)
        self.conn.row_factory = sqlite3.Row
        self._init_db()

    def _init_db(self):
        sql = Path(__file__).parent / "schema.sql"
        with open(sql, "r", encoding="utf-8") as fh:
            self.conn.executescript(fh.read())
        # set defaults if not present
        cur = self.conn.cursor()
        cur.execute("INSERT OR IGNORE INTO config(key, value) VALUES(?, ?)", ("token_budget", "256000"))
        cur.execute("INSERT OR IGNORE INTO config(key, value) VALUES(?, ?)", ("toolbox_k", "5"))
        cur.execute("INSERT OR IGNORE INTO config(key, value) VALUES(?, ?)", ("kb_k", "3"))
        cur.execute("INSERT OR IGNORE INTO config(key, value) VALUES(?, ?)", ("distance_strategy", "cosine"))
        self.conn.commit()

    def get_config(self, key: str) -> Optional[str]:
        cur = self.conn.cursor()
        cur.execute("SELECT value FROM config WHERE key = ?", (key,))
        r = cur.fetchone()
        return r["value"] if r else None

    def set_config(self, key: str, value: str):
        cur = self.conn.cursor()
        cur.execute("INSERT OR REPLACE INTO config(key, value) VALUES(?, ?)", (key, value))
        self.conn.commit()

    # Conversation operations (R1, R2, R3)
    def append_conversation_record(self, thread_id: str, role: str, content: str, metadata: Dict = None) -> str:
        rid = uuid.uuid4().hex[:8]
        ts = _now_iso()
        cur = self.conn.cursor()
        cur.execute(
            "INSERT INTO conversational_memory(id, thread_id, role, content, timestamp, metadata, summary_id) VALUES(?,?,?,?,?,?,?)",
            (rid, thread_id, role, content, ts, json.dumps(metadata or {}), None),
        )
        self.conn.commit()
        return rid

    def read_conversation(self, thread_id: str, limit: int = 10) -> List[Dict]:
        cur = self.conn.cursor()
        cur.execute(
            "SELECT id, thread_id, role, content, timestamp, metadata, summary_id FROM conversational_memory "
            "WHERE thread_id = ? AND (summary_id IS NULL) ORDER BY timestamp ASC LIMIT ?",
            (thread_id, limit),
        )
        rows = cur.fetchall()
        return [dict(r) for r in rows]

    # Simple semantic store read (brute-force cosine)
    def _load_embeddings_for_store(self, store_name: str) -> List[Dict]:
        cur = self.conn.cursor()
        cur.execute(
            "SELECT id, text_content, metadata, embedding FROM semantic_store_kv WHERE store_name = ?",
            (store_name,),
        )
        rows = cur.fetchall()
        results = []
        for r in rows:
            emb = json.loads(r["embedding"]) if r["embedding"] else None
            results.append({"id": r["id"], "text": r["text_content"], "metadata": json.loads(r["metadata"] or "{}"), "embedding": emb})
        return results

    def _cosine_similarity(self, a: List[float], b: List[float]) -> float:
        if not a or not b:
            return 0.0
        # simple dot / (||a||*||b||)
        dot = sum(x * y for x, y in zip(a, b))
        na = sqrt(sum(x * x for x in a))
        nb = sqrt(sum(y * y for y in b))
        if na == 0 or nb == 0:
            return 0.0
        return dot / (na * nb)

    def semantic_search(self, store_name: str, query: str, k: int) -> List[Dict]:
        # embed the query
        q_emb = embeddings.embed_text([query])[0]
        entries = self._load_embeddings_for_store(store_name)
        scored = []
        for e in entries:
            score = self._cosine_similarity(q_emb, e["embedding"]) if e["embedding"] else 0.0
            scored.append((score, e))
        scored.sort(key=lambda x: x[0], reverse=True)
        return [item for score, item in scored[:k]]

    def write_semantic(self, store_name: str, text_content: str, metadata: Dict = None) -> str:
        rid = uuid.uuid4().hex[:8]
        created = _now_iso()
        emb = embeddings.embed_text([text_content])[0]
        cur = self.conn.cursor()
        cur.execute(
            "INSERT INTO semantic_store_kv(id, store_name, text_content, metadata, embedding, created_at) VALUES(?,?,?,?,?,?)",
            (rid, store_name, text_content, json.dumps(metadata or {}), json.dumps(emb), created),
        )
        self.conn.commit()
        return rid

    # Toolbox registration & retrieval (R4, R5, D11)
    def register_tool(self, name: str, description: str, signature: Dict = None, metadata: Dict = None, augment_text: Optional[str] = None) -> str:
        """Idempotent registration by tool name. Stores name in metadata for deduplication."""
        # check existing by name
        cur = self.conn.cursor()
        cur.execute("SELECT id FROM semantic_store_kv WHERE store_name = ? AND json_extract(metadata, '$.name') = ?", ("TOOLBOX_MEMORY", name))
        if cur.fetchone():
            # return existing id
            cur.execute("SELECT id FROM semantic_store_kv WHERE store_name = ? AND json_extract(metadata, '$.name') = ?", ("TOOLBOX_MEMORY", name))
            return cur.fetchone()[0]
        text = augment_text if augment_text else description
        meta = metadata or {}
        meta.update({"name": name, "signature": signature or {}})
        return self.write_semantic("TOOLBOX_MEMORY", text, meta)

    def toolbox_retrieve(self, query: str, k: Optional[int] = None) -> List[Dict]:
        k = k or int(self.get_config("toolbox_k") or 5)
        results = self.semantic_search("TOOLBOX_MEMORY", query, k * 3)  # fetch more, dedupe
        seen = set()
        out = []
        for r in results:
            name = r["metadata"].get("name")
            if not name:
                continue
            if name in seen:
                continue
            seen.add(name)
            # map to RetrievedToolSchema-like dict
            func = {"name": name, "description": r["text"], "parameters": r["metadata"].get("signature", {})}
            out.append({"type": "function", "function": func})
            if len(out) >= k:
                break
        return out

    def count_toolbox_entries(self) -> int:
        cur = self.conn.cursor()
        cur.execute("SELECT COUNT(*) FROM semantic_store_kv WHERE store_name = ?", ("TOOLBOX_MEMORY",))
        return cur.fetchone()[0]

    # Tool log (R10)
    def write_tool_log(self, thread_id: str, tool_name: str, tool_args: Dict, result: str, status: str = "success", error_message: str = None) -> str:
        rid = uuid.uuid4().hex[:8]
        ts = _now_iso()
        preview = (result.encode("utf-8")[:2000]).decode("utf-8", errors="ignore")
        cur = self.conn.cursor()
        cur.execute(
            "INSERT INTO tool_log(id, thread_id, tool_name, tool_args, result, result_preview, status, error_message, metadata, timestamp) VALUES(?,?,?,?,?,?,?,?,?,?)",
            (rid, thread_id, tool_name, json.dumps(tool_args or {}), result, preview, status, error_message, json.dumps({}), ts),
        )
        self.conn.commit()
        return rid

    # Summaries (R8, R9) - create a summary and mark source rows
    def create_summary_and_mark_rows(self, thread_id: str, summary_text: str, description: str, source_row_ids: List[str]) -> str:
        sid = uuid.uuid4().hex[:8]
        ts = _now_iso()
        cur = self.conn.cursor()
        cur.execute(
            "INSERT INTO summary_memory(id, summary, description, full_content, thread_id, created_at) VALUES(?,?,?,?,?,?)",
            (sid, summary_text, description, summary_text, thread_id, ts),
        )
        # mark rows
        for rid in source_row_ids:
            cur.execute("UPDATE conversational_memory SET summary_id = ? WHERE id = ?", (sid, rid))
        self.conn.commit()
        return sid

    def expand_summary(self, summary_id: str) -> Dict:
        cur = self.conn.cursor()
        cur.execute("SELECT summary, full_content, thread_id FROM summary_memory WHERE id = ?", (summary_id,))
        row = cur.fetchone()
        if not row:
            return {}
        cur.execute("SELECT id, role, content, timestamp FROM conversational_memory WHERE thread_id = ? AND summary_id = ? ORDER BY timestamp ASC", (row["thread_id"], summary_id))
        rows = [dict(r) for r in cur.fetchall()]
        return {"summary": row["summary"], "full_content": row["full_content"], "messages": rows}

    # Budget monitor (R6)
    def estimate_token_usage(self, text: str) -> int:
        return max(1, len(text) // 4)

    def budget_status_for_text(self, text: str, override_budget: Optional[int] = None) -> Dict:
        est = self.estimate_token_usage(text)
        budget = int(override_budget or int(self.get_config("token_budget") or 256000))
        pct = est * 100.0 / budget
        if pct < 50:
            status = "ok"
        elif pct < 80:
            status = "warning"
        else:
            status = "critical"
        return {"status": status, "estimate": est, "budget": budget, "pct": pct}

