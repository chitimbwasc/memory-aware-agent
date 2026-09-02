import json
from typing import List, Dict, Any, Optional
from .memory_manager import MemoryManager


def run_workflow_record(mm: MemoryManager, thread_id: str, steps: List[Dict[str, Any]], final_answer: str) -> str:
    """
    Persist a workflow run: steps is a list of {'description': str, 'outcome': 'success'|'failed'}
    This writes a workflow_memory row only when len(steps) > 0 per R11.
    Returns the workflow id.
    """
    if not steps:
        # do not write zero-step workflows
        return None
    wid = uuid = __import__('uuid').uuid4().hex[:8]
    ts = __import__('datetime').datetime.utcnow().isoformat() + 'Z'
    steps_text = json.dumps(steps)
    answer_excerpt = final_answer[:200]
    num_steps = len(steps)
    success = any(s.get('outcome') == 'success' for s in steps)
    cur = mm.conn.cursor()
    cur.execute(
        "INSERT INTO workflow_memory(id, query, steps, answer_excerpt, num_steps, success, timestamp) VALUES(?,?,?,?,?,?,?)",
        (uuid, thread_id, steps_text, answer_excerpt, num_steps, 1 if success else 0, ts),
    )
    mm.conn.commit()
    return uuid
