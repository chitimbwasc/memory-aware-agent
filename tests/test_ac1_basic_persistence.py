import os
import uuid
from pathlib import Path
import json

import pytest

from src.memory_manager import MemoryManager


def test_basic_persistence(tmp_path):
    # create a temporary DB path
    db_file = tmp_path / "agent_memory.db"
    db_path = str(db_file)

    # first process: write seed thread of 12 messages
    mm1 = MemoryManager(db_path=db_path)
    thread_id = "seed-01"
    messages = []
    for i in range(12):
        role = "user" if i % 2 == 0 else "assistant"
        content = f"Message {i} from {role}"
        rid = mm1.append_conversation_record(thread_id=thread_id, role=role, content=content)
        messages.append({"id": rid, "role": role, "content": content})

    # second process: new MemoryManager instance reading the same DB
    mm2 = MemoryManager(db_path=db_path)
    conv = mm2.read_conversation(thread_id, limit=20)

    # assert we retrieved all 12 messages in chronological order
    assert len(conv) == 12
    for i, row in enumerate(conv):
        expected_role = "user" if i % 2 == 0 else "assistant"
        assert row["role"] == expected_role
        assert row["content"] == f"Message {i} from {expected_role}"
