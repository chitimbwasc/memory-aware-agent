-- Database schema for the memory core (SQLite)

CREATE TABLE IF NOT EXISTS conversational_memory (
  id TEXT PRIMARY KEY,
  thread_id TEXT NOT NULL,
  role TEXT CHECK(role IN ('user','assistant')) NOT NULL,
  content TEXT NOT NULL,
  timestamp TEXT NOT NULL,
  metadata TEXT,
  summary_id TEXT
);

CREATE INDEX IF NOT EXISTS idx_conv_thread_ts ON conversational_memory(thread_id, timestamp);

CREATE TABLE IF NOT EXISTS tool_log (
  id TEXT PRIMARY KEY,
  thread_id TEXT NOT NULL,
  tool_name TEXT NOT NULL,
  tool_args TEXT,
  result TEXT,
  result_preview TEXT,
  status TEXT CHECK(status IN ('success','failed')) NOT NULL,
  error_message TEXT,
  metadata TEXT,
  timestamp TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS summary_memory (
  id TEXT PRIMARY KEY,
  summary TEXT NOT NULL,
  description TEXT NOT NULL,
  full_content TEXT NOT NULL,
  thread_id TEXT,
  created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS workflow_memory (
  id TEXT PRIMARY KEY,
  query TEXT NOT NULL,
  steps TEXT NOT NULL, -- JSON array
  answer_excerpt TEXT,
  num_steps INTEGER NOT NULL,
  success INTEGER NOT NULL,
  timestamp TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS entity_memory (
  id TEXT PRIMARY KEY,
  name TEXT NOT NULL,
  type TEXT CHECK(type IN ('PERSON','PLACE','SYSTEM','UNKNOWN')) NOT NULL,
  description TEXT,
  embedding TEXT, -- JSON list of floats
  metadata TEXT,
  created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS semantic_store_kv (
  id TEXT PRIMARY KEY,
  store_name TEXT NOT NULL, -- e.g., 'SEMANTIC_MEMORY', 'TOOLBOX_MEMORY'
  text_content TEXT NOT NULL,
  metadata TEXT,
  embedding TEXT,
  created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS config (
  key TEXT PRIMARY KEY,
  value TEXT NOT NULL
);
