"""
Tool runner: executes a tool function, persists full result to tool_log, creates a model-facing excerpt <=3000 chars
and ensures the stored preview is <=2000 bytes (MemoryManager.write_tool_log handles preview truncation).

API:
- run_tool_and_log(mm, thread_id, tool_name, tool_func, tool_args_dict) -> (model_excerpt, log_id)

The tool_func may raise; on exception we still write a failed tool_log with error message.
"""
from typing import Callable, Any, Dict, Tuple


def run_tool_and_log(mm, thread_id: str, tool_name: str, tool_func: Callable[..., Any], tool_args: Dict[str, Any]) -> Tuple[str, str]:
    try:
        result = tool_func(**tool_args)
        result_text = result if isinstance(result, str) else str(result)
        status = "success"
        error = None
    except Exception as e:
        result_text = str(e)
        status = "failed"
        error = str(e)
    # write full result and preview
    log_id = mm.write_tool_log(thread_id=thread_id, tool_name=tool_name, tool_args=tool_args, result=result_text, status=status, error_message=error)
    # prepare model-facing excerpt: <=3000 chars
    if len(result_text) <= 3000:
        excerpt = result_text
    else:
        excerpt = result_text[:3000]
        excerpt += f"\n\n[Truncated: full results in tool_log id={log_id}]"
    return excerpt, log_id
