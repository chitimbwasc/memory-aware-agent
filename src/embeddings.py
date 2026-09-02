"""
Embeddings helper: wraps sentence-transformers model and exposes encode(text)->list[float].

Default model: sentence-transformers/paraphrase-mpnet-base-v2
"""

from typing import List
import threading
import numpy as np

_model = None
_model_lock = threading.Lock()


def load_model(model_name: str = "sentence-transformers/paraphrase-mpnet-base-v2"):
    global _model
    with _model_lock:
        if _model is None:
            from sentence_transformers import SentenceTransformer
            _model = SentenceTransformer(model_name)
    return _model


def embed_text(texts: List[str], model_name: str = None) -> List[List[float]]:
    model = _model if _model is not None else load_model(model_name or "sentence-transformers/paraphrase-mpnet-base-v2")
    embs = model.encode(texts, show_progress_bar=False, convert_to_numpy=True)
    # convert to regular Python lists for JSON storage
    return [list(map(float, e.tolist())) for e in np.atleast_2d(embs)]
