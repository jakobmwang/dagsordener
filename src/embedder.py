"""BGE-M3 embedder for dense + sparse vectors."""

import os
import warnings
import logging

# Suppress all transformer/tokenizer noise
os.environ["TRANSFORMERS_VERBOSITY"] = "error"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
logging.getLogger("transformers").setLevel(logging.ERROR)
warnings.filterwarnings("ignore")

# Suppress multiprocess ResourceTracker cleanup noise (Python 3.12 bug)
# https://github.com/python/cpython/issues/140485
def _silence_resource_tracker():
    try:
        from multiprocess import resource_tracker
        # Disable the tracker's destructor
        resource_tracker.ResourceTracker.__del__ = lambda self: None
    except Exception:
        pass

_silence_resource_tracker()

from FlagEmbedding import BGEM3FlagModel


class Embedder:
    def __init__(self, device: str = "cpu", use_fp16: bool = False):
        self.model = BGEM3FlagModel("BAAI/bge-m3", device=device, use_fp16=use_fp16)

    def embed(self, texts: list[str]) -> dict:
        """
        Embed texts and return both dense and sparse vectors.

        Returns:
            {
                "dense": [[float, ...], ...],  # one per text
                "sparse": [{"indices": [...], "values": [...]}, ...]
            }
        """
        output = self.model.encode(
            texts,
            return_dense=True,
            return_sparse=True,
            return_colbert_vecs=False,
        )

        dense = output["dense_vecs"].tolist()

        # Convert sparse format
        sparse = []
        for sp in output["lexical_weights"]:
            indices = list(sp.keys())
            values = list(sp.values())
            sparse.append({"indices": indices, "values": values})

        return {"dense": dense, "sparse": sparse}

    def embed_single(self, text: str) -> dict:
        """Embed a single text."""
        result = self.embed([text])
        return {
            "dense": result["dense"][0],
            "sparse": result["sparse"][0],
        }
