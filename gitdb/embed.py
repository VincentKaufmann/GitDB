"""Embedding backends — Arctic Embed (default) and NV-EmbedQA.

Lazy-loaded: no heavy imports until you actually call embed().
Supports text-in for add, query, cherry-pick-by-semantics, semantic blame, etc.

Matryoshka Representation Learning (MRL):
  NV-EmbedQA supports truncating embeddings to smaller dims with minimal quality loss.
  Use dim=384/512/768/1024/2048 to trade quality for speed/storage.
  GitDB stores track their embedding dim, so Matryoshka lets you pick the sweet spot
  per-store: 384d for fast prototyping, 1024d for production, 2048d for max quality.
"""

import os
from typing import Dict, List, Optional, Union

import torch

# ── Model registry ──────────────────────────────────────────

MODELS: Dict[str, dict] = {
    "arctic": {
        "hf_name": "Snowflake/snowflake-arctic-embed-l",
        "dim": 1024,
        "max_tokens": 512,
        "params": "335M",
        "matryoshka": False,
        "description": "Fast, lightweight, strong retrieval (default)",
    },
    "nv-embed-qa": {
        "hf_name": "nvidia/llama-3.2-nv-embedqa-1b-v2",
        "dim": 2048,
        "max_tokens": 512,
        "params": "1.2B",
        "matryoshka": True,
        "matryoshka_dims": [384, 512, 768, 1024, 2048],
        "description": "NVIDIA retrieval QA, Matryoshka dims (384-2048)",
    },
}

DEFAULT_MODEL = "arctic"

# ── Singleton model cache ───────────────────────────────────

_loaded_model = None
_loaded_model_name = None
_loaded_tokenizer = None


def _get_device() -> str:
    """Pick the best available device."""
    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def _load_model(model_name: str = DEFAULT_MODEL):
    """Load a model on first use. Returns (model, tokenizer)."""
    global _loaded_model, _loaded_model_name, _loaded_tokenizer

    if _loaded_model is not None and _loaded_model_name == model_name:
        return _loaded_model, _loaded_tokenizer

    if model_name not in MODELS:
        raise ValueError(f"Unknown model: {model_name}. Available: {list(MODELS.keys())}")

    spec = MODELS[model_name]

    try:
        from sentence_transformers import SentenceTransformer
    except ImportError:
        raise ImportError(
            "sentence-transformers is required for embedding. "
            "Install with: pip install sentence-transformers"
        )

    device = _get_device()
    model = SentenceTransformer(spec["hf_name"], device=device, trust_remote_code=True)

    _loaded_model = model
    _loaded_model_name = model_name
    _loaded_tokenizer = None  # SentenceTransformer handles tokenization internally
    return model, None


# ── Public API ──────────────────────────────────────────────

def embed(
    texts: Union[str, List[str]],
    model_name: str = DEFAULT_MODEL,
    dim: Optional[int] = None,
    normalize: bool = True,
) -> torch.Tensor:
    """Embed text(s) into vectors.

    Args:
        texts: Single string or list of strings.
        model_name: Model key from MODELS registry.
        dim: Target dimension (for Matryoshka truncation). None = model default.
        normalize: L2-normalize output vectors.

    Returns:
        Tensor of shape (N, dim) on CPU.
    """
    if isinstance(texts, str):
        texts = [texts]

    model, _ = _load_model(model_name)
    spec = MODELS[model_name]

    # Validate Matryoshka dim if requested
    target_dim = dim or spec["dim"]
    if dim is not None and spec.get("matryoshka"):
        valid_dims = spec["matryoshka_dims"]
        if dim not in valid_dims:
            raise ValueError(
                f"Matryoshka dim {dim} not supported by {model_name}. "
                f"Valid: {valid_dims}"
            )

    # Encode
    embeddings = model.encode(
        texts,
        convert_to_tensor=True,
        normalize_embeddings=normalize,
        show_progress_bar=len(texts) > 100,
    )

    # Matryoshka truncation — slice to target dim and re-normalize
    if target_dim < embeddings.shape[1]:
        embeddings = embeddings[:, :target_dim]
        if normalize:
            embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)

    return embeddings.cpu()


def embed_query(
    query: str,
    model_name: str = DEFAULT_MODEL,
    dim: Optional[int] = None,
) -> torch.Tensor:
    """Embed a single query string. Returns shape (dim,)."""
    return embed(query, model_name=model_name, dim=dim).squeeze(0)


def get_model_info(model_name: str = DEFAULT_MODEL) -> dict:
    """Return model spec for display."""
    if model_name not in MODELS:
        raise ValueError(f"Unknown model: {model_name}")
    return dict(MODELS[model_name])


def list_models() -> Dict[str, dict]:
    """List all available embedding models."""
    return {k: dict(v) for k, v in MODELS.items()}


def similarity(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Cosine similarity between two sets of embeddings.

    Args:
        a: (N, D) or (D,)
        b: (M, D) or (D,)
    Returns:
        (N, M) similarity matrix or scalar.
    """
    if a.dim() == 1:
        a = a.unsqueeze(0)
    if b.dim() == 1:
        b = b.unsqueeze(0)
    a = torch.nn.functional.normalize(a, p=2, dim=1)
    b = torch.nn.functional.normalize(b, p=2, dim=1)
    return torch.mm(a, b.t())


def semantic_filter(
    embeddings: torch.Tensor,
    query: str,
    threshold: float = 0.5,
    model_name: str = DEFAULT_MODEL,
    dim: Optional[int] = None,
) -> List[int]:
    """Return indices of embeddings semantically similar to query.

    Args:
        embeddings: (N, D) tensor to search.
        query: Text query to match against.
        threshold: Minimum cosine similarity.
        model_name: Embedding model to use.
        dim: Target dimension.

    Returns:
        List of row indices where similarity >= threshold.
    """
    q = embed_query(query, model_name=model_name, dim=dim)
    sims = similarity(q, embeddings).squeeze(0)
    mask = sims >= threshold
    return mask.nonzero(as_tuple=True)[0].tolist()
