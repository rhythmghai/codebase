"""
Embedding layer, built behind a single interface so the backend is swappable
without touching any retrieval code.

Why pluggable: this was built in a sandboxed dev environment without access
to huggingface.co, so TfidfEmbedder is the default for local dev/testing.
On a real machine or in deployment, swap to SentenceTransformerEmbedder
(all-MiniLM-L6-v2, 384-dim, free/local) or GeminiEmbedder
(gemini-embedding-001, 768-dim, matches the CareerRadar embedding setup).

Every embedder exposes the same contract:
    fit(corpus: list[str]) -> None          # only meaningful for TF-IDF
    encode(texts: list[str]) -> np.ndarray   # shape (n, dim), L2-normalized
"""

import numpy as np
from abc import ABC, abstractmethod


class BaseEmbedder(ABC):
    dim: int

    def fit(self, corpus: list[str]) -> None:
        pass

    @abstractmethod
    def encode(self, texts: list[str]) -> np.ndarray:
        ...


class TfidfEmbedder(BaseEmbedder):
    """
    Offline stand-in for a dense embedding model. Not semantically as strong
    as a neural embedder, but it's a real, defensible vector representation
    (sparse -> dense via SVD) and it lets the full hybrid pipeline run
    end-to-end without network access.
    """

    def __init__(self, dim: int = 256):
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.decomposition import TruncatedSVD

        self.dim = dim
        self.vectorizer = TfidfVectorizer(
            max_features=20000,
            ngram_range=(1, 2),
            token_pattern=r"(?u)\b\w[\w_]+\b",  # keep underscores -> code identifiers survive
        )
        self.svd = TruncatedSVD(n_components=dim, random_state=42)
        self._fitted = False

    def fit(self, corpus: list[str]) -> None:
        tfidf = self.vectorizer.fit_transform(corpus)
        self.svd.fit(tfidf)
        self._fitted = True

    def encode(self, texts: list[str]) -> np.ndarray:
        if not self._fitted:
            raise RuntimeError("TfidfEmbedder.fit() must be called on the corpus before encode()")
        tfidf = self.vectorizer.transform(texts)
        dense = self.svd.transform(tfidf)
        norms = np.linalg.norm(dense, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        return dense / norms


class SentenceTransformerEmbedder(BaseEmbedder):
    """Real neural embedder. Use this outside the sandbox / in deployment."""

    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        from sentence_transformers import SentenceTransformer
        self.model = SentenceTransformer(model_name)
        self.dim = self.model.get_sentence_embedding_dimension()

    def encode(self, texts: list[str]) -> np.ndarray:
        return self.model.encode(texts, normalize_embeddings=True, show_progress_bar=False)


class GeminiEmbedder(BaseEmbedder):
    def __init__(self, api_key: str, model_name: str = "gemini-embedding-001"):
        from google import genai
        self.client = genai.Client(api_key=api_key)
        self.model_name = model_name
        self.dim = 768

    def encode(self, texts: list[str]) -> np.ndarray:
        resp = self.client.models.embed_content(model=self.model_name, contents=texts)
        arr = np.array([e.values for e in resp.embeddings])
        norms = np.linalg.norm(arr, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        return arr / norms


def get_embedder(backend: str = "tfidf", **kwargs) -> BaseEmbedder:
    if backend == "tfidf":
        return TfidfEmbedder(**kwargs)
    if backend == "sentence-transformer":
        return SentenceTransformerEmbedder(**kwargs)
    if backend == "gemini":
        return GeminiEmbedder(**kwargs)
    raise ValueError(f"Unknown embedder backend: {backend}")
