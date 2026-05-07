"""Retriever implementations: TF-IDF (cơ bản) và Semantic (nâng cao)."""

from __future__ import annotations

import importlib
import time
from abc import ABC, abstractmethod
from typing import Dict, List

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from .schemas import SearchResult

faiss = None
try:
    faiss = importlib.import_module("faiss")
    FAISS_AVAILABLE = True
except ModuleNotFoundError:
    FAISS_AVAILABLE = False


class BaseRetriever(ABC):
    """Abstract base class cho các retriever."""

    @abstractmethod
    def build_index(self, documents: List[str]) -> None:
        """Xây dựng index từ danh sách documents."""
        pass

    @abstractmethod
    def search(self, query: str, top_k: int = 5) -> Dict[str, object]:
        """Tìm kiếm top-k kết quả cho query."""
        pass


class TFIDFRetriever(BaseRetriever):
    """TF-IDF retriever: Phương pháp tìm kiếm từ khóa truyền thống.

    Sử dụng:
    - TfidfVectorizer từ scikit-learn
    - Unigrams + bigrams (1-2 grams)
    - Cosine similarity ranking
    """

    def __init__(self) -> None:
        self.vectorizer: TfidfVectorizer | None = None
        self.matrix = None
        self.documents: List[str] = []

    def build_index(self, documents: List[str]) -> None:
        """Xây dựng TF-IDF index từ documents.

        Args:
            documents: Danh sách các documents đã được làm sạch
        """
        self.documents = documents
        self.vectorizer = TfidfVectorizer(
            max_features=20000,
            ngram_range=(1, 2),
        )
        self.matrix = self.vectorizer.fit_transform(documents)

    def search(self, query: str, top_k: int = 5) -> Dict[str, object]:
        """Tìm kiếm TF-IDF cho query.

        Args:
            query: Query đã được làm sạch
            top_k: Số kết quả cần trả về

        Returns:
            Dict chứa:
            - results: List[SearchResult]
            - inference_ms: Thời gian xử lý (ms)

        Raises:
            RuntimeError: Nếu index chưa được khởi tạo
        """
        if self.vectorizer is None or self.matrix is None:
            raise RuntimeError("TF-IDF index chưa được khởi tạo. Gọi build_index() trước.")

        start = time.perf_counter()
        query_vector = self.vectorizer.transform([query])
        scores = cosine_similarity(query_vector, self.matrix).flatten()

        top_indices = np.argsort(scores)[-top_k:][::-1]
        elapsed_ms = (time.perf_counter() - start) * 1000

        results = [
            SearchResult(
                rank=rank + 1,
                score=float(scores[idx]),
                text=self.documents[idx],
            )
            for rank, idx in enumerate(top_indices)
        ]

        return {
            "results": results,
            "inference_ms": elapsed_ms,
        }


class SemanticRetriever(BaseRetriever):
    """Semantic retriever: Tìm kiếm dựa trên ý nghĩa sử dụng embeddings.

    Sử dụng:
    - Sentence-Transformers để tạo embeddings
    - FAISS để tìm kiếm nhanh (nếu có)
    - Cosine similarity ranking
    """

    def __init__(self, model_name: str = "all-MiniLM-L6-v2", use_faiss: bool = True) -> None:
        """Khởi tạo SemanticRetriever.

        Args:
            model_name: Tên model sentence-transformer
            use_faiss: Có sử dụng FAISS nếu available không
        """
        self.model_name = model_name
        self.use_faiss = use_faiss and FAISS_AVAILABLE
        self.model = None
        self.vectors: np.ndarray | None = None
        self.faiss_index = None
        self.documents: List[str] = []

    def build_index(self, documents: List[str]) -> None:
        """Xây dựng semantic index từ documents.

        Args:
            documents: Danh sách các documents đã được làm sạch
        """
        self.documents = documents

        try:
            st_module = importlib.import_module("sentence_transformers")
            SentenceTransformer = getattr(st_module, "SentenceTransformer")
        except ModuleNotFoundError as exc:
            raise ImportError(
                "Missing dependency: sentence-transformers. "
                "Please run: pip install -r requirements.txt"
            ) from exc

        self.model = SentenceTransformer(self.model_name)
        vectors = self.model.encode(
            documents,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        self.vectors = vectors.astype("float32")

        # Xây dựng FAISS index nếu available
        if self.use_faiss:
            dimension = self.vectors.shape[1]
            self.faiss_index = faiss.IndexFlatIP(dimension)
            self.faiss_index.add(self.vectors)

    def search(self, query: str, top_k: int = 5) -> Dict[str, object]:
        """Tìm kiếm semantic cho query.

        Args:
            query: Query đã được làm sạch
            top_k: Số kết quả cần trả về

        Returns:
            Dict chứa:
            - results: List[SearchResult]
            - inference_ms: Thời gian xử lý (ms)
            - backend: "FAISS" hoặc "NumPy"

        Raises:
            RuntimeError: Nếu index chưa được khởi tạo
        """
        if self.model is None or self.vectors is None:
            raise RuntimeError("Semantic index chưa được khởi tạo. Gọi build_index() trước.")

        start = time.perf_counter()
        query_vec = self.model.encode(
            [query],
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        ).astype("float32")

        if self.faiss_index is not None:
            scores, indices = self.faiss_index.search(query_vec, top_k)
            top_scores = scores[0]
            top_indices = indices[0]
        else:
            top_scores = np.dot(self.vectors, query_vec[0])
            top_indices = np.argsort(top_scores)[-top_k:][::-1]
            top_scores = top_scores[top_indices]

        elapsed_ms = (time.perf_counter() - start) * 1000

        results = [
            SearchResult(
                rank=rank + 1,
                score=float(score),
                text=self.documents[idx],
            )
            for rank, (idx, score) in enumerate(zip(top_indices, top_scores))
        ]

        return {
            "results": results,
            "inference_ms": elapsed_ms,
            "backend": "FAISS" if self.faiss_index is not None else "NumPy",
        }
