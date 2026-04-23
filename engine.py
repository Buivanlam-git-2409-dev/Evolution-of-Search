"""Các bộ máy tìm kiếm để so sánh truy hồi theo từ khóa và theo ngữ nghĩa.

Project: Evolution of Search: From Keyword-based to Semantic Search
"""

from __future__ import annotations

import re
import time
import importlib
from dataclasses import dataclass
from typing import Dict, List

import numpy as np
from datasets import load_dataset
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS, TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

faiss = None
try:
    faiss = importlib.import_module("faiss")

    FAISS_AVAILABLE = True
except ModuleNotFoundError:
    FAISS_AVAILABLE = False


@dataclass
class SearchResult:
    """Một kết quả tìm kiếm."""

    rank: int
    score: float
    text: str


class TextProcessor:
    """Làm sạch và chuẩn hóa văn bản trước khi lập chỉ mục/tìm kiếm.

    Pipeline:
    1) Chuyển về chữ thường và loại bỏ khoảng trắng thừa
    2) Làm sạch bằng Regex (xóa URL, dấu câu, khoảng trắng thừa)
    3) Loại bỏ stopwords
    """

    def __init__(self) -> None:
        self.stopwords = set(ENGLISH_STOP_WORDS)
        self.url_pattern = re.compile(r"https?://\S+|www\.\S+")
        self.non_alpha_pattern = re.compile(r"[^a-z0-9\s]")
        self.multi_space_pattern = re.compile(r"\s+")

    def clean(self, text: str) -> str:
        if not text:
            return ""

        # EN: Normalize surface form. VI: Chuẩn hóa văn bản trước khi tạo vector.
        normalized = text.lower().strip()
        normalized = self.url_pattern.sub(" ", normalized)
        normalized = self.non_alpha_pattern.sub(" ", normalized)
        normalized = self.multi_space_pattern.sub(" ", normalized).strip()

        tokens = [token for token in normalized.split() if token not in self.stopwords]
        return " ".join(tokens)


class SearchEngine:
    """Xây dựng và truy vấn cho cả 2 bộ máy: Cơ bản (TF-IDF) và Nâng cao (Semantic)."""

    def __init__(
        self,
        sample_size: int = 1500,
        model_name: str = "all-MiniLM-L6-v2",
        use_faiss: bool = True,
    ) -> None:
        self.sample_size = sample_size
        self.model_name = model_name
        self.use_faiss = use_faiss and FAISS_AVAILABLE

        self.processor = TextProcessor()

        self.raw_documents: List[str] = []
        self.cleaned_documents: List[str] = []

        self.tfidf_vectorizer: TfidfVectorizer | None = None
        self.tfidf_matrix = None

        self.semantic_model = None
        self.semantic_vectors: np.ndarray | None = None
        self.faiss_index = None

    def prepare(self) -> None:
        """Tải dữ liệu và tạo index cho cả 2 cách tiếp cận truy hồi."""
        # Dùng chung một corpus để kết quả so sánh công bằng giữa hai phương pháp.
        self.raw_documents = self._load_ms_marco_passages(self.sample_size)
        self.cleaned_documents = [self.processor.clean(text) for text in self.raw_documents]

        self._build_tfidf_index()
        self._build_semantic_index()

    def _load_ms_marco_passages(self, sample_size: int) -> List[str]:
        """Tải khoảng 1000-2000 đoạn văn từ MS MARCO cho mục đích demo."""
        dataset = load_dataset(
            "microsoft/ms_marco",
            "v2.1",
            split=f"validation[:{sample_size}]",
        )

        passages: List[str] = []
        seen = set()

        for row in dataset:
            row_passages = row.get("passages", {}).get("passage_text", [])
            for passage in row_passages:
                if not passage:
                    continue
                text = passage.strip()
                if text and text not in seen:
                    passages.append(text)
                    seen.add(text)
                if len(passages) >= sample_size:
                    return passages

        return passages

    def _build_tfidf_index(self) -> None:
        self.tfidf_vectorizer = TfidfVectorizer(
            max_features=20000,
            ngram_range=(1, 2),
        )
        self.tfidf_matrix = self.tfidf_vectorizer.fit_transform(self.cleaned_documents)

    def _build_semantic_index(self) -> None:
        try:
            st_module = importlib.import_module("sentence_transformers")
            sentence_transformer_cls = getattr(st_module, "SentenceTransformer")
        except ModuleNotFoundError as exc:
            raise ImportError(
                "Missing dependency: sentence-transformers. "
                "Please run: pip install -r requirements.txt"
            ) from exc

        self.semantic_model = sentence_transformer_cls(self.model_name)
        vectors = self.semantic_model.encode(
            self.cleaned_documents,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        self.semantic_vectors = vectors.astype("float32")

        # FAISS speeds up nearest-neighbor search for larger corpora.
        # FAISS giúp tăng tốc tìm kiếm lân cận gần nhất khi corpus lớn.
        if self.use_faiss:
            dimension = self.semantic_vectors.shape[1]
            self.faiss_index = faiss.IndexFlatIP(dimension)
            self.faiss_index.add(self.semantic_vectors)

    def search_tfidf(self, query: str, top_k: int = 5) -> Dict[str, object]:
        if not self.tfidf_vectorizer or self.tfidf_matrix is None:
            raise RuntimeError("Index TF-IDF chưa được khởi tạo.")

        # Thời gian này bao gồm làm sạch query và truy hồi kết quả.
        start = time.perf_counter()
        cleaned_query = self.processor.clean(query)
        query_vector = self.tfidf_vectorizer.transform([cleaned_query])
        scores = cosine_similarity(query_vector, self.tfidf_matrix).flatten()

        top_indices = np.argsort(scores)[-top_k:][::-1]
        elapsed_ms = (time.perf_counter() - start) * 1000

        results = [
            SearchResult(
                rank=rank + 1,
                score=float(scores[idx]),
                text=self.raw_documents[idx],
            )
            for rank, idx in enumerate(top_indices)
        ]

        return {
            "results": results,
            "inference_ms": elapsed_ms,
        }

    def search_semantic(self, query: str, top_k: int = 5) -> Dict[str, object]:
        if self.semantic_model is None or self.semantic_vectors is None:
            raise RuntimeError("Index Semantic chưa được khởi tạo.")

        # Đo độ trễ của quá trình embedding + tìm kiếm lân cận để so sánh runtime.
        start = time.perf_counter()
        cleaned_query = self.processor.clean(query)
        query_vec = self.semantic_model.encode(
            [cleaned_query],
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        ).astype("float32")

        if self.faiss_index is not None:
            scores, indices = self.faiss_index.search(query_vec, top_k)
            top_scores = scores[0]
            top_indices = indices[0]
        else:
            top_scores = np.dot(self.semantic_vectors, query_vec[0])
            top_indices = np.argsort(top_scores)[-top_k:][::-1]
            top_scores = top_scores[top_indices]

        elapsed_ms = (time.perf_counter() - start) * 1000

        results = [
            SearchResult(
                rank=rank + 1,
                score=float(score),
                text=self.raw_documents[idx],
            )
            for rank, (idx, score) in enumerate(zip(top_indices, top_scores))
        ]

        return {
            "results": results,
            "inference_ms": elapsed_ms,
            "backend": "FAISS" if self.faiss_index is not None else "NumPy",
        }

    def compare(self, query: str, top_k: int = 5) -> Dict[str, Dict[str, object]]:
        return {
            "baseline": self.search_tfidf(query, top_k=top_k),
            "advanced": self.search_semantic(query, top_k=top_k),
        }