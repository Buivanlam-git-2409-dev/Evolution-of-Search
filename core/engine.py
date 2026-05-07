"""Các bộ máy tìm kiếm để so sánh truy hồi theo từ khóa và theo ngữ nghĩa.

Project: Evolution of Search: From Keyword-based to Semantic Search
"""

from __future__ import annotations

from typing import Dict, List

from datasets import load_dataset

from .processor import TextProcessor
from .retrievers import TFIDFRetriever, SemanticRetriever


class SearchEngine:
    """Xây dựng và truy vấn cho cả 2 bộ máy: Cơ bản (TF-IDF) và Nâng cao (Semantic)."""

    def __init__(
        self,
        sample_size: int = 1500,
        model_name: str = "all-MiniLM-L6-v2",
        use_faiss: bool = True,
    ) -> None:
        """Khởi tạo SearchEngine.

        Args:
            sample_size: Số lượng documents từ MS MARCO dataset
            model_name: Tên model sentence-transformer
            use_faiss: Có sử dụng FAISS nếu available không
        """
        self.sample_size = sample_size
        self.model_name = model_name

        self.processor = TextProcessor()
        self.tfidf_retriever = TFIDFRetriever()
        self.semantic_retriever = SemanticRetriever(
            model_name=model_name,
            use_faiss=use_faiss,
        )

        self.raw_documents: List[str] = []
        self.cleaned_documents: List[str] = []

    def prepare(self) -> None:
        """Tải dữ liệu và tạo index cho cả 2 cách tiếp cận truy hồi."""
        # Dùng chung một corpus để kết quả so sánh công bằng giữa hai phương pháp.
        self.raw_documents = self._load_ms_marco_passages(self.sample_size)
        self.cleaned_documents = [self.processor.clean(text) for text in self.raw_documents]

        self.tfidf_retriever.build_index(self.cleaned_documents)
        self.semantic_retriever.build_index(self.cleaned_documents)

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

    def search_tfidf(self, query: str, top_k: int = 5) -> Dict[str, object]:
        """Tìm kiếm bằng TF-IDF.

        Args:
            query: Query (sẽ được làm sạch tự động)
            top_k: Số kết quả cần trả về

        Returns:
            Dict chứa results và inference_ms
        """
        cleaned_query = self.processor.clean(query)
        return self.tfidf_retriever.search(cleaned_query, top_k=top_k)

    def search_semantic(self, query: str, top_k: int = 5) -> Dict[str, object]:
        """Tìm kiếm bằng Semantic.

        Args:
            query: Query (sẽ được làm sạch tự động)
            top_k: Số kết quả cần trả về

        Returns:
            Dict chứa results, inference_ms, và backend
        """
        cleaned_query = self.processor.clean(query)
        return self.semantic_retriever.search(cleaned_query, top_k=top_k)

    def compare(self, query: str, top_k: int = 5) -> Dict[str, Dict[str, object]]:
        """So sánh kết quả tìm kiếm TF-IDF vs Semantic.

        Args:
            query: Query cần so sánh
            top_k: Số kết quả cần trả về

        Returns:
            Dict với keys "baseline" và "advanced"
        """
        return {
            "baseline": self.search_tfidf(query, top_k=top_k),
            "advanced": self.search_semantic(query, top_k=top_k),
        }