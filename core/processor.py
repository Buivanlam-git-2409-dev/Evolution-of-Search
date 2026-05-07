"""Text processing và normalization pipeline."""

from __future__ import annotations

import re

from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS


class TextProcessor:
    """Làm sạch và chuẩn hóa văn bản trước khi lập chỉ mục/tìm kiếm.

    Pipeline:
    1) Chuyển về chữ thường và loại bỏ khoảng trắng thừa
    2) Làm sạch bằng Regex (xóa URL, dấu câu, khoảng trắng thừa)
    3) Loại bỏ stopwords

    Example:
        >>> processor = TextProcessor()
        >>> cleaned = processor.clean("Visit https://example.com. Hello, world!")
        >>> print(cleaned)
        'visit hello world'
    """

    def __init__(self) -> None:
        """Khởi tạo TextProcessor với stopwords và regex patterns."""
        self.stopwords = set(ENGLISH_STOP_WORDS)
        self.url_pattern = re.compile(r"https?://\S+|www\.\S+")
        self.non_alpha_pattern = re.compile(r"[^a-z0-9\s]")
        self.multi_space_pattern = re.compile(r"\s+")

    def clean(self, text: str) -> str:
        """Làm sạch và chuẩn hóa một chuỗi văn bản.

        Args:
            text: Văn bản cần làm sạch

        Returns:
            Văn bản đã làm sạch, loại bỏ URL, dấu câu, stopwords
        """
        if not text:
            return ""

        # EN: Normalize surface form. VI: Chuẩn hóa văn bản trước khi tạo vector.
        normalized = text.lower().strip()
        normalized = self.url_pattern.sub(" ", normalized)
        normalized = self.non_alpha_pattern.sub(" ", normalized)
        normalized = self.multi_space_pattern.sub(" ", normalized).strip()

        tokens = [token for token in normalized.split() if token not in self.stopwords]
        return " ".join(tokens)
