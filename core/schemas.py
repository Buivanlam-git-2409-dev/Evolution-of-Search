"""Data structures và Response models cho search engine."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class SearchResult:
    """Một kết quả tìm kiếm.
    
    Attributes:
        rank: Vị trí trong danh sách kết quả (bắt đầu từ 1)
        score: Điểm độ tương đồng (0-1)
        text: Văn bản của đoạn kết quả
    """

    rank: int
    score: float
    text: str
