"""Utility functions cho Streamlit UI."""

from typing import List, Dict, Any


def format_inference_ms(milliseconds: float) -> str:
    """Format milliseconds với unit phù hợp.

    Args:
        milliseconds: Thời gian tính bằng ms

    Returns:
        String formatted (ms hoặc µs)
    """
    if milliseconds >= 1000:
        return f"{milliseconds / 1000:.2f}s"
    elif milliseconds < 1:
        return f"{milliseconds * 1000:.2f}µs"
    else:
        return f"{milliseconds:.2f}ms"


def calculate_speedup(baseline_ms: float, advanced_ms: float) -> float:
    """Tính speedup của advanced so với baseline.

    Args:
        baseline_ms: Thời gian baseline (ms)
        advanced_ms: Thời gian advanced (ms)

    Returns:
        Speedup factor
    """
    if advanced_ms == 0:
        return 0.0
    return baseline_ms / advanced_ms


def get_performance_label(speedup: float) -> str:
    """Nhận label cho mức performance.

    Args:
        speedup: Speedup factor

    Returns:
        Performance label (Slower, Similar, Faster, Much Faster)
    """
    if speedup < 0.9:
        return "⚠️ Slower"
    elif speedup < 1.1:
        return "≈ Similar"
    elif speedup < 2.0:
        return "✅ Faster"
    else:
        return "🚀 Much Faster"


def truncate_text(text: str, max_length: int = 100) -> str:
    """Truncate text dengan ellipsis.

    Args:
        text: Văn bản cần truncate
        max_length: Độ dài tối đa

    Returns:
        Truncated text
    """
    if len(text) <= max_length:
        return text
    return text[:max_length] + "..."


def compare_results_quality(baseline_results: List[Dict[str, Any]], advanced_results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """So sánh chất lượng kết quả giữa hai phương pháp.

    Args:
        baseline_results: Kết quả từ TF-IDF
        advanced_results: Kết quả từ Semantic

    Returns:
        Dict với các metrics so sánh
    """
    # Lấy top-1 documents
    baseline_top = baseline_results[0].text if baseline_results else ""
    advanced_top = advanced_results[0].text if advanced_results else ""

    # Kiểm tra overlap
    baseline_ids = {i for i in range(len(baseline_results))}
    advanced_ids = {i for i in range(len(advanced_results))}
    overlap = len(baseline_ids & advanced_ids)

    return {
        "top_1_same": baseline_top == advanced_top,
        "overlap_count": overlap,
        "baseline_avg_score": sum(r.score for r in baseline_results) / len(baseline_results) if baseline_results else 0,
        "advanced_avg_score": sum(r.score for r in advanced_results) / len(advanced_results) if advanced_results else 0,
    }
