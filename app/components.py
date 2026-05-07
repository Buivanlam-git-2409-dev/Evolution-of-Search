"""Reusable Streamlit components."""

import streamlit as st
from typing import List

from engine import SearchResult


def render_search_result(result: SearchResult) -> None:
    """Render một search result.

    Args:
        result: SearchResult object
    """
    col1, col2 = st.columns([1, 10])

    with col1:
        st.metric("Rank", result.rank)

    with col2:
        st.metric("Score", f"{result.score:.4f}")

    st.write(result.text)
    st.divider()


def render_results_table(results: List[SearchResult], title: str = "Results") -> None:
    """Render danh sách kết quả trong một container.

    Args:
        results: Danh sách SearchResult
        title: Tiêu đề section
    """
    with st.container():
        st.subheader(title)
        for result in results:
            render_search_result(result)


def render_comparison_metrics(baseline_ms: float, advanced_ms: float, backend: str = "N/A") -> None:
    """Render metrics so sánh giữa hai phương pháp.

    Args:
        baseline_ms: Thời gian TF-IDF (ms)
        advanced_ms: Thời gian Semantic (ms)
        backend: Backend sử dụng (FAISS/NumPy)
    """
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("TF-IDF (ms)", f"{baseline_ms:.2f}")

    with col2:
        st.metric("Semantic (ms)", f"{advanced_ms:.2f}")

    with col3:
        speedup = baseline_ms / advanced_ms if advanced_ms > 0 else 0
        st.metric("Speedup", f"{speedup:.2f}x")

    with col4:
        st.metric("Backend", backend)


def render_model_info(model_name: str, use_faiss: bool, num_docs: int) -> None:
    """Render thông tin model.

    Args:
        model_name: Tên embedding model
        use_faiss: Có dùng FAISS không
        num_docs: Số documents
    """
    with st.expander("📋 Model Information"):
        st.write(f"**Embedding Model:** {model_name}")
        st.write(f"**FAISS:** {'✅ Enabled' if use_faiss else '❌ Disabled'}")
        st.write(f"**Documents:** {num_docs}")
