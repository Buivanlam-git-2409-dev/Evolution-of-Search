"""Giao diện Streamlit để so sánh song song: TF-IDF và Semantic Search."""

from __future__ import annotations

import streamlit as st

from core.engine import SearchEngine


st.set_page_config(
    page_title="Evolution of Search: TF-IDF vs Semantic",
    page_icon="🔎",
    layout="wide",
)

st.title("Evolution of Search: Từ khóa dựa trên từ khóa đến Tìm kiếm Ngữ nghĩa")
st.caption(
    "So sánh phiên bản cơ bản (TF-IDF + Cosine Similarity) và Nâng cao (Sentence Embeddings + FAISS) trong việc truy vấn tập dữ liệu MS MARCO. Model semantic sử dụng Sentence-Transformers để mã hóa câu thành vector ý nghĩa, giúp cải thiện độ chính xác và tốc độ truy vấn so với phương pháp truyền thống."
    "(Sentence-Transformers all-MiniLM-L6-v2)."
)


@st.cache_resource(show_spinner=True)
def load_engine(sample_size: int) -> SearchEngine:
    # Lưu cache tài nguyên nặng (dataset/model) để tăng tốc các lần chạy lại.
    engine = SearchEngine(sample_size=sample_size, model_name="all-MiniLM-L6-v2", use_faiss=True)
    engine.prepare()
    return engine


with st.sidebar:
    st.header("Cấu hình")
    sample_size = st.slider("Số lượng đoạn văn MS MARCO", min_value=1000, max_value=2000, value=1500, step=100)
    top_k = st.slider("Số kết quả Top K", min_value=3, max_value=10, value=5, step=1)
    st.info("Model: sentence-transformers/all-MiniLM-L6-v2")

    st.markdown("### Hướng dẫn lưu trữ (bản Nâng cao)")
    st.markdown(
        "- **Faiss (đã tích hợp):** Tìm kiếm độ tương đồng vector nhanh trong bộ nhớ.\n"
        "- **ChromaDB (tùy chọn):** Lưu trữ bền vững và lọc metadata tốt hơn cho demo production."
    )

    with st.expander("Cách sử dụng ChromaDB"):
        st.code(
            """
from chromadb import PersistentClient

client = PersistentClient(path="./chroma_store")
collection = client.get_or_create_collection(name="msmarco")

collection.add(
    ids=[str(i) for i in range(len(passages))],
    documents=passages,
    embeddings=vectors.tolist(),
)

results = collection.query(query_embeddings=[query_vec.tolist()], n_results=5)
            """.strip(),
            language="python",
        )

engine = load_engine(sample_size=sample_size)

query = st.text_input("Nhập truy vấn", placeholder="Ví dụ: Làm sao để cải thiện trí nhớ khi học?")

if query:
    # Chạy đồng thời 2 phiên bản trên cùng truy vấn để so sánh trực tiếp.
    outputs = engine.compare(query=query, top_k=top_k)
    baseline = outputs["baseline"]
    advanced = outputs["advanced"]

    metric_col1, metric_col2 = st.columns(2)
    metric_col1.metric("Thời gian suy luận TF-IDF (ms)", f"{baseline['inference_ms']:.2f}")
    metric_col2.metric(
        f"Thời gian suy luận Semantic (ms) [{advanced.get('backend', 'N/A')} ]",
        f"{advanced['inference_ms']:.2f}",
    )

    tab1, tab2 = st.tabs(["Cơ bản: TF-IDF", "Nâng cao: Semantic Search"])

    with tab1:
        st.subheader("Kết quả đối khớp theo từ khóa")
        for item in baseline["results"]:
            st.markdown(
                f"**#{item.rank} | Điểm: {item.score:.4f}**\n\n"
                f"{item.text}"
            )
            st.divider()

    with tab2:
        st.subheader("Kết quả truy hồi theo ngữ nghĩa")
        for item in advanced["results"]:
            st.markdown(
                f"**#{item.rank} | Điểm: {item.score:.4f}**\n\n"
                f"{item.text}"
            )
            st.divider()

st.markdown("---")
st.markdown("### Vì sao Semantic Search thường tốt hơn với truy vấn có ngữ cảnh?")
st.markdown(
    "1. TF-IDF phụ thuộc vào từ xuất hiện chính xác, nên dễ bỏ sót từ đồng nghĩa và câu diễn đạt lại.\n"
    "2. Sentence Embeddings mã hóa ý nghĩa câu, nên các cụm từ liên quan sẽ nằm gần nhau trong không gian vector.\n"
    "3. Kết hợp FAISS/Chroma giúp Semantic Search mở rộng tốt hơn mà vẫn giữ độ liên quan theo ngữ cảnh."
)

st.caption(
    "Lưu ý: Lần chạy đầu có thể chậm hơn vì cần tải model và dataset từ Hugging Face."
)