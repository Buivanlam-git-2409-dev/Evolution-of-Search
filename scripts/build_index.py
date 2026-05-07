"""Xây dựng TF-IDF và Semantic indices."""

import sys
from pathlib import Path

import pickle


def build_indices(sample_size: int = 1500, output_dir: str = "models") -> None:
    """Xây dựng indices cho cả TF-IDF và Semantic search.

    Args:
        sample_size: Số lượng documents
        output_dir: Thư mục lưu indices
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    print(f"🔄 Đang xây dựng indices ({sample_size} documents)...")

    try:
        # Import từ core package
        sys.path.insert(0, str(Path(__file__).parent.parent))
        from core.engine import SearchEngine

        # Khởi tạo engine
        engine = SearchEngine(sample_size=sample_size, use_faiss=True)
        print("📦 Đang tải dữ liệu và xây dựng indices...")
        engine.prepare()

        # Lưu TF-IDF retriever
        tfidf_path = output_path / "tfidf_retriever.pkl"
        with open(tfidf_path, "wb") as f:
            pickle.dump(engine.tfidf_retriever, f)
        print(f"✅ TF-IDF retriever lưu tại: {tfidf_path}")

        # Lưu Semantic retriever
        semantic_path = output_path / "semantic_retriever.pkl"
        with open(semantic_path, "wb") as f:
            pickle.dump(engine.semantic_retriever, f)
        print(f"✅ Semantic retriever lưu tại: {semantic_path}")

        print(f"\n📊 Index info:")
        print(f"   - Documents: {len(engine.cleaned_documents)}")
        print(f"   - TF-IDF matrix shape: {engine.tfidf_retriever.matrix.shape}")
        print(f"   - Semantic vectors shape: {engine.semantic_retriever.vectors.shape}")
        print(f"   - FAISS available: {engine.semantic_retriever.faiss_index is not None}")

    except Exception as e:
        print(f"❌ Lỗi khi xây dựng indices: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Xây dựng search indices")
    parser.add_argument(
        "--sample-size",
        type=int,
        default=1500,
        help="Số lượng documents (default: 1500)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="models",
        help="Thư mục output (default: models)",
    )

    args = parser.parse_args()
    build_indices(sample_size=args.sample_size, output_dir=args.output_dir)
