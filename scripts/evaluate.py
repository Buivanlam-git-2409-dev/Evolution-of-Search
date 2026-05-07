"""Đánh giá chất lượng truy hồi của cả hai phương pháp."""

import sys
from pathlib import Path

from core.engine import SearchEngine


def evaluate_search(queries: list[str], top_k: int = 5) -> None:
    """Đánh giá chất lượng truy hồi.

    Args:
        queries: Danh sách các queries để kiểm tra
        top_k: Số kết quả cần trả về
    """
    print("🔄 Đang khởi tạo engine...")
    engine = SearchEngine(sample_size=1500, use_faiss=True)
    engine.prepare()

    print(f"\n📊 Đánh giá với {len(queries)} queries (top_k={top_k}):\n")

    total_tfidf_time = 0.0
    total_semantic_time = 0.0

    for i, query in enumerate(queries, 1):
        print(f"Query {i}: \"{query}\"")

        try:
            results = engine.compare(query, top_k=top_k)
            baseline = results["baseline"]
            advanced = results["advanced"]

            tfidf_time = baseline["inference_ms"]
            semantic_time = advanced["inference_ms"]

            total_tfidf_time += tfidf_time
            total_semantic_time += semantic_time

            print(f"  TF-IDF:    {tfidf_time:.2f} ms")
            print(f"  Semantic:  {semantic_time:.2f} ms ({advanced.get('backend', 'N/A')})")
            print(f"  Top result (TF-IDF): {baseline['results'][0].text[:60]}...")
            print(f"  Top result (Semantic): {advanced['results'][0].text[:60]}...")
            print()

        except Exception as e:
            print(f"  ❌ Lỗi: {e}\n")

    print(f"\n📈 Summary:")
    print(f"  - Total TF-IDF time: {total_tfidf_time:.2f} ms")
    print(f"  - Total Semantic time: {total_semantic_time:.2f} ms")
    print(f"  - Avg TF-IDF time: {total_tfidf_time/len(queries):.2f} ms")
    print(f"  - Avg Semantic time: {total_semantic_time/len(queries):.2f} ms")

    speedup = total_tfidf_time / total_semantic_time
    print(f"  - Speedup (TF-IDF / Semantic): {speedup:.2f}x")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Đánh giá chất lượng truy hồi")
    parser.add_argument(
        "--queries",
        nargs="+",
        default=[
            "machine learning algorithms",
            "artificial intelligence applications",
            "deep neural networks",
            "computer vision tasks",
            "natural language processing",
        ],
        help="Danh sách queries (default: pre-defined)",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=5,
        help="Số kết quả (default: 5)",
    )

    args = parser.parse_args()
    evaluate_search(queries=args.queries, top_k=args.top_k)
