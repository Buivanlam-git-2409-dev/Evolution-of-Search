"""Benchmark hiệu năng search: TF-IDF vs Semantic."""

import sys
import time
import json
from pathlib import Path

from core.engine import SearchEngine


def benchmark_search(
    sample_sizes: list[int] = None,
    num_queries: int = 10,
    top_k: int = 5,
    output_file: str = "logs/benchmark_results.json",
) -> None:
    """Benchmark hiệu năng của hai phương pháp search.

    Args:
        sample_sizes: Danh sách các sample sizes để test
        num_queries: Số lượng queries per sample size
        top_k: Số kết quả trả về
        output_file: File lưu kết quả
    """
    if sample_sizes is None:
        sample_sizes = [1000, 1500, 2000]

    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    results = []

    for sample_size in sample_sizes:
        print(f"\n🔄 Benchmark với sample_size={sample_size}...")

        try:
            engine = SearchEngine(sample_size=sample_size, use_faiss=True)
            engine.prepare()

            # Test queries
            test_queries = [
                "machine learning",
                "artificial intelligence",
                "neural networks",
                "deep learning",
                "data science",
                "computer vision",
                "nlp",
                "embeddings",
                "vectors",
                "similarity",
            ][:num_queries]

            tfidf_times = []
            semantic_times = []

            for query in test_queries:
                comparison = engine.compare(query, top_k=top_k)
                tfidf_times.append(comparison["baseline"]["inference_ms"])
                semantic_times.append(comparison["advanced"]["inference_ms"])

            avg_tfidf = sum(tfidf_times) / len(tfidf_times)
            avg_semantic = sum(semantic_times) / len(semantic_times)

            result = {
                "sample_size": sample_size,
                "num_queries": len(test_queries),
                "tfidf_avg_ms": round(avg_tfidf, 3),
                "semantic_avg_ms": round(avg_semantic, 3),
                "speedup": round(avg_tfidf / avg_semantic, 2),
                "backend": engine.semantic_retriever.faiss_index is not None and "FAISS" or "NumPy",
            }
            results.append(result)

            print(f"  ✅ TF-IDF avg: {avg_tfidf:.3f} ms")
            print(f"  ✅ Semantic avg: {avg_semantic:.3f} ms")
            print(f"  ✅ Speedup: {avg_tfidf / avg_semantic:.2f}x")

        except Exception as e:
            print(f"  ❌ Lỗi: {e}")

    # Lưu kết quả
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\n\n📊 Benchmark Summary:")
    print(f"{'Sample Size':<15} {'TF-IDF (ms)':<15} {'Semantic (ms)':<15} {'Speedup':<10} {'Backend':<10}")
    print("-" * 70)
    for r in results:
        print(
            f"{r['sample_size']:<15} {r['tfidf_avg_ms']:<15} {r['semantic_avg_ms']:<15} "
            f"{r['speedup']:<10} {r['backend']:<10}"
        )

    print(f"\n✅ Kết quả đã lưu tại: {output_path}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Benchmark search performance")
    parser.add_argument(
        "--sample-sizes",
        nargs="+",
        type=int,
        default=[1000, 1500, 2000],
        help="Sample sizes (default: 1000 1500 2000)",
    )
    parser.add_argument(
        "--num-queries",
        type=int,
        default=10,
        help="Number of queries per sample (default: 10)",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=5,
        help="Top K results (default: 5)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="logs/benchmark_results.json",
        help="Output file (default: logs/benchmark_results.json)",
    )

    args = parser.parse_args()
    benchmark_search(
        sample_sizes=args.sample_sizes,
        num_queries=args.num_queries,
        top_k=args.top_k,
        output_file=args.output,
    )
