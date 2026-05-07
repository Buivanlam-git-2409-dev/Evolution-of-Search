"""Tải MS MARCO dataset cho demo."""

import sys
from pathlib import Path

from datasets import load_dataset


def download_ms_marco(sample_size: int = 1500, output_dir: str = "data/raw") -> None:
    """Tải MS MARCO dataset.

    Args:
        sample_size: Số lượng samples cần tải
        output_dir: Thư mục lưu dataset
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    print(f"Đang tải MS MARCO dataset ({sample_size} samples)...")

    try:
        # Tải từ Hugging Face
        dataset = load_dataset(
            "microsoft/ms_marco",
            "v2.1",
            split=f"validation[:{sample_size}]",
        )

        # Lưu dataset
        dataset.save_to_disk(str(output_path / "ms_marco"))
        print(f"Dataset đã lưu tại: {output_path / 'ms_marco'}")

        print(f"Dataset info:")
        print(f"   - Samples: {len(dataset)}")
        print(f"   - Columns: {dataset.column_names}")

    except Exception as e:
        print(f"Lỗi khi tải dataset: {e}")
        sys.exit(1)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Tải MS MARCO dataset")
    parser.add_argument(
        "--sample-size",
        type=int,
        default=1500,
        help="Số lượng samples (default: 1500)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="data/raw",
        help="Thư mục output (default: data/raw)",
    )

    args = parser.parse_args()
    download_ms_marco(sample_size=args.sample_size, output_dir=args.output_dir)
