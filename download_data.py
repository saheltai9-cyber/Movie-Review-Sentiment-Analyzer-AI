"""Stanford IMDB Dataset Download and Extraction Script.

Official dataset details:
- Large Movie Review Dataset (Andrew Maas et al., Stanford University)
- 50,000 polarized reviews (25,000 train, 25,000 test; 12,500 pos / 12,500 neg in each)
- Primary canonical source: https://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz
- Verified backup mirrors:
  - Zenodo curated release (DOI 10.5281/zenodo.7929635)
  - Official Hugging Face stanfordnlp/imdb parquet mirror
"""

from __future__ import annotations

import argparse
import io
import logging
import os
import shutil
import tarfile
import urllib.request
import zipfile
from pathlib import Path
from typing import Optional

from src.data_loader import validate_dataset_structure

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

PRIMARY_STANFORD_URL = "https://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz"
HTTP_STANFORD_URL = "http://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz"
ZENODO_ZIP_URL = "https://zenodo.org/api/records/7929635/files/large_movie_review_dataset.zip/content"
HF_TRAIN_PARQUET = "https://huggingface.co/datasets/stanfordnlp/imdb/resolve/main/plain_text/train-00000-of-00001.parquet"
HF_TEST_PARQUET = "https://huggingface.co/datasets/stanfordnlp/imdb/resolve/main/plain_text/test-00000-of-00001.parquet"


def download_file_with_progress(url: str, dest_path: Path, timeout: int = 15) -> bool:
    """Download a file with user-agent and progress reporting.

    Args:
        url: URL to download.
        dest_path: File path on disk.
        timeout: Socket timeout in seconds.

    Returns:
        True if successfully downloaded, False otherwise.
    """
    logger.info("Attempting download from: %s", url)
    headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) SentimentAnalyzer/1.0"}
    req = urllib.request.Request(url, headers=headers)

    try:
        with urllib.request.urlopen(req, timeout=timeout) as response:
            total_size = int(response.headers.get("content-length", 0))
            block_size = 1024 * 64  # 64 KB chunks
            downloaded = 0

            dest_path.parent.mkdir(parents=True, exist_ok=True)
            temp_file = dest_path.with_suffix(".tmp")

            with open(temp_file, "wb") as f:
                while True:
                    chunk = response.read(block_size)
                    if not chunk:
                        break
                    f.write(chunk)
                    downloaded += len(chunk)
                    if total_size > 0:
                        pct = (downloaded / total_size) * 100
                        if downloaded % (1024 * 1024 * 5) < block_size:
                            logger.info(
                                "Downloaded %5.1f MB / %5.1f MB (%.1f%%)",
                                downloaded / (1024 * 1024),
                                total_size / (1024 * 1024),
                                pct,
                            )

            # Rename temp file to final destination once complete
            if temp_file.exists():
                if dest_path.exists():
                    dest_path.unlink()
                temp_file.rename(dest_path)

        logger.info("Successfully downloaded to %s (%d bytes)", dest_path, dest_path.stat().st_size)
        return True

    except Exception as exc:
        logger.warning("Download failed for %s: %s", url, exc)
        temp_file = dest_path.with_suffix(".tmp")
        if temp_file.exists():
            temp_file.unlink()
        return False


def extract_tar_gz(archive_path: Path, extract_to: Path) -> bool:
    """Safely extract a .tar.gz archive into target directory."""
    logger.info("Extracting %s to %s...", archive_path, extract_to)
    try:
        with tarfile.open(archive_path, "r:gz") as tar:
            tar.extractall(path=extract_to)
        logger.info("Extraction complete.")
        return True
    except Exception as exc:
        logger.error("Error extracting tar.gz archive: %s", exc)
        return False


def extract_zip(archive_path: Path, extract_to: Path) -> bool:
    """Safely extract a .zip archive into target directory and standardize paths."""
    logger.info("Extracting %s to %s...", archive_path, extract_to)
    try:
        with zipfile.ZipFile(archive_path, "r") as z:
            z.extractall(path=extract_to)

        # Standardize folder naming: if extracted as large_movie_review_dataset, normalize to aclImdb
        extracted_folder = extract_to / "large_movie_review_dataset"
        target_acl = extract_to / "aclImdb"

        if extracted_folder.exists() and not target_acl.exists():
            extracted_folder.rename(target_acl)
        elif extracted_folder.exists() and target_acl.exists():
            # Merge if necessary
            for item in extracted_folder.iterdir():
                dest = target_acl / item.name
                if not dest.exists():
                    shutil.move(str(item), str(dest))
            shutil.rmtree(extracted_folder, ignore_errors=True)

        logger.info("ZIP extraction and standardization complete.")
        return True
    except Exception as exc:
        logger.error("Error extracting ZIP archive: %s", exc)
        return False


def reconstruct_raw_from_hf_parquet(
    train_parquet: Path, test_parquet: Path, target_dir: Path
) -> bool:
    """Reconstruct the official aclImdb directory structure from Hugging Face IMDB parquet files.

    Writes all 50,000 individual review text files into:
    - target_dir/train/pos/*.txt (12,500)
    - target_dir/train/neg/*.txt (12,500)
    - target_dir/test/pos/*.txt (12,500)
    - target_dir/test/neg/*.txt (12,500)
    """
    try:
        import pandas as pd
    except ImportError:
        logger.error("Pandas is required to reconstruct raw dataset from Parquet files.")
        return False

    logger.info("Reconstructing official aclImdb directory structure at %s...", target_dir)
    splits = [("train", train_parquet), ("test", test_parquet)]

    for split_name, parquet_path in splits:
        logger.info("Processing %s split from %s...", split_name, parquet_path)
        df = pd.read_parquet(parquet_path)

        pos_dir = target_dir / split_name / "pos"
        neg_dir = target_dir / split_name / "neg"
        pos_dir.mkdir(parents=True, exist_ok=True)
        neg_dir.mkdir(parents=True, exist_ok=True)

        pos_idx = 0
        neg_idx = 0

        for _, row in df.iterrows():
            text = row["text"]
            label = int(row["label"])
            if label == 1:
                file_path = pos_dir / f"{pos_idx}_pos.txt"
                pos_idx += 1
            else:
                file_path = neg_dir / f"{neg_idx}_neg.txt"
                neg_idx += 1

            with open(file_path, "w", encoding="utf-8") as f:
                f.write(text)

        logger.info(
            "Created %d positive and %d negative reviews in %s/%s",
            pos_idx,
            neg_idx,
            target_dir,
            split_name,
        )

    return True


def ensure_imdb_dataset(data_dir: Path | str = "data/raw") -> Path:
    """Ensure the Stanford IMDB dataset is downloaded and extracted.

    Checks:
    1. If dataset already exists and is valid, skips download.
    2. Tries primary Stanford URLs first.
    3. Falls back gracefully to Zenodo archive or HuggingFace parquet mirror if Stanford server is down.
    4. Validates final extracted structure.

    Args:
        data_dir: Directory where raw dataset should reside.

    Returns:
        Path to the verified aclImdb dataset folder.

    Raises:
        RuntimeError: If dataset could not be downloaded or validated.
    """
    raw_path = Path(data_dir)
    raw_path.mkdir(parents=True, exist_ok=True)
    target_acl = raw_path / "aclImdb"

    # Step 1: Check existing dataset
    val = validate_dataset_structure(target_acl)
    if val["valid"] and val["total"] == 50_000:
        logger.info(
            "Stanford IMDB dataset already verified at %s (%d files). Skipping download.",
            target_acl,
            val["total"],
        )
        return target_acl

    # Step 2: Try Stanford primary .tar.gz
    tar_archive = raw_path / "aclImdb_v1.tar.gz"
    downloaded = False

    for st_url in [PRIMARY_STANFORD_URL, HTTP_STANFORD_URL]:
        if download_file_with_progress(st_url, tar_archive, timeout=10):
            if extract_tar_gz(tar_archive, raw_path):
                downloaded = True
                break

    # Step 3: Fallback to Zenodo ZIP if Stanford is unreachable
    if not downloaded:
        logger.warning(
            "Stanford primary server did not respond. Trying official Zenodo mirror (DOI 10.5281/zenodo.7929635)..."
        )
        zenodo_archive = raw_path / "large_movie_review_dataset.zip"
        if download_file_with_progress(ZENODO_ZIP_URL, zenodo_archive, timeout=30):
            if extract_zip(zenodo_archive, raw_path):
                downloaded = True

    # Step 4: Fallback to Hugging Face verified Parquet mirror if Zenodo is slow
    if not downloaded:
        logger.warning(
            "Zenodo mirror unavailable. Using official Stanfordnlp IMDB dataset mirror from Hugging Face..."
        )
        train_pq = raw_path / "train.parquet"
        test_pq = raw_path / "test.parquet"

        if download_file_with_progress(HF_TRAIN_PARQUET, train_pq, timeout=20) and \
           download_file_with_progress(HF_TEST_PARQUET, test_pq, timeout=20):
            if reconstruct_raw_from_hf_parquet(train_pq, test_pq, target_acl):
                downloaded = True

    # Step 5: Final Validation
    val_final = validate_dataset_structure(target_acl)
    if val_final["valid"] and val_final["total"] == 50_000:
        logger.info(
            "Dataset verification successful! 50,000 files verified at %s.",
            target_acl,
        )
        return target_acl
    elif val_final["valid"] and val_final["total"] > 0:
        logger.warning(
            "Dataset loaded with %d reviews. (Full expected count is 50,000).",
            val_final["total"],
        )
        return target_acl
    else:
        raise RuntimeError(
            f"Failed to prepare Stanford IMDB dataset. Validation errors: {val_final['errors']}"
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download Stanford Large Movie Review Dataset")
    parser.add_argument(
        "--data-dir",
        type=str,
        default="data/raw",
        help="Target folder for raw dataset (default: data/raw)",
    )
    args = parser.parse_args()

    try:
        acl_path = ensure_imdb_dataset(args.data_dir)
        print(f"SUCCESS: Stanford IMDB dataset ready at: {acl_path.resolve()}")
    except Exception as e:
        print(f"ERROR: {e}")
        exit(1)

