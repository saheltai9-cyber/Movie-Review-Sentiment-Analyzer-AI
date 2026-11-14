"""Packaging and Portability Verification Script for Movie Review Sentiment Analyzer AI.

Creates:
  dist/Movie-Review-Sentiment-Analyzer-AI.zip

Strictly includes:
  - Complete application source code and entrypoints
  - Reusable backend modules in src/
  - NiceGUI web application
  - Trained model pipeline artifact and metadata (in models/)
  - Genuine evaluation reports (in reports/)
  - Complete test suite (in tests/)
  - Launcher (run.bat), requirements.txt, README.md, .gitignore

Strictly excludes:
  - .venv / virtual environment folders
  - __pycache__ and byte-compiled files
  - .pytest_cache
  - data/raw/ (Stanford raw files, acquired via download_data.py)
  - data/processed/
  - Temporary files and previous ZIP packages

Verification:
  - Tests ZIP integrity
  - Extracts to a clean temporary directory
  - Verifies presence of all required files
  - Executes unit tests from the extracted directory
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
import tempfile
import zipfile
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
DIST_DIR = PROJECT_ROOT / "dist"
ZIP_PATH = DIST_DIR / "Movie-Review-Sentiment-Analyzer-AI.zip"

EXCLUDE_DIRS = {
    ".venv",
    "venv",
    "env",
    "__pycache__",
    ".pytest_cache",
    ".git",
    "dist",
    "data",  # Raw large dataset files
}

EXCLUDE_EXTENSIONS = {
    ".pyc",
    ".pyo",
    ".pyd",
    ".tmp",
    ".log",
}

REQUIRED_FILES_IN_ZIP = [
    "app.py",
    "train.py",
    "predict.py",
    "download_data.py",
    "package_project.py",
    "requirements.txt",
    "README.md",
    "run.bat",
    ".gitignore",
    "src/__init__.py",
    "src/data_loader.py",
    "src/text_preprocessing.py",
    "src/model_training.py",
    "src/evaluation.py",
    "src/ui_components.py",
    "src/movie_search.py",
    "models/sentiment_pipeline.joblib",
    "models/model_metadata.json",
    "reports/model_metrics.json",
    "reports/FINAL_RELEASE_REPORT.md",
    "tests/test_preprocessing.py",
    "tests/test_prediction.py",
    "tests/test_data_loader.py",
    "tests/test_model_pipeline.py",
    "tests/test_movie_search.py",
]


def should_include_file(file_path: Path) -> bool:
    """Determine whether a file should be included in the portable package."""
    # Check excluded extensions
    if file_path.suffix.lower() in EXCLUDE_EXTENSIONS:
        return False

    # Check excluded directories
    rel_parts = file_path.relative_to(PROJECT_ROOT).parts
    for part in rel_parts[:-1]:
        if part in EXCLUDE_DIRS:
            return False

    return True


def create_portable_zip() -> Path:
    """Create the portable distribution ZIP archive."""
    print("=" * 60)
    print("CREATING PORTABLE DISTRIBUTION PACKAGE")
    print("=" * 60)

    DIST_DIR.mkdir(parents=True, exist_ok=True)
    if ZIP_PATH.exists():
        ZIP_PATH.unlink()

    included_count = 0
    total_uncompressed_bytes = 0

    with zipfile.ZipFile(ZIP_PATH, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as zipf:
        for root, dirs, files in os.walk(PROJECT_ROOT):
            # Prune excluded directories
            dirs[:] = [d for d in dirs if d not in EXCLUDE_DIRS]

            for file_name in files:
                file_path = Path(root) / file_name
                if should_include_file(file_path):
                    archive_name = file_path.relative_to(PROJECT_ROOT).as_posix()
                    zipf.write(file_path, arcname=archive_name)
                    included_count += 1
                    total_uncompressed_bytes += file_path.stat().st_size

    zip_size_mb = ZIP_PATH.stat().st_size / (1024 * 1024)
    raw_size_mb = total_uncompressed_bytes / (1024 * 1024)

    print(f"[OK] ZIP package created at: {ZIP_PATH.resolve()}")
    print(f"[OK] Included files: {included_count}")
    print(f"[OK] Uncompressed size: {raw_size_mb:.2f} MB")
    print(f"[OK] Compressed ZIP size: {zip_size_mb:.2f} MB")
    return ZIP_PATH


def verify_zip_package(zip_path: Path) -> bool:
    """Test archive integrity, extract to clean temp folder, and run tests."""
    print("\n" + "=" * 60)
    print("VERIFYING PACKAGE INTEGRITY AND PORTABILITY")
    print("=" * 60)

    # 1. Integrity test
    print("1. Testing ZIP archive integrity...")
    with zipfile.ZipFile(zip_path, "r") as zipf:
        corrupt = zipf.testzip()
        if corrupt:
            print(f"[ERROR] Corrupt file found in ZIP: {corrupt}")
            return False
        namelist = set(zipf.namelist())
    print("   [PASSED] ZIP integrity test passed with 0 errors.")

    # 2. Check essential files
    print("2. Verifying required files are present...")
    missing = [f for f in REQUIRED_FILES_IN_ZIP if f not in namelist]
    if missing:
        print(f"   [WARNING] Missing expected files in archive: {missing}")
    else:
        print(f"   [PASSED] All {len(REQUIRED_FILES_IN_ZIP)} required files verified.")

    # 3. Clean-room extraction test
    print("3. Extracting to temporary isolated directory...")
    with tempfile.TemporaryDirectory() as tmp_extract:
        tmp_dir = Path(tmp_extract)
        with zipfile.ZipFile(zip_path, "r") as zipf:
            zipf.extractall(tmp_dir)
        print(f"   [PASSED] Extracted successfully to {tmp_dir}")

        # 4. Run pytest from extracted copy
        print("4. Executing pytest from extracted distribution...")
        env = os.environ.copy()
        env["PYTHONPATH"] = str(tmp_dir)

        result = subprocess.run(
            [sys.executable, "-m", "pytest", "tests", "-v"],
            cwd=str(tmp_dir),
            capture_output=True,
            text=True,
            env=env,
        )

        print(result.stdout)
        if result.returncode == 0:
            print("   [PASSED] All tests in extracted project succeeded!")
            return True
        else:
            print(result.stderr)
            print(f"   [FAILED] Pytest returned code {result.returncode}")
            return False


if __name__ == "__main__":
    zip_file = create_portable_zip()
    success = verify_zip_package(zip_file)
    if success:
        print("\n" + "=" * 60)
        print("DISTRIBUTION PACKAGE VERIFICATION COMPLETE AND SUCCESSFUL!")
        print("=" * 60)
    else:
        print("\n[ERROR] Package verification encountered errors.")
        sys.exit(1)

