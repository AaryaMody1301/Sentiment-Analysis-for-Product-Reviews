from __future__ import annotations

import logging
import os
import time
import zipfile
from datetime import datetime
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)


def create_directory_if_not_exists(directory: str | os.PathLike[str]) -> None:
    Path(directory).mkdir(parents=True, exist_ok=True)


def format_file_size(size_bytes: int | float) -> str:
    size = float(size_bytes)
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if abs(size) < 1024.0:
            return f"{size:.2f} {unit}"
        size /= 1024.0
    return f"{size:.2f} PB"


def get_file_info(file_path: str | os.PathLike[str]) -> dict[str, object] | None:
    path = Path(file_path)
    if not path.is_file():
        return None
    stat = path.stat()
    row_count = None
    columns = None
    if path.suffix.lower() == ".csv":
        try:
            header = pd.read_csv(path, nrows=0)
            columns = header.columns.tolist()
            row_count = sum(len(chunk) for chunk in pd.read_csv(path, chunksize=100_000))
        except (OSError, UnicodeError, pd.errors.ParserError) as exc:
            logger.warning("Could not inspect CSV %s: %s", path, exc)
    return {
        "name": path.name,
        "path": str(path),
        "size": stat.st_size,
        "size_human": format_file_size(stat.st_size),
        "extension": path.suffix,
        "last_modified": datetime.fromtimestamp(stat.st_mtime).strftime("%Y-%m-%d %H:%M:%S"),
        "row_count": row_count,
        "columns": columns,
    }


def compress_directory(
    directory: str | os.PathLike[str], output_path: str | os.PathLike[str] | None = None
) -> str:
    source = Path(directory)
    if not source.is_dir():
        raise ValueError(f"Directory not found: {source}")
    destination = Path(output_path) if output_path is not None else source.with_suffix(".zip")
    with zipfile.ZipFile(destination, "w", zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(source.rglob("*")):
            if path.is_file():
                archive.write(path, path.relative_to(source.parent))
    return str(destination)


def extract_zip_file(
    zip_path: str | os.PathLike[str], extract_to: str | os.PathLike[str] | None = None
) -> str:
    archive_path = Path(zip_path)
    if not archive_path.is_file():
        raise ValueError(f"Zip file not found: {archive_path}")
    destination = Path(extract_to) if extract_to is not None else archive_path.with_suffix("")
    destination.mkdir(parents=True, exist_ok=True)
    root = destination.resolve()
    with zipfile.ZipFile(archive_path, "r") as archive:
        for member in archive.infolist():
            target = (root / member.filename).resolve()
            if target != root and root not in target.parents:
                raise ValueError(f"Unsafe zip member path: {member.filename!r}")
        archive.extractall(root)
    return str(destination)


def list_files_by_extension(
    directory: str | os.PathLike[str], extension: str
) -> list[str]:
    suffix = extension if extension.startswith(".") else f".{extension}"
    return [str(path) for path in sorted(Path(directory).glob(f"*{suffix}")) if path.is_file()]


def create_sample_dataset(
    input_csv: str | os.PathLike[str],
    output_csv: str | os.PathLike[str],
    sample_size: int = 1000,
    random_seed: int = 42,
) -> str:
    """Write an exact-size deterministic sample of a CSV dataset."""

    source = Path(input_csv)
    destination = Path(output_csv)
    if not source.is_file():
        raise ValueError(f"Input CSV file not found: {source}")
    if sample_size <= 0:
        raise ValueError("sample_size must be positive.")

    frame = pd.read_csv(source)
    sample = frame.sample(n=min(sample_size, len(frame)), random_state=random_seed)
    destination.parent.mkdir(parents=True, exist_ok=True)
    sample.to_csv(destination, index=False)
    return str(destination)


def get_process_memory_usage() -> float | None:
    try:
        import psutil
    except ImportError:
        return None
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / (1024 * 1024)


def log_performance_metrics(
    operation: str,
    start_time: float,
    end_time: float | None = None,
    extra_info: dict[str, object] | None = None,
) -> None:
    end = time.time() if end_time is None else end_time
    parts = [f"Performance - {operation}: {end - start_time:.2f} seconds"]
    memory_usage = get_process_memory_usage()
    if memory_usage is not None:
        parts.append(f"Memory: {memory_usage:.2f} MB")
    if extra_info:
        parts.extend(f"{key}: {value}" for key, value in extra_info.items())
    logger.info(", ".join(parts))


class ProgressTracker:
    def __init__(self, total_steps: int = 100, operation_name: str = "Operation"):
        if total_steps <= 0:
            raise ValueError("total_steps must be positive.")
        self.total_steps = total_steps
        self.current_step = 0
        self.operation_name = operation_name
        self.start_time = time.time()
        self.last_update_time = self.start_time
        self.update_interval = 1.0

    def update(self, step: int | None = None, message: str | None = None):
        now = time.time()
        self.current_step = self.current_step + 1 if step is None else step
        progress = min(1.0, max(0.0, self.current_step / self.total_steps))
        elapsed = now - self.start_time
        remaining = None
        if self.current_step > 0:
            remaining = (elapsed / self.current_step) * max(self.total_steps - self.current_step, 0)
        if now - self.last_update_time >= self.update_interval or progress >= 1.0:
            self.last_update_time = now
            suffix = f" - {message}" if message else ""
            logger.info(
                "Progress - %s: %.1f%% (%s/%s)%s",
                self.operation_name,
                progress * 100,
                self.current_step,
                self.total_steps,
                suffix,
            )
        return progress, elapsed, remaining

    def complete(self, message: str | None = None) -> float:
        total_time = time.time() - self.start_time
        suffix = f" - {message}" if message else ""
        logger.info("Completed - %s in %s%s", self.operation_name, format_time(total_time), suffix)
        return total_time


def format_time(seconds: float) -> str:
    if seconds < 60:
        return f"{seconds:.1f} seconds"
    if seconds < 3600:
        return f"{seconds / 60:.1f} minutes"
    return f"{seconds / 3600:.1f} hours"


def estimate_memory_usage(n_rows: int, n_features: int) -> float:
    if n_rows < 0 or n_features < 0:
        raise ValueError("n_rows and n_features must be non-negative.")
    bytes_per_row_raw = 100
    bytes_per_row_features = 20 * (8 + 8)
    bytes_model = n_features * 8
    total_bytes = (bytes_per_row_raw + bytes_per_row_features) * n_rows + bytes_model + n_rows * 50
    return total_bytes / (1024 * 1024)


def deduplicate_column_names(
    df: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[object, object], bool]:
    """Rename duplicate columns deterministically without changing cell values."""

    if df.columns.is_unique:
        return df, {}, False

    used: set[object] = set()
    next_suffix: dict[object, int] = {}
    renamed_columns: dict[object, object] = {}
    new_names: list[object] = []

    for original in df.columns:
        if original not in used:
            candidate = original
            used.add(candidate)
            next_suffix.setdefault(original, 1)
        else:
            suffix = next_suffix.get(original, 1)
            candidate = f"{original}_{suffix}"
            while candidate in used:
                suffix += 1
                candidate = f"{original}_{suffix}"
            next_suffix[original] = suffix + 1
            used.add(candidate)
            renamed_columns[original] = candidate
        new_names.append(candidate)

    result = df.copy()
    result.columns = new_names
    return result, renamed_columns, True


def safe_display_dataframe(df, st_container, max_rows: int = 10, max_cols: int | None = None) -> None:
    display_df, _, _ = deduplicate_column_names(df)
    if max_cols is not None:
        display_df = display_df.iloc[:, :max_cols]
    st_container.dataframe(display_df.head(max_rows), use_container_width=True)
