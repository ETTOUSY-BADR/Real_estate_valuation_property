"""Failure-safe writers for generated project artifacts."""

from __future__ import annotations

from collections.abc import Callable
import json
from pathlib import Path
from typing import Protocol


class ParquetWritable(Protocol):
    """Minimal interface required by :func:`write_parquet_atomic`."""

    def to_parquet(self, path: Path, **kwargs: object) -> object: ...


def temporary_path(path: Path) -> Path:
    """Return the sibling staging path used for an artifact."""
    return path.with_suffix(path.suffix + ".part")


def write_atomic(path: Path, writer: Callable[[Path], None]) -> None:
    """Write to a sibling temporary file and replace ``path`` after success."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = temporary_path(path)
    temporary.unlink(missing_ok=True)
    try:
        writer(temporary)
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def write_json_atomic(path: Path, payload: object) -> None:
    """Serialize JSON without replacing a valid artifact on failure."""

    def write(temporary: Path) -> None:
        serialized = json.dumps(payload, indent=2, ensure_ascii=False)
        temporary.write_text(serialized, encoding="utf-8")

    write_atomic(path, write)


def write_parquet_atomic(data: ParquetWritable, path: Path) -> None:
    """Serialize Parquet without replacing a valid artifact on failure."""

    def write(temporary: Path) -> None:
        data.to_parquet(temporary, index=False, compression="zstd")

    write_atomic(path, write)
