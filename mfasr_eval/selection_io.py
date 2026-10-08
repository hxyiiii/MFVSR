from __future__ import annotations

import csv
from pathlib import Path


def load_video_ids(path: str | Path) -> set[str]:
    """Load IDs from one-per-line text or a CSV containing ``video_id``."""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"ID file does not exist: {path}")
    if path.suffix.lower() == ".csv":
        with path.open("r", encoding="utf-8", newline="") as handle:
            rows = list(csv.DictReader(handle))
        if rows and "video_id" not in rows[0]:
            raise ValueError(f"CSV ID file lacks video_id column: {path}")
        return {str(row["video_id"]).strip() for row in rows if str(row["video_id"]).strip()}
    output = set()
    for line in path.read_text(encoding="utf-8").splitlines():
        value = line.strip()
        if value and not value.startswith("#"):
            output.add(value)
    return output

