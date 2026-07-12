import math
import os
from dataclasses import dataclass


@dataclass(frozen=True)
class CoordinateRecord:
    filename: str
    x: float
    y: float


def normalize_origin(origin):
    value = (origin or "top-left").strip().lower().replace("_", "-")
    if value not in {"top-left", "bottom-left"}:
        raise ValueError("coordinate origin must be 'top-left' or 'bottom-left'")
    return value


def resolve_coordinate_origin(coordinate_path, requested="auto"):
    value = (requested or "auto").strip().lower().replace("_", "-")
    if value == "auto":
        return "bottom-left" if os.path.basename(os.fspath(coordinate_path)).lower() == "invert.star" else "top-left"
    return normalize_origin(value)


def normalize_xy(record, mrc_shape, origin="top-left"):
    height = int(mrc_shape[0])
    y = float(record.y)
    if normalize_origin(origin) == "bottom-left":
        y = height - y
    return _nearest_pixel(y), _nearest_pixel(float(record.x))


def read_coordinate_file(path):
    with open(path, "r", encoding="utf-8") as handle:
        text = handle.read()
    records = list(_read_star_coordinates(text) if _looks_like_star(text) else _read_legacy_coordinates(text))
    if not records:
        raise ValueError(f"No valid coordinates found in {path}")
    return records


def _nearest_pixel(value):
    return int(math.floor(float(value) + 0.5))


def _looks_like_star(text):
    return any(
        line.strip().startswith(("data_", "loop_", "_rln"))
        for line in text.splitlines()
    )


def _read_legacy_coordinates(text):
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        parts = stripped.split()
        if len(parts) < 3:
            continue
        try:
            yield CoordinateRecord(os.path.basename(parts[0]), float(parts[1]), float(parts[2]))
        except ValueError:
            continue


def _read_star_coordinates(text):
    for columns, rows in _iter_star_loops(text):
        names = [column.lstrip("_").split("#", 1)[0] for column in columns]
        required = ("rlnMicrographName", "rlnCoordinateX", "rlnCoordinateY")
        if not all(name in names for name in required):
            continue
        name_index = names.index("rlnMicrographName")
        x_index = names.index("rlnCoordinateX")
        y_index = names.index("rlnCoordinateY")
        max_index = max(name_index, x_index, y_index)
        for row in rows:
            if len(row) <= max_index:
                continue
            try:
                yield CoordinateRecord(
                    os.path.basename(row[name_index]),
                    float(row[x_index]),
                    float(row[y_index]),
                )
            except ValueError:
                continue


def _iter_star_loops(text):
    lines = text.splitlines()
    index = 0
    while index < len(lines):
        if lines[index].strip() != "loop_":
            index += 1
            continue
        index += 1
        columns = []
        while index < len(lines):
            stripped = lines[index].strip()
            if not stripped:
                index += 1
                continue
            if not stripped.startswith("_"):
                break
            columns.append(stripped.split()[0])
            index += 1
        rows = []
        while index < len(lines):
            stripped = lines[index].strip()
            if not stripped or stripped.startswith("#"):
                index += 1
                continue
            if stripped == "loop_" or stripped.startswith(("data_", "_")):
                break
            rows.append(stripped.split())
            index += 1
        yield columns, rows
