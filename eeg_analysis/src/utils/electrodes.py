from __future__ import annotations

from typing import Dict, List, Tuple, Optional

import re

from src.utils.logger import setup_logger

logger = setup_logger(__name__)


def _parse_floats_from_line(line: str) -> Optional[Tuple[float, float, float]]:
    parts = line.strip().split()
    if len(parts) != 3:
        return None
    try:
        x, y, z = float(parts[0]), float(parts[1]), float(parts[2])
        return (x, y, z)
    except Exception:
        return None


def parse_asa_text_to_unit_sphere(asa_text: str) -> Dict[str, Tuple[float, float, float]]:
    """
    Parse an ASA electrode file content (as text) into a mapping of LABEL -> (x, y, z) on the unit sphere.
    Positions are assumed to be in mm; each vector is normalized to unit length.
    If number of labels mismatches number of positions, extra items are ignored.
    """
    lines = [ln.rstrip() for ln in asa_text.splitlines()]
    # Find "Positions" and "Labels" section indices
    try:
        pos_idx = next(i for i, ln in enumerate(lines) if ln.strip().lower().startswith("positions"))
    except StopIteration:
        raise ValueError("ASA text missing 'Positions' section")
    try:
        labels_idx = next(i for i, ln in enumerate(lines) if ln.strip().lower().startswith("labels"))
    except StopIteration:
        raise ValueError("ASA text missing 'Labels' section")

    # Collect positions between sections
    positions_raw: List[Tuple[float, float, float]] = []
    for ln in lines[pos_idx + 1 : labels_idx]:
        triple = _parse_floats_from_line(ln)
        if triple is not None:
            positions_raw.append(triple)

    # Collect labels to end
    labels: List[str] = []
    for ln in lines[labels_idx + 1 :]:
        name = ln.strip()
        if not name:
            continue
        labels.append(name)

    if not positions_raw or not labels:
        raise ValueError("No positions or labels parsed from ASA text")

    n = min(len(positions_raw), len(labels))
    if len(positions_raw) != len(labels):
        logger.warning(f"Positions ({len(positions_raw)}) and Labels ({len(labels)}) count mismatch; using first {n}")

    # Normalize to unit sphere and build mapping
    mapping: Dict[str, Tuple[float, float, float]] = {}
    for i in range(n):
        lbl = labels[i].strip().upper()
        x, y, z = positions_raw[i]
        norm = (x * x + y * y + z * z) ** 0.5
        if norm <= 0:
            unit = (0.0, 0.0, 0.0)
        else:
            unit = (x / norm, y / norm, z / norm)
        # Keep first occurrence; skip duplicates
        if lbl not in mapping:
            mapping[lbl] = unit
    return mapping


def load_asa_file_to_unit_sphere(asa_path: str) -> Dict[str, Tuple[float, float, float]]:
    """
    Load an ASA electrode file from disk and convert to LABEL -> (x, y, z) unit vectors.
    """
    with open(asa_path, "r", encoding="utf-8") as f:
        content = f.read()
    return parse_asa_text_to_unit_sphere(content)


