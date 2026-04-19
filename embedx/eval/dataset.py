"""Evaluation dataset loader (YAML / JSON)."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional


@dataclass
class EvalCase:
    query: str
    expected: list[str]
    metadata: dict = field(default_factory=dict)


@dataclass
class EvalDataset:
    name: str
    cases: list[EvalCase]
    corpus: list[str] = field(default_factory=list)


def load_dataset(path: str) -> EvalDataset:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Dataset not found: {path}")

    suffix = p.suffix.lower()
    if suffix in (".yaml", ".yml"):
        try:
            import yaml  # type: ignore
        except ImportError as exc:
            raise ImportError(
                "PyYAML is required to load YAML datasets.\n"
                "Install with: pip install pyyaml"
            ) from exc
        with p.open() as f:
            raw = yaml.safe_load(f)
    elif suffix == ".json":
        with p.open() as f:
            raw = json.load(f)
    else:
        raise ValueError(f"Unsupported dataset format: {suffix}. Use .yaml or .json")

    cases = [
        EvalCase(
            query=c["query"],
            expected=c.get("expected", []),
            metadata=c.get("metadata", {}),
        )
        for c in raw.get("cases", [])
    ]
    return EvalDataset(
        name=raw.get("name", p.stem),
        cases=cases,
        corpus=raw.get("corpus", []),
    )
