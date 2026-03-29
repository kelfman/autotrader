"""
track_config.py — per-track configuration schema for V2 multi-track research.

Each research track is an independent agent instance seeded with a different
signal class vocabulary. This module defines the TrackConfig model that
parameterises the agent system prompt, allowed libraries, df column contract,
and exploration hints for each track.

Usage:
    from track_config import TrackConfig, load_track

    config = load_track("tracks/track_a_vol_regime.json")
    print(config.track_id, config.signal_class)
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Annotated

from pydantic import BaseModel, Field


class TrackConfig(BaseModel):
    """
    Configuration for one V2 research track.

    Fields:
        track_id:             Single letter identifier ("A"–"E").
        signal_class:         Snake-case signal class name for logs and filenames.
        description:          One-line human-readable description.
        allowed_libraries:    Python libraries the agent may import in compute_signals.
                              The track runner enforces this at prompt level; the
                              backtest harness does not restrict imports.
        df_columns_extra:     Column names beyond standard OHLCV that will be present
                              in the df passed to compute_signals. The track runner is
                              responsible for providing these columns.
        signal_class_brief:   2–4 paragraph explanation of the signal class mechanism,
                              why it is expected to work, and what market structure
                              property it exploits. Injected verbatim into system prompt.
        reducibility_note:    1–2 sentences on whether the signal is reducible (does not
                              depend on other participants watching the same chart).
                              Injected verbatim into system prompt.
        exploration_hints:    Ordered list of specific ideas for the agent to explore.
                              More specific = more useful.
        ta_baseline_fitness:  V1 TA baseline fitness score. All tracks must beat this
                              to be considered improvements.
    """

    track_id: Annotated[str, Field(min_length=1, max_length=2)]
    signal_class: str
    description: str
    allowed_libraries: list[str]
    df_columns_extra: list[str] = Field(default_factory=list)
    signal_class_brief: str
    reducibility_note: str
    exploration_hints: list[str]
    ta_baseline_fitness: float = 0.4714


def load_track(path: str | Path) -> TrackConfig:
    """Load a TrackConfig from a JSON file."""
    return TrackConfig(**json.loads(Path(path).read_text()))


def list_tracks(tracks_dir: str | Path = "tracks") -> list[Path]:
    """Return sorted list of track JSON files found in tracks_dir."""
    return sorted(Path(tracks_dir).glob("track_*.json"))
