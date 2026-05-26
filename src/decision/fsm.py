from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class PipelineFSM:
    """Detects driver switching by tracking track_id changes."""

    current_track_id: int | None = None
    reset_count: int = 0
    history: list[int] = field(default_factory=list)

    def on_track(self, track_id: int) -> bool:
        if self.current_track_id is None:
            self.current_track_id = track_id
            self.history.append(track_id)
            return False
        if track_id != self.current_track_id:
            self.current_track_id = track_id
            self.history.append(track_id)
            self.reset_count += 1
            return True
        return False


@dataclass
class FatigueStateFSM:
    """Debounce filter for fatigue state transitions.

    Upgrades state immediately (safety-first).
    Downgrades only after `downgrade_frames` consecutive frames of a lower
    assessment, preventing alert flicker from single-frame noise.
    """

    downgrade_frames: int = 10
    _state: str = field(default="NORMAL", init=False)
    _downgrade_count: int = field(default=0, init=False)

    _LEVELS: dict[str, int] = field(
        default_factory=lambda: {"NORMAL": 0, "WARNING": 1, "ALERT": 2},
        init=False,
        repr=False,
        compare=False,
    )

    def update(self, raw_state: str) -> str:
        raw_level = self._LEVELS.get(raw_state, 0)
        cur_level = self._LEVELS.get(self._state, 0)

        if raw_level >= cur_level:
            # Upgrade immediately
            self._state = raw_state
            self._downgrade_count = 0
        else:
            self._downgrade_count += 1
            if self._downgrade_count >= self.downgrade_frames:
                self._state = raw_state
                self._downgrade_count = 0

        return self._state

    def reset(self) -> None:
        self._state = "NORMAL"
        self._downgrade_count = 0

    @property
    def state(self) -> str:
        return self._state


@dataclass
class TimedStateFSM:
    """Time-based label stabilizer for flickery categorical states."""

    initial_state: str = "NORMAL"
    min_state_seconds: float = 3.0
    candidate_seconds: float = 0.8
    _state: str = field(init=False)
    _state_since: float | None = field(default=None, init=False)
    _candidate: str | None = field(default=None, init=False)
    _candidate_since: float | None = field(default=None, init=False)

    def __post_init__(self) -> None:
        self._state = self.initial_state

    def update(self, raw_state: str, ts: float) -> str:
        if self._state_since is None:
            self._state_since = ts

        if raw_state == self._state:
            self._candidate = None
            self._candidate_since = None
            return self._state

        if raw_state != self._candidate:
            self._candidate = raw_state
            self._candidate_since = ts
            return self._state

        candidate_age = ts - (self._candidate_since if self._candidate_since is not None else ts)
        state_age = ts - (self._state_since if self._state_since is not None else ts)
        if candidate_age >= self.candidate_seconds and state_age >= self.min_state_seconds:
            self._state = raw_state
            self._state_since = ts
            self._candidate = None
            self._candidate_since = None

        return self._state

    def reset(self) -> None:
        self._state = self.initial_state
        self._state_since = None
        self._candidate = None
        self._candidate_since = None

    @property
    def state(self) -> str:
        return self._state
