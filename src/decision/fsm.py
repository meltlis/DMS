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
