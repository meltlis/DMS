from src.decision.fsm import PipelineFSM


def test_fsm_track_switch_behavior() -> None:
    fsm = PipelineFSM()

    assert fsm.on_track(1) is False
    assert fsm.current_track_id == 1
    assert fsm.history == [1]
    assert fsm.reset_count == 0

    assert fsm.on_track(1) is False
    assert fsm.reset_count == 0

    assert fsm.on_track(2) is True
    assert fsm.current_track_id == 2
    assert fsm.history == [1, 2]
    assert fsm.reset_count == 1
