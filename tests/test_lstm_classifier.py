import numpy as np

from src.decision.lstm_classifier import LSTMClassifier


def test_feature_mapping_uses_degree_thresholds() -> None:
    clf = LSTMClassifier(
        model_path="",
        thresholds={
            "ear_threshold": 0.16,
            "mar_threshold": 0.50,
            "lstm_pitch_threshold_deg": 18.0,
            "lstm_yaw_threshold_deg": 15.0,
        },
    )

    focused = clf.features_to_yolo_classes([0.30, 0.10, -0.3, 0.4, 0.0])
    assert int(np.argmax(focused)) == 3

    head_down = clf.features_to_yolo_classes([0.30, 0.10, -20.0, 0.0, 0.0])
    assert int(np.argmax(head_down)) == 4

    look_right = clf.features_to_yolo_classes([0.30, 0.10, 0.0, 18.0, 0.0])
    assert int(np.argmax(look_right)) == 7


def test_sequence_model_disabled_without_path() -> None:
    clf = LSTMClassifier(model_path="", thresholds={"lstm_enabled": True})
    assert clf.model_loaded is False
    assert clf.predict([[0.3, 0.1, 0.0, 0.0, 0.0]]) == (0.0, 0)
