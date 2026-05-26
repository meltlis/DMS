from pathlib import Path

from src.pipeline import resolve_external_project_root, resolve_optional_model_path, resolve_sequence_model_path


def test_resolve_external_project_root_prefers_runtime_path(tmp_path: Path) -> None:
    external = tmp_path / "Drowsiness-Detection-based-on-yolo11-and-LSTM-main"
    external.mkdir()

    assert resolve_external_project_root({"external_project_dir": str(external)}) == external


def test_resolve_optional_model_path_uses_external_default(tmp_path: Path) -> None:
    project = tmp_path / "project"
    external = tmp_path / "external"
    model = external / "runs" / "detect" / "train16" / "weights" / "best.pt"
    model.parent.mkdir(parents=True)
    model.write_bytes(b"model")

    assert (
        resolve_optional_model_path(
            "",
            project_root=project,
            external_root=external,
            external_default="runs/detect/train16/weights/best.pt",
        )
        == model
    )


def test_resolve_sequence_model_prefers_lstm(tmp_path: Path) -> None:
    external = tmp_path / "external"
    external.mkdir()
    lstm = external / "lstm_model.pth"
    transformer = external / "transformer_model.pth"
    lstm.write_bytes(b"lstm")
    transformer.write_bytes(b"transformer")

    assert resolve_sequence_model_path({}, external) == lstm
