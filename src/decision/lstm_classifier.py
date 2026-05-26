import logging
import os
from typing import Mapping

import numpy as np
import torch
import torch.nn as nn


# 9-class labels matching the external sequence training labels:
# 0 eyes_closed, 1 eyes_closed_head_left, 2 eyes_closed_head_right,
# 3 focused, 4 head_down, 5 head_up, 6 seeing_left, 7 seeing_right, 8 yawning
_FATIGUE_CLASSES = [0, 1, 2, 8]


class SimpleLSTM(nn.Module):
    def __init__(self, input_size: int = 9, hidden_size: int = 64, num_layers: int = 1, num_classes: int = 9):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :])


class SimpleTransformerClassifier(nn.Module):
    def __init__(
        self,
        input_dim: int = 9,
        num_classes: int = 9,
        num_layers: int = 2,
        nhead: int = 3,
        hidden_dim: int = 64,
        seq_len: int = 30,
    ):
        super().__init__()
        self.pos_embedding = nn.Parameter(torch.randn(1, seq_len, input_dim))
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=input_dim,
            nhead=nhead,
            dim_feedforward=hidden_dim,
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.fc = nn.Linear(input_dim, num_classes)

    def forward(self, x):
        x = x + self.pos_embedding
        x = self.transformer(x)
        return self.fc(x.mean(dim=1))


class LSTMClassifier:
    """Optional sequence model wrapper.

    The external training script may save either an LSTM checkpoint
    (``lstm_model.pth``) or a Transformer checkpoint (``transformer_model.pth``).
    This wrapper inspects the state dict and loads the matching architecture.
    Because the provided training set is tiny, the pipeline treats this model as
    diagnostic unless ``lstm_can_warn``/``lstm_can_alert`` are explicitly enabled.
    """

    def __init__(self, model_path: str, seq_len: int = 30, thresholds: Mapping[str, object] | None = None):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model_loaded = False
        self.model_kind = "disabled"
        self.model_path = model_path
        self.seq_len = seq_len
        self.model: nn.Module = SimpleLSTM(input_size=9, hidden_size=64, num_layers=1, num_classes=9)

        t = thresholds or {}
        self.enabled = bool(t.get("lstm_enabled", True))
        self.ear_threshold = float(t.get("ear_threshold", 0.16))
        self.mar_threshold = float(t.get("mar_threshold", 0.60))
        self.pitch_threshold_deg = abs(float(t.get("lstm_pitch_threshold_deg", t.get("phone_head_down_pitch_deg", 18.0))))
        self.yaw_threshold_deg = abs(float(t.get("lstm_yaw_threshold_deg", max(15.0, float(t.get("yaw_threshold_deg", 15.0))))))

        if self.enabled and model_path and os.path.exists(model_path):
            try:
                state_dict = torch.load(model_path, map_location=self.device)
                self.model = self._build_model_from_state_dict(state_dict)
                self.model.load_state_dict(state_dict)
                self.model_loaded = True
                self.model_kind = "transformer" if "pos_embedding" in state_dict else "lstm"
            except Exception as exc:
                logging.getLogger(__name__).warning("Sequence model load failed: %s; model disabled", exc)

        self.model.to(self.device)
        self.model.eval()

    def _build_model_from_state_dict(self, state_dict: Mapping[str, torch.Tensor]) -> nn.Module:
        if "pos_embedding" in state_dict:
            pos = state_dict["pos_embedding"]
            seq_len = int(pos.shape[1])
            input_dim = int(pos.shape[2])
            num_classes = int(state_dict["fc.bias"].shape[0])
            layer_ids = {
                int(key.split(".")[2])
                for key in state_dict
                if key.startswith("transformer.layers.") and key.split(".")[2].isdigit()
            }
            num_layers = max(layer_ids) + 1 if layer_ids else 2
            self.seq_len = seq_len
            return SimpleTransformerClassifier(
                input_dim=input_dim,
                num_classes=num_classes,
                num_layers=num_layers,
                nhead=3 if input_dim % 3 == 0 else 1,
                hidden_dim=64,
                seq_len=seq_len,
            )

        hidden_size = int(state_dict["lstm.weight_hh_l0"].shape[1])
        input_size = int(state_dict["lstm.weight_ih_l0"].shape[1])
        num_classes = int(state_dict["fc.bias"].shape[0])
        return SimpleLSTM(input_size=input_size, hidden_size=hidden_size, num_layers=1, num_classes=num_classes)

    def features_to_yolo_classes(self, feats):
        """Map [ear, mar, pitch_deg, yaw_deg, roll_deg] to the 9 one-hot labels."""
        ear, mar, pitch, yaw, _roll = feats
        vec = np.zeros(9, dtype=np.float32)

        is_closed = ear < self.ear_threshold
        is_yawn = mar > self.mar_threshold

        if is_closed:
            if yaw <= -self.yaw_threshold_deg:
                vec[1] = 1.0
            elif yaw >= self.yaw_threshold_deg:
                vec[2] = 1.0
            else:
                vec[0] = 1.0
        elif is_yawn:
            vec[8] = 1.0
        elif pitch <= -self.pitch_threshold_deg:
            vec[4] = 1.0
        elif pitch >= self.pitch_threshold_deg:
            vec[5] = 1.0
        elif yaw <= -self.yaw_threshold_deg:
            vec[6] = 1.0
        elif yaw >= self.yaw_threshold_deg:
            vec[7] = 1.0
        else:
            vec[3] = 1.0

        return vec

    def predict(self, raw_sequence):
        """Return (fatigue_score, pred), where fatigue_score is P(fatigue)."""
        if not self.model_loaded or len(raw_sequence) == 0:
            return 0.0, 0

        mapped = [self.features_to_yolo_classes(f) for f in raw_sequence]
        if len(mapped) < self.seq_len:
            pad = [np.zeros(9, dtype=np.float32)] * (self.seq_len - len(mapped))
            mapped = mapped + pad
        else:
            mapped = mapped[-self.seq_len :]

        tensor_x = torch.tensor(np.asarray(mapped, dtype=np.float32)).unsqueeze(0).to(self.device)
        with torch.no_grad():
            outputs = self.model(tensor_x)
            prob = torch.softmax(outputs, dim=1)[0].cpu().numpy()

        fatigue_score = float(sum(prob[c] for c in _FATIGUE_CLASSES))
        return fatigue_score, 1 if fatigue_score > 0.5 else 0
