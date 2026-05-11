import torch
import torch.nn as nn
import numpy as np
import os

# 9-class labels matching training data:
# 0: eyes_closed, 1: eyes_closed_head_left, 2: eyes_closed_head_right,
# 3: focused, 4: head_down, 5: head_up, 6: seeing_left, 7: seeing_right, 8: yawning
_FATIGUE_CLASSES = [0, 1, 2, 8]  # classes that indicate fatigue


class SimpleLSTM(nn.Module):
    def __init__(self, input_size=9, hidden_size=64, num_layers=1, num_classes=9):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.fc(out[:, -1, :])
        return out


class LSTMClassifier:
    def __init__(self, model_path, seq_len=30, thresholds: dict | None = None):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model_loaded = False
        self.model = SimpleLSTM(input_size=9, hidden_size=64, num_layers=1, num_classes=9)
        if os.path.exists(model_path):
            try:
                self.model.load_state_dict(torch.load(model_path, map_location=self.device))
                self.model_loaded = True
            except Exception as e:
                import logging
                logging.getLogger(__name__).warning(f"LSTM model load failed: {e} — LSTM disabled")
        self.model.to(self.device)
        self.model.eval()
        self.seq_len = seq_len
        # Use same thresholds as rule engine for consistent feature encoding
        t = thresholds or {}
        self.ear_threshold = float(t.get("ear_threshold", 0.16))
        self.mar_threshold = float(t.get("mar_threshold", 0.60))

    def features_to_yolo_classes(self, feats):
        """Map [ear, mar, pitch, yaw, roll] to 9-dim one-hot matching training labels."""
        ear, mar, pitch, yaw, roll = feats
        vec = np.zeros(9, dtype=np.float32)

        is_closed = ear < self.ear_threshold
        is_yawn = mar > self.mar_threshold

        if is_closed:
            if yaw < -0.2:   vec[1] = 1.0  # eyes_closed_head_left
            elif yaw > 0.2:  vec[2] = 1.0  # eyes_closed_head_right
            else:            vec[0] = 1.0  # eyes_closed
        elif is_yawn:        vec[8] = 1.0  # yawning
        elif pitch < -0.2:   vec[4] = 1.0  # head_down
        elif pitch > 0.2:    vec[5] = 1.0  # head_up
        elif yaw < -0.2:     vec[6] = 1.0  # seeing_left
        elif yaw > 0.2:      vec[7] = 1.0  # seeing_right
        else:                vec[3] = 1.0  # focused

        return vec

    def predict(self, raw_sequence):
        """Return (fatigue_score, pred) where fatigue_score is P(fatigue) in [0,1]."""
        if not self.model_loaded or len(raw_sequence) == 0:
            return 0.0, 0

        mapped = [self.features_to_yolo_classes(f) for f in raw_sequence]

        if len(mapped) < self.seq_len:
            pad = [np.zeros(9, dtype=np.float32)] * (self.seq_len - len(mapped))
            mapped = mapped + pad
        else:
            mapped = mapped[-self.seq_len:]

        mapped_arr = np.array(mapped, dtype=np.float32)
        tensor_x = torch.tensor(mapped_arr).unsqueeze(0).to(self.device)

        with torch.no_grad():
            outputs = self.model(tensor_x)
            prob = torch.softmax(outputs, dim=1)[0].cpu().numpy()

        # Fatigue score = sum of probabilities for fatigue-related classes
        fatigue_score = float(sum(prob[c] for c in _FATIGUE_CLASSES))
        pred = 1 if fatigue_score > 0.5 else 0

        return fatigue_score, pred
