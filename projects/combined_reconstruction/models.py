import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

# Add stag_original directory to path to load original model classes
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'stag_original')))
from src.models.slu_dnn import SLUModel, PaperSLUModel, CharacterTokenizer
from projects.stag_original.src.pipeline.features import extract_spectrogram

class CNNUpscaler(nn.Module):
    """
    1D CNN regressor for local sliding window context.
    """
    def __init__(self, window_size=11, num_channels=4):
        super(CNNUpscaler, self).__init__()
        self.conv1 = nn.Conv1d(num_channels, 16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(16, 32, kernel_size=3, padding=1)
        self.fc1 = nn.Linear(32 * window_size, 64)
        self.fc2 = nn.Linear(64, 1)
        
    def forward(self, x):
        # x shape: (batch, num_channels, window_size)
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        return self.fc2(x).squeeze(-1) # shape: (batch,)

class RNNUpscaler(nn.Module):
    """
    RNN (BiGRU) regressor for local sliding window context.
    """
    def __init__(self, num_channels=4, hidden_size=32):
        super(RNNUpscaler, self).__init__()
        self.gru = nn.GRU(
            input_size=num_channels,
            hidden_size=hidden_size,
            num_layers=2,
            batch_first=True,
            bidirectional=True
        )
        self.fc = nn.Sequential(
            nn.Linear(hidden_size * 2, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )
        
    def forward(self, x):
        # x shape: (batch, window_size, num_channels)
        out, _ = self.gru(x) # out shape: (batch, window_size, hidden_size * 2)
        # We predict using the last hidden state of the sequence
        # or the hidden state corresponding to the middle element (index W)
        mid_idx = x.size(1) // 2
        mid_features = out[:, mid_idx, :]
        return self.fc(mid_features).squeeze(-1) # shape: (batch,)

class SLUEncoderRegressionEngine(nn.Module):
    """
    A regression model wrapping a pre-trained SLU encoder.
    Extracts deep representations from signal spectrograms and projects them to acc_even values.
    """
    def __init__(self, slu_model, encoder_dim=256):
        super(SLUEncoderRegressionEngine, self).__init__()
        self.encoder = slu_model.encoder
        # Freeze the encoder weights to preserve speech-learned representations
        for param in self.encoder.parameters():
            param.requires_grad = False
            
        self.regressor = nn.Sequential(
            nn.Linear(encoder_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )
        
    def forward(self, imu_spec, target_len):
        """
        imu_spec shape: (1, frames, 30)
        target_len: length of target even timestamps sequence N
        """
        # Get encoder states
        # encoder_states shape: (1, pooled_frames, encoder_dim)
        encoder_states, _ = self.encoder(imu_spec)
        
        # Interpolate encoder states from pooled_frames to target_len (N)
        # Interpolate expects (batch, channels, length)
        encoder_states = encoder_states.transpose(1, 2) # (1, encoder_dim, pooled_frames)
        encoder_states_interp = F.interpolate(
            encoder_states,
            size=target_len,
            mode='linear',
            align_corners=True
        ) # shape: (1, encoder_dim, target_len)
        
        encoder_states_interp = encoder_states_interp.transpose(1, 2).squeeze(0) # shape: (target_len, encoder_dim)
        
        # Predict acc_even
        predictions = self.regressor(encoder_states_interp).squeeze(-1) # shape: (target_len,)
        return predictions

def load_teacher_regression_engine(checkpoint_path):
    """
    Loads pre-trained PaperSLUModel and wraps it in a regression engine.
    """
    tokenizer = CharacterTokenizer()
    slu_model = PaperSLUModel(vocab_size=tokenizer.vocab_size)
    slu_model.load_state_dict(torch.load(checkpoint_path, map_location='cpu'))
    slu_model.eval()
    return SLUEncoderRegressionEngine(slu_model, encoder_dim=256)

def load_student_regression_engine(checkpoint_path):
    """
    Loads pre-trained SLUModel (student) and wraps it in a regression engine.
    """
    tokenizer = CharacterTokenizer()
    slu_model = SLUModel(vocab_size=tokenizer.vocab_size)
    slu_model.load_state_dict(torch.load(checkpoint_path, map_location='cpu'))
    slu_model.eval()
    return SLUEncoderRegressionEngine(slu_model, encoder_dim=256)
