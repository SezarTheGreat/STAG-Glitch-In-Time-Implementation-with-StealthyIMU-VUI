import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
import logging
from config import BranchAConfig, BranchBConfig

logger = logging.getLogger("Day13_Module5")

class BranchA_Seq2Seq(nn.Module):
    """
    Branch A: For unconstrained speech mapping.
    Architecture: CNN Encoder -> BLSTM -> GRU Decoder with Attention.
    """
    def __init__(self, config: BranchAConfig = None):
        super(BranchA_Seq2Seq, self).__init__()
        self.config = config if config else BranchAConfig()
        
        # 1D CNN Encoder
        channels = self.config.cnn_channels
        self.conv1 = nn.Conv1d(self.config.input_dim, channels[0], kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(channels[0], channels[1], kernel_size=3, padding=1, stride=2)
        self.conv3 = nn.Conv1d(channels[1], channels[2], kernel_size=3, padding=1, stride=2)
        
        # BLSTM
        self.blstm = nn.LSTM(
            input_size=channels[2], 
            hidden_size=self.config.blstm_hidden, 
            bidirectional=True, 
            batch_first=True
        )
        
        self.embed = nn.Embedding(self.config.vocab_size, 128)
        self.gru = nn.GRUCell(128 + self.config.blstm_hidden * 2, self.config.gru_hidden)
        self.fc_out = nn.Linear(self.config.gru_hidden, self.config.vocab_size)
        
        self.hidden_proj = None
        if self.config.blstm_hidden * 2 != self.config.gru_hidden:
            self.hidden_proj = nn.Linear(self.config.blstm_hidden * 2, self.config.gru_hidden)
            self.encoder_proj = nn.Linear(self.config.blstm_hidden * 2, self.config.gru_hidden)
        else:
            self.encoder_proj = nn.Identity()
        
    def forward(self, x, max_len=50):
        """
        x shape: (batch, 1, time)
        """
        batch_size = x.size(0)
        
        # CNN
        c = F.relu(self.conv1(x))
        c = F.relu(self.conv2(c))
        c = F.relu(self.conv3(c))
        
        # BLSTM expects (batch, seq, features)
        c = c.transpose(1, 2)
        encoder_outputs, (h_n, c_n) = self.blstm(c)
        encoder_outputs = self.encoder_proj(encoder_outputs)
        
        # Init Decoder hidden state
        # h_n shape for 1-layer bi-LSTM: (2, batch_size, hidden_size)
        hidden = torch.cat([h_n[0], h_n[1]], dim=-1) # (batch, blstm_hidden*2)
        
        if self.hidden_proj is not None:
            hidden = self.hidden_proj(hidden)
            
        outputs = []
        dec_input = torch.zeros(batch_size, dtype=torch.long, device=x.device)
        
        for t in range(max_len):
            embedded = self.embed(dec_input)
            
            # Simple Dot-Product Attention
            attn_weights = torch.bmm(encoder_outputs, hidden.unsqueeze(2)).squeeze(2)
            attn_weights = F.softmax(attn_weights, dim=1)
            context = torch.bmm(attn_weights.unsqueeze(1), encoder_outputs).squeeze(1)
            
            gru_input = torch.cat([embedded, context], dim=1)
            hidden = self.gru(gru_input, hidden)
            
            out = self.fc_out(hidden)
            outputs.append(out.unsqueeze(1))
            
            # Autoregressive teacher forcing (argmax)
            dec_input = out.argmax(1)
            
        return torch.cat(outputs, dim=1)


class BranchB_DenseNet(nn.Module):
    """
    Branch B: DenseNet for targeted vocabulary using 244x244 spectrograms.
    """
    def __init__(self, config: BranchBConfig = None):
        super(BranchB_DenseNet, self).__init__()
        self.config = config if config else BranchBConfig()
        
        # Spectrogram converter
        self.mel_spec = torchaudio.transforms.MelSpectrogram(
            sample_rate=self.config.sample_rate,
            n_fft=self.config.n_fft,
            hop_length=self.config.hop_length,
            n_mels=self.config.n_mels
        )
        
        # Backbone (DenseNet121)
        import torchvision.models as models
        self.backbone = models.densenet121(weights=None)
        # Adapt 3-channel input to 1-channel spectrogram
        self.backbone.features.conv0 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.backbone.classifier = nn.Linear(self.backbone.classifier.in_features, self.config.num_classes)
        
    def _pad_or_trim(self, spec):
        """Forces the spectrogram to be exactly target_size x target_size."""
        target_size = self.config.spectrogram_size
        _, _, h, w = spec.size()
        
        if w < target_size[1]:
            pad = target_size[1] - w
            spec = F.pad(spec, (0, pad))
        elif w > target_size[1]:
            spec = spec[:, :, :, :target_size[1]]
            
        if h < target_size[0]:
            pad = target_size[0] - h
            spec = F.pad(spec, (0, 0, 0, pad))
        elif h > target_size[0]:
            spec = spec[:, :, :target_size[0], :]
            
        return spec
        
    def forward(self, x):
        """
        x shape: (batch, 1, time)
        """
        spec = self.mel_spec(x)
        spec = torchaudio.functional.amplitude_to_DB(spec, multiplier=10.0, amin=1e-10, db_multiplier=0.0, top_db=80.0)
        
        spec = self._pad_or_trim(spec)
        
        logits = self.backbone(spec)
        return logits
