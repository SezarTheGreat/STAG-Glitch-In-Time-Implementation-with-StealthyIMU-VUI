import torch
import torch.nn as nn
import torch.nn.functional as F

class CharacterTokenizer:
    """
    Lightweight character-level tokenizer for private entity list target texts.
    Predefines a vocabulary of ASCII printable characters for stability.
    """
    def __init__(self):
        self.pad_token = "<pad>"
        self.sos_token = "<sos>"
        self.eos_token = "<eos>"
        self.unk_token = "<unk>"
        
        self.special_tokens = [self.pad_token, self.sos_token, self.eos_token, self.unk_token]
        
        # ASCII printable characters: space (32) to ~ (126)
        self.vocab = self.special_tokens + [chr(i) for i in range(32, 127)]
        
        self.char_to_id = {char: idx for idx, char in enumerate(self.vocab)}
        self.id_to_char = {idx: char for idx, char in enumerate(self.vocab)}
        
        self.pad_id = self.char_to_id[self.pad_token]
        self.sos_id = self.char_to_id[self.sos_token]
        self.eos_id = self.char_to_id[self.eos_token]
        self.unk_id = self.char_to_id[self.unk_token]
        
        self.vocab_size = len(self.vocab)

    def encode(self, text):
        token_ids = [self.sos_id]
        for char in text:
            token_ids.append(self.char_to_id.get(char, self.unk_id))
        token_ids.append(self.eos_id)
        return token_ids

    def decode(self, token_ids):
        chars = []
        for tid in token_ids:
            if tid in [self.pad_id, self.sos_id, self.eos_id]:
                continue
            chars.append(self.id_to_char.get(tid, ''))
        return "".join(chars)

class Attention(nn.Module):
    """
    Key-Value Attention mechanism.
    """
    def __init__(self, encoder_dim, decoder_dim, attn_dim):
        super(Attention, self).__init__()
        self.query_proj = nn.Linear(decoder_dim, attn_dim)
        self.key_proj = nn.Linear(encoder_dim, attn_dim)
        self.v = nn.Linear(attn_dim, 1, bias=False)

    def forward(self, query, keys):
        # query shape: (batch, decoder_dim)
        # keys shape: (batch, seq_len, encoder_dim)
        
        # Project query and keys
        q = self.query_proj(query).unsqueeze(1) # Shape: (batch, 1, attn_dim)
        k = self.key_proj(keys) # Shape: (batch, seq_len, attn_dim)
        
        # Calculate attention scores (additive attention)
        scores = self.v(torch.tanh(q + k)).squeeze(-1) # Shape: (batch, seq_len)
        
        # Softmax to get weights
        weights = F.softmax(scores, dim=-1) # Shape: (batch, seq_len)
        
        # Weighted sum of keys to get context vector
        context = torch.bmm(weights.unsqueeze(1), keys).squeeze(1) # Shape: (batch, encoder_dim)
        
        return context, weights

class SLUEncoder(nn.Module):
    """
    CNN-RNN-DNN Encoder.
    """
    def __init__(self, input_bins=30, encoder_dim=256, num_cnn_blocks=1, num_rnn_layers=2):
        super(SLUEncoder, self).__init__()
        
        # CNN block: reduce input dimension along frequency/time
        # Input shape: (batch, 1, seq_len, input_bins)
        self.cnn_layers = nn.ModuleList()
        in_channels = 1
        out_channels = 64
        
        for i in range(num_cnn_blocks):
            self.cnn_layers.append(
                nn.Sequential(
                    nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
                    nn.BatchNorm2d(out_channels),
                    nn.LeakyReLU(0.2),
                    nn.MaxPool2d(kernel_size=(2, 2)) # TimePooling and FrequencyPooling
                )
            )
            in_channels = out_channels
            out_channels *= 2
            
        # Calculate final frequency feature size after pooling
        freq_dim = input_bins
        for _ in range(num_cnn_blocks):
            freq_dim = freq_dim // 2
            
        rnn_input_dim = in_channels * freq_dim
        
        # RNN block: BiLSTM
        self.lstm = nn.LSTM(
            input_size=rnn_input_dim,
            hidden_size=encoder_dim // 2, # BiLSTM -> hidden_size * 2 = encoder_dim
            num_layers=num_rnn_layers,
            batch_first=True,
            bidirectional=True
        )
        
        # DNN block: Linear projection
        self.dnn = nn.Linear(encoder_dim, encoder_dim)

    def forward(self, x):
        # x shape: (batch, seq_len, bins)
        batch_size, seq_len, bins = x.size()
        
        # Prepare for CNN: add channel dimension
        x = x.unsqueeze(1) # Shape: (batch, 1, seq_len, bins)
        
        for layer in self.cnn_layers:
            x = layer(x)
            
        # x shape: (batch, channels, pooled_seq_len, pooled_bins)
        batch_size, channels, pooled_seq_len, pooled_bins = x.size()
        
        # Transpose and flatten channels/bins into RNN features
        x = x.transpose(1, 2).contiguous() # (batch, pooled_seq_len, channels, pooled_bins)
        x = x.view(batch_size, pooled_seq_len, -1) # (batch, pooled_seq_len, channels * pooled_bins)
        
        # LSTM forward
        lstm_out, (h_n, c_n) = self.lstm(x) # (batch, pooled_seq_len, encoder_dim)
        
        # DNN projection
        encoder_states = self.dnn(lstm_out)
        
        # Final encoder hidden state for decoder initialization
        # Average bidirectional LSTM last states
        last_hidden = h_n.view(self.lstm.num_layers, 2, batch_size, self.lstm.hidden_size)
        last_hidden = last_hidden[-1] # take last layer: shape (2, batch_size, hidden_size)
        last_hidden = torch.cat([last_hidden[0], last_hidden[1]], dim=-1) # (batch_size, encoder_dim)
        
        return encoder_states, last_hidden

class SLUDecoder(nn.Module):
    """
    GRU Decoder with Key-Value Attention.
    """
    def __init__(self, vocab_size, embed_dim=128, encoder_dim=256, decoder_dim=256):
        super(SLUDecoder, self).__init__()
        
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.gru = nn.GRUCell(embed_dim + encoder_dim, decoder_dim)
        self.attention = Attention(encoder_dim, decoder_dim, attn_dim=128)
        self.out_proj = nn.Linear(decoder_dim + encoder_dim, vocab_size)

    def forward(self, prev_token, hidden, encoder_states, last_context):
        # prev_token: (batch,)
        # hidden: (batch, decoder_dim)
        # encoder_states: (batch, seq_len, encoder_dim)
        # last_context: (batch, encoder_dim)
        
        # 1. Embed previous token
        embedded = self.embedding(prev_token) # (batch, embed_dim)
        
        # 2. Concatenate embedding and last attention context
        gru_input = torch.cat([embedded, last_context], dim=-1) # (batch, embed_dim + encoder_dim)
        
        # 3. Step GRU
        new_hidden = self.gru(gru_input, hidden) # (batch, decoder_dim)
        
        # 4. Calculate attention over encoder states
        context, weights = self.attention(new_hidden, encoder_states) # (batch, encoder_dim)
        
        # 5. Output projection
        out_input = torch.cat([new_hidden, context], dim=-1) # (batch, decoder_dim + encoder_dim)
        logits = self.out_proj(out_input) # (batch, vocab_size)
        
        return logits, new_hidden, context, weights

class SLUModel(nn.Module):
    """
    Seq2Seq Spoken Language Understanding Model.
    """
    def __init__(self, vocab_size, input_bins=30, encoder_dim=256, decoder_dim=256, num_cnn_blocks=1):
        super(SLUModel, self).__init__()
        
        self.encoder = SLUEncoder(input_bins=input_bins, encoder_dim=encoder_dim, num_cnn_blocks=num_cnn_blocks)
        self.decoder = SLUDecoder(vocab_size=vocab_size, encoder_dim=encoder_dim, decoder_dim=decoder_dim)
        self.vocab_size = vocab_size

    def forward(self, src_spectrogram, trg_tokens, teacher_forcing_ratio=0.5):
        # src_spectrogram: (batch, seq_len, bins)
        # trg_tokens: (batch, trg_len)
        
        batch_size = src_spectrogram.size(0)
        trg_len = trg_tokens.size(1)
        
        # 1. Encode source
        encoder_states, last_hidden = self.encoder(src_spectrogram)
        
        # 2. Initialize decoder hidden state and context
        decoder_hidden = last_hidden # (batch, decoder_dim)
        context = torch.zeros(batch_size, self.encoder.dnn.out_features, device=src_spectrogram.device)
        
        outputs = []
        
        # First input token is <sos>
        dec_input = trg_tokens[:, 0]
        
        for t in range(1, trg_len):
            logits, decoder_hidden, context, _ = self.decoder(dec_input, decoder_hidden, encoder_states, context)
            outputs.append(logits)
            
            # Predict token
            top1 = logits.argmax(1)
            
            # Teacher forcing decision
            is_teacher = torch.rand(1).item() < teacher_forcing_ratio
            dec_input = trg_tokens[:, t] if is_teacher else top1
            
        return torch.stack(outputs, dim=1) # (batch, trg_len-1, vocab_size)

    def predict(self, src_spectrogram, max_len=150, sos_id=1, eos_id=2):
        """
        Greedy decoding prediction logic.
        """
        self.eval()
        with torch.no_grad():
            batch_size = src_spectrogram.size(0)
            
            encoder_states, last_hidden = self.encoder(src_spectrogram)
            decoder_hidden = last_hidden
            context = torch.zeros(batch_size, self.encoder.dnn.out_features, device=src_spectrogram.device)
            
            dec_input = torch.tensor([sos_id] * batch_size, device=src_spectrogram.device)
            
            predictions = []
            for t in range(max_len):
                logits, decoder_hidden, context, _ = self.decoder(dec_input, decoder_hidden, encoder_states, context)
                top1 = logits.argmax(1)
                predictions.append(top1)
                dec_input = top1
                
            predictions = torch.stack(predictions, dim=1) # (batch, max_len)
            return predictions
