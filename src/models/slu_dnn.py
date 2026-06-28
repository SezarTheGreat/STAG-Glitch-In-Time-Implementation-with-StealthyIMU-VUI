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

    def predict(self, src_spectrogram, max_len=150, sos_id=1, eos_id=2, valid_sequences=None):
        """
        Greedy decoding prediction logic with optional trie constraints.
        """
        self.eval()
        with torch.no_grad():
            batch_size = src_spectrogram.size(0)
            
            encoder_states, last_hidden = self.encoder(src_spectrogram)
            decoder_hidden = last_hidden
            context = torch.zeros(batch_size, self.encoder.dnn.out_features, device=src_spectrogram.device)
            
            dec_input = torch.tensor([sos_id] * batch_size, device=src_spectrogram.device)
            
            # Build the trie if valid_sequences is provided
            trie = None
            if valid_sequences is not None:
                trie = {}
                for seq in valid_sequences:
                    curr = trie
                    # We start from after the first sos_id because dec_input starts with sos_id
                    for token_id in seq[1:]:
                        if token_id not in curr:
                            curr[token_id] = {}
                        curr = curr[token_id]
            
            # Track active trie nodes for each batch item
            active_nodes = [trie] * batch_size if trie is not None else None
            
            predictions = []
            for t in range(max_len):
                logits, decoder_hidden, context, _ = self.decoder(dec_input, decoder_hidden, encoder_states, context)
                
                if active_nodes is not None:
                    # Apply trie constraints to the logits
                    for b in range(batch_size):
                        curr_node = active_nodes[b]
                        if curr_node is not None and len(curr_node) > 0:
                            # Create a mask of invalid tokens (non-keys of the current trie node)
                            mask = torch.ones(self.vocab_size, dtype=torch.bool, device=logits.device)
                            for valid_token in curr_node.keys():
                                mask[valid_token] = False
                            logits[b, mask] = -float('inf')
                        elif curr_node is not None:
                            # Node is empty, meaning target sequence ended. Only allow eos_id.
                            mask = torch.ones(self.vocab_size, dtype=torch.bool, device=logits.device)
                            mask[eos_id] = False
                            logits[b, mask] = -float('inf')
                
                top1 = logits.argmax(1)
                predictions.append(top1)
                dec_input = top1
                
                # Update the active trie nodes based on the chosen token
                if active_nodes is not None:
                    for b in range(batch_size):
                        curr_node = active_nodes[b]
                        chosen_token = top1[b].item()
                        if curr_node is not None and chosen_token in curr_node:
                            active_nodes[b] = curr_node[chosen_token]
                        else:
                            active_nodes[b] = None # outside trie or finished
                            
            predictions = torch.stack(predictions, dim=1) # (batch, max_len)
            return predictions

class PaperSLUEncoder(nn.Module):
    """
    Encoder as described in the paper:
    - 2 convolutional layers (kernel size 3x3, max pooling 2x2)
    - 3-layer bidirectional Gated Recurrent Unit (BiGRU) with 256 hidden units.
    """
    def __init__(self, input_bins=30, encoder_dim=256, num_rnn_layers=3):
        super(PaperSLUEncoder, self).__init__()
        
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2),
            nn.MaxPool2d(kernel_size=(2, 2)), # bins -> 15
            
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2),
            nn.MaxPool2d(kernel_size=(2, 2))  # bins -> 7
        )
        
        # input_bins // 4 = 7
        rnn_input_dim = 128 * 7
        
        self.gru = nn.GRU(
            input_size=rnn_input_dim,
            hidden_size=encoder_dim // 2, # BiGRU -> hidden_size * 2 = encoder_dim
            num_layers=num_rnn_layers,
            batch_first=True,
            bidirectional=True
        )
        
        self.dnn = nn.Linear(encoder_dim, encoder_dim)

    def forward(self, x):
        batch_size, seq_len, bins = x.size()
        x = x.unsqueeze(1) # (batch, 1, seq_len, bins)
        
        x = self.cnn(x) # (batch, 128, seq_len // 4, 7)
        batch_size, channels, pooled_seq_len, pooled_bins = x.size()
        
        x = x.transpose(1, 2).contiguous() # (batch, pooled_seq_len, channels, pooled_bins)
        x = x.view(batch_size, pooled_seq_len, -1) # (batch, pooled_seq_len, channels * pooled_bins)
        
        gru_out, h_n = self.gru(x) # (batch, pooled_seq_len, encoder_dim)
        encoder_states = self.dnn(gru_out)
        
        # Average last states of bidirectional GRU
        last_hidden = h_n.view(self.gru.num_layers, 2, batch_size, self.gru.hidden_size)
        last_hidden = last_hidden[-1] # take last layer: shape (2, batch_size, hidden_size)
        last_hidden = torch.cat([last_hidden[0], last_hidden[1]], dim=-1) # (batch_size, encoder_dim)
        
        return encoder_states, last_hidden

class PaperSLUModel(SLUModel):
    def __init__(self, vocab_size, input_bins=30, encoder_dim=256, decoder_dim=256):
        super(SLUModel, self).__init__()
        self.encoder = PaperSLUEncoder(input_bins=input_bins, encoder_dim=encoder_dim)
        self.decoder = SLUDecoder(vocab_size=vocab_size, encoder_dim=encoder_dim, decoder_dim=decoder_dim)
        self.vocab_size = vocab_size
