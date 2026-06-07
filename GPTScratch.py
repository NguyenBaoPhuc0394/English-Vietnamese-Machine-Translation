import torch
import torch.nn as nn

class GPTDecoderLayer(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward, dropout=0.1):
        super(GPTDecoderLayer, self).__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.feed_forward = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, d_model)
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, tgt_mask=None, tgt_key_padding_mask=None):
        attn_output, _ = self.self_attn(x, x, x, attn_mask=tgt_mask, key_padding_mask=tgt_key_padding_mask)
        x = self.norm1(x + self.dropout(attn_output))
        ff_output = self.feed_forward(x)
        x = self.norm2(x + self.dropout(ff_output))
        return x


class GPTModel(nn.Module):
    def __init__(self, vocab_size, d_model=256, nhead=8, num_layers=6, dim_feedforward=1024, max_length=50):
        super(GPTModel, self).__init__()
        self.d_model = d_model
        self.max_length = max_length

        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.pos_embedding = nn.Embedding(max_length, d_model)
        self.dropout = nn.Dropout(0.1)

        self.decoder_layers = nn.ModuleList([
            GPTDecoderLayer(d_model, nhead, dim_feedforward) for _ in range(num_layers)
        ])

        self.output_layer = nn.Linear(d_model, vocab_size)
        self.final_norm = nn.LayerNorm(d_model)  # Optional but useful for training stability

    def forward(self, input_ids, attention_mask=None):
        batch_size, seq_len = input_ids.size()

        # Causal mask: shape (seq_len, seq_len)
        tgt_mask = torch.triu(
            torch.full((seq_len, seq_len), float('-inf'), device=input_ids.device),
            diagonal=1
        )

        # Token + position embedding
        position_ids = torch.arange(seq_len, device=input_ids.device).unsqueeze(0).expand(batch_size, seq_len)
        x = self.token_embedding(input_ids) * (self.d_model ** 0.5)
        x = x + self.pos_embedding(position_ids)
        x = self.dropout(x)

        # Transpose for multihead attention: (seq_len, batch_size, d_model)
        x = x.transpose(0, 1)

        key_padding_mask = ~attention_mask.bool() if attention_mask is not None else None

        for layer in self.decoder_layers:
            x = layer(x, tgt_mask=tgt_mask, tgt_key_padding_mask=key_padding_mask)

        x = x.transpose(0, 1)
        x = self.final_norm(x)
        logits = self.output_layer(x)
        return logits
