import pandas as pd
import torch
import torch.nn as nn
import os


class TransformerModel(nn.Module):
    def __init__(self, vocab_size = 20000, d_model = 256, nhead = 8,num_encoder_layers = 4, num_decoder_layers = 4, dim_feedforward = 1024, dropout: float = 0.1):
        super(TransformerModel, self).__init__()
        self.transformer = torch.nn.Transformer(
            d_model=d_model,
            nhead=nhead,
            num_encoder_layers=num_encoder_layers,
            num_decoder_layers=num_decoder_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True
        )
        self.embedding = torch.nn.Embedding(vocab_size, d_model)
        self.layer_norm = torch.nn.LayerNorm(d_model)  # Added LayerNorm after embedding
        self.positional_encoding = self.create_positional_encoding(max_len=5000, d_model=d_model)  # Changed max_len to 5000
        self.fc_out = torch.nn.Linear(d_model, vocab_size)
        self.d_model = d_model
        
    def create_positional_encoding(self, max_len, d_model):
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-torch.log(torch.tensor(10000.0)) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        return pe.unsqueeze(0)
    
    def forward(self, src: torch.Tensor, tgt: torch.Tensor, src_mask: torch.Tensor = None, tgt_mask: torch.Tensor = None, src_key_padding_mask: torch.Tensor = None, tgt_key_padding_mask: torch.Tensor = None):
        src = self.embedding(src) * torch.sqrt(torch.tensor(self.d_model, dtype=torch.float))
        tgt = self.embedding(tgt) * torch.sqrt(torch.tensor(self.d_model, dtype=torch.float))
        
        # Apply LayerNorm after embedding
        src = self.layer_norm(src)
        tgt = self.layer_norm(tgt)
        
        src = src + self.positional_encoding[:, :src.size(1), :].to(src.device)
        tgt = tgt + self.positional_encoding[:, :tgt.size(1), :].to(tgt.device)
        
        output = self.transformer(
            src, tgt,
            src_mask=src_mask,
            tgt_mask=tgt_mask,
            src_key_padding_mask=src_key_padding_mask,
            tgt_key_padding_mask=tgt_key_padding_mask
        )
        return self.fc_out(output)
    

def create_transformer_masks(src: torch.Tensor, tgt: torch.Tensor, attention_mask: torch.Tensor,device: str):
    # Target mask (prevent attending to future tokens)
    tgt_seq_len = tgt.size(1)
    tgt_mask = torch.nn.Transformer.generate_square_subsequent_mask(tgt_seq_len).to(device)
    
    # Source padding mask
    src_key_padding_mask = ~attention_mask.bool()
    
    # Target padding mask (ignore padding tokens)
    tgt_key_padding_mask = ~(tgt != 0)
    
    return tgt_mask, src_key_padding_mask, tgt_key_padding_mask

