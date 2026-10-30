import torch
from torch import nn

class TransformerEncoder(nn.Module):
    def __init__(self, d_model, nhead, d_ff, dropout=0.1):
        super().__init__()

        self.norm1 = nn.LayerNorm(normalized_shape=d_model)
        self.norm2 = nn.LayerNorm(normalized_shape=d_model)

        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model)
        )

        self.attention = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=nhead,
            dropout=dropout,
            batch_first=True
        )

        self.dropout = nn.Dropout(dropout)

    def forward(self, x, padding_mask=None):
        normalized = self.norm1(x)

        attn, _ = self.attention(normalized, normalized, normalized, key_padding_mask=padding_mask, need_weights=False)

        x = x + self.dropout(attn)

        normalized = self.norm2(x)
        ffop = self.ffn(normalized)

        x = x + self.dropout(ffop)

        return x


class TransformerDecoder(nn.Module):
    def __init__(self, d_model, nhead, d_ff, dropout=0.1):
        super().__init__()

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model)
        )

        self.attention = nn.MultiheadAttention(
            d_model, nhead, dropout=dropout,
            batch_first=True
        )

        self.dropout = nn.Dropout(dropout)

    def forward(self, x, padding_mask=None):
        seq_len = x.size(1)

        casual_mask = torch.triu(
            torch.ones(seq_len, seq_len, dtype=torch.bool, device=x.device),
            diagonal=1
        )

        normalized = self.norm1(x)

        attn, _ = self.attention(
            normalized, normalized, normalized,
            need_weights=False, attn_mask=casual_mask,
            key_padding_mask=padding_mask
        )

        x = x + self.dropout(attn)

        normalized = self.norm2(x)
        ffop = self.ffn(normalized)

        x = x + self.dropout(ffop)

        return x
        



decoder = TransformerDecoder(8, 2, 32)
x = torch.randn(2, 4, 8)

op = decoder(x)
print(op.shape)