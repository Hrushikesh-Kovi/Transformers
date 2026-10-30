import torch
import torch.nn.functional as F
from torch import nn


text = "I love deep learning I love machine learning"

words = text.lower().split()
vocab = {word : i for i, word in enumerate(sorted(set(words)))}
id_to_word = {i : word for word, i in vocab.items()}

tokens = torch.tensor([[vocab[word] for word in words]])

vocab_size = len(vocab)
d_model = 32
d_ff = 128
nhead = 4
num_layers = 2
max_seq_len = 16


class GPTDecoderBlock(nn.Module):
    def __init__(self, d_model, nhead, d_ff, dropout=0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.attention = nn.MultiheadAttention(
            d_model,
            nhead,
            dropout=dropout,
            batch_first=True
        )
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model)
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        seq_len = x.size(1)
        causal_mask = torch.triu(
            torch.ones(
                seq_len, seq_len,
                dtype=torch.bool,
                device=x.device
            ),
            diagonal=1
        )
        normalized = self.norm1(x)
        attn_output, _ = self.attention(
            normalized, normalized, normalized,
            attn_mask=causal_mask,
            need_weights=False
        )
        x = x + self.dropout(attn_output)
        ffn_output = self.ffn(self.norm2(x))
        x = x + self.dropout(ffn_output)
        return x

class GPT(nn.Module):
    def __init__(self, vocab_size, d_model, nhead, d_ff, num_layers, max_seq_len):
        super().__init__()
        self.token_embedding = nn.Embedding(
            vocab_size, d_model
        )
        self.positional_embedding = nn.Embedding(
            max_seq_len, d_model
        )
        self.blocks = nn.Sequential(
            *[
                GPTDecoderBlock(d_model, nhead, d_ff)
                for _ in range(num_layers)
            ]
        )
        self.final_norm = nn.LayerNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size)

    def forward(self, tokens):
        batch_size, seq_len = tokens.shape
        positions = torch.arange(seq_len, device=tokens.device)
        token_emb = self.token_embedding(tokens)
        positional_emb = self.positional_embedding(positions)

        x = token_emb + positional_emb
        x = self.blocks(x)
        x = self.final_norm(x)
        logits = self.lm_head(x)

        return logits


model = GPT(
    vocab_size=vocab_size,
    d_model=d_model,
    d_ff=d_ff,
    num_layers=num_layers,
    max_seq_len=max_seq_len,
    nhead=nhead
)
x = tokens[:, :-1]
y = tokens[:, 1:]
optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)

for step in range(300):
    model.train()
    logits = model(x)
    loss = F.cross_entropy(logits.reshape(-1, vocab_size), y.reshape(-1))
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    if step == 0 or (step+1) % 10 == 0:
        print(f"Epoch : {step+1} | Loss : {loss.item():.4f}")

model.eval()

test_corpus = "I love"
test_tokens = torch.tensor([[vocab[word] for word in sorted(set(test_corpus.lower().split()))]])


with torch.no_grad():
    logits = model(test_tokens)
    next_token_logits = logits[:, -1, :]
    probs = torch.softmax(next_token_logits, dim=-1)
    predicted_id = probs.argmax(dim=-1).item()

print(f"Next Token : {id_to_word[predicted_id]}")