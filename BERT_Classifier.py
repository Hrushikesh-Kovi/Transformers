import torch
import torch.nn as nn

class T5Transformer(nn.Module):
    def __init__(self, vocab_size, d_model=64, d_ff=256, num_layers=2, max_seq_len=20, nhead=4, dropout=0.1):
        super().__init__()
        self.encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            batch_first=True,
            dim_feedforward=d_ff,
            dropout=dropout
        )
        self.decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model,
            dim_feedforward=d_ff,
            nhead=nhead,
            dropout=dropout,
            batch_first=True
        )
        self.token_embeddings = nn.Embedding(
            vocab_size, d_model
        )
        self.positional_embeddings = nn.Embedding(
            max_seq_len, d_model
        )
        self.encoder = nn.TransformerEncoder(
            self.encoder_layer, num_layers, enable_nested_tensor=False
        )
        self.decoder = nn.TransformerDecoder(
            self.decoder_layer, num_layers
        )
        self.output_layer = nn.Linear(d_model, vocab_size)

    def embed(self, tokens):
        positions = torch.arange(tokens.size(1), device=tokens.device).unsqueeze(0)
        return (self.token_embeddings(tokens) + self.positional_embeddings(positions))

    def forward(self, source, target):
        source_embeddings = self.embed(source)
        target_embeddings = self.embed(target)

        memory = self.encoder(source_embeddings)

        target_length = target.size(1)

        causal_mask = nn.Transformer.generate_square_subsequent_mask(
            target_length,
            device=target.device
        )
        decoder_output = self.decoder(
            tgt=target_embeddings,
            memory=memory,
            tgt_mask=causal_mask
        )
        return self.output_layer(decoder_output)

model = T5Transformer(vocab_size=15)

source = torch.tensor([[1, 2, 3, 4]])
target = torch.tensor([[10, 3, 2, 1]])

logits = model(source, target)

print("Source shape:", source.shape)
print("Target shape:", target.shape)
print("Output shape:", logits.shape)