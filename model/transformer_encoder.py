import math

import torch


class PositionalEncoding(torch.nn.Module):
    def __init__(self, d_model: int, dropout: float, max_len=500):
        super(PositionalEncoding, self).__init__()
        self.dropout = torch.nn.Dropout(p=dropout)
        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2) * -(math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer("pe", pe)

    def forward(self, x):
        pos = self.pe[:, : x.size(1)].requires_grad_(False)
        x = x + pos  # Add the position encoding to original vector x
        return self.dropout(x)

class EtymologyTransformerEncoder(torch.nn.Module):
    def __init__(
        self,
        vocab_size: int,
        num_classes: int,
        num_layers: int = 6,
        embedding_size: int = 512,
        nhead: int = 4,
        d_ff: int = 2048,
        padding_idx: int = 0,
    ):
        super().__init__()
        self.padding_idx = padding_idx
        self.nhead = nhead

        self.embedding = torch.nn.Embedding(
            vocab_size, embedding_size, padding_idx=padding_idx
        )
        self.src_pe = PositionalEncoding(embedding_size, 0.1)
        self.encoder = torch.nn.TransformerEncoder(
            torch.nn.TransformerEncoderLayer(
                d_model=embedding_size,
                nhead=nhead,
                dim_feedforward=d_ff,
                batch_first=True,
            ),
            num_layers=num_layers,
        )
        self.proj = torch.nn.Linear(embedding_size, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        mask = torch.repeat_interleave(
            x[:, None, :].repeat(1, x.size(dim=-1), 1) == self.padding_idx,
            self.nhead,
            dim=0,
        )
        x = self.embedding(x)
        x = self.src_pe(x)
        x = self.encoder(x, mask=mask)
        x = torch.mean(x, dim=-2)
        x = self.proj(x)
        return x
