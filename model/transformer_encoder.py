import torch


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
        x = self.encoder(x, mask=mask)
        x = torch.mean(x, dim=-2)
        x = self.proj(x)
        return x
