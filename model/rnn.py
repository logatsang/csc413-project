from typing import Literal
import torch


class EtymologyRNN(torch.nn.Module):
    """
    RNN-based etymological classifier for input terms as token sequences.

    Architecture:
        embedding: Linear embedding layer.
        rnn: RNN, GRU, or LSTM layers. Number of layers is a hyperparameter.
        proj: Linear projection layer for class prediction.
    """

    def __init__(
        self,
        vocab_size: int,
        embedding_size: int,
        hidden_size: int,
        num_classes: int,
        padding_idx: int = 0,
        num_layers: int = 6,
        bidirectional: bool = True,
        rnn_type: Literal["rnn", "gru", "lstm"] = "gru",
    ) -> None:
        """
        Initialize EtymologyRNN classifier model.

        Parameters:
            vocab_size: Input vocabulary size.
            embedding_size: Embedding feature dimension.
            hidden_size: Hidden feature dimension for RNN.
            num_classes: Number of output classes.
            padding_idx: Index of padding token.
            num_layers: Number of RNN layers.
            bidirectional: Whether to use bidirectional RNN.
            rnn_type: Type of RNN architecture to use.
        """
        super().__init__()
        self.padding_idx = padding_idx

        self.embedding = torch.nn.Embedding(
            vocab_size, embedding_size, padding_idx=padding_idx
        )
        self.rnn = {"rnn": torch.nn.RNN, "gru": torch.nn.GRU, "lstm": torch.nn.LSTM}[
            rnn_type
        ](
            input_size=embedding_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            bidirectional=bidirectional,
            batch_first=True,
        )
        self.proj = torch.nn.Linear(hidden_size * (1 + bidirectional), num_classes)

    def forward(self, x: torch.Tensor, pack_sequence: bool = True) -> torch.Tensor:
        """
        Compute logits for a batch of input sequences.

        Parameters:
            x: Batch of padded input sequences, shape (batch_size, max_seq_len).
            pack_sequence: Whether to pack the data with torch.nn.utils.rnn.pack_padded_sequence.

        Return: Tensor of output logits, shape (batch_size, num_classes).
        """
        lens = None
        if pack_sequence:
            lens = torch.tensor(
                [s.tolist().index(True) if any(s) else s.size(dim=0) for s in x == self.padding_idx]
            ).cpu()
        x = self.embedding(x)
        if pack_sequence:
            x = torch.nn.utils.rnn.pack_padded_sequence(x, lens, batch_first=True, enforce_sorted=False)  # type: ignore
        x, _ = self.rnn(x)
        if pack_sequence:
            x, _ = torch.nn.utils.rnn.pad_packed_sequence(x, batch_first=True)  # type: ignore
        # TODO: experiment with alternative methods for condensing RNN outputs?
        x = torch.mean(x, dim=-2)
        x = self.proj(x)
        return x
