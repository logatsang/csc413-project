from typing import Sequence
import torch


class EtymologyCNN(torch.nn.Module):
    """
    CNN-based etymological classifier for input terms as token sequences.

    Architecture:
        embedding: Linear embedding layer.
        conv: List of convolutional layers and max pooling layers.
        proj: Linear projection layer for class prediction.
    """
    def __init__(
        self,
        vocab_size: int,
        embedding_size: int,
        num_classes: int,
        conv_layers: Sequence[tuple[int, ...]],
        conv_filter_count: int = 1,
        padding_idx: int = 0
    ) -> None:
        """
        Initialize EtymologyCNN classifier model.
        
        Parameters:
            vocab_size: Input vocabulary size.
            embedding_size: Embedding feature dimension.
            num_classes: Number of output classes.
            conv_layers: Tuple of tuples describing each pair of convolutional/pooling layers. Each pair has kernel size, stride, and padding.
            padding_idx: Index of padding token.
        """
        super().__init__()
        self.padding_idx = padding_idx

        self.embedding = torch.nn.Embedding(
            vocab_size, embedding_size, padding_idx=padding_idx
        )

        self.conv = torch.nn.ModuleList()
        for index, layer_settings in enumerate(conv_layers):
            (
                conv_kernel_size,
                conv_stride,
                conv_padding,
                pool_kernel_size,
                pool_stride,
                pool_padding
            ) = layer_settings

            conv_layer = torch.nn.Conv1d(
                in_channels=conv_filter_count if index else embedding_size,
                out_channels=conv_filter_count,
                kernel_size=conv_kernel_size,
                stride=conv_stride,
                padding=conv_padding
            )

            pool_layer = torch.nn.AvgPool1d(
                kernel_size=pool_kernel_size,
                stride=pool_stride,
                padding=pool_padding
            )

            self.conv.append(conv_layer)
            self.conv.append(pool_layer)

        self.proj = torch.nn.Linear(
            in_features=conv_filter_count,
            out_features=num_classes
        )

    def forward(self, x: torch.Tensor):
        x = self.embedding(x)
        x = x.permute(0, 2, 1)

        for layer in self.conv:
            x = layer(x)
    
        x, _ = x.max(dim=-1)
        x = self.proj(x)

        return x