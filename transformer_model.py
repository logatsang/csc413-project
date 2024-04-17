import math

import numpy as np
import torch
import torch.nn as nn


class LayerNorm(nn.Module):
    def __init__(self, num_features, eps=1e-6):
        super(LayerNorm, self).__init__()
        self.gamma = nn.Parameter(torch.ones(num_features))
        self.beta = nn.Parameter(torch.zeros(num_features))
        self.eps = eps

    def forward(self, x):
        """
        Compute layer normalization
            y = gamma * (x - mu) / (sigma + eps) + beta where mu and sigma are computed over the feature dimension

        x: torch.Tensor, shape [batch_size, seq_len, d_model]
        return: torch.Tensor, shape [batch_size, seq_len, d_model]
        """
        mu = torch.mean(x, dim=2, keepdim=True)
        sigma = torch.std(x, dim=2, keepdim=True)

        return (self.gamma * (x - mu)) / (sigma - self.eps) + self.beta


class MultiHeadAttention(nn.Module):
    """
    Multi-head attention for both self-attention and cross-attention
    """

    def __init__(
        self,
        num_heads,
        d_model,
        dropout=0.0,
        atten_dropout=0.0,
        store_attention_scores=False,
    ):
        """
        num_heads: int, the number of heads
        d_model: int, the dimension of the model
        dropout: float, the dropout rate
        atten_dropout: float, the dropout rate for the attention i.e. drops out full tokens
        store_attention_scores: bool, whether to store the attention scores for visualization
        """
        super(MultiHeadAttention, self).__init__()
        assert d_model % num_heads == 0
        # Assume values and keys are the same size
        self.d_head = d_model // num_heads
        self.num_heads = num_heads

        # Note for students, for self-attention, it is more efficient to treat q, k, and v as one matrix
        # but this way allows us to use the same attention function for cross-attention
        self.q_linear = nn.Linear(d_model, d_model)
        self.k_linear = nn.Linear(d_model, d_model)
        self.v_linear = nn.Linear(d_model, d_model)

        self.atten_dropout = nn.Dropout(p=atten_dropout)  # applied after softmax

        # applied at the end
        self.out_linear = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(p=dropout)

        # used for visualization
        self.store_attention_scores = store_attention_scores
        self.attention_scores = None  # set by set_attention_scores

    def set_attention_scores(self, scores):
        """
        A helper function for visualization of attention scores.
        These are stored as attributes so that students do not need to deal with passing them around.

        The attention scores should be given after masking but before the softmax.
        scores: torch.Tensor, shape [batch_size, num_heads, query_seq_len, key_seq_len]
        return: None
        """
        if scores is None:  # for clean up
            self.attention_scores = None
        if self.store_attention_scores and not self.training:
            self.attention_scores = scores.cpu().detach().numpy()

    def attention(self, query, key, value, mask=None):
        """
        Scaled dot product attention
        Hint: the mask is applied before the softmax.
        Hint: attention dropout `self.atten_dropout` is applied to the attention weights after the softmax.

        You are required to make comments about the shapes of the tensors at each step of the way
        in order to assist the markers.  Does a tensor change shape?  Make a comment.

        You are required to call set_attention_scores with the correct tensor before returning from this function.
        The attention scores should be given after masking but before the softmax.

        query: torch.Tensor, shape [batch_size, num_heads, query_seq_len, d_head]
        key: torch.Tensor, shape [batch_size, num_heads, key_seq_len, d_head]
        value: torch.Tensor, shape [batch_size, num_heads, key_seq_len, d_head]
        mask:  torch.Tensor, shape [batch_size, query_seq_len, key_seq_len,], True, where masked or None

        return torch.Tensor, shape [batch_size, num_heads, query_seq_len, d_head]
        """

        # dot product
        dot_product_attention_score = query @ torch.transpose(key, 2, 3) # shape [batch_size, num_heads, query_seq_len, key_seq_len]

        # raw_score_shape = (len(query), query.shape[1], query.shape[2], key.shape[2])
        # assert(dot_product_attention_score.shape == raw_score_shape)

        # mask
        if mask is not None:
            masked_attention_score = dot_product_attention_score + (torch.unsqueeze(mask, 1) * -1e9)
            # assert(masked_attention_score.shape == raw_score_shape)
        else:
            masked_attention_score = dot_product_attention_score
        self.set_attention_scores(masked_attention_score)

        # scale and softmaxs
        prob_attention = torch.softmax(masked_attention_score / math.sqrt(self.d_head), dim=-1)
        # assert(prob_attention.shape == raw_score_shape)

        dropout_attention = self.atten_dropout(prob_attention)

        attention = dropout_attention @ value # shape [batch_size, num_heads, query_seq_len, d_head]
        assert(attention.shape == query.shape)

        return attention

    def forward(self, query, key=None, value=None, mask=None):
        """
        If the key and values are None, assume self-attention is being applied.  Otherwise, assume cross-attention.

        Note we only need one mask, which will work for either causal self-attention or cross-attention as long as
        the mask is set up properly beforehand.

        You are required to make comments about the shapes of the tensors at each step of the way
        in order to assist the markers.  Does a tensor change shape?  Make a comment.

        query: torch.Tensor, shape [batch_size, query_seq_len, d_model]
        key: torch.Tensor, shape [batch_size, key_seq_len, d_model] or None
        value: torch.Tensor, shape [batch_size, key_seq_len, d_model] or None
        mask: torch.Tensor, shape [batch_size, query_seq_len, key_seq_len,], True where masked or None

        return: torch.Tensor, shape [batch_size, query_seq_len, d_model]
        """
        q = self.q_linear(query)
        k = self.k_linear(key if key is not None else query)
        v = self.v_linear(value if value is not None else query)

        # split into heads
        q = q.transpose(-1, -2).reshape(len(q), self.num_heads, self.d_head, -1).transpose(-1, -2)
        k = k.transpose(-1, -2).reshape(len(k), self.num_heads, self.d_head, -1).transpose(-1, -2)
        v = v.transpose(-1, -2).reshape(len(v), self.num_heads, self.d_head, -1).transpose(-1, -2)

        # call attention
        attention = self.attention(q, k, v, mask)

        # recombine
        combined = attention.transpose(-1, -2).reshape(query.shape[0], query.shape[2], query.shape[1]).transpose(-1, -2)
        return self.dropout(self.out_linear(combined))


class FeedForwardLayer(nn.Module):
    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.1):
        super(FeedForwardLayer, self).__init__()
        self.w_1 = nn.Linear(d_model, d_ff)
        self.f = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)
        self.w_2 = nn.Linear(d_ff, d_model)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x):
        """
        Compute the feedforward sublayer.
        Dropout is applied after the activation function and after the second linear layer

        x: torch.Tensor, shape [batch_size, seq_len, d_model]
        return: torch.Tensor, shape [batch_size, seq_len, d_m odel]
        """
        h = self.dropout1(self.f(self.w_1(x)))
        return self.dropout2(self.w_2(h))


class TransformerEncoderLayer(nn.Module):
    """

    Idea if we can give this init done, then the students can fill in the decoder init in the same way but add in cross attention


    Performs multi-head self attention and FFN with the desired pre- or post-layer norm and residual connections.
    """

    def __init__(
        self,
        d_model: int,
        d_ff: int,
        num_heads: int,
        dropout: float = 0.1,
        atten_dropout: float = 0.0,
    ):
        """
        d_model: int, the dimension of the model
        d_ff: int, the dimension of the feedforward network interior projection
        num_heads: int, the number of heads for the multi-head attention
        dropout: float, the dropout rate
        atten_dropout: float, the dropout rate for the attention i.e. drops out full tokens
            Hint:  be careful about zeroing out tokens.  How does this affect the softmax?
        """
        super(TransformerEncoderLayer, self).__init__()
        assert d_model % num_heads == 0
        self.d_model = d_model
        self.d_ff = d_ff
        self.num_head = num_heads

        self.ln1 = LayerNorm(d_model)
        self.self_attn = MultiHeadAttention(
            num_heads, d_model, dropout=dropout, atten_dropout=atten_dropout
        )

        self.ln2 = LayerNorm(d_model)
        self.ff = FeedForwardLayer(d_model, d_ff, dropout=dropout)

    def forward(self, x, mask):
        self_attn_out = self.self_attn(self.ln1(x), mask=mask) + x
        ff_out = self.ff(self.ln2(self_attn_out)) + self_attn_out
        return ff_out


class TransformerEncoder(nn.Module):
    """
    Stacks num_layers of TransformerEncoderLayer and applies layer norm at the correct place.
    """

    def __init__(
        self,
        num_layers: int,
        d_model: int,
        d_ff: int,
        num_heads: int,
        dropout: float = 0.1,
        atten_dropout: float = 0.0,
    ):
        super(TransformerEncoder, self).__init__()
        self.layers = torch.nn.ModuleList()
        for l in range(num_layers):
            self.layers.append(
                TransformerEncoderLayer(
                    d_model, d_ff, num_heads, dropout, atten_dropout
                )
            )
        self.norm = LayerNorm(d_model)

    def forward(self, x, mask):
        """
        x: torch.Tensor, the input to the encoder
        mask: torch.Tensor, the mask to apply to the attention
        """
        for layer in self.layers:
            x = layer(x, mask)
        x = self.norm(x)
        return x


class TransformerEmbeddings(nn.Module):
    def __init__(self, vocab_size: int, d_model: int):
        super(TransformerEmbeddings, self).__init__()
        self.lookup = nn.Embedding(vocab_size, d_model)
        self.d_model = d_model

    def forward(self, x):
        """
        x: torch.Tensor, shape [batch_size, seq_len] of int64 in range [0, vocab_size)
        return torch.Tensor, shape [batch_size, seq_len, d_model]
        """
        return self.lookup(x) * math.sqrt(self.d_model)


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, dropout: float, max_len=500):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
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


class Transformer(nn.Module):
    def __init__(
            self,
            src_vocab_size: int,
            d_model: int,
            output_classes: int,
            padding_idx: int,
            num_heads: int,
            d_ff,
            num_layers,
            dropout: float = 0.1,
            atten_dropout: float = 0.0):
        super(Transformer, self).__init__()
        self.padding_idx = padding_idx

        self.src_embed = TransformerEmbeddings(src_vocab_size, d_model)
        self.src_pe = PositionalEncoding(d_model, dropout)
        self.encoder = TransformerEncoder(
            num_layers,
            d_model,
            d_ff,
            num_heads,
            dropout,
            atten_dropout,
        )

        self.out = nn.Linear(d_model, output_classes)

    def create_pad_mask(self, tokens):
        """
        Create a padding mask using pad_idx (an attribute of the class)
        Hint: respect the output shape

        tokens: torch.Tensor, [batch_size, seq_len]
        return: torch.Tensor, [batch_size, 1, seq_len] where True means to mask, and on the same device as tokens
        """
        return torch.unsqueeze((tokens == self.padding_idx), 1).to(torch.bool).to(tokens.device)


    def forward(self, x):
        encode_mask = self.create_pad_mask(x)
        encode_rep = self.src_pe(self.src_embed(x))

        encode_output = self.encoder(encode_rep, encode_mask)
        # print(f"encode output shape: {encode_output.shape}")

        pooled = torch.mean(encode_output, dim=1)
        # print(f"pool shape: {pooled.shape}")

        return self.out(pooled)
