import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class Attention(nn.Module):
    """Scaled Dot-Product Attention

    Args:
        input_shape: the embedding shape of each token
        hidden_shape: the embedding shape of query and key
        value_shape: the embedding shape of value
    """

    def __init__(self, input_shape: int, hidden_shape: int, value_shape: int):
        super().__init__()
        self.input_shape = input_shape
        self.hidden_shape = hidden_shape
        self.value_shape = value_shape

        self.q = nn.Linear(in_features=input_shape, out_features=hidden_shape)
        self.k = nn.Linear(in_features=input_shape, out_features=hidden_shape)
        self.v = nn.Linear(in_features=input_shape, out_features=value_shape)

    def forward(self, x):
        """Perform forward pass

        Args:
            x: shape (batch size x channels x in_dim)

        Returns:
            tensor of shape (batch size x channels x value_shape)
        """
        q, k, v = self.q(x), self.k(x), self.v(x)  # NxTxhidden, NxTxvalue
        att = F.softmax(torch.matmul(q, k.transpose(1, 2)), dim=-1) / math.sqrt(
            self.hidden_shape
        )
        return torch.matmul(att, v)


class SimplifiedMultiHeadAttention(nn.Module):
    """Multihead attention from scaled dot-product attention

    @NOTE: this implementation does not guarantee the input and output to have same
    dimension. However, each transformer block has the positional addition after self-
    attention, which implies that the input and output of self-attention must have the
    same shape (unless, we patch with linear to force the input and output to have the
    same dimension).

    Args:
        n_heads: number of parallel attention heads
    """

    def __init__(
        self, n_heads: int, input_shape: int, hidden_shape: int, value_shape: int
    ):
        super().__init__()
        self.n_heads = n_heads
        self.atts = [
            Attention(
                input_shape=input_shape,
                hidden_shape=hidden_shape,
                value_shape=value_shape,
            )
            for _ in range(n_heads)
        ]

    def forward(self, x):
        """Perform the forward pass

        Args:
            x: shape (batch size x channels x in_dim)

        Returns:
            tensor of shape (batch size x channels x value_shape * n_heads)
        """
        items = [each(x) for each in self.atts]
        return torch.cat(items, dim=-1)


class MaskedMultiHeadAttention(nn.Module):
    """Multi-head attention block

    Unlike SimplifiedMultiHeadAttention, this implementation guarantees the input and
    output have same dimension. This implementation can better be executed on GPU.
    Also, the mask is there to make sure that the future does not come back to the
    past.

    Args:
        input_shape: dimension of the input
        n_heads: the number of attention heads
    """

    def __init__(self, input_shape: int, n_heads: int):
        if input_shape % n_heads:
            raise AttributeError(f"{input_shape=} must be divisible by {n_heads=}")

        super().__init__()
        self.input_shape = input_shape
        self.n_heads = n_heads
        self.kqv = nn.Linear(in_features=input_shape, out_features=input_shape * 3)

    def forward(self, x):
        """Forward through MultiHeadAttention block

        Args:
            x: the input with shape (batch size x sequence length x in dims)

        Returns:
            the output with shape (batch size x sequence length x in dims)
        """
        N, C, D = x.shape
        z = self.kqv(x)

        k = z[:, :, :D].reshape(N, C, self.n_heads, D // self.n_heads).transpose(1, 2)
        q = (
            z[:, :, D : D * 2]
            .reshape(N, C, self.n_heads, D // self.n_heads)
            .transpose(1, 2)
        )
        v = (
            z[:, :, D * 2 :]
            .reshape(N, C, self.n_heads, D // self.n_heads)
            .transpose(1, 2)
        )

        z = torch.matmul(q, k.transpose(2, 3)) / math.sqrt(D)
        mask = torch.tril(torch.ones(C, C, device=x.device)).view(1, 1, C, C)
        z = z.masked_fill(mask == 0, float("-inf"))
        z = F.softmax(z, dim=-1)
        z = torch.matmul(z, v)
        return z.transpose(1, 2).reshape(N, C, D)


class DecoderBlock(nn.Module):
    """Transformer decoder block

    Input ---> MaskedMultiSelfAttention -+-> FeedForward ---> LayerNorm -+-> Output
           |                                |                 |              |
            --------------------------------                   --------------
    """

    def __init__(self, n_heads: int, input_shape: int):
        super().__init__()
        self.attention = MaskedMultiHeadAttention(input_shape, n_heads)
        self.norm_01 = nn.LayerNorm(normalized_shape=input_shape)
        self.linear = nn.Linear(in_features=input_shape, out_features=input_shape)
        self.norm_02 = nn.LayerNorm(normalized_shape=input_shape)

    def forward(self, x):
        z = self.attention(x)
        x = self.norm_01(x + z)
        z = self.linear(x)
        return self.norm_02(x + z)


class EmbeddingAndPositionalEncoding(nn.Module):
    """Input embedding and positional encoding"""

    def __init__(self, num_embeddings, embedding_dim, sequence_length):
        super(EmbeddingAndPositionalEncoding, self).__init__()
        self.text_embedding = nn.Embedding(
            num_embeddings=num_embeddings, embedding_dim=embedding_dim
        )
        self.position_embedding = nn.Embedding(
            num_embeddings=sequence_length, embedding_dim=embedding_dim
        )

    def forward(self, x):
        pos = torch.arange(0, x.shape[1], device=x.device).unsqueeze(0)
        tok_emb = self.text_embedding(x)
        pos_emb = self.position_embedding(pos)
        return tok_emb + pos_emb


class TransformerDecoder(nn.Module):
    """The transformer decoder block

    Args:
        n_blocks: the number of decoder blocks in the transformer decoder
    """

    def __init__(self, n_blocks: int, n_heads: int, input_shape: int):
        super().__init__()
        self.blocks = nn.Sequential(
            *[DecoderBlock(n_heads, input_shape) for _ in range(n_blocks)]
        )

    def forward(self, x):
        return self.blocks(x)


class Transformer(nn.Module):
    """The transformer model used in GPT"""

    def __init__(self, vocab_size, embedding_dim, sequence_length, n_blocks, n_heads):
        super(Transformer, self).__init__()
        self.embedding = EmbeddingAndPositionalEncoding(
            num_embeddings=vocab_size,
            embedding_dim=embedding_dim,
            sequence_length=sequence_length,
        )
        self.transformer = TransformerDecoder(
            n_blocks=n_blocks,
            n_heads=n_heads,
            input_shape=embedding_dim,
        )
        self.linear = nn.Linear(in_features=embedding_dim, out_features=vocab_size)

    def forward(self, x):
        z = self.embedding(x)
        z = self.transformer(z)
        return self.linear(z)


if __name__ == "__main__":
    def test_TransformerDecoder():
        n_heads = 2
        input_shape = 8
        seq_length = 3
        x = torch.ones(1, seq_length, input_shape)
        mha = TransformerDecoder(n_blocks=12, n_heads=n_heads, input_shape=input_shape)
        return mha(x)

    def test_Transformer():
        model = Transformer(
            vocab_size=1000,
            embedding_dim=192,
            sequence_length=1024,
            n_blocks=6,
            n_heads=6,
        )
        input_ = torch.zeros(1, 1024, dtype=torch.int64)
        input_[0,0] = 10
        input_[0,2] = 120
        output = model(input_)
        return output

    output = test_Transformer()
