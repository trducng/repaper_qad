import torch
import torch.nn as nn


class MultiHeadAttention(nn.Module):
    """The multihead attention module
    """

    def __init__(self, dim, n_heads, dropout):
        super(MultiHeadAttention, self).__init__()

        self.dim = dim
        self.n_heads = n_heads
        self.dropout = dropout
        self.scale = dim ** -0.5

        self.key_project = nn.Linear(dim, n_heads * dim)
        self.query_project = nn.Linear(dim, n_heads * dim)
        self.value_project = nn.Linear(dim, n_heads * dim)
        self.final_project = nn.Linear(n_heads * dim, dim)
        self.drop_layer = nn.Dropout(dropout)

    def forward(self, x):
        """Perform the forward pass

        # Args
            x <Tensor>: input of shape b x t x dim
        """
        batch, n_steps, dim = x.shape

        key = self.key_project(x)       # b x t x hidden_dim
        query = self.query_project(x)
        value = self.value_project(x)

        # b x n_heads x t x dim
        key = key.view(batch, n_steps, self.n_heads, dim).transpose(1, 2)
        query = query.view(batch, n_steps, self.n_heads, dim).transpose(1, 2)
        value = value.view(batch, n_steps, self.n_heads, dim).transpose(1, 2)

        kq = torch.matmul(query, key.transpose(2, 3)) * self.scale  # b x n_heads x t x t
        kq = torch.softmax(kq, dim=-1)      # b x n_heads x t x t
        kq = self.drop_layer(kq)
        value = kq @ value                  # b x n_heads x t x dim

        value = value.transpose(1, 2) # b x t x n_heads x dim 
        value = value.reshape(batch, n_steps, -1).contiguous()  # b x t x hidden_dim
        output = self.final_project(value)  # b x t x dim

        return output


class EncoderLayer(nn.Module):
    """Each transformer encoder layer
    """
    def __init__(self, dim, n_heads, attention_dropout, mlp_dim, dropout):
        super(EncoderLayer, self).__init__()
        self.attention = MultiHeadAttention(dim, n_heads, attention_dropout)
        self.drop_layer1 = nn.Dropout(dropout)
        self.drop_layer2 = nn.Dropout(dropout)
        self.linear = nn.Sequential(
                nn.Linear(dim, mlp_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(mlp_dim, dim))
        self.layer_norm1 = nn.LayerNorm(dim)
        self.layer_norm2 = nn.LayerNorm(dim)

    def forward(self, x):
        """Perform the input pass"""
        hidden1 = self.layer_norm1(x)        # b x t x dim
        hidden1 = self.attention(hidden1)
        hidden1 = self.drop_layer1(hidden1)
        hidden1 += x

        hidden2 = self.layer_norm2(hidden1)
        hidden2 = self.linear(hidden2)
        hidden2 = self.drop_layer2(hidden2)
        hidden2 += hidden1

        return hidden2


class Encoder(nn.Module):
    """Encoder in Transformer
    """
    def __init__(self, dim, n_heads, attention_dropout, mlp_dim, dropout, n_layers):
        super(Encoder, self).__init__()
        self.blocks = nn.Sequential(*[
                EncoderLayer(dim, n_heads, attention_dropout, mlp_dim, dropout)
                for _ in range(n_layers)])
        self.norm = nn.LayerNorm(dim)

    def forward(self, x):
        """Perform the forward pass"""
        hidden = self.blocks(x)
        output = self.norm(hidden)

        return output


class VisionTransformer(nn.Module):
    """The vision transformer

    # Args
        input_size <int>: the height/width of the input image
        patch_size <int>: the number of patches to construct
    """

    def __init__(self,
            input_size=224,
            patch_size=28,
            hidden_dim=768,
            n_layers=8,
            n_attention_heads=12,
            attention_dropout=0.1,
            mlp_dropout=0.1,
            mlp_dim=3072):

        super(VisionTransformer, self).__init__()

        self.input_size = input_size
        self.patch_size = patch_size
        self.hidden_dim = hidden_dim

        self.patch_embedding = nn.Conv2d(
                in_channels=3,
                out_channels=self.hidden_dim,
                kernel_size=self.patch_size,
                stride=self.patch_size,
                bias=True)  # b x hidden_dim x n_patch/2 x n_patch/2

        n_patches = int((input_size / patch_size) ** 2)
        self.cls_embedding = nn.Parameter(torch.randn(1, 1, self.hidden_dim))
        self.position_embedding = nn.Parameter(torch.randn(
                1,
                n_patches + 1,
                self.hidden_dim))

        self.encoder = Encoder(
                dim=hidden_dim,
                n_heads=n_attention_heads,
                attention_dropout=attention_dropout,
                mlp_dim=mlp_dim,
                dropout=mlp_dropout,
                n_layers=n_layers)

        self.linear = nn.Sequential(
                nn.Linear(in_features=self.hidden_dim, out_features=mlp_dim),
                nn.ReLU(),
                nn.Linear(in_features=mlp_dim, out_features=1000))

    def forward(self, input_x):
        """Perform the forward pass"""
        batch = input_x.size(0)

        # patch embedding
        hidden = self.patch_embedding(input_x) # b x hidden_dim x n_patch/2 x n_patch/2
        hidden = hidden.view(batch, self.hidden_dim, -1) # b x hidden_dim x T
        hidden = hidden.permute(0, 2, 1) # b x T x hidden_dim

        # add cls embedding
        cls_token = self.cls_embedding.expand(batch, -1, -1)
        hidden = torch.cat([cls_token, hidden], dim=1)  # b x (T+1) x hidden_dim

        # positional encoding
        hidden += self.position_embedding     # b x (T+1) x hidden_dim

        # transformer
        hidden1 = self.encoder(hidden)  # b x (T+1) x hidden_dim
        hidden1 = hidden1[:,0,:]  # b x hidden_dim

        # output
        output = self.linear(hidden1)    # b x 1000

        return output


if __name__ == '__main__':
    input_x = torch.randn(10, 3, 224, 224)
    model = VisionTransformer()
    output = model(input_x)
