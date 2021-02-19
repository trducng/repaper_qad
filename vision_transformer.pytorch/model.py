import torch
import torch.nn as nn


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
            n_layers=12,
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

        encoder_layer = nn.TransformerEncoderLayer(
                d_model=self.hidden_dim,
                nhead=n_attention_heads,
                dim_feedforward=mlp_dim,
                dropout=attention_dropout,
                activation='relu')
        self.encoder = nn.TransformerEncoder(
                encoder_layer=encoder_layer,
                num_layers=n_layers)

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
