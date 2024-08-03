import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.encoding = torch.zeros(max_len, d_model)
        self.encoding.requires_grad = False

        pos = torch.arange(0, max_len).unsqueeze(1)
        _2i = torch.arange(0, d_model, 2)
        self.encoding[:, 0::2] = torch.sin(pos / (10000 ** (_2i / d_model)))
        self.encoding[:, 1::2] = torch.cos(pos / (10000 ** (_2i / d_model)))

    def forward(self, x):
        seq_len = x.size(1)
        return x + self.encoding[:seq_len, :].to(x.device)


class TransformerModel(nn.Module):
    def __init__(self, feat_dim, time_dim, d_model, nhead, num_encoder_layers, num_decoder_layers, dim_feedforward, out_dim, max_len=5000):
        super(TransformerModel, self).__init__()
        self.positional_encoding = PositionalEncoding(d_model, max_len)
        self.input_projection = nn.Linear(feat_dim, d_model)
        self.transformer = nn.Transformer(d_model, nhead, num_encoder_layers, num_decoder_layers, dim_feedforward, batch_first=True)
        self.fc = nn.Linear(time_dim * d_model, out_dim)

    def forward(self, x: torch.Tensor):
        """
        Args:
            x: [Tensor] (batch_size, feat_dim, time_dim)
        return:
            output representation [Tensor] (batch_size, out_dim)
        """
        B = x.size(0)
        x = x.permute(0, 2, 1)  # (batch_size, time_dim, feat_dim)

        x = self.input_projection(x)  # (batch_size, time_dim, d_model)
        x = self.positional_encoding(x)  # (batch_size, time_dim, d_model)

        x = x.permute(1, 0, 2)  # (time_dim, batch_size, d_model)
        transformer_out = self.transformer(x, x)  # (time_dim, batch_size, d_model)
        transformer_out = transformer_out.permute(1, 0, 2)  # (batch_size, time_dim, d_model)

        transformer_out = transformer_out.reshape(B, -1)  # (batch_size, time_dim * d_model)
        out = self.fc(transformer_out)  # (batch_size, out_dim)

        return out


if __name__ == "__main__":
    model = TransformerModel(
        feat_dim=40,
        time_dim=972,
        d_model=64,
        nhead=8,
        num_encoder_layers=3,
        num_decoder_layers=3,
        dim_feedforward=256,
        out_dim=1,
        max_len=972
    )
    x = torch.Tensor(np.random.rand(8, 40, 972))
    y = model(x)
    print(y.shape)  # Expected output: torch.Size([8, 1])
    print(y)
