import torch
import torch.nn as nn
import torch.nn.functional as F

class Autoencoder(nn.Module):
    def __init__(self, latent_dim=32, input_length=512):
        super().__init__()

        self.encoder_convs = nn.Sequential(
            nn.Conv1d(1, 16, 9, stride=2, padding=4), nn.BatchNorm1d(16), nn.ReLU(),
            nn.Conv1d(16, 32, 9, stride=2, padding=4), nn.BatchNorm1d(32), nn.ReLU(),
            nn.Conv1d(32, 64, 9, stride=2, padding=4), nn.BatchNorm1d(64), nn.ReLU()
        )

        with torch.no_grad():
            dummy = torch.zeros(1, 1, input_length)
            conv_out = self.encoder_convs(dummy)
            self.flat_dim = conv_out.numel()

        self.fc_enc = nn.Linear(self.flat_dim, latent_dim)
        self.fc_dec = nn.Linear(latent_dim, self.flat_dim)

        self.decoder_convs = nn.Sequential(
            nn.ConvTranspose1d(64, 32, 9, stride=2, padding=4, output_padding=1),
            nn.BatchNorm1d(32), nn.ReLU(),
            nn.ConvTranspose1d(32, 16, 9, stride=2, padding=4, output_padding=1),
            nn.BatchNorm1d(16), nn.ReLU(),
            nn.ConvTranspose1d(16, 1, 9, stride=2, padding=4, output_padding=1)
        )

    def encode(self, x):
        x = self.encoder_convs(x)
        x = x.view(x.size(0), -1)
        z = self.fc_enc(x)
        return z

    def decode(self, z):
        x = self.fc_dec(z)
        x = x.view(z.size(0), 64, -1)     # 64 channels recovered
        x = self.decoder_convs(x)
        return x

    def forward(self, x):
        z = self.encode(x)
        x_hat = self.decode(z)
        if x_hat.size(-1) > x.size(-1):
            x_hat = x_hat[..., :x.size(-1)]
        elif x_hat.size(-1) < x.size(-1):
            pad = x.size(-1) - x_hat.size(-1)
            x_hat = F.pad(x_hat, (0, pad))
        return x_hat