import torch
import torch.nn as nn


class ConvBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, stride: int = 2):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(
                in_ch,
                out_ch,
                kernel_size=3,
                stride=stride,
                padding=1,
                bias=False
            ),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class DeconvBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, stride: int = 2, activation: nn.Module = None):
        super().__init__()
        act = activation if activation is not None else nn.ReLU(inplace=True)
        self.block = nn.Sequential(
            nn.ConvTranspose2d(
                in_ch,
                out_ch,
                kernel_size=4,
                stride=stride,
                padding=1,
                bias=False
            ),
            nn.BatchNorm2d(out_ch),
            act,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class KuroshioAutoencoder(nn.Module):
    def __init__(self, in_channels: int = 2, base_filters: int = 32):
        super().__init__()
        f = base_filters

        self.enc1 = ConvBlock(in_channels, f)
        self.enc2 = ConvBlock(f, f * 2)
        self.enc3 = ConvBlock(f * 2, f * 4)
        self.enc4 = ConvBlock(f * 4, f * 8)

        self.dec4 = DeconvBlock(f * 8, f * 4)
        self.dec3 = DeconvBlock(f * 4, f * 2)
        self.dec2 = DeconvBlock(f * 2, f)
        self.dec1 = DeconvBlock(f, in_channels, activation=nn.Tanh())

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        z = self.enc1(x)
        z = self.enc2(z)
        z = self.enc3(z)
        z = self.enc4(z)
        return z

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        x = self.dec4(z)
        x = self.dec3(x)
        x = self.dec2(x)
        x = self.dec1(x)
        return x

    def _crop_to_match(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Center-crop decoder output to match the original input size.
        This fixes size mismatches when H/W are not multiples of 16.
        """
        _, _, h_t, w_t = target.shape
        _, _, h_p, w_p = pred.shape

        if h_p == h_t and w_p == w_t:
            return pred

        start_h = max((h_p - h_t) // 2, 0)
        start_w = max((w_p - w_t) // 2, 0)

        end_h = start_h + h_t
        end_w = start_w + w_t

        return pred[:, :, start_h:end_h, start_w:end_w]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        pred = self.decode(self.encode(x))
        pred = self._crop_to_match(pred, x)
        return pred


def masked_mse_loss(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    err = (pred - target) ** 2
    err = err.mean(dim=1)
    err = err * mask.unsqueeze(0).float()
    loss = err.sum() / (mask.sum() * pred.size(0))
    return loss


def pixel_error_map(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return ((pred - target) ** 2).sum(dim=1)