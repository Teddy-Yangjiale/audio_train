import torch
import torch.nn as nn


def stft_mag(x, n_fft, hop_length, win_length=None):
    window = torch.hann_window(n_fft if win_length is None else win_length).to(x.device)
    stft = torch.stft(
        x.squeeze(1),
        n_fft=n_fft,
        hop_length=hop_length,
        win_length=win_length or n_fft,
        window=window,
        return_complex=True,
        pad_mode="reflect",
    )
    return stft.abs()


def spectral_convergence(pred_mag, target_mag):
    return (torch.norm(target_mag - pred_mag, p="fro") /
            torch.norm(target_mag, p="fro").clamp(min=1e-6))


def log_stft_magnitude(pred_mag, target_mag, eps=1e-7):
    return torch.mean(torch.abs(torch.log(pred_mag + eps) - torch.log(target_mag + eps)))


class MultiResolutionSTFTLoss(nn.Module):
    """Multi-resolution STFT loss (spectral convergence + log-magnitude)."""

    def __init__(self, fft_sizes, hop_ratios):
        super().__init__()
        self.fft_sizes = fft_sizes
        self.hop_ratios = hop_ratios

    def forward(self, pred, target):
        sc_loss, mag_loss = 0.0, 0.0
        for n_fft, ratio in zip(self.fft_sizes, self.hop_ratios):
            hop = int(n_fft * ratio)
            pred_mag = stft_mag(pred, n_fft, hop)
            target_mag = stft_mag(target, n_fft, hop)
            sc_loss += spectral_convergence(pred_mag, target_mag)
            mag_loss += log_stft_magnitude(pred_mag, target_mag)
        return sc_loss / len(self.fft_sizes), mag_loss / len(self.fft_sizes)


class CombinedLoss(nn.Module):
    """L1 (time) + Multi-resolution STFT (freq)"""

    def __init__(self, l1_weight=100.0, stft_weight=1.0,
                 fft_sizes=None, hop_ratios=None):
        super().__init__()
        self.l1 = nn.L1Loss()
        self.l1_weight = l1_weight
        self.stft_weight = stft_weight
        self.stft_loss = MultiResolutionSTFTLoss(
            fft_sizes or [2048, 1024, 512],
            hop_ratios or [0.25, 0.25, 0.25],
        )

    def forward(self, pred, target):
        l1 = self.l1(pred, target)
        sc, mag = self.stft_loss(pred, target)
        total = self.l1_weight * l1 + self.stft_weight * (sc + mag)
        return total, {"l1": l1.item(), "sc": sc.item(), "mag": mag.item()}


def compute_snr(pred, target):
    """Signal-to-noise ratio in dB."""
    noise = pred - target
    snr = 10 * torch.log10(
        (target ** 2).sum() / (noise ** 2).sum().clamp(min=1e-10)
    )
    return snr.item()


def compute_lsd(pred, target, n_fft=2048, hop=512):
    """Log Spectral Distance."""
    pred_mag = stft_mag(pred.unsqueeze(0), n_fft, hop)
    target_mag = stft_mag(target.unsqueeze(0), n_fft, hop)
    lsd = torch.mean(torch.sqrt(
        torch.mean((torch.log(pred_mag + 1e-7) - torch.log(target_mag + 1e-7)) ** 2,
                   dim=0)
    ))
    return lsd.item()
