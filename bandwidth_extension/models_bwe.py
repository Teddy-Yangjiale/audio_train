import torch
import torch.nn as nn
import torch.nn.functional as F


class EncoderBlock(nn.Module):
    """Conv1d -> BatchNorm -> LeakyReLU -> downsample"""
    def __init__(self, in_ch, out_ch, kernel_size, stride):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(in_ch, out_ch, kernel_size, stride=stride,
                      padding=kernel_size // 2),
            nn.BatchNorm1d(out_ch),
            nn.LeakyReLU(0.2),
        )

    def forward(self, x):
        return self.conv(x)


class DecoderBlock(nn.Module):
    """Upsample -> concat skip -> Conv1d -> BatchNorm -> LeakyReLU"""
    def __init__(self, in_ch, skip_ch, out_ch, kernel_size, stride):
        super().__init__()
        self.upsample = nn.ConvTranspose1d(in_ch, in_ch, kernel_size=stride,
                                           stride=stride, padding=0)
        self.conv = nn.Sequential(
            nn.Conv1d(in_ch + skip_ch, out_ch, kernel_size, stride=1,
                      padding=kernel_size // 2),
            nn.BatchNorm1d(out_ch),
            nn.LeakyReLU(0.2),
        )

    def forward(self, x, skip):
        x = self.upsample(x)
        # Trim to match skip length
        if x.shape[-1] > skip.shape[-1]:
            x = x[..., :skip.shape[-1]]
        elif x.shape[-1] < skip.shape[-1]:
            skip = skip[..., :x.shape[-1]]
        x = torch.cat([x, skip], dim=1)
        return self.conv(x)


class AudioUNet(nn.Module):
    """U-Net for 1D audio bandwidth extension.

    Input:  low-res audio upsampled to target sample rate  (B, 1, T)
    Output: high-res audio residual added to input          (B, 1, T)
    """

    def __init__(self, base_channels=64, num_encoders=5,
                 kernel_size=15, stride=4):
        super().__init__()
        self.stride = stride
        ch = base_channels

        # --- Encoder ---
        self.encoders = nn.ModuleList()
        self.enc_channels = []  # track per-layer channels
        in_ch = 1
        for i in range(num_encoders):
            out_ch = min(ch * (2 ** i), 512)
            self.enc_channels.append(out_ch)
            self.encoders.append(
                EncoderBlock(in_ch, out_ch, kernel_size,
                             stride if i < num_encoders - 1 else 1)
            )
            in_ch = out_ch

        # --- Bottleneck ---
        bottleneck_ch = in_ch
        self.bottleneck = nn.Sequential(
            nn.Conv1d(bottleneck_ch, bottleneck_ch, kernel_size,
                      padding=kernel_size // 2),
            nn.BatchNorm1d(bottleneck_ch),
            nn.LeakyReLU(0.2),
            nn.Conv1d(bottleneck_ch, bottleneck_ch, kernel_size,
                      padding=kernel_size // 2),
            nn.BatchNorm1d(bottleneck_ch),
            nn.LeakyReLU(0.2),
        )

        # --- Decoder ---
        self.decoders = nn.ModuleList()
        decoder_ch = self.enc_channels[-1]  # match bottleneck
        for dec_idx in range(num_encoders - 1, 0, -1):  # 4, 3, 2, 1
            if dec_idx >= 2:
                skip_ch = self.enc_channels[dec_idx - 2]
            else:
                skip_ch = 1  # original input
            out_ch = skip_ch
            self.decoders.append(
                DecoderBlock(decoder_ch, skip_ch, out_ch, kernel_size, stride)
            )
            decoder_ch = out_ch

        # --- Output ---
        self.output_conv = nn.Sequential(
            nn.Conv1d(decoder_ch, ch, kernel_size, padding=kernel_size // 2),
            nn.LeakyReLU(0.2),
            nn.Conv1d(ch, 1, 1),
            nn.Tanh(),
        )

    def forward(self, x):
        # Encoder
        skips = []
        for enc in self.encoders:
            skips.append(x)
            x = enc(x)

        # Bottleneck
        x = self.bottleneck(x) + x

        # Decoder
        for i, dec in enumerate(self.decoders):
            skip = skips[-(i + 2)]
            x = dec(x, skip)

        return self.output_conv(x)

    def infer(self, low_res_audio):
        """Process variable-length audio in overlapping chunks."""
        self.eval()
        segment_len = 32768
        overlap = 8192
        audio = low_res_audio.unsqueeze(0)  # (1, 1, T)

        output = torch.zeros_like(audio)
        weight = torch.zeros_like(audio)
        window = torch.hann_window(segment_len).to(audio.device)

        stride = segment_len - overlap
        for start in range(0, audio.shape[-1], stride):
            end = min(start + segment_len, audio.shape[-1])
            chunk = audio[..., start:end]
            if chunk.shape[-1] < segment_len:
                pad_len = segment_len - chunk.shape[-1]
                chunk = F.pad(chunk, (0, pad_len))

            with torch.no_grad():
                pred = self(chunk)

            if end - start < segment_len:
                pred = pred[..., :end - start]

            output[..., start:end] += pred * window[:end - start]
            weight[..., start:end] += window[:end - start]

        output = output / weight.clamp(min=1e-6)
        return output.squeeze(0)
