import torch
import torch.nn as nn

class BiGRUAttention(nn.Module):
    def __init__(self, in_dim=54, hidden_dim=96, num_layers=2, num_classes=12, dropout=0.3):
        super().__init__()
        self.input_norm = nn.LayerNorm(in_dim)
        self.gru = nn.GRU(
            input_size=in_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.attn = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1),
        )
        self.head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, num_classes),
        )

    def forward(self, x):
        # x: [B, T, 54]
        x = self.input_norm(x)
        h, _ = self.gru(x)                     # [B, T, 2H]
        w = torch.softmax(self.attn(h), dim=1) # [B, T, 1]
        ctx = (h * w).sum(dim=1)               # [B, 2H]
        return self.head(ctx)

class _TCNBlock(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size=3, dilation=1, dropout=0.25):
        super().__init__()
        pad = (kernel_size - 1) * dilation // 2
        self.net = nn.Sequential(
            nn.Conv1d(in_ch, out_ch, kernel_size, padding=pad, dilation=dilation),
            nn.BatchNorm1d(out_ch),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Conv1d(out_ch, out_ch, kernel_size, padding=pad, dilation=dilation),
            nn.BatchNorm1d(out_ch),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.shortcut = nn.Conv1d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()

    def forward(self, x):
        return self.net(x) + self.shortcut(x)

class TemporalConvNet(nn.Module):
    def __init__(self, in_dim=54, channels=96, num_classes=12, dropout=0.3):
        super().__init__()
        self.input_norm = nn.LayerNorm(in_dim)
        self.blocks = nn.Sequential(
            _TCNBlock(in_dim, channels, kernel_size=3, dilation=1, dropout=dropout),
            _TCNBlock(channels, channels, kernel_size=3, dilation=2, dropout=dropout),
            _TCNBlock(channels, channels, kernel_size=3, dilation=4, dropout=dropout),
        )
        self.head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(channels * 2, num_classes),
        )

    def forward(self, x):
        # x: [B, T, 54] -> [B, 54, T]
        x = self.input_norm(x).transpose(1, 2)
        h = self.blocks(x)                     # [B, C, T]
        avg_p = h.mean(dim=-1)
        max_p = h.max(dim=-1).values
        return self.head(torch.cat([avg_p, max_p], dim=-1))
