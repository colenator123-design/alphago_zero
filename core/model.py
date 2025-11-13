import torch
import torch.nn as nn
from typing import Tuple

from .encode import STATE_CHANNELS, POLICY_DIM

class ResBlock(nn.Module):
    def __init__(self, num_filters):
        super().__init__()
        self.conv1 = nn.Conv2d(num_filters, num_filters, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(num_filters)
        self.conv2 = nn.Conv2d(num_filters, num_filters, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(num_filters)

    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = nn.functional.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out += identity
        return nn.functional.relu(out)

class ChessNet(nn.Module):
    """
    CNN + ResNet 架構
    - 輸入: (N, 13, 8, 8) 的棋盤狀態
    - 輸出:
        - value: (N, 1) 的局面價值
        - policy_logits: (N, 4096) 的走法 logits
    """
    def __init__(self, num_res_blocks=5, num_filters=128):
        super().__init__()
        self.initial_conv = nn.Sequential(
            nn.Conv2d(STATE_CHANNELS, num_filters, kernel_size=3, padding=1),
            nn.BatchNorm2d(num_filters),
            nn.ReLU()
        )
        
        self.res_blocks = nn.ModuleList([ResBlock(num_filters) for _ in range(num_res_blocks)])
        
        # Value Head
        self.value_head = nn.Sequential(
            nn.Conv2d(num_filters, 1, kernel_size=1),
            nn.BatchNorm2d(1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(64, num_filters),
            nn.ReLU(),
            nn.Linear(num_filters, 1),
            nn.Tanh()
        )
        
        # Policy Head
        self.policy_head = nn.Sequential(
            nn.Conv2d(num_filters, 2, kernel_size=1),
            nn.BatchNorm2d(2),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(2 * 8 * 8, POLICY_DIM)
        )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # Add batch dimension if not present
        if x.dim() == 3:
            x = x.unsqueeze(0)

        x = self.initial_conv(x)
        for res_block in self.res_blocks:
            x = res_block(x)
        
        value = self.value_head(x)
        policy_logits = self.policy_head(x)
        
        return value.squeeze(-1), policy_logits
