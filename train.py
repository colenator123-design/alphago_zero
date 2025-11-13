# train.py
# -*- coding: utf-8 -*-
"""
AlphaZero 主訓練程式
"""

import math
import random
from dataclasses import dataclass
from typing import Dict, List, Tuple
from collections import deque

import torch
import torch.nn as nn
import torch.optim as optim
import chess

# 從 core 模組導入核心元件
from core.encode import encode_board, move_to_policy_index, POLICY_DIM
from core.model import ChessNet
from core.mcts import MCTS

# ---------------------------------------------------------
# 經驗回放池 & 訓練流程
# ---------------------------------------------------------

@dataclass
class GameSample:
    fen: str
    policy_target: torch.Tensor
    player_sign: float
    result_white: float

class ReplayBuffer:
    def __init__(self, capacity: int):
        self.capacity = capacity
        self.buffer = deque(maxlen=capacity)

    def push(self, game: List[GameSample]):
        for sample in game:
            self.buffer.append(sample)

    def sample(self, batch_size: int) -> List[GameSample]:
        return random.sample(self.buffer, batch_size)

    def __len__(self) -> int:
        return len(self.buffer)

def self_play_game(mcts: MCTS, temperature: float = 1.0, max_moves: int = 200) -> List[GameSample]:
    board = chess.Board()
    history: List[GameSample] = []

    for _ply in range(max_moves):
        if board.is_game_over():
            break

        # Pass is_self_play=True to MCTS.run
        visits = mcts.run(board, is_self_play=True)
        total_visits = sum(visits.values())
        if total_visits == 0:
            break
            
        policy_target = torch.zeros(POLICY_DIM, dtype=torch.float32)
        for move, count in visits.items():
            policy_target[move_to_policy_index(move)] = count / total_visits

        moves = list(visits.keys())
        probs = [visits[m] for m in moves]

        if temperature > 0:
            move = random.choices(moves, weights=probs, k=1)[0]
        else:
            move = max(moves, key=lambda m: visits[m])

        player_sign = 1.0 if board.turn == chess.WHITE else -1.0
        history.append(GameSample(
            fen=board.fen(),
            policy_target=policy_target,
            player_sign=player_sign,
            result_white=0.0,
        ))
        board.push(move)

    if board.is_checkmate():
        result_white = 1.0 if not board.turn == chess.WHITE else -1.0
    else: # Draw or other terminal conditions
        result_white = 0.0

    for sample in history:
        sample.result_white = result_white

    return history


def train_on_buffer(
    net: ChessNet,
    optimizer: optim.Optimizer,
    buffer: ReplayBuffer,
    steps: int,
    batch_size: int,
    device: str = "cpu",
    value_loss_weight: float = 1.0,
):
    net.train()
    total_v_loss = 0.0
    total_p_loss = 0.0
    
    for _ in range(steps):
        batch = buffer.sample(batch_size)
        if not batch:
            continue
        
        state_tensors = torch.stack([encode_board(chess.Board(s.fen)) for s in batch]).to(device)
        policy_targets = torch.stack([s.policy_target for s in batch]).to(device)
        value_targets = torch.tensor([s.result_white * s.player_sign for s in batch], dtype=torch.float32).to(device)

        v_preds, policy_logits = net(state_tensors)

        v_loss = nn.functional.mse_loss(v_preds, value_targets)
        p_loss = nn.functional.cross_entropy(policy_logits, policy_targets)
        loss = value_loss_weight * v_loss + p_loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_v_loss += v_loss.item()
        total_p_loss += p_loss.item()

    avg_v_loss = total_v_loss / steps if steps > 0 else 0
    avg_p_loss = total_p_loss / steps if steps > 0 else 0
    print(f"[train] steps = {steps}, avg_v_loss = {avg_v_loss:.4f}, avg_p_loss = {avg_p_loss:.4f}")


# ---------------------------------------------------------
# main：主訓練迴圈
# ---------------------------------------------------------

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"使用裝置：{device}")

    net = ChessNet().to(device)
    optimizer = optim.Adam(net.parameters(), lr=1e-4)

    # --- 訓練參數 ---
    n_iterations = 100
    games_per_iter = 5
    mcts_simulations = 50
    
    buffer_capacity = 20000
    min_buffer_size = 500
    train_batch_size = 64
    train_steps = 100
    # ---

    replay_buffer = ReplayBuffer(capacity=buffer_capacity)

    for it in range(1, n_iterations + 1):
        print(f"\n========== Iteration {it}/{n_iterations} ==========")

        net.eval()
        mcts = MCTS(net, n_simulations=mcts_simulations, c_puct=1.5, device=device)

        for g in range(1, games_per_iter + 1):
            print(f"[self-play] game {g}/{games_per_iter} ... (buffer size: {len(replay_buffer)})")
            game_samples = self_play_game(mcts, max_moves=150)
            replay_buffer.push(game_samples)
            print(f"  產生步數：{len(game_samples)}")

        if len(replay_buffer) < min_buffer_size:
            print(f"Buffer size ({len(replay_buffer)}) is not large enough for training (min: {min_buffer_size}). Skipping training.")
            continue
            
        print(f"Buffer is ready. Starting training on {len(replay_buffer)} samples.")
        train_on_buffer(
            net, 
            optimizer, 
            replay_buffer, 
            steps=train_steps, 
            batch_size=train_batch_size, 
            device=device
        )

        if it % 10 == 0:
            torch.save(net.state_dict(), f"chess_mcts_net_iter_{it}.pt")
            print(f"模型已儲存為 chess_mcts_net_iter_{it}.pt")

    torch.save(net.state_dict(), "chess_mcts_net.pt")
    print("訓練完成，最終模型已儲存為 chess_mcts_net.pt")


if __name__ == "__main__":
    main()
