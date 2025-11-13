import math
import torch
from dataclasses import dataclass
from typing import Dict, List, Tuple
import chess
import numpy as np # 導入 numpy

from .model import ChessNet
from .encode import encode_board, move_to_policy_index

@dataclass
class MCTSNode:
    board: chess.Board
    parent: 'MCTSNode' = None
    children: Dict[chess.Move, 'MCTSNode'] = None
    N: Dict[chess.Move, int] = None      # visit counts
    W: Dict[chess.Move, float] = None    # total value
    Q: Dict[chess.Move, float] = None    # mean value
    P: Dict[chess.Move, float] = None    # prior from NN
    is_expanded: bool = False

    def __post_init__(self):
        if self.children is None:
            self.children = {}
        if self.N is None:
            self.N = {}
        if self.W is None:
            self.W = {}
        if self.Q is None:
            self.Q = {}
        if self.P is None:
            self.P = {}


class MCTS:
    def __init__(self, net: ChessNet, n_simulations: int = 50, c_puct: float = 1.5, device: str = "cpu"):
        self.net = net
        self.n_simulations = n_simulations
        self.c_puct = c_puct
        self.device = device

    def _terminal_value(self, board: chess.Board) -> float:
        if board.is_checkmate(): return -1.0
        return 0.0

    def _expand(self, node: MCTSNode) -> float:
        board = node.board
        if board.is_game_over():
            return self._terminal_value(board)

        state_vec = encode_board(board).to(self.device)
        with torch.no_grad():
            value, policy_logits = self.net(state_vec)
        
        value = value.item()
        policy_logits = policy_logits.squeeze(0)

        legal_moves = list(board.legal_moves)
        if not legal_moves:
            return self._terminal_value(board)

        # 只保留合法走法的 logits，並重新 softmax
        legal_move_indices = [move_to_policy_index(m) for m in legal_moves]
        legal_logits = policy_logits[legal_move_indices]
        probs = torch.softmax(legal_logits, dim=0).cpu().numpy()

        for move, p in zip(legal_moves, probs):
            child_board = board.copy()
            child_board.push(move)
            child_node = MCTSNode(board=child_board, parent=node)
            node.children[move] = child_node
            node.N[move] = 0
            node.W[move] = 0.0
            node.Q[move] = 0.0
            node.P[move] = float(p)

        node.is_expanded = True
        return value

    def _simulate(self, root: MCTSNode):
        node = root
        path: List[Tuple[MCTSNode, chess.Move]] = []
        while True:
            if node.board.is_game_over():
                v = self._terminal_value(node.board)
                break
            if not node.is_expanded:
                v = self._expand(node)
                break
            
            sum_N = sum(node.N.values()) + 1e-8
            best_score, best_move, best_child = -1e9, None, None
            for move, child in node.children.items():
                N = node.N.get(move, 0)
                Q = node.Q.get(move, 0.0)
                P = node.P.get(move, 0.0)
                u = self.c_puct * P * math.sqrt(sum_N) / (1 + N)
                score = Q + u
                if score > best_score:
                    best_score, best_move, best_child = score, move, child
            
            if best_child is None:
                v = self._terminal_value(node.board)
                break
            
            path.append((node, best_move))
            node = best_child

        value = v
        for parent, move in reversed(path):
            parent.N[move] = parent.N.get(move, 0) + 1
            parent.W[move] = parent.W.get(move, 0.0) + value
            parent.Q[move] = parent.W[move] / parent.N[move]
            value = -value

    def run(self, board: chess.Board, is_self_play: bool = False) -> Dict[chess.Move, int]:
        root = MCTSNode(board=board.copy())
        _ = self._expand(root)

        if is_self_play:
            epsilon = 0.25  # AlphaZero uses 0.25
            alpha = 0.3     # AlphaZero uses 0.3 for chess

            moves = list(root.P.keys())
            if moves: # Only apply if there are legal moves
                prior_probs = np.array([root.P[m] for m in moves])

                # Generate Dirichlet noise
                noise = np.random.dirichlet([alpha] * len(moves))

                # Mix noise with prior probabilities
                mixed_probs = (1 - epsilon) * prior_probs + epsilon * noise

                # Re-normalize to ensure sum is 1 (due to floating point, might be slightly off)
                mixed_probs /= mixed_probs.sum()

                # Update root.P
                for i, move in enumerate(moves):
                    root.P[move] = mixed_probs[i]
        
        for _ in range(self.n_simulations):
            self._simulate(root)
        
        visits = {move: root.N.get(move, 0) for move in root.children.keys()}
        return visits
