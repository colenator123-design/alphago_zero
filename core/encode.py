import torch
import chess

POLICY_DIM = 64 * 64  # from-square, to-square
STATE_CHANNELS = 13   # 12 for pieces, 1 for side-to-move

def encode_board(board: chess.Board) -> torch.Tensor:
    """
    將 python-chess 的棋盤轉成一個 (C, H, W) = (13, 8, 8) 的張量 for CNN
    - 0-5: 白方棋子 (P, N, B, R, Q, K)
    - 6-11: 黑方棋子 (P, N, B, R, Q, K)
    - 12: 當前下子方顏色 (白方=1, 黑方=-1)
    """
    tensor = torch.zeros(STATE_CHANNELS, 8, 8, dtype=torch.float32)
    for square in chess.SQUARES:
        piece = board.piece_at(square)
        if piece:
            rank, file = chess.square_rank(square), chess.square_file(square)
            offset = 0 if piece.color == chess.WHITE else 6
            plane_idx = offset + (piece.piece_type - 1)
            tensor[plane_idx, rank, file] = 1.0

    # Side-to-move plane
    stm = 1.0 if board.turn == chess.WHITE else -1.0
    tensor[12, :, :] = stm
    return tensor

def move_to_policy_index(move: chess.Move) -> int:
    """將走法轉為 4096 維 policy 向量的索引"""
    return move.from_square * 64 + move.to_square
