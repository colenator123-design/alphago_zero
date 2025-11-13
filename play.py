import math
import torch
import torch.nn as nn
import chess
from dataclasses import dataclass
from typing import Dict, List, Tuple

# 從 core 模組導入核心元件
from core.encode import encode_board
from core.model import ChessNet
from core.mcts import MCTS

# ===================================================================
# 2. 新增的人機對弈主程式 (介面美化版)
# ===================================================================

UNICODE_PIECES = {
    "P": "♙", "R": "♖", "N": "♘", "B": "♗", "Q": "♕", "K": "♔",
    "p": "♟", "r": "♜", "n": "♞", "b": "♝", "q": "♛", "k": "♚",
}

def print_pretty_board(board: chess.Board):
    """用 Unicode 符號來印出更美觀的棋盤"""
    print()
    # We print the board from rank 8 (index 7) down to 1 (index 0)
    for rank in range(7, -1, -1):
        print(f" {rank+1} |", end="")
        for file in range(8):
            square = chess.square(file, rank)
            piece = board.piece_at(square)
            if piece:
                # Use two spaces for alignment, one for the piece
                print(f" {UNICODE_PIECES[piece.symbol()]}", end="")
            else:
                print(" .", end="")
        print(" |")
    print("   +-----------------+")
    print("     a b c d e f g h")


def play_game():
    """主遊戲迴圈"""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model_path = "chess_mcts_net.pt"
    
    # 載入模型
    try:
        net = ChessNet().to(device)
        net.load_state_dict(torch.load(model_path, map_location=device))
        net.eval()
        print(f"成功從 {model_path} 載入模型。")
    except FileNotFoundError:
        print(f"錯誤：找不到模型檔案 {model_path}。請先執行 `train.py` 進行訓練。")
        return
    except Exception as e:
        print(f"載入模型時發生錯誤：{e}")
        return

    # 增加模擬次數以提高 AI 棋力
    mcts = MCTS(net, n_simulations=100, c_puct=1.5, device=device)
    board = chess.Board()

    print("\n--- 西洋棋人機對弈 ---")
    print("您執白棋 (White)，AI 執黑棋 (Black)。")
    print("請使用標準代數記譜法 (SAN) 輸入您的走法 (例如: e4, Nf3, O-O)。")
    print("輸入 'exit' 或 'quit' 來結束遊戲。")

    while not board.is_game_over():
        print("\n" + "="*20)
        print_pretty_board(board)
        print(f"\n輪到 {'白方 (您)' if board.turn == chess.WHITE else '黑方 (AI)'} 下棋。")

        if board.turn == chess.WHITE: # 人類玩家
            move_san = input("您的走法: ").strip()

            if move_san.lower() in ['quit', 'exit']:
                print("遊戲結束。")
                break

            try:
                # 使用 push_san，更直覺
                board.push_san(move_san)
            except ValueError:
                print("錯誤：無效或不合法的走法，請再試一次 (例如: e4, Nf3, O-O)。")

        else: # AI 玩家
            print("AI 正在思考中...")
            policy = mcts.run(board)
            
            if not policy:
                print("AI 找不到任何走法，遊戲可能已結束。")
                break

            # 選擇訪問次數最多的走法
            ai_move = max(policy, key=policy.get)
            # 輸出也使用 SAN
            print(f"AI 下子: {board.san(ai_move)}")
            board.push(ai_move)

    # 遊戲結束
    print("\n" + "="*20)
    print("遊戲結束！")
    print_pretty_board(board)
    result = board.result()
    print(f"\n結果: {result}")
    if board.is_checkmate():
        winner = "AI (黑方)" if board.turn == chess.WHITE else "您 (白方)"
        print(f"{winner} 將死對方！")
    elif board.is_stalemate():
        print("和局 (逼和)。")
    elif board.is_insufficient_material():
        print("和局 (子力不足)。")
    else:
        print("和局 (其他原因)。")


if __name__ == "__main__":
    play_game()


# ===================================================================
# 2. 新增的人機對弈主程式 (介面美化版)
# ===================================================================

UNICODE_PIECES = {
    "P": "♙", "R": "♖", "N": "♘", "B": "♗", "Q": "♕", "K": "♔",
    "p": "♟", "r": "♜", "n": "♞", "b": "♝", "q": "♛", "k": "♚",
}

def print_pretty_board(board: chess.Board):
    """用 Unicode 符號來印出更美觀的棋盤"""
    print()
    # We print the board from rank 8 (index 7) down to 1 (index 0)
    for rank in range(7, -1, -1):
        print(f" {rank+1} |", end="")
        for file in range(8):
            square = chess.square(file, rank)
            piece = board.piece_at(square)
            if piece:
                # Use two spaces for alignment, one for the piece
                print(f" {UNICODE_PIECES[piece.symbol()]}", end="")
            else:
                print(" .", end="")
        print(" |")
    print("   +-----------------+")
    print("     a b c d e f g h")


def play_game():
    """主遊戲迴圈"""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model_path = "chess_mcts_net.pt"
    
    # 載入模型
    try:
        net = ChessNet().to(device)
        net.load_state_dict(torch.load(model_path, map_location=device))
        net.eval()
        print(f"成功從 {model_path} 載入模型。")
    except FileNotFoundError:
        print(f"錯誤：找不到模型檔案 {model_path}。請先執行 `chess_mcts_rl.py` 進行訓練。")
        return
    except Exception as e:
        print(f"載入模型時發生錯誤：{e}")
        return

    # 增加模擬次數以提高 AI 棋力
    mcts = MCTS(net, n_simulations=100, c_puct=1.5, device=device)
    board = chess.Board()

    print("\n--- 西洋棋人機對弈 ---")
    print("您執白棋 (White)，AI 執黑棋 (Black)。")
    print("請使用標準代數記譜法 (SAN) 輸入您的走法 (例如: e4, Nf3, O-O)。")
    print("輸入 'exit' 或 'quit' 來結束遊戲。")

    while not board.is_game_over():
        print("\n" + "="*20)
        print_pretty_board(board)
        print(f"\n輪到 {'白方 (您)' if board.turn == chess.WHITE else '黑方 (AI)'} 下棋。")

        if board.turn == chess.WHITE: # 人類玩家
            move_san = input("您的走法: ").strip()

            if move_san.lower() in ['quit', 'exit']:
                print("遊戲結束。")
                break

            try:
                # 使用 push_san，更直覺
                board.push_san(move_san)
            except ValueError:
                print("錯誤：無效或不合法的走法，請再試一次 (例如: e4, Nf3, O-O)。")

        else: # AI 玩家
            print("AI 正在思考中...")
            policy = mcts.run(board)
            
            if not policy:
                print("AI 找不到任何走法，遊戲可能已結束。")
                break

            # 選擇訪問次數最多的走法
            ai_move = max(policy, key=policy.get)
            # 輸出也使用 SAN
            print(f"AI 下子: {board.san(ai_move)}")
            board.push(ai_move)

    # 遊戲結束
    print("\n" + "="*20)
    print("遊戲結束！")
    print_pretty_board(board)
    result = board.result()
    print(f"\n結果: {result}")
    if board.is_checkmate():
        winner = "AI (黑方)" if board.turn == chess.WHITE else "您 (白方)"
        print(f"{winner} 將死對方！")
    elif board.is_stalemate():
        print("和局 (逼和)。")
    elif board.is_insufficient_material():
        print("和局 (子力不足)。")
    else:
        print("和局 (其他原因)。")


if __name__ == "__main__":
    play_game()
