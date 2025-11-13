# alphago_zero

## 專案簡介

這是一個基於 AlphaZero 思想實現的西洋棋 AI 專案。它結合了卷積神經網路 (CNN)、蒙地卡羅樹搜尋 (MCTS) 和強化學習 (Reinforcement Learning) 的自對弈機制，旨在從零開始學習如何下西洋棋。

專案的目標是提供一個清晰、模組化的 AlphaZero 實現，展示現代 AI 如何在複雜的策略遊戲中達到超人類的表現。

## 功能特色

*   **AlphaZero 架構**：完整的 MCTS + 深度神經網路 + 自對弈強化學習循環。
*   **CNN 棋盤表示**：使用多通道卷積神經網路處理棋盤狀態，有效捕捉空間特徵。
*   **ResNet 網路骨幹**：採用殘差網路 (ResNet) 結構，支持更深層次的網路訓練。
*   **蒙地卡羅樹搜尋 (MCTS)**：利用 MCTS 進行棋步推演，並由神經網路的價值和策略輸出引導。
*   **狄利克雷噪聲 (Dirichlet Noise)**：在自對弈的 MCTS 根節點加入噪聲，促進探索多樣化的棋步。
*   **經驗回放池 (Replay Buffer)**：儲存過去的自對弈數據，用於穩定和高效地訓練神經網路。
*   **人機對弈介面**：提供命令列介面，支援 Unicode 棋盤顯示和直觀的標準代數記譜法 (SAN) 輸入。
*   **模組化設計**：核心演算法和網路結構被拆分為獨立模組，便於維護和擴展。

## 核心演算法與技術

### 1. 遊戲引擎 (`python-chess`)
*   **用途**：處理西洋棋的規則、棋盤狀態、合法走法生成、棋譜記錄 (FEN, SAN, UCI) 等。
*   **優勢**：提供強大且經過驗證的西洋棋邏輯，讓開發者能專注於 AI 演算法本身。

### 2. 神經網路 (`PyTorch`, `CNN`, `ResNet`)
*   **框架**：使用 `PyTorch` 構建和訓練深度學習模型。
*   **棋盤編碼**：
    *   將 `chess.Board` 轉換為 `(13, 8, 8)` 的張量作為 CNN 輸入。
    *   `12` 個通道用於表示白方和黑方的六種棋子類型 (每個棋子一個 8x8 的 one-hot 平面)。
    *   `1` 個通道用於表示當前下子方 (白方為 `1.0`，黑方為 `-1.0`)。
*   **網路架構 (`ChessNet`)**：
    *   **骨幹**：由一個初始卷積層和多個 `ResBlock` (殘差區塊) 組成，用於提取棋盤特徵。
    *   **價值頭 (Value Head)**：輸出一個介於 `-1` (黑方勝) 到 `1` (白方勝) 之間的浮點數，評估當前局面的勝率。
    *   **策略頭 (Policy Head)**：輸出一個 `4096` 維的向量，表示所有 `(from_square, to_square)` 走法的對數機率 (logits)。

### 3. 蒙地卡羅樹搜尋 (MCTS)
*   **用途**：在神經網路的引導下，進行棋步的深度推演和評估。
*   **搜尋策略**：
    *   **選擇 (Selection)**：使用 `PUCT` (Polynomial Upper Confidence Trees) 公式平衡探索 (exploration) 和利用 (exploitation)。
    *   **擴展 (Expansion)**：當 MCTS 遇到未訪問過的節點時，使用神經網路的策略頭來初始化該節點的先驗機率。
    *   **模擬 (Simulation)**：在 AlphaZero 中，模擬階段被神經網路的價值頭取代，直接評估葉節點的價值。
    *   **反向傳播 (Backpropagation)**：將模擬結果（價值）沿著搜尋路徑反向傳播，更新節點的訪問次數和總價值。
*   **探索機制**：在自對弈模式下，MCTS 根節點的先驗機率會混入 `狄利克雷噪聲 (Dirichlet Noise)`，鼓勵 AI 探索更多樣的開局和走法。

### 4. 強化學習 (Reinforcement Learning)
*   **學習方式**：透過「自對弈」(Self-Play) 產生大量的訓練數據。
*   **數據生成**：AI 使用當前訓練中的模型，自己與自己對弈，記錄每一步的棋盤狀態、MCTS 產生的策略分佈和最終的勝負結果。
*   **經驗回放池 (Replay Buffer)**：
    *   儲存數萬步棋的 `(棋盤狀態, MCTS策略分佈, 最終結果)` 數據。
    *   訓練時從中隨機抽樣小批次數據，打破數據的時序相關性，穩定訓練過程。
*   **損失函數**：
    *   **價值損失 (Value Loss)**：使用均方誤差 (MSE) 衡量神經網路預測的局面價值與實際遊戲結果之間的差異。
    *   **策略損失 (Policy Loss)**：使用交叉熵 (Cross-Entropy) 衡量神經網路預測的走法機率分佈與 MCTS 產生的策略分佈之間的差異。
*   **優化器**：使用 `Adam` 優化器更新神經網路的權重。

## 專案結構

```
chess_mcts_rl/
├── core/
│   ├── __init__.py
│   ├── encode.py       # 棋盤與走法編碼函數
│   ├── model.py        # 神經網路模型 (ResBlock, ChessNet)
│   └── mcts.py         # MCTS 演算法實現 (MCTSNode, MCTS)
├── train.py            # AI 訓練主程式 (自對弈, Replay Buffer, 訓練循環)
├── play.py             # 人機對弈介面
├── chess_mcts_net.pt   # 訓練好的模型權重檔案
└── README.md           # 專案說明文件
```

## 環境設定與安裝

1.  **克隆專案**：
    ```bash
    git clone git@github.com:colenator123-design/alphago_zero.git
    cd alphago_zero
    ```

2.  **安裝依賴**：
    ```bash
    pip install torch chess numpy
    ```
    *   `torch`: PyTorch 深度學習框架。
    *   `chess`: `python-chess` 庫，用於處理西洋棋邏輯。
    *   `numpy`: 數值運算庫，MCTS 中用於狄利克雷噪聲。

## 使用方式

### 1. 訓練 AI

執行 `train.py` 腳本來啟動 AI 的自對弈和訓練過程。這會生成一個 `chess_mcts_net.pt` 檔案，其中包含訓練好的模型權重。

```bash
python3 train.py
```

*   **注意**：訓練過程可能非常耗時，尤其是在 CPU 上運行時。您可以修改 `train.py` 中的參數（例如 `n_iterations`, `games_per_iter`, `mcts_simulations`）來調整訓練時間和強度。
*   如果您有 GPU，請確保 PyTorch 已正確配置以使用 GPU，這將大幅加速訓練。

### 2. 與 AI 對弈

執行 `play.py` 腳本來與訓練好的 AI 進行對弈。您將執白棋，AI 執黑棋。

```bash
python3 play.py
```

*   **輸入走法**：請使用標準代數記譜法 (SAN) 輸入您的走法，例如 `e4`, `Nf3`, `O-O` (短易位)。
*   **棋盤顯示**：棋盤會以美觀的 Unicode 符號顯示。

## 未來可能的改進

*   **MCTS 批次化**：優化 MCTS 擴展階段，將多個葉節點的網路評估打包成批次處理，提高 GPU 利用率。
*   **更複雜的棋盤特徵**：加入重複次數、50 步規則計數、王車易位權限等更多棋盤特徵作為 CNN 輸入。
*   **UCI 協議支援**：實現 Universal Chess Interface (UCI) 協議，讓 AI 能夠連接到標準的西洋棋 GUI 或在線平台。
*   **模型評估**：建立自動化的模型評估流程，例如讓不同版本的 AI 互相對弈，以追蹤棋力進步。
*   **超參數調優**：對網路結構、MCTS 參數、訓練參數等進行系統性調優。
*   **更強大的網路**：增加 ResNet 區塊數量、過濾器數量，或嘗試其他先進的 CNN 架構。

## 授權

[請在此處填寫您的授權資訊，例如 MIT License]