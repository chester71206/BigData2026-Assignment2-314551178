# GNN 與 LightGBM 比較：實際執行結果

本實驗於 2026/11/17 以現有 `data/extended/db/` 八張老師預處理表執行，沒有重新下載 GH Archive，也沒有修改老師原始程式。原始資料範圍仍為 2026/09/01 00:00–11/16 15:00 UTC。所有分數來自 `scripts/evaluate.py` 對 4,465 位正式測試使用者的歷史週標籤評估；正式測試週尚未開始。

## 圖資料與防洩漏

每個 prediction_time 只保留 row time 早於起點的表列。`Users`、`Repositories`、`PullRequests`、`Issues`、`PRReviews` 使用 `first_seen_at`；`IssueComments`、`PRReviewComments`、`Pushes` 使用 `created_at`。從老師 `preprocess.TABLES` 的 FK 定義建立 **8 種節點、19 種安全 FK**，訊息傳遞時各加一條反向關係，共 **38 種有向 edge type**。`PullRequests.merged_by_id` 是事後才得知的關係，故排除；repo owner login、head repo code 等不是 FK，未偽裝成 FK 邊。跨表 FK 的兩端都必須在起點前可見。節點不使用 `merged_at`、`closed_at` 等事後狀態。

現有 9,000 位 cohort 中，起點前沒有 `Users` 列的人以無邊查詢節點存在，特徵全 0，不會連到未來才出現的實體。User 使用既有 18 維累積特徵；B/C 再加既有 44 維 7／14／30 天、recency 等 Temporal 特徵。其他節點只用節點種類與其列在起點前已存在的時間長度。固定的 signed `log1p` 數值轉換不從驗證週估計統計量。檢查腳本刪除 10/17 起點後才可見的所有列後重建圖，**全部節點特徵、查詢索引與 38 種邊逐值不變**，見 `check_graph_invariance.py`。

| 快照起點 UTC | 節點 | 原方向 FK 邊 | 全節點孤立比例 | 可觀測測試者 |
| --- | ---: | ---: | ---: | ---: |
| 9/5 16:00 | 28,893 | 38,155 | 25.82% | 1,662／4,465 |
| 9/26 16:00 | 132,015 | 236,941 | 3.61% | 3,836／4,465 |
| 10/10 16:00 | 185,125 | 334,882 | 2.26% | 4,181／4,465 |
| 10/17 16:00 | 202,528 | 366,521 | 1.98% | 4,250／4,465 |

完整七個時間點、每種節點與每種邊的數量見 [graph_statistics.csv](metrics/graph_statistics.csv)、[graph_nodes.csv](metrics/graph_nodes.csv)、[graph_relations.csv](metrics/graph_relations.csv)。全節點孤立比例較低，部分原因是事件節點都有 repo／actor FK；**使用者覆蓋率**更能表達早期圖的稀疏問題。10/17 時 36,057 個可見 PR 中，只有 **14,979 個（41.54%）**有可連至當時可見 `Users` 的作者 FK；近期 reviewer–PR 作者關係更稀疏，見既有 `relational_study_summary.md`。

## 模型與時間切分

- A：兩層標準同質 `GCNConv`，使用 User 累積特徵、節點類型與節點年齡。
- B：同樣兩層 GCN，另加入 User Temporal 特徵。
- C：兩層 `HeteroConv` + `SAGEConv`，逐節點／邊類型傳遞訊息，加入 Temporal 特徵。

三者均為 hidden dimension 32、dropout 0.25、Adam learning rate 0.003、weight decay 0.0001、最多 24 epochs、patience 5；seed 42。現有 `.venv` 的 PyTorch 2.7.0 是 CPU build（`torch.cuda.is_available() == False`）；CPU 4 threads 可在每折約 5–27 秒內訓練完成，故本次未另裝 CUDA build。早停使用 threshold 週 BCE loss；各活動 threshold 也只在同一較早週調整，**不使用獨立驗證週挑模型或 threshold**。

| 折次 | 訓練週起點 UTC | 早停／threshold 週 | 獨立驗證週 | Push 處理 |
| --- | --- | --- | --- | --- |
| sep26_six | 9/5、9/12 16:00 | 9/19 16:00 | 9/26 16:00 | 六類可評估 |
| oct10_five | 9/12、9/19、9/26 16:00 | 10/3 16:00 | 10/10 16:00 | 不訓練／評估 push |
| oct17_five | 9/19、9/26、10/3 16:00 | 10/10 16:00 | 10/17 16:00 | 不訓練／評估 push |

同一折只要訓練、早停或驗證週出現整天零 Pushes，就完全排除該折 push loss 與評分。五類比較在全部三折一致。九月較早的訓練週缺完整 30 天歷史，且五類也可能有部分漏記；分數是八表**可觀測標籤**的代理評估。

## 三週比較

下表為老師評分公式 `(Macro-F1 + Micro-F1) / 2`，統一**排除 push 的五類 Score**，單位百分比。LightGBM 使用既有同折實驗結果；不從 GNN 驗證週重新挑 LightGBM。

| 模型 | 9/26 | 10/10 | 10/17 |
| --- | ---: | ---: | ---: |
| GNN A：GCN | 36.98 | 34.08 | 34.45 |
| GNN B：GCN + Temporal | 39.31 | 36.25 | 35.05 |
| GNN C：HeteroSAGE + Temporal | **40.77** | **37.55** | **35.71** |
| LightGBM B：User | **45.24** | 39.76 | 38.97 |
| LightGBM C：User + Temporal | 37.57 | **41.12** | 40.30 |
| LightGBM F：User + Temporal + 關聯 | 38.09 | 40.70 | **41.44** |

9/26 另有六類 Score：GCN **49.61%**、GCN + Temporal **51.73%**、HeteroSAGE **53.09%**；既有最佳 LightGBM B 為 **56.23%**。每折 Macro-F1、Micro-F1、每活動 F1、早停 epoch 與推論時間見 [gnn_fold_metrics.csv](metrics/gnn_fold_metrics.csv)、[gnn_per_activity.csv](metrics/gnn_per_activity.csv)、[gnn_vs_lightgbm.csv](metrics/gnn_vs_lightgbm.csv)及各折 `logs/*_evaluate.log`。[跨模型圖](figures/gnn_vs_lightgbm.png)、[架構圖](figures/architecture.svg)可供報告使用。

**回答 1–3：**GCN A 三週都低於 LightGBM User，也略低於同週 Copy Last Week。HeteroSAGE 雖在 9/26 超過 LightGBM 的 C 與 F 兩個特徵組，仍低於該週最佳 LightGBM B，且後兩週也低於最佳 LightGBM。Temporal 使 GCN B 相對 A 增加 **2.34、2.17、0.61 個百分點**。異質 C 相對 B 再增加 **1.46、1.30、0.66 個百分點**。這些是單一 seed 的逐折結果，不能推論所有設定都有效。

## 關係重要性、稀疏與穩定性

將 seed 42 的 C 模型在各獨立週**推論時**逐一遮蔽一種 FK 及其反向邊，保持 checkpoint 與 threshold 不變。三週平均五類 Score 降幅最大的邊為 `PRReviewComments.author_id` **12.18**、`PRReviews.reviewer_id` **10.54**、`IssueComments.author_id` **8.50**、`PullRequests.author_id` **7.74** 個百分點。repo 邊的降幅普遍較小或為負。這表示已訓練模型強烈依賴活動→使用者的直接邊；**推論遮蔽造成分布轉移，不等於重新訓練的消融或因果重要度**。逐週、逐 FK 結果見 [relation_ablation.csv](metrics/relation_ablation.csv)與[關係遮蔽圖](figures/relation_masking.png)。PR 作者邊雖重要，但有大量缺值，限制可傳遞到其他節點的互動資訊。

另外使用 seed **7、42、123** 重訓 C，固定架構、資料折次與 threshold 規則。五類 Score 的平均±樣本標準差為 9/26 **39.74±1.60**、10/10 **34.69±3.10**、10/17 **35.40±2.52** 個百分點；三個 seed 在三週均低於同週最佳 LightGBM。10/10 的範圍 **31.40–37.55%**，顯示 GNN 在稀疏圖和少量正例下有明顯訓練不穩定性。全部九次結果見 [seed_stability.csv](metrics/seed_stability.csv)。

## 訓練成本與推論

同機 CPU 4 threads、特徵／圖已載入後，seed 42 三折平均如下。GNN 的 peak RSS 包含七個圖快照；LightGBM 的 RSS 包含預先建立的七組表格特徵，兩者皆是**整個 Python process**，不是模型本體記憶體。LightGBM 每活動以兩組候選參數訓練一個 binary classifier；GNN 一個模型同時輸出六維。推論對同一批 4,465 位使用者計時，包含 GNN 全圖 forward。

| 方法 | 平均訓練秒／折 | 平均推論秒／折 | 最高 RSS 約 |
| --- | ---: | ---: | ---: |
| GCN A | 18.51 | 0.183 | 2.46 GB |
| GCN B | 16.22 | 0.149 | 2.39 GB |
| HeteroSAGE C | 17.87 | 0.264 | 2.51 GB |
| LightGBM User | 2.31 | 0.027 | 0.77 GB |
| LightGBM User + Temporal | 3.31 | 0.032 | 0.77 GB |
| LightGBM 完整關聯 | 3.83 | 0.040 | 0.77 GB |

實測原值見 [GNN metrics](metrics/gnn_fold_metrics.csv)與[LightGBM benchmark](metrics/lightgbm_benchmark.csv)。圖建立時間、資料載入時間和開啟 Python 的時間未算入；不同 process 的 peak RSS 定義相同，但不能把表中秒數當作端到端服務延遲。以目前資料規模與硬體，GNN 並未換得更高分，還需要較多計算與記憶體。

## 可重現產物與限制

- [Graph Dataset 程式](graph_dataset.py)、[訓練程式](train_gnn.py)、[checkpoint 推論程式](predict_gnn.py)、[圖防洩漏檢查](check_graph_invariance.py)、[關係遮蔽](relation_ablation.py)、[LightGBM 成本對照](benchmark_lightgbm.py)。執行指令見 [README.md](README.md)。
- `snapshots/*.pt` 為七個歷史圖快照；`models/*.pt` 為九個主實驗 checkpoint，額外兩個 seed 的 checkpoint 位於 `stability_seed7/`、`stability_seed123/`；`predictions/*.csv` 為逐折預測和答案；`logs/` 保存 epoch loss、threshold 與老師評分輸出。
- seed 42 的三折 C 預測均已從 checkpoint 重新產生，CSV 與原始預測逐位元組一致。沒有產生 GNN 正式週提交檔；現有 LightGBM 暫定預測未改寫。
- 關係遮蔽不能識別因果貢獻；三個 seed 也不足以估計極端尾部分布。圖中 PR 作者缺值、早期未觀測使用者與近期 GH Archive 活動下降仍是主要限制。若有更完整的合法歷史資料，可重跑同樣折次與品質稽核，再嘗試簡單的 relation dropout 或更長訓練，但不能用獨立驗證週挑閾值。
