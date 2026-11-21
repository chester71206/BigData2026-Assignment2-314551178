# 最新可得資料的 LightGBM 重訓結果（2026/11/20）

本次確實執行下載、老師原始 `preprocess.py`、資料品質稽核、四個時間回測、最後完整週調參、全部合格標籤週重訓，並產生 4,465 人的預測。**正式預測起點 2026/11/21 16:00 UTC 尚未到來；目前 CSV 是截至 2026/11/20 06:00 UTC 的暫定候選，不能宣稱已有正式測試分數。**

## 資料與品質

- 原始 GH Archive：2026/09/01 00:00–11/20 05:00 UTC，共 **1,926 個連續小時檔**；本次新下載 54 個、238,791,401 bytes，全部通過 gzip 完整讀取，無小於 1 MB 的新檔。
- 八張老師 Parquet 在 `data/through_2026-11-20-05/db/`，原程式未修改；cohort 事件 188,224 筆。八表列數：Users 10,469、Repositories 20,928、PullRequests 44,685、Issues 28,884、IssueComments 16,011、PRReviews 13,337、PRReviewComments 10,215、Pushes 89,803。
- `final_refresh/quality/` 有逐日、逐月及每週資料品質 CSV 與圖。到 11/19 完整 UTC 日，push 有 15 個整天零列斷層。其他五類沒有整天零列，但近月活動量明顯下降，故五類歷史 F1 仍是八表可觀測標籤的代理值。
- `scripts/check_temporal_invariance.py` 在 11/7 16:00 UTC 檢查通過：刪除該時刻後才可見的列，全部 User、Temporal、Relational、repo-only、interaction 特徵不變。

## 用哪些週訓練、驗證？

一筆樣本是 `(UserID, prediction_time)`；每週對 9,000 位 cohort 使用者建立特徵，後續七天六類活動作標籤。所有輸入特徵只用起點前已可見的八表列；從未使用 random split。四個回測折次均以較早週訓練、下一週調 early stopping 和 threshold、再以更晚週獨立評估：

| 獨立評估週起點 UTC | 訓練週起點 | 調參週起點 | 評分活動 |
| --- | --- | --- | --- |
| 9/26 16:00 | 9/5、9/12 | 9/19 | 統一五類 |
| 10/10 16:00 | 9/5～9/26 四週 | 10/3 | 統一五類 |
| 10/17 16:00 | 9/5～10/3 五週 | 10/10 | 統一五類 |
| 10/31 16:00 | 9/5～10/17 七週 | 10/24 | 統一五類 |

每類各訓一個 LightGBM binary classifier。沿用原本兩組小範圍超參數候選、25 輪 early stopping、最多 250 輪、固定 seed 42 與四執行緒；以調參週 logloss 選配置與輪數，該週 F1 選各類 threshold。最後完整標籤週 **11/7 16:00～11/14 16:00 UTC** 專供最終調參，沒有參與四折方法選擇；最終五類模型再用 **9/5～11/7 共十個週**的全部已知標籤重訓，固定前一步選出的參數、輪數及 threshold。11/14 開始的一週在原始檔截止時尚未完整，沒有當訓練負例。

## 四折消融與方法選擇

下表是老師 `evaluate.py --skip push` 計算的五類 Score，單位百分比。方法選擇規則是**四折等權平均**；不以單一較晚週或最後調參週挑選。

| 方法 | 9/26 | 10/10 | 10/17 | 10/31 | 四折平均 |
| --- | ---: | ---: | ---: | ---: | ---: |
| Copy Last Week | 37.82 | 35.32 | 34.60 | 34.48 | 35.56 |
| **B User** | **45.24** | 39.41 | 38.59 | 40.57 | **40.95** |
| C User + Temporal | 37.57 | 39.98 | 40.94 | 41.27 | 39.94 |
| D 原始關聯特徵 | 37.75 | 40.50 | **41.80** | 40.48 | 40.13 |
| R repo 活動量 | 38.00 | **40.96** | 41.13 | 40.76 | 40.21 |
| I 實際互動 | 38.00 | 39.47 | 39.94 | 40.21 | 39.41 |
| F repo + 互動 | 38.09 | 40.74 | 40.09 | **41.98** | 40.23 |

因此最終五類選 **B User**。它的 9/26 表現明顯較好；較晚三折的 Temporal／Relational 組則較有競爭力，不能宣稱 B 在最近月份一定最好。每折 Macro-F1、Micro-F1、每類 F1、預測 CSV 與老師評估輸出見 `final_refresh/metrics/backtest_scores.csv`、`backtest_per_activity.csv`、`predictions/`、`logs/`；圖見 `final_refresh/figures/backtests.png`。

Push 單獨處理：只有 **9/5、9/12、9/19、9/26** 四個標籤週沒有整天零 Pushes；後期缺口週**完全不拿來訓練或調 push threshold**。9/19、9/26 兩個較早的獨立 push 回測比較 B、C、F，平均 F1 最高為 B。以 9/26 作最終早停與 threshold，再用四週全部標籤重訓。這個模型只代表早期較完整的 push regime，能否外推到 11 月尚無證據。

## 目前預測與限制

- [暫定候選 CSV](final_refresh/predictions/provisional_prediction.csv) 含 **4,465** 位正式名單使用者、欄位順序正確、沒有重複，通過老師 `evaluate.py --input` 格式檢查；相同內容也另存為 [submission/output_candidate_2026-11-20.csv](submission/output_candidate_2026-11-20.csv)。[checkpoint 重建 CSV](final_refresh/predictions/reproduced_prediction.csv) 與它逐位元組一致。正式繳交仍需以自己的學號命名 `output_{studentID}.csv`。
- 預測起點為 **2026/11/20 06:00 UTC**，只使用此前可取得的八表資料；**148** 位起點前沒有 Users 列的測試者六類均設 0。
- 六類預測正例數依序為 open_pr **195**、open_issue **108**、issue_comment **169**、pr_review **108**、pr_review_comment **21**、push **3,433**。Push 正例偏高：近七天實際八表只觀測到 394 位測試者有 push，且來源已知有系統性漏記。不能據此判定模型一定錯或一定對；若正式週 push 資料同樣缺漏，作業規則可能不評該類，但目前未知。請勿把這個 push 預測解讀成已驗證的 11 月效果。
- `final_refresh/metrics/final_model_details.csv` 記錄每活動特徵組、訓練週數、threshold、最佳輪數及正例數；`models/` 保存六個 checkpoint，`logs/` 保存訓練、預處理、品質與評分紀錄。舊版 LightGBM 和 GNN 成果都未覆寫。
- **正式測試週答案不存在**，所以沒有正式 Score。正式提交前須再次補到當時可取得的、早於 **2026/11/21 16:00 UTC** 的連續小時檔，重建新八表、稽核、重跑 pipeline。課程繳交截止若早於正式起點，應在截止前用最新合法連續時點產生提交檔。

## 重跑指令

在 repository 根目錄執行；使用新輸出資料夾保存這次結果：

```bash
python scripts/download_gharchive.py --start 2026-11-20 --end 2026-11-21 --end-hour 14 --out raw_data
.venv/bin/python scripts/preprocess.py --raw raw_data --out data/through_2026-11-21-14
.venv/bin/python scripts/audit_event_quality.py --db data/through_2026-11-21-14/db --raw raw_data --out final_refresh_2026-11-21/quality
.venv/bin/python scripts/train_final_lightgbm.py --db data/through_2026-11-21-14/db --raw raw_data --quality final_refresh_2026-11-21/quality/weekly_gap_flags.csv --out final_refresh_2026-11-21 --threads 4
.venv/bin/python scripts/predict_final_lightgbm.py --db data/through_2026-11-21-14/db --artifacts final_refresh_2026-11-21
.venv/bin/python scripts/evaluate.py --pred final_refresh_2026-11-21/predictions/provisional_prediction.csv --input submission/test_input.csv
```

這些命令只適用於指定的小時檔皆已發布且下載成功的情況；不能為了取得尚未發布的檔案越過正式預測起點，亦不能把缺檔時段當負樣本。
