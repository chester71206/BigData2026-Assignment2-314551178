# Assignment 2 LightGBM 實驗摘要

以下是 **2026/11/17 實際執行**的結果。時間皆為 UTC，台灣時間加 8 小時；正式測試週尚未開始，沒有正式測試分數。

**後續稽核修正：**發現下述 10/10 驗證週含整天零 Pushes 的斷層，原六類 51.66% 只能作歷史紀錄，不能當可靠的六類效果結論。新的 [資料品質稽核](data_quality_report.md) 與 [三個時間週的關聯特徵比較](relational_study_summary.md) 已排除受影響週的 push，並新增 9/26 六類驗證。請以這兩份後續報告解讀模型效果。

## 資料與環境

- GH Archive 原始小時檔：**2026/09/01 00:00–11/16 15:00 UTC**，連續 1,840 個檔案，共 29,779,439,958 bytes（約 27.7 GiB）。執行時 11/16 16–23 UTC 尚回傳 404，沒有把未取得時段當成零活動。
- 老師的 scripts/preprocess.py 未修改，輸出在 data/extended/db/；原 data/db/ 保留。八表列數依序為 Users 10,380、Repositories 20,752、PullRequests 43,787、Issues 28,383、IssueComments 15,823、PRReviews 13,057、PRReviewComments 10,176、Pushes 89,707；預處理讀取 186,679 筆 cohort 事件。
- Users 表包含 5,316 位 cohort 使用者，正式測試名單可對應 4,315／4,465 人。缺席者仍保留特徵與預測列。
- Python 3.13、LightGBM 4.7.0；最多 4 個 LightGBM 執行緒。模型只讀老師產生的八表。

## 時間切分與模型

每個樣本是 (UserID, prediction_time)，起點 UTC 週六 16:00，對應台灣週日 00:00；標籤是後續七天的觀測活動。

| 用途 | 週起點（UTC） |
| --- | --- |
| 訓練 | 9/12 16:00、9/19 16:00、9/26 16:00 |
| Early stopping、兩組參數選擇、六個 threshold | 10/03 16:00 |
| 獨立時間驗證 | 10/10 16:00；標籤至 10/17 16:00 |
| 暫定預測 | 11/16 16:00（台灣 11/17 00:00） |

每種活動各訓練一個 LightGBM binary classifier。兩組候選配置的樹葉數為 15 或 31，最多 250 輪；以 threshold 週的 binary logloss 選配置，25 輪無改善就 early stop。六個 threshold 在同一週按 F1 選出，於**不同**的驗證週評估。訓練使用 9,000 位 cohort 使用者的三個歷史週；驗證只看正式名單的 4,465 人。選定最佳組後，以所有已知歷史標籤重新訓練，沿用先前選好的參數、樹數與 threshold 產生暫定預測；重訓後模型的正式測試表現未知。

B 特徵包含活動總數、比例、活躍天數與參與 repo 數。C 加入 7／14／30 天次數、最近活動距離、前後週變化與頻率。D 加入透過八表 FK 建立的 repo 鄰居活動量、PR 收到 review／程式碼留言、issue／PR 討論留言、指派 issue 等聚合。模型不使用 masked 文字、最新 login／repo 名稱，也不使用無法安全還原歷史狀態的 merged_at、merged_by_id、closed_at、state_reason、dismissed_at。

## 初版四組消融（10/10；push 不可靠）

以下由老師 scripts/evaluate.py 對 **10/10 16:00–10/17 16:00 的八表可觀測標籤**評估，單位為百分比。

| 實驗 | Macro-F1 | Micro-F1 | Score |
| --- | ---: | ---: | ---: |
| A Copy Last Week | 40.43 | 53.38 | 46.91 |
| B LightGBM + User | 44.22 | 56.58 | 50.40 |
| **C LightGBM + User + Temporal** | **45.58** | **57.74** | **51.66** |
| D LightGBM + User + Temporal + Relational | 44.56 | 56.73 | 50.65 |

數值見 [ablation.csv](artifacts/metrics/ablation.csv)、[per_activity.csv](artifacts/metrics/per_activity.csv)，圖見 [ablation_comparison.png](artifacts/figures/ablation_comparison.png)。

1. B 比 Copy Last Week 高 **3.49 個百分點**；最佳 C 高 **4.75 個百分點**。
2. C 比 B 高 **1.26 個百分點**，此週的 Temporal 特徵有幫助。
3. D 比 C 低 **1.02 個百分點**，關聯特徵在主驗證週沒有改善。較晚、資料更稀疏的 11/7 驗證週 D 反比 C 高 0.86 個百分點，但不適合作可靠結論。
4. D 六模型 gain 加總中，時間特徵約占 **72.5%**、使用者特徵 **25.8%**、關聯特徵 **1.8%**。最佳 C 的前幾名是最近活動距離、issue／review 留言距離及使用者活動頻率。gain 不是因果影響，相關特徵會分攤重要度。見 [最佳模型重要度圖](artifacts/figures/best_feature_importance.png)、[完整關聯模型重要度圖](artifacts/figures/feature_importance.png) 與 [CSV](artifacts/metrics/feature_importance.csv)。
5. C 的 issue_comment F1 **39.04%** 最低；pr_review **39.79%**、pr_review_comment **40.25%** 也較難。原始 push F1 **68.38%** 受零紀錄日影響，不宜解讀。pr_review_comment 在驗證週僅 **186／4,465** 人為正例，類別不平衡明顯。
6. 最早訓練週距原始資料起點只有 11 天，30 天視窗尚未覆蓋完整 30 天；模型有可觀測天數特徵。D 加特徵後在獨立週下降，可能是關聯訊號弱、樣本少或擬合，單一驗證週不能證明原因。

## 缺漏與解讀限制

全部 1,840 個小時檔皆存在且沒有低於 1 MB 的檔案，但**檔案存在不代表事件種類完整**。每週原始壓縮量由 9/5 起約 **7.05 GB** 降到 11/7 起約 **0.70 GB**。push 觀測正例由 9/26 週的 **2,718／9,000** 人降至 10/31 週 **283** 人、11/7 週 **277** 人。這與作業 PDF 對近期 GH Archive 缺漏的警告一致；不能把所有未觀測活動解釋成真實負例，其他活動也有下降。證據見 [activity_coverage.csv](artifacts/metrics/activity_coverage.csv) 與 [activity_coverage.png](artifacts/figures/activity_coverage.png)。

後續逐日檢查確認，**10/10 週本身也含 push 斷層日**；上表六類數值不應用來選定最終方法。跨週研究改以 9/26 做相對較可用的六類驗證，10/10 與 10/17 用老師的 --skip push 僅計五類。所有 F1 仍是八表可觀測標籤的代理評估，不能宣稱是完整 GitHub 活動的真實 F1。較晚 11/7 週的探索性結果保存在 artifacts_recent/：A **33.76%**、B **38.54%**、C **39.89%**、D **40.75%**；其 push 更不可靠。無法從八張表識別每個使用者的未觀測事件，因此不能安全補正所有標籤。

## 資料洩漏檢查

- 每個特徵同時要求 event_at 及該表 row time 都早於 prediction_time。PR、issue、review 以 first_seen_at 判斷列可見時間，沒有只看較早的 created_at／submitted_at。三張即時事件表使用 created_at。
- 在 10/10 16:00 驗證起點，有 **692 筆**活動時間較早、但資料列到起點後才可見的事件，已被排除於特徵。
- 執行 scripts/check_temporal_invariance.py：分別刪除八表在 **10/10 16:00**、**11/7 16:00** 以後才出現的所有列，18 個使用者、44 個時間、16 個關聯特徵在兩次檢查中**完全不變**。
- 關聯只 join 當時可見的 PR、issue、repository 與 review；晚得知的狀態欄位不用。訓練、threshold、驗證週依時間先後切分，沒有 random split。
- 七天標籤要求活動時間落在該週且資料列於週末前可見。這個保守規則可能漏掉事後才進入八表的真實活動，列為未解限制。

## 成果與未完成事項

- 程式：scripts/activity_features.py、scripts/run_lightgbm.py、scripts/predict_from_checkpoints.py、scripts/audit_activity_coverage.py、scripts/check_temporal_invariance.py、scripts/download_gharchive.py、scripts/make_diagrams.py；Python 依賴列於 requirements.txt，圖形產生另需 Graphviz dot。老師兩支腳本未修改。
- 主實驗的 artifacts/metrics/ 含四組 CSV、每活動 F1、threshold 與覆蓋率；artifacts/logs/ 含訓練與老師評估輸出；artifacts/models/ 含驗證 checkpoint 與最佳方法的六個重訓 checkpoint；artifacts/figures/ 含消融、重要度、覆蓋率圖，以及從老師 TABLES 定義產生、含八表全部 20 條 FK 邊的 [schema graph](artifacts/figures/schema_graph.svg) 與 [方法框架圖](artifacts/figures/framework.svg)。
- [暫定預測](artifacts/predictions/provisional_prediction.csv) 有 **4,465 列**，通過老師 evaluate.py 的格式檢查。從六個最佳 checkpoint 與 threshold 重新載入產生的 reproduced_prediction.csv 與原檔逐位元組一致。它只用 **11/16 16:00 UTC** 前的資料，**不是** 11/22 台灣時間正式提交檔。近期資料稀疏，須謹慎解讀。
- **尚未完成：**補到截止前能取得的最新原始資料；重跑老師預處理與 pipeline；重新稽核覆蓋率並檢查 4,465 列格式；依學號輸出 output_{studentID}.csv。正式測試答案尚不存在，不能計算正式分數。英文簡報與本人錄影也需另行完成。
- **後續工作：**已另做逐日資料稽核、三個時間週消融與 reviewer–repo–PR 互動統計，見前述兩份新報告。仍可補更早資料以完整覆蓋首週 30 天歷史，並改善缺失偵測；這些尚未實驗，不能視為已證實的提升。
