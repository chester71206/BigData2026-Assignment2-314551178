# 跨週驗證、關聯特徵與缺席使用者

本報告於 2026/11/17 使用已存在的八張老師預處理表實際執行。細節、模型與所有老師 evaluate.py 輸出在 [relational_study](relational_study)；資料品質判定見 [data_quality_report.md](data_quality_report.md)。

## 研究設計

比較六組模型：A Copy Last Week、B 累積 User 特徵、C 加 Temporal 特徵、R 在 C 上只加 repo 活躍度、I 在 C 上只加實際互動特徵、F 在 C 上同時加 R 與 I。每活動各自訓練 LightGBM binary classifier；每折訓練週、threshold 週、驗證週按時間排列。每次仍只用預測起點前可見的列；早停及 threshold 用調參週，最終數字由較晚的驗證週與老師 evaluate.py 產生。沒有 random split。

| 折次 | 訓練週起點（UTC） | Threshold 週 | 驗證週 | 可評活動 |
| --- | --- | --- | --- | --- |
| sep26_six | 9/5、9/12 的 16:00 | 9/19 16:00 | 9/26 16:00 | 六類 |
| oct10_five | 9/12、9/19、9/26 的 16:00 | 10/3 16:00 | 10/10 16:00 | 排除 push 的五類 |
| oct17_five | 9/19、9/26、10/3 的 16:00 | 10/10 16:00 | 10/17 16:00 | 排除 push 的五類 |

9/26 驗證週的六類均無整天零紀錄斷層，是現有資料中**相對較可用**的完整六類週，但無法證明源資料百分之百完整。後兩折的訓練、調參或驗證視窗含整天零 Pushes，故這兩折的 push 不作負樣本、不訓練、不計分。三折的第一個訓練週都缺完整 30 天歷史；這是固定資料起點的限制。

## 實際消融結果

為讓三折可比較，下表統一顯示**五類 Score**（排除 push），單位百分比。9/26 的六類 Score 另列於下方。

| 特徵組 | 9/26 | 10/10 | 10/17 |
| --- | ---: | ---: | ---: |
| A Copy | 37.82 | 35.32 | 34.60 |
| B User | **45.24** | 39.76 | 38.97 |
| C User + Temporal | 37.57 | **41.12** | 40.30 |
| R C + repo 活躍度 | 38.00 | 40.51 | 40.26 |
| I C + reviewer/PR 互動 | 38.00 | 39.46 | 40.47 |
| F C + 兩種關聯 | 38.09 | 40.70 | **41.44** |

9/26 六類 Score：A **49.97%**、B **56.23%**、C **51.77%**、R **52.81%**、I **52.11%**、F **52.87%**。每折 Macro-F1、Micro-F1、六類個別 F1 及使用的 threshold 分別見 [fold_metrics.csv](relational_study/metrics/fold_metrics.csv)、[per_activity.csv](relational_study/metrics/per_activity.csv)、[tuning.csv](relational_study/metrics/tuning.csv)。[跨週比較圖](relational_study/figures/fold_comparison.png)使用相同五類，避免把五類與六類 Score 混在同一尺度。

**Temporal 的效果並不穩定：**C 相對 B 在 9/26 的五類 Score 低 **7.67 個百分點**，在 10/10、10/17 則分別高 **1.36**、**1.33** 個百分點。9/26 是較早驗證週，訓練歷史較短且資料分布不同；不能用 10/10 單週宣稱時間特徵一定有效。

**repo 活躍度本身未有穩定幫助：**R 相對 C 於 9/26 +0.43、10/10 −0.61、10/17 −0.04 個百分點。新 reviewer–repo–PR 互動特徵 I 的三週變化為 +0.43、−1.66、+0.17。合併 R+I 的 F 為 +0.52、−0.42、+1.14。最後一週的改善有趣，但不能視為確定提升。

以 4,465 位測試使用者為配對單位、固定隨機種子做 **1,000 次 bootstrap**，F 相對 C 在 10/17 的五類 Score 差是 **+1.14 個百分點**，95% 區間 **−0.14 到 +2.44**；10/10 是 −0.42，區間 −1.52 到 +0.75。兩個區間都跨 0。I 在 10/10 的下降 −1.66，區間 −3.11 到 −0.21。詳見 [bootstrap_differences.csv](relational_study/metrics/bootstrap_differences.csv)。這些是使用者抽樣不確定性，不包含 GH Archive 系統性缺漏風險。

## 真正可觀測到多少關聯？

新 I 特徵包括 reviewer 在同一 repo 反覆 review 不同 PR、repo 中其他 reviewer 的 review 數、同一 PR 的共同 reviewer、reviewer 對同一 PR 的程式碼留言及討論留言、reviewer–PR 作者重複互動等。所有 join 都要求相關 PR、issue、review 已在截止時間前可見。為避免把原有活動次數換個名稱當關聯，I 組沒有加入單純的 review 總次數。

| 截止時間（UTC） | 近 30 天 review | PR 作者已知比例 | reviewer–repo 配對 | 同 repo ≥2 PR 的配對 | ≥2 reviewer 的 PR |
| --- | ---: | ---: | ---: | ---: | ---: |
| 9/26 16:00 | 9,054 | 28.6% | 2,124 | 867 | 38 |
| 10/10 16:00 | 6,799 | 19.2% | 1,980 | 773 | 22 |
| 10/17 16:00 | 3,022 | 7.1% | 1,284 | 444 | 6 |
| 11/16 16:00 | 1,895 | **2.2%** | 941 | 260 | **2** |

10/10 時，repo-only 特徵對 **4,033／4,465** 位測試者有非零值；I 特徵只對 **1,172** 位有值，其中同 PR 多 reviewer 只涉及 **21** 位。11/16 時 I 有值者降為 **630** 位，同 PR 多 reviewer 只剩 **1** 位。因此 reviewer–repo 重複活動仍有可觀測訊號，但 PR 作者與共同 reviewer 關係相當稀疏；「使用者建立 PR 後由誰 review」不能作主要特徵來源。見 [relation_observability.csv](relational_study/metrics/relation_observability.csv)、[feature_observability.csv](relational_study/metrics/feature_observability.csv)。

在 10/17 折的 F 模型 gain 加總中，新 I 特徵占約 **3.8%**、repo-only 約 **1.9%**；新特徵以 reviewed PR 數與同 PR 程式碼留言較重要。gain 不代表因果貢獻，也不能取代上面的獨立週消融。見 [關聯特徵重要度圖](relational_study/figures/relation_feature_importance.png)。

## 150 位缺少歷史事件的測試者

正式 test_input.csv 仍有完整 **4,465** 人；其中 **150 人在截至 11/16 15:00 UTC 的八張表內沒有 Users 列**，不是 CSV 遺漏。pipeline 現在明確規定：預測起點前沒有 Users 列時，六類預測採 **全 0**。這避免 LightGBM 只憑「全缺失特徵」將同一群人任意判為正例；目前的暫定 CSV 本來就對這 150 人輸出全 0，因此檔案內容未變。

用歷史週「起點前還沒有 Users 列」的人模擬冷啟動：9/26 有 629 人、10/10 有 284 人、10/17 有 215 人。9/26 的六類可靠觀測比較中，對這群人統一給 0 的全體 F1 **每類都不低於**統一給 1；即使 push 有 **217／629** 位後續被觀測為正例，push 全體 F1 仍是 0 預設 **72.49%**、1 預設 **72.13%**。另外五類的冷啟動正例率較低。詳見 [cold_start_strategy.csv](relational_study/metrics/cold_start_strategy.csv)。這是**保守的預設**，會漏掉首次活躍的人；歷史未見者與目前長期未見的 150 人不完全相同，正式週效果未知。

## 結論與下一步

目前沒有足夠證據宣稱 Temporal 或任一關聯特徵在所有週穩定提升。關聯特徵的可用性受 PR 作者缺值、共同 reviewer 稀少及近期資料稀疏限制。可在繳交前用新資料重跑，但**不要**把含整天零 Pushes 的週當成完整 push 負樣本。主模型暫定預測仍是既有 C 模型；本次研究沒有用這些驗證週直接挑選或改寫正式測試結果。需待正式預測起點前補資料，並重做品質稽核與格式檢查。
