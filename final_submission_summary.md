# Assignment 2 提交版本紀錄（314551178）

## 資料與時間

- 資料來源：[GH Archive](https://www.gharchive.org/) 每小時原始檔，2026/09/01 00:00–11/21 02:00 UTC，共 1,947 個連續小時。僅使用作業允許的事件及老師原始 `scripts/preprocess.py` 產生的八張 Parquet 表；沒有使用 GitHub API。
- 預測特徵截止於 **2026/11/21 03:00 UTC**（台灣時間 11:00）。正式測試起點為 11/21 16:00 UTC；因此本檔是繳交期限前以已取得資料產生的 provisional prediction，沒有正式測試分數。
- 五類非 Push 活動使用起點為 9/5、9/12、9/19、9/26、10/3、10/10、10/17、10/24、10/31、11/7 16:00 UTC 的十個完整標籤週。11/7 週調整樹輪數與各類 threshold，然後十週全部重訓。Push 僅使用前四個沒有整天零紀錄斷層的標籤週。
- 逐日資料品質稽核見 `final_submission/quality/`。完整原始小時檔不表示所有活動類型完整；Push 有 15 個完整 UTC 日零列，近期 Push 預測的外推風險較大。

## 模型選擇與結果

四折 chronological holdout 起點為 9/26、10/10、10/17、10/31 16:00 UTC。每折用更早一週選設定與 threshold，以獨立的較晚週評分。以下是老師 `scripts/evaluate.py` 的五類 Score（Push 因品質問題排除）；各折數值和每類 F1 在 `final_submission/metrics/`，評分原始紀錄在 `final_submission/logs/`。

| 特徵組 | 9/26 | 10/10 | 10/17 | 10/31 | 四折平均 |
| --- | ---: | ---: | ---: | ---: | ---: |
| Copy last week | 37.82% | 35.32% | 34.60% | 34.48% | 35.56% |
| User history | 45.24% | 39.41% | 38.59% | 40.57% | **40.95%** |
| User + Temporal | 37.57% | 39.98% | 40.94% | 41.27% | 39.94% |
| User + Temporal + original relations | 37.75% | 40.50% | 41.80% | 40.48% | 40.13% |
| User + Temporal + repo volume | 38.00% | 40.96% | 41.13% | 40.76% | 40.21% |
| User + Temporal + interactions | 38.00% | 39.47% | 39.94% | 40.21% | 39.41% |
| User + Temporal + repo + interactions | 38.09% | 40.74% | 40.09% | 41.98% | 40.23% |

四折等權平均選出 **User history**；Push 的兩個安全週回測也選到 User history。最終是六個獨立 LightGBM 二元分類器，各自以調參週決定 threshold 和最佳 rounds，再用所有合格週重訓。較複雜特徵在部分週有效，但沒有提升四折平均；詳細關聯可觀測性和時間消融見 `relational_study_summary.md`、`report/temporal_ablation_summary.md`。

## 提交檔與重現

- `submission/output_314551178.csv`：4,465 人、7 欄，老師 `scripts/evaluate.py --pred ... --input ...` 回報 `Format check: no errors (4465 users)`。六類預測正例依序為 196、109、167、109、21、3,428。
- 146 位測試者在預測截止前沒有可見的 Users 列，保留於 CSV，六類採全 0 預設。這不是 `test_input.csv` 漏人。
- `scripts/predict_final_lightgbm.py` 由保存的 checkpoint 重建 `final_submission/predictions/reproduced_prediction.csv`，已用 `cmp` 確認與訓練時寫出的預測檔位元組完全相同。
- 資料、品質稽核、訓練和推論的完整指令見根目錄 `README.md`；模型設定在 `final_submission/metrics/final_config.json`，最佳模型在 `final_submission/models/`，訓練紀錄在 `final_submission/logs/`。原始 GH Archive 和 Parquet 因體積較大不提交到 GitHub，可由腳本重建。
- 英文 16 頁報告：`report/Asg2_314551178.pdf`；可編輯來源、圖和中文講稿在 `report/`。

## 尚未完成與限制

正式測試週尚未開始，無法計算正式 Score。Push 在 9–11 月有系統性觀測缺口，模型只用早期安全週，近期預測需審慎解讀。講解影片 `Asg2_314551178.mp4` 必須由學生本人錄製；CSV、PDF、MP4 仍須分別上傳 E3。改善方向包括在截止前取得更晚且完整的原始檔後重建八表和預測，以及在可觀測關係較充足的資料區間評估更細的圖與時間特徵。
