# 2026/11/18 資料更新紀錄

本次只使用 GH Archive 官方小時檔與老師原始 `scripts/preprocess.py`，沒有下載其他資料，也沒有覆寫既有 LightGBM、GNN 模型或舊版 Parquet。

## 實際執行

- 原有小時檔：2026/09/01 00:00–11/16 15:00 UTC，共 1,840 個。
- 執行 `python scripts/download_gharchive.py --start 2026-11-16 --end 2026-11-17 --out raw_data --workers 8`；16 個既有檔跳過，**32 個新檔下載成功、0 個失敗**，新增 **118,990,605 bytes**。
- 更新後原始檔：2026/09/01 00:00–11/17 23:00 UTC，共 **1,872 個連續小時檔**，沒有缺少的中間小時；32 個新檔全部通過 gzip 完整讀取，且沒有小於 1 MB 的新檔。
- 執行 `.venv/bin/python scripts/preprocess.py --raw raw_data --out data/through_2026-11-17`，老師腳本讀取到 **187,166 筆 cohort 事件**。舊版 `data/extended/db/` 保留。

| 表 | 舊版列數 | 新版列數 | 增量 |
| --- | ---: | ---: | ---: |
| Users | 10,380 | 10,408 | 28 |
| Repositories | 20,752 | 20,798 | 46 |
| PullRequests | 43,787 | 44,091 | 304 |
| Issues | 28,383 | 28,524 | 141 |
| IssueComments | 15,823 | 15,892 | 69 |
| PRReviews | 13,057 | 13,147 | 90 |
| PRReviewComments | 10,176 | 10,188 | 12 |
| Pushes | 89,707 | 89,730 | 23 |

最新版仍只有 **4,315／4,465** 位正式測試者在 Users 表有紀錄，另 150 位需保留冷啟動預設。新版逐日檢查在 `quality_study_2026-11-17/`：11/17 UTC 八表可見列中，open_pr 82、open_issue 66、issue_comment 48、pr_review 67、pr_review_comment 7、push **22**。push 並非整天零列，但仍比九月明顯稀疏；不能把下載齊全等同於六類事件完整。

本次**尚未用新版資料重訓模型或更新預測 CSV**。原有模型與分數仍對應較早的 `data/extended/db/`。正式預測起點是 2026/11/21 16:00 UTC，之後仍應補齊起點前可取得的合法資料，重新跑老師預處理與品質檢查。
