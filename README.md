# BigData2026-Assignment2

## 中文導讀：這份作業在做什麼？

以下依 repo 中的修訂版作業 PDF、`SCHEMA.md` 與提供的腳本整理。這是**個人作業**，占學期成績 15%。你要對 `submission/test_input.csv` 的 **4,465 位 GitHub 使用者**，預測他們在 **2026/11/22 00:00 至 11/29 00:00（台灣時間）** 是否會做六種活動。每個人、每種活動各輸出 `0` 或 `1`：

| 欄位 | 活動 |
| --- | --- |
| `open_pr` | 開新的 pull request |
| `open_issue` | 開新的 issue（不含 PR） |
| `issue_comment` | 在 issue 或 PR 討論串留言 |
| `pr_review` | 提交 PR review |
| `pr_review_comment` | 在 PR 程式碼差異的特定行留言 |
| `push` | 將 commit push 到 branch 或 tag |

模型可自行設計；重點是合理使用**資料表之間的關係**與**時間資訊**。PDF 提到 GNN、relational deep learning 作為例子，沒有規定只能使用特定架構。

**截止與繳交：**2026/11/21 23:55 前，把完整、可重現的原始碼與執行說明放到**公開 GitHub repo**，所有 commit 也須在截止前完成。另上傳 `output_{studentID}.csv`、`Asg2_{studentID}.mp4`、`Asg2_{studentID}.pdf` 到 E3。逾期不收。

### 目前 repo 的狀態

| 路徑 | 用途 |
| --- | --- |
| `scripts/preprocess.py`、`scripts/cohort.txt` | 把 GH Archive 事件轉成匿名化的八張關聯表；cohort 含 9,000 位使用者 |
| `scripts/download_gharchive.py` | 依 UTC 日期範圍批次下載 GH Archive 每小時原始檔 |
| `scripts/evaluate.py` | 檢查 CSV 格式，有驗證答案時可算分 |
| `submission/test_input.csv` | 正式測試使用者名單 |
| `submission/sample_submission.csv` | **隨機預測**的格式範例，不是模型結果 |
| `SCHEMA.md` | 各表欄位、主鍵、外鍵與匿名化說明 |

**提交版本：學號 314551178；[公開 GitHub Repository](https://github.com/chester71206/BigData2026-Assignment2-314551178)。** `raw_data/` 現有 **2026/09/01 00:00–11/21 02:00（UTC）** 共 1,947 個連續小時檔，老師腳本產生的最新八表在 `data/final_2026-11-21-02/db/`。本次重訓產物在 `final_submission/`，提交用 CSV 為 [`submission/output_314551178.csv`](submission/output_314551178.csv)，英文簡報 PDF 為 [`report/Asg2_314551178.pdf`](report/Asg2_314551178.pdf)。由於課程繳交截止早於正式預測起點，此 CSV 使用當時最新可得資料，屬 provisional prediction；正式測試分數尚不可得。[11/20 訓練紀錄](final_refresh_summary.md)、[較早實驗](experiment_summary.md)、[資料品質報告](data_quality_report.md)及 GNN 實驗保留各自當時的資料範圍。`raw_data/`、`data/`、`.venv/` 不上傳 GitHub；錄影需本人完成，並將三個指定檔案分別上傳 E3。

### 建議流程

1. **準備資料。**老師腳本需 Python 3.9+；本次模型實驗使用 **Python 3.13** 與 `requirements.txt` 中的確切套件版本。已下載 2026/09/01 00:00–11/21 02:00 UTC 的連續小時檔；已存在的檔案會由下載器跳過。檔名日期與小時是 **UTC**。`preprocess.py` 會忽略 2025/12/01 00:00 UTC 以前的檔案。
2. **執行指定預處理。**PDF 規定必須先使用提供的腳本，且只可用其產生的八張表。舊版資料集須重新建立。執行 `.venv/bin/python scripts/preprocess.py --raw raw_data --out data/extended`，得到 `data/extended/db/*.parquet`；本次已完成。若下載更多資料後要保留本次結果，請換一個新的 `--out` 目錄。八表為 Users、Repositories、PullRequests、Issues、IssueComments、PRReviews、PRReviewComments、Pushes。`test_input.csv` 的 `UserID` 對應 `Users.github_user_id`，**不是** `Users.user_id`（後者是當次建表的行號）。
3. **做時間切分與特徵。**選數個歷史週作訓練／驗證；每個預測起點 `t` 只用當時已出現的列，後續七天活動作標籤。**不要隨機切分**。列的可見時間看 `SCHEMA.md` 的 *row time*：Users、Repositories、PullRequests、Issues、PRReviews 用 `first_seen_at`；IssueComments、PRReviewComments、Pushes 用 `created_at`。不可只因 PR、issue 的 `created_at` 或 review 的 `submitted_at` 早於 `t`，就使用實際在 `t` 後才出現的列。
4. **避免時間洩漏。**`PullRequests.merged_at`、`merged_by_id`、`closed_at`，`Issues.closed_at`、`state_reason`，`PRReviews.dismissed_at` 等欄位可能日後才知道；只有能確認在 `t` 前已知時才可用，否則排除。PDF 禁止蒐集額外資料，包括其他 GH Archive event 類型、GitHub API 或其他網站。
5. **建立模型並驗證。**可先試「前七天做過某活動就預測 1」的基線，再加入跨表關聯與時間特徵。自製驗證預測檔及答案檔（格式同正式輸出），用 `python scripts/evaluate.py --pred validation_pred.csv --truth validation_truth.csv` 計算每類 F1，以及 `Score = (macro-F1 + micro-F1) / 2`。驗證資料須有該週事件，並記錄 GH Archive 時段缺漏；若某類活動缺漏，可用 `--skip 活動欄位` 排除。正式測試若某類活動資料缺漏，該類不計分。
6. **產生與檢查正式預測。**以 **2026/11/22 00:00 台灣時間（2026/11/21 16:00 UTC）之前**可取得的八表資料，對所有測試使用者輸出一列。檔名為 `output_{studentID}.csv`（替換成自己的學號），欄名與順序必須完全是 `UserID,open_pr,open_issue,issue_comment,pr_review,pr_review_comment,push`。六個預測欄只能有 `0`、`1`，不得空白；使用者不得重複、缺漏或多出。執行 `python scripts/evaluate.py --pred output_{studentID}.csv --input submission/test_input.csv` 檢查格式。這個指令**不會計算正式測試分數**；格式錯誤會讓模型表現項目得 0 分。`sample_submission.csv` 只有隨機值，不應當成完成的預測交出。
7. **完成報告與繳交。**公開 repo 需包含讀資料、建特徵、訓練與預測的全部程式及明確執行指令；助教會以指定腳本準備資料並檢查結果能否重現。用課程 `Assignment2_report_template` 製作**英文、最多 16 頁**投影片，輸出 `Asg2_{studentID}.pdf`；錄製本人講解的 **16–20 分鐘**影片（中文／英文皆可），輸出 `Asg2_{studentID}.mp4`。PDF 指定可用 Webex 或 Google Meet 錄製；AI 生成影片得 0 分。簡報須列出公開 GitHub repo，四題都需在簡報與影片中說明。

| 報告題目 | 應說明內容 | 配分 |
| --- | --- | ---: |
| Q1 | 畫八表 schema graph，標明每表主鍵、外鍵，畫出每條外鍵指向所參照主鍵的邊；提供模型所用資料的時間範圍 | 1% |
| Q2 | 如何加強不同表之間關係的建模 | 3% |
| Q3 | 如何加強時間資訊的建模 | 5% |
| Q4 | 用框架圖說明完整方法流程 | 5% |

**修訂後評分：**正式預測分數僅供參考，不計入成績；README 致謝資料來源占 **1%**，影片與簡報占 **14%**。可用驗證實驗支撐報告，但投影片和影片都應呈現實際做過的內容。

### Data Source

This assignment uses data from [GH Archive](https://www.gharchive.org/). The analysis is intended for educational use within this course.

## LightGBM 實驗與重現

**目前提交版本：**[final_submission_summary.md](final_submission_summary.md) 記錄 2026/11/21 以最新可得資料重訓的流程與結果。它使用 9/5～11/7 起始的十個完整歷史週訓練五類，另以無整天零紀錄斷層的四週訓練 push；四折回測先選特徵組，11/7 週調參後再全部重訓。下列 `artifacts/` 與 `final_refresh/` 為保留的較早實驗。

新增 pipeline 只讀取老師 `preprocess.py` 產生的八張 Parquet 表。`scripts/activity_features.py` 建立使用者、時間與關聯特徵；`scripts/run_lightgbm.py` 執行四組消融、時間驗證、threshold tuning、畫圖及產生預測。老師原始腳本未修改。

```bash
python -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
python scripts/download_gharchive.py --start 2026-09-01 --end 2026-11-16 --out raw_data
.venv/bin/python scripts/preprocess.py --raw raw_data --out data/rebuild
.venv/bin/python scripts/check_temporal_invariance.py --db data/rebuild/db
.venv/bin/python scripts/run_lightgbm.py --db data/rebuild/db --raw raw_data --out artifacts_rerun --threads 4
.venv/bin/python scripts/audit_activity_coverage.py --db data/rebuild/db --raw raw_data --out artifacts_rerun/metrics
.venv/bin/python scripts/predict_from_checkpoints.py --db data/rebuild/db --artifacts artifacts_rerun --output artifacts_rerun/predictions/reproduced_prediction.csv
.venv/bin/python scripts/make_diagrams.py --out artifacts_rerun/figures
```

主實驗訓練週起點（UTC）為 9/12、9/19、9/26 的 16:00；10/3 16:00 的一週用於早停、有限參數選擇與六個 threshold；10/10 16:00 的一週獨立評估。每個視窗均為七天，週六 16:00 UTC 對應台灣週日 00:00。訓練使用 `cohort.txt` 的 9,000 人，評估與輸出使用 `test_input.csv` 的 4,465 人。A 為 Copy Last Week，B 為使用者累積特徵，C 加入 7/14/30 天與趨勢，D 再加入八表關聯聚合。晚得知的狀態欄位不當特徵。

主實驗產物位於 `artifacts/`：`metrics/` 指標、`figures/` 消融／重要度／覆蓋率圖，以及可供報告使用的 schema graph 與框架圖（PNG、SVG）、`logs/` 訓練與評估紀錄、`models/` checkpoint、`predictions/` 預測。`predict_from_checkpoints.py` 能重現暫定預測，本次已確認兩份 CSV 完全一致。繪製 schema graph 與框架圖需要系統安裝 Graphviz 的 `dot` 指令。`artifacts_recent/` 是 11/7 週的探索性結果；近期活動紀錄明顯稀疏，解讀限制見 [experiment_summary.md](experiment_summary.md)。`artifacts/predictions/provisional_prediction.csv` 是 11/16 16:00 UTC 時點的暫定預測，尚非 11/22 的正式提交檔。

**資料品質修正：**逐日稽核後發現 10/10 週也有整天零 Pushes；原六類 Score 不宜當完整六類結論。請閱讀 [data_quality_report.md](data_quality_report.md) 與 [relational_study_summary.md](relational_study_summary.md)。後者新增 9/26 六類驗證，以及 10/10、10/17 排除 push 的五類驗證，分開比較 repo 活躍度和真正的 reviewer–repo–PR 互動特徵。這些結果顯示特徵效果跨週不穩定。

`test_input.csv` 並未漏人；較早的八表中有 **150／4,465** 位測試者沒有 Users 列，11/20 版為 **148 位**。推論程式對這些人採六類全 0 預設；歷史冷啟動稽核、風險與逐活動 F1 比較見 [relational_study_summary.md](relational_study_summary.md)。提交版本的實際人數見 [final_submission_summary.md](final_submission_summary.md)；預測檔含全部 4,465 人。

另有獨立的 [GNN 實驗](gnn_study/gnn_study_summary.md)：沿用相同三個時間折次與五類比較，實際訓練兩層 GCN、GCN + Temporal、HeteroConv + SAGEConv，並在本機保存圖快照、checkpoint、老師評分、三個 seed、關係遮蔽及速度／記憶體對照。圖快照約 291 MB、可由程式重建，因此不納入 Git；重跑指令見 [gnn_study/README.md](gnn_study/README.md)。三週中最佳 LightGBM 仍高於 GNN；既有 LightGBM 預測未被覆寫。

若要重做新稽核與關聯研究，使用新輸出資料夾避免覆寫既有成果：

```bash
.venv/bin/python scripts/audit_event_quality.py --db data/extended/db --raw raw_data --out quality_study_rerun
.venv/bin/python scripts/run_relational_study.py --db data/extended/db --raw raw_data --quality quality_study_rerun/weekly_gap_flags.csv --out relational_study_rerun --threads 4
.venv/bin/python scripts/bootstrap_study.py --study relational_study_rerun --replicates 1000
.venv/bin/python scripts/audit_cold_start.py --db data/extended/db --study relational_study_rerun
```

### 重現本次提交版本

以下使用截至 **2026/11/21 02:00 UTC** 的原始小時檔；會下載約 1,947 小時的 GH Archive 資料，需足夠磁碟與時間。已有檔案會跳過。老師原始 `preprocess.py`、`evaluate.py` 均未修改。使用新的輸出資料夾保留 Git 中的實驗結果：

```bash
python -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
python scripts/download_gharchive.py --start 2026-09-01 --end 2026-11-21 --end-hour 2 --out raw_data
.venv/bin/python scripts/preprocess.py --raw raw_data --out data/final_2026-11-21-02
.venv/bin/python scripts/audit_event_quality.py --db data/final_2026-11-21-02/db --raw raw_data --out final_submission_rerun/quality
.venv/bin/python scripts/train_final_lightgbm.py --db data/final_2026-11-21-02/db --raw raw_data --quality final_submission_rerun/quality/weekly_gap_flags.csv --out final_submission_rerun --threads 4
.venv/bin/python scripts/predict_final_lightgbm.py --db data/final_2026-11-21-02/db --artifacts final_submission_rerun --output final_submission_rerun/predictions/reproduced_prediction.csv
.venv/bin/python scripts/evaluate.py --pred final_submission_rerun/predictions/reproduced_prediction.csv --input submission/test_input.csv
```

執行後檢查 `final_submission_rerun/metrics/final_config.json` 的實際 `prediction_time_utc`，並與 `submission/output_314551178.csv` 比對。若在繳交截止前取得更晚、仍早於 **2026/11/21 16:00 UTC** 的完整原始檔，可另建新資料夾重跑；不得把測試週資料用於特徵。

---

## 原 repo 英文說明與指令

**Updates are marked with ⭐.**

In this assignment you predict what GitHub users will do in the coming week. For each test
user, you predict which of 6 activities (opening a pull request, pushing code, and so on) the
user will perform during the test week.

This repository contains:

```
assignment2/
+-- README.md                  # download, build the tables, submit
+-- SCHEMA.md                  # every column, explained
+-- scripts/
|   +-- preprocess.py          # builds the 8 tables from GH Archive
|   +-- evaluate.py            # checks the format, computes the score
|   +-- cohort.txt             # contains 9,000 anonymized user IDs (read by preprocess.py)
+-- submission/
    +-- test_input.csv         # official test users
    +-- sample_submission.csv  # example output in the correct format
```

## ⭐ 1. Download the GH Archive files

Download the hourly files of the time period you want from [GH Archive](https://www.gharchive.org)
into your own folder `RAW_DATA_FOLDER`. GH Archive provides one file per hour, named by UTC date
and hour.

For example, based on the command instructions from GH Archive, you can download all the data in
Oct. 2026 using the **GH Archive API**:

```bash
wget -P RAW_DATA_FOLDER https://data.gharchive.org/2026-10-{01..31}-{0..23}.json.gz
```

For more download options, see the GH Archive website.

## ⭐ 2. Build the tables

```bash
python scripts/preprocess.py --raw RAW_DATA_FOLDER --out YOUR_OUTPUT_FOLDER
```

- `RAW_DATA_FOLDER`: your own folder used in Section 1.
- `YOUR_OUTPUT_FOLDER`: your own folder for the tables.

You are required to run our provided preprocessing script before using the data for your model.
The preprocessing script generates the 8 required tables and includes anonymization.

## 3. The tables

| Table | Each row is | Row time |
|---|---|---|
| `Users` | a GitHub user | `first_seen_at` |
| `Repositories` | a repository | `first_seen_at` |
| `PullRequests` | a pull request | `first_seen_at` |
| `Issues` | an issue (the conversation of a pull request is also stored here, with `is_pull_request = true`) | `first_seen_at` |
| `IssueComments` | a comment in an issue or pull request conversation | `created_at` |
| `PRReviews` | a review of a pull request | `first_seen_at` |
| `PRReviewComments` | a comment on specific lines of a pull request's code | `created_at` |
| `Pushes` | a push of commits to a branch or tag | `created_at` |

The **row time** is the time at which a row appears in the data. For a prediction at time t, use
only rows whose row time is before t. Do not filter pull requests, issues or reviews by
`created_at` or `submitted_at`: they can enter the data long after they were created.

**Which users are included.** The tables contain all the activity of 9,000 fixed users (the
*cohort*, marked with `Users.is_cohort = true`), plus the users related to their activity.

**Times** are in UTC (no time zone is stored). Taiwan time is UTC+8.

## 4. Test users and output

⭐ **Test week:** 2026/11/22 00:00 – 11/29 00:00 (Taiwan time)

⭐ **Missing data:** Some hours are missing or incomplete in GH Archive itself. If GH Archive is
missing the data of an activity during the test week, that activity is not scored.

`submission/test_input.csv` lists the test users. Its `UserID` column is `Users.github_user_id`.

Your output is a CSV file with one row per test user, exactly this header, and 1 (the user will
do it) or 0 (the user will not) in each activity column, with no empty cells:

```
UserID,open_pr,open_issue,issue_comment,pr_review,pr_review_comment,push
```

**Example output file:** `submission/sample_submission.csv` (random predictions in the correct
format).

| Column | 1 means that during the test week the user ... |
|---|---|
| `open_pr` | opens a pull request |
| `open_issue` | opens an issue (not a pull request) |
| `issue_comment` | comments in an issue or pull request conversation |
| `pr_review` | submits a review of a pull request |
| `pr_review_comment` | comments on specific lines of a pull request's code |
| `push` | pushes commits to a branch or tag |

**Checking and scoring.** A file with a format error is not scored.

```bash
# Check the format of your output file
python scripts/evaluate.py --pred output_{studentID}.csv --input submission/test_input.csv

# Score your predictions for a validation week against your own answer file (same format)
python scripts/evaluate.py --pred YOUR_PREDICTION_FILE --truth YOUR_ANSWER_FILE
```

**Scoring.** F1 for each activity; `Score = (macro-F1 + micro-F1) / 2`.

**Reference scores.** These are the scores of five baseline methods that we tested on the
week of 2026/09/27 – 10/04.

| Method | macro-F1 | micro-F1 | Score |
|---|---|---|---|
| Random | 19.77% | 22.74% | 21.26% |
| Predict all 0 | 0.00% | 0.00% | 0.00% |
| Copy last week | 43.01% | 56.20% | 49.61% |
| GNN without time encoding | 22.80% | 48.61% | 35.70% |
| GNN with time encoding | 50.90% | 58.74% | 54.82% |

- **Copy last week**: for each activity, 1 if the user did it in the 7 days before the
  prediction time, otherwise 0.
- **GNN**: a graph neural network on the 8 tables, as in the
  [RelBench](https://relbench.stanford.edu) examples. The version with time encoding also
  tells the model how long before the prediction time each row happened.

## Data Claim

The data used in this assignment is sourced from [GH Archive](https://www.gharchive.org), a
publicly available archive of GitHub events. This assignment is intended for educational
purposes within this course.
