"""Build an editable 16-slide PPTX and matching PDF from the official template."""

import json
import re
from pathlib import Path
from xml.sax.saxutils import escape

from PIL import Image
import pandas as pd
from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.enum.shapes import MSO_SHAPE
from pptx.enum.text import PP_ALIGN, MSO_ANCHOR
from pptx.util import Inches, Pt
from reportlab.lib import colors
from reportlab.lib.utils import ImageReader
from reportlab.lib.enums import TA_CENTER, TA_LEFT
from reportlab.lib.styles import ParagraphStyle
from reportlab.platypus import Paragraph
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.pdfgen import canvas
from reportlab.lib.utils import simpleSplit

ROOT = Path(__file__).resolve().parents[1]
DIR = ROOT / "report"
ASSETS = DIR / "assets"
STUDENT_ID = "314551178"
PUBLIC_REPO = "https://github.com/chester71206/BigData2026-Assignment2-314551178"
TEMPLATE = ASSETS / "Assignment2_report_template.pptx"
PPTX = DIR / "Assignment2_report.pptx"
PDF = DIR / "Assignment2_report.pdf"
NOTES = DIR / "speaker_notes_zh_TW.md"
W, H = 10.0, 5.625
FONT = "Arial"
CODE_FONT = "DejaVu Sans Mono"
INK = RGBColor(0, 0, 0)
GRAY = RGBColor(90, 90, 90)


def tx(text, x, y, w, h, size=16, bold=False, color="black", align="left"):
    return {"kind": "text", "text": text, "x": x, "y": y, "w": w, "h": h,
            "size": size, "bold": bold, "color": color, "align": align}


def img(name, x, y, w, h):
    return {"kind": "image", "name": name, "x": x, "y": y, "w": w, "h": h}


def tbl(headers, rows, x, y, w, h, widths, size=11):
    return {"kind": "table", "headers": headers, "rows": rows,
            "x": x, "y": y, "w": w, "h": h, "widths": widths, "size": size}


def slide(title, question, blocks, source, seconds, note):
    return {"title": title, "question": question, "blocks": blocks,
            "source": source, "seconds": seconds, "note": note}


def temporal_ablation_rows():
    data = pd.read_csv(DIR / "temporal_ablation/metrics/scores.csv")
    arms = [
        ("U_history", "User history (18)"),
        ("U_plus_7d", "+ 7d counts (26)"),
        ("U_plus_7_14d", "+ 14d counts (34)"),
        ("U_plus_7_14_30d", "+ 30d counts (42)"),
        ("U_plus_windows_recency", "+ recency (49)"),
        ("U_plus_windows_recency_trend", "+ trend / delta (56)"),
    ]
    folds = ["sep26", "oct10", "oct17", "oct31"]
    pivot = data.pivot(index="arm", columns="fold", values="score")
    if len(data) != 24 or any(arm not in pivot.index for arm, _ in arms):
        raise ValueError("Temporal ablation requires six complete arms on four folds")
    return [[label] + [f"{100 * pivot.loc[arm, fold]:.2f}" for fold in folds]
            + [f"{100 * pivot.loc[arm, folds].mean():.2f}"] for arm, label in arms]


SLIDES = [
    slide("Predicting Seven-Day GitHub User Activity", "Assignment 2",
          [tx("Big Data Analytics Techniques and Applications · Fall 2026", 1.08, 2.15, 7.9, 0.35, 17),
           tx("LightGBM with leakage-aware relational and temporal features", 1.08, 2.75, 7.9, 0.58, 18),
           tx(f"Student ID: {STUDENT_ID}", 1.08, 3.56, 7.9, 0.35, 14),
           tx(f"Public GitHub repository: {PUBLIC_REPO}", 1.08, 4.02, 8.3, 0.52, 12)],
          "Official Assignment 2 report template; GH Archive data.", 45,
          "這份報告說明我如何預測 GitHub 使用者在未來七天的六種活動。主方法是以老師提供的八張關聯式資料表，建立只有預測起點以前可見的使用者特徵，再訓練六個 LightGBM 分類器。我也實作了關聯特徵與 GNN 做對照。這份簡報的數字來自歷史週驗證；正式測試週還沒有答案。封面列出學號與個人公開 GitHub repository。"),
    slide("The prediction task", "Context",
          [tx("For each of 4,465 test users, predict six binary outcomes over the next seven days.",
              0.55, 1.13, 8.9, 0.64, 19, True),
           tx("`open_pr`     `open_issue`     `issue_comment`\n`pr_review`   `pr_review_comment`   `push`",
              0.72, 2.00, 8.7, 1.0, 18),
           tx("One sample = (`UserID`, `prediction_time`). A label is 1 if the activity is observed in [t, t + 7 days).",
              0.72, 3.25, 8.6, 0.88, 15),
           tx("Official test period: 2026-11-21 16:00 to 11-28 16:00 UTC.\nNo official test labels or score are available yet.",
              0.72, 4.25, 8.6, 0.6, 13)],
          "Instructor test_input.csv; Assignment 2 specification.", 60,
          "題目給我們四千四百六十五位正式測試使用者，每人要輸出六個零或一。六類分別是開 PR、開 Issue、Issue 或 PR 討論留言、PR Review、PR 程式碼留言，以及 Push。預測對象不是活動次數，也不是下一次活動的日期，而是接下來七天有沒有發生。正式測試週從台灣時間十一月二十二日零點開始，現在還沒有正式答案，所以後面的 F1 都是歷史週的回測結果。"),
    slide("Q1. Schema graph: eight tables, 20 foreign keys", "Question 1",
          [img("schema_graph_bw.png", 0.22, 1.07, 9.56, 3.96),
           tx("20 arrows point FK → PK. Dashed lines target Repositories; the self-loop is a comment reply.",
              0.45, 5.02, 9.1, 0.18, 9)],
          "Graph reflects instructor preprocess.py TABLES metadata; all 20 FKs are shown.", 90,
          "這張 Schema Graph 畫出老師八張表與全部二十條外鍵邊。Users 和 Repositories 放在右側，其他活動表依 review、issue、PR、push 路徑由左向右排列。為了讓線條清楚，採用水平與垂直的直角連線；指向 Repositories 的邊改用虛線，方便和 Users 等其他邊區分。每個框內用等寬字體標示 primary key 與所有 foreign keys；每條箭頭由持有外鍵的表指向被參照表。欄位名稱放在節點內，不在二十條線上重複標字。例如 PRReviews.pr_id 指向 PullRequests.pr_id，PRReviewComments.pull_request_review_id 指向 PRReviews.review_id。左側自我迴圈是 in_reply_to_id。PullRequests.merged_by_id 也在 schema 裡，但它可能是事後才知道的資訊，建模時會排除。下一頁會示範一條實際 join 路徑。"),
    slide("Q1→Q2. One observable join path", "Questions 1–2",
          [tx("Example: aggregate a reviewer's PR activity through typed foreign keys.",
              0.55, 1.10, 8.9, 0.38, 16, True),
           tbl(["Foreign key → referenced primary key", "Information gained"], [
               ["`PRReviews.reviewer_id` → `Users.user_id`", "Who reviewed"],
               ["`PRReviews.pr_id` → `PullRequests.pr_id`", "Which PR"],
               ["`PullRequests.repo_id` → `Repositories.repo_id`", "Which repository"],
               ["`PullRequests.author_id` → `Users.user_id`", "PR author, if visible"],
               ["`PRReviewComments.pull_request_review_id` → `PRReviews.review_id`", "Review discussion"]],
               0.55, 1.65, 8.9, 2.76, [5.35, 3.55], 10),
           tx("At cutoff t, every joined row must already be visible. Test `UserID` maps to `Users.github_user_id`.",
              0.58, 4.62, 8.8, 0.48, 12, True)],
          "All eight tables, PKs/FKs and 20 arrow directions remain on slide 3; SCHEMA.md.", 60,
          "第三頁交代完整 schema，這一頁用一條實際路徑示範怎麼做 join。從 PRReviews 的 reviewer_id 找到使用者，再以 pr_id 找到被審查的 PR，由 PR 的 repo_id 找到 Repository。若 PR 作者關係在當時已出現，還能查作者；PRReviewComments 則可以接回該次 review 的討論。這樣可統計使用者審查過幾個 PR、參與哪些 repo，以及是否有程式碼留言。每段路徑都必須在預測時間 t 以前可見。另有一個常見的 ID 陷阱：正式 test_input.csv 的 UserID 對應 Users.github_user_id，不是表的 user_id。"),
    slide("Q1. Dataset period and observed tables", "Question 1",
          [tx("Raw GH Archive: 2026-09-01 00:00 to 11-21 02:00 UTC\n1,947 consecutive hourly files; 188,688 cohort events after preprocessing.",
              0.55, 1.08, 8.9, 0.87, 15, True),
           tbl(["Table", "Rows"], [
               ["`Users`", "10,501"], ["`Repositories`", "20,976"],
               ["`PullRequests`", "44,900"], ["`Issues`", "29,035"],
               ["`IssueComments`", "16,060"], ["`PRReviews`", "13,406"],
               ["`PRReviewComments`", "10,233"], ["`Pushes`", "89,885"]],
               0.55, 2.12, 4.2, 2.90, [2.8, 1.4], 11),
           tx("Modeling windows\n• Ten complete labeled weeks: Sep 5–Nov 14 UTC\n• Last complete week tunes rounds and thresholds\n• Submission feature cutoff: Nov 21, 03:00 UTC\n• Official cutoff: Nov 21, 16:00 UTC",
              5.10, 2.22, 4.25, 2.68, 13)],
          "data/final_2026-11-21-02/db/; final_submission/metrics/final_config.json.", 65,
          "這次實驗只用 GH Archive 官方小時資料和老師原始 preprocess.py。資料從九月一日零點 UTC 到十一月二十一日二點 UTC，共一千九百四十七個連續小時檔。老師的程式篩出十八萬八千六百八十八筆 cohort 事件，做成右圖所示的八張 Parquet 表。對五種非 Push 活動，我們可以使用從九月五日開始、十一月十四日結束的十個完整標籤週。十一月十四日以後的週在這次資料截止時尚未完整，不能拿來當負例，但仍可用已到手的事件建立提交候選的預測特徵。"),
    slide("Q2. Three levels of relation modeling", "Question 2",
          [tx("1  Direct history", 0.60, 1.14, 3.0, 0.35, 16, True),
           tx("Counts by activity, distinct repositories, active days, and activity shares.",
              0.90, 1.56, 8.4, 0.54, 14),
           tx("2  Repository neighborhood", 0.60, 2.28, 4.0, 0.35, 16, True),
           tx("30-day activity of repositories the user touched; subtract the user's own events.",
              0.90, 2.70, 8.4, 0.56, 14),
           tx("3  PR / issue interactions", 0.60, 3.45, 4.0, 0.35, 16, True),
           tx("Reviewers received, PR comments, issue assignees, repeat reviewer–repo–PR paths.",
              0.90, 3.88, 8.4, 0.65, 14),
           tx("Selected User history still joins event tables to Users; higher-order paths are tested by ablation.",
              0.60, 4.88, 8.7, 0.25, 11, True)],
          "scripts/activity_features.py: user (18), temporal (44), original relation (16), repo (6), interaction (14).", 85,
          "第二題研究三種不同深度的跨表關聯，而不是預先假設關聯越多越好。第一層把各事件表透過作者、reviewer、user_id 等外鍵歸到 Users，做十八個使用者歷史特徵；這就是最後選上的模型，仍然使用多表關聯。第二層經由 Repository 鄰域，計算最近三十天參與的 repo 有多少其他人的事件，並扣掉本人的事件避免重複計數。第三層沿 PR、Issue、Review 與 Comment 的外鍵建立互動統計，例如 PR 收到的 review 和不同 reviewer 人數。第二、三層並非沒有實作，而是要由後面的可觀測性分析與消融實驗決定是否留下。所有 join 都只用預測起點前可見的列。"),
    slide("Q2. Are rich relations actually observable?", "Question 2",
          [tbl(["30-day relation evidence", "Sep 26", "Nov 20"], [
               ["Reviews", "9,054", "1,929"],
               ["Reviews with known PR author", "28.6%", "2.2%"],
               ["Reviewer–repository pairs", "2,124", "962"],
               ["PRs with >=2 reviewers", "38", "1"],
               ["Test users with repo features", "3,796", "2,470"],
               ["Test users with interaction features", "1,212", "651"]],
               0.55, 1.18, 8.90, 2.83, [5.2, 1.85, 1.85], 11),
           tx("The reviewer–repo path is observable; PR-author and co-reviewer paths are sparse.",
              0.70, 4.29, 8.6, 0.50, 15, True)],
          "Observed from the eight Parquet tables at each cutoff; no GitHub API data.", 85,
          "關係特徵要有足夠觀測，否則再複雜的模型也學不到。這張表比較九月二十六日和十一月二十日的三十天視窗。近月 review 從九千多筆降到一千九百多筆；其中能連到已知 PR 作者的比例從百分之二十八點六降到百分之二點二。有兩位以上 reviewer 的 PR 在最新視窗只剩一個。Repository 聚合對兩千四百七十位測試者有值，但更細的互動特徵只有六百五十一位有值。所以真正可利用的訊號主要是 reviewer 與 repo 的重複活動，PR 作者及共同 reviewer 關係非常稀疏。"),
    slide("Q2. Relation-feature ablation", "Question 2",
          [img("backtest_bw.png", 0.48, 1.10, 9.04, 3.43),
           tx("Five-class Score = (Macro-F1 + Micro-F1) / 2. Four-fold mean: User 40.95%; repo 40.21%;\ninteraction 39.41%; combined 40.23%. Combined wins Oct 31 (41.98% vs 40.57%).",
              0.55, 4.52, 8.9, 0.65, 10)],
          "Teacher evaluate.py, five activities with push excluded; final_submission/metrics/backtest_scores.csv.", 80,
          "這張圖是方法比較的主要證據。所有折次都用同樣五種活動評分，排除已知資料缺漏的 Push。四個較晚的驗證週是九月二十六日、十月十日、十月十七日和十月三十一日；每折的訓練、調參和評估依時間排列。若四折等權平均，只用 User 特徵的模型是百分之四十點九五，加入完整關聯的 F 模型是四十點二三。但最新的十月三十一日折，F 反而以四十一點九八勝過 User 的四十點五七。這表示關聯特徵並非完全沒用，只是目前觀測稀疏、跨週效果不穩定，不能宣稱穩定提升。"),
    slide("Q3. A feature must exist at prediction time", "Question 3",
          [tx("Feature rule at cutoff t", 0.62, 1.17, 4.0, 0.38, 17, True),
           tx("`event_at` < t  AND  `row_visible_at` < t", 0.92, 1.62, 8.0, 0.42, 17),
           tx("Label rule for the following week", 0.62, 2.36, 5.1, 0.38, 17, True),
           tx("t <= `event_at` < t + 7 days; visible by the week's end.",
              0.92, 2.81, 8.2, 0.52, 16),
           tx("Example: a PR may have `created_at` < t but `first_seen_at` > t.\nIt is excluded from the feature snapshot.",
              0.62, 3.65, 8.8, 0.79, 15),
           tx("Later state fields such as `merged_at` and `closed_at` are not features.",
              0.62, 4.72, 8.8, 0.30, 11, True)],
          "scripts/activity_features.py; SCHEMA.md row-time definitions.", 75,
          "第三題的核心是避免時間洩漏。每個樣本都有預測起點 t。只有事件發生時間早於 t，並且那筆資料首次出現在八表中的時間也早於 t，才能進入特徵。PR 與 Issue 尤其重要：某個 PR 可能較早建立，但老師資料是在較晚的 review 事件才第一次看到它。若只看 created_at，就會把當時根本不知道的 PR 放進特徵。標籤只看 t 到七天後的事件，且資料列在那週結束前可見。合併時間、關閉時間等事後狀態一律不當特徵。我們也刪掉截止後才可見的所有列重建特徵，確認結果完全不變。"),
    slide("Q3. Incremental temporal ablation", "Question 3",
          [tbl(["Cumulative features", "Sep 26", "Oct 10", "Oct 17", "Oct 31", "Mean"],
               temporal_ablation_rows(), 0.55, 1.14, 8.9, 2.88,
               [3.20, 1.14, 1.14, 1.14, 1.14, 1.14], 10),
           tx("+7d helps all three October holdouts, but Sep 26 falls 7.86 points.",
              0.60, 4.16, 8.8, 0.37, 13, True),
           tx("Adding 14d and 30d lowers the mean further; recency recovers 0.80 points,\nwhile trend adds only 0.02. None exceeds User history's 40.95% mean.",
              0.60, 4.54, 8.8, 0.60, 11)],
          "Teacher evaluate.py; report/temporal_ablation/metrics/scores.csv; Push excluded.", 85,
          "這頁是這次補跑的細分 Temporal 消融，所有數值都來自同樣四個時間回測與老師的 evaluate.py。第一列是十八個累積使用者特徵，平均四十點九五。只加最近七天各活動次數、總事件和活躍天數後，十月三個評估週都進步，但九月二十六日由四十五點二四降到三十七點三八，少七點八六個百分點；當週五個活動的 F1 都下降，因此平均反而變三十九點九二。再依序加十四天、三十天視窗，平均降到三十九點四四和三十八點八七。加入距離上次活動時間，平均從三十八點八七回升到三十九點六七；最後加最近週對前週的差值，只小幅升到三十九點六九。這是逐層加入特徵，不能把相鄰差值解釋成獨立的因果效應。最早週歷史視窗較短，可能加劇不穩定，但七天特徵也在那週大跌，所以不能只怪三十天視窗。三十天週頻率是三十天次數的線性換算，這次細分消融不重複放入。"),
    slide("Q3. Coverage gaps change the meaning of zero", "Question 3",
          [img("push_quality_bw.png", 0.52, 1.18, 8.95, 3.15),
           tx("15 full UTC days have zero observed Pushes despite complete hourly files and other activity.\nA missing push is not a reliable negative label; late push weeks are excluded from its training.",
              0.58, 4.48, 8.85, 0.62, 12)],
          "final_submission/quality/daily_activity_quality.csv; audited through Nov 20 UTC.", 85,
          "時間資訊不只是做七天視窗，還要判斷這個視窗的標籤可不可信。我們檢查所有原始小時檔，發現它們雖然連續，但處理後 Pushes 有十五個完整 UTC 日完全零筆，而其他活動仍有資料。九月每日 Push 中位數是一千四百二十筆，十一月只有四十四筆。這不能完全證明其他類別沒有部分漏記，但 Push 的整天零列已是明確斷層。因此後期缺漏週不能把未看到 Push 的人當作負例。跨週比較統一只看五類；Push 模型只使用四個沒有整天零列的較早週，並在結論中標示外推風險。"),
    slide("Q3. Chronological validation and final tuning", "Question 3",
          [tbl(["Holdout", "Train starts", "Tune", "Evaluate"], [
               ["Sep 26", "Sep 5, Sep 12", "Sep 19", "Sep 26"],
               ["Oct 10", "Sep 5–Sep 26", "Oct 3", "Oct 10"],
               ["Oct 17", "Sep 5–Oct 3", "Oct 10", "Oct 17"],
               ["Oct 31", "Sep 5–Oct 17", "Oct 24", "Oct 31"]],
               0.55, 1.15, 8.9, 2.25, [1.5, 3.1, 2.15, 2.15], 11),
           tx("Method selection: equal mean of four held-out five-class Scores.",
              0.62, 3.62, 8.8, 0.42, 15, True),
           tx("Final tuning week: Nov 7–14. Then refit all ten complete labeled weeks.\nLightGBM: two small configurations; <=250 rounds; stop after 25 unimproved rounds; class-specific thresholds.",
              0.62, 4.10, 8.8, 0.87, 12)],
          "scripts/train_final_lightgbm.py; teacher evaluate.py; no random split.", 85,
          "這張表說明怎麼避免拿同一週反覆調參又宣稱是獨立驗證。每一折都把較早的週拿來訓練，下一週做早停和調每活動的分類門檻，再用更晚的一週評估。我們看四折五類平均分數選特徵組，不做 random split。選完方法後，再用最後一個完整週，也就是十一月七日至十四日，調整參數、樹的輪數和門檻。最後把這週併回所有已知的十個完整週，固定輪數重新訓練。正式預測則只需對測試 ID 建立最新合法特徵，不需要知道測試週答案。"),
    slide("Q4. Model architecture and training strategy", "Question 4",
          [img("model_architecture_bw.png", 0.35, 1.05, 9.3, 3.77),
           tx("Only the selected feature group enters the final classifiers; other groups remain tested alternatives.",
              0.52, 4.92, 8.96, 0.28, 11, True)],
          "Reproducible scripts: preprocess.py, activity_features.py, train_final_lightgbm.py, evaluate.py.", 85,
          "第四題這張改成真正的模型架構圖。最上方八張 Parquet 表先依預測時間切成可觀測快照，再形成三組候選特徵：累積 User History、短中期 Temporal，以及 Repository 與 PR/Issue/Review 互動。三組特徵都經四次時間回測的消融比較；四折等權平均選出 User History，所以只有這十八個特徵會進入最後的六個 LightGBM 二元分類器。六個分類器各輸出一種未來七天活動的機率，再用每類門檻轉成零或一。圖下方獨立列出訓練策略：滾動回測選方法、最後完整週定 rounds 和門檻、再用所有合格已知週重訓。這區分了模型如何預測和我們如何評估模型。"),
    slide("Q4. Final model, output, and cold start", "Question 4",
          [tx("Six LightGBM binary classifiers", 0.60, 1.15, 8.5, 0.42, 17, True),
           tx("Learning rate 0.05; 15 or 31 leaves; max 250 rounds; early stopping patience 25.\nThreshold tuned separately for each activity.",
              0.84, 1.65, 8.4, 0.80, 14),
           tx("Refit with eligible historical labels", 0.60, 2.76, 5.0, 0.42, 17, True),
           tx("Five activities: 10 labeled weeks (Sep 5–Nov 14). Push: 4 early safe weeks.\nChosen: User history (18 features); richer groups did not improve four-fold mean.",
              0.84, 3.23, 8.4, 0.78, 14),
           tx("Output: 4,465 rows; 146 users with no visible history receive six zeros.\nCurrent CSV is provisional (Nov 21 03:00 UTC cutoff), not an official test score.",
              0.60, 4.45, 8.8, 0.60, 12)],
          "final_submission/metrics/final_config.json and final_model_details.csv.", 85,
          "實際訓練時，每種活動各有一個二元 LightGBM。只試兩組樹參數，學習率零點零五，最多兩百五十輪，二十五輪沒有改善就停。每類的最佳門檻不是固定零點五，而是在調參週找 F1 較好的值。四折等權平均選出累積 User 特徵組，所以五類模型最後用十個完整標籤週重訓。Push 因後期資料缺口，只用四個早期安全週，這是目前最大限制。輸出檔四千四百六十五列，有一百四十六位使用者在八表完全沒出現，就對六類輸出零。這份檔案使用截至十一月二十一日零三點 UTC 前可見的資料，仍是暫定候選，沒有正式分數。"),
    slide("Q2/Q3. GNN versus LightGBM", "Questions 2–3",
          [tbl(["Model / five-class Score", "Sep 26", "Oct 10", "Oct 17"], [
               ["Two-layer GCN", "36.98", "34.08", "34.45"],
               ["GCN + temporal", "39.31", "36.25", "35.05"],
               ["HeteroSAGE + temporal", "40.77", "37.55", "35.71"],
               ["Best LightGBM in same fold", "45.24", "41.12", "41.44"]],
               0.55, 1.20, 8.9, 2.35, [4.55, 1.45, 1.45, 1.45], 11),
           tx("Temporal and typed relations improve GCN, but the tested GNNs do not beat LightGBM.\nSparse PR-author edges and weaker late coverage constrain message passing.",
              0.64, 3.88, 8.72, 0.90, 14)],
          "gnn_study/gnn_study_summary.md; earlier three-fold study, not the later final refit.", 80,
          "在關聯建模之後，我們也比較圖神經網路，看看明確的圖結構能否超過聚合特徵。兩層 GCN 加入時間特徵後，在三個評估週全部提升；區分節點和關係類型的 HeteroSAGE 又進一步提高。但是這三個週的最佳 LightGBM 仍然較高。因此只能說我們測試的 GNN 沒有打敗 LightGBM，不能推論所有 GNN 都不適合。可能限制包括 PR 作者邊覆蓋率低、近期互動觀測變少。這也支持我們的主軸：關聯越複雜不保證越有效，要看圖的觀測密度。這是較早三折研究，不能和最後的四折模型選擇當成同一次實驗。"),
    slide("Findings, limitations, and repository", "Conclusion",
          [tx("• The eight-table joins and temporal snapshots are reproducible and leakage-aware.\n• User history is the best four-fold mean: 40.95% five-class Score; relations help some later weeks.\n• Push labels have systematic gaps; its current positive predictions are uncertain.\n• The official test week has no labels yet; this presentation reports historical validation only.",
              0.60, 1.15, 8.8, 2.20, 14),
           tx("Public GitHub repository", 0.60, 3.62, 5.0, 0.43, 17, True),
           tx(PUBLIC_REPO, 0.60, 4.10, 8.7, 0.55, 13),
           tx("Data source: GH Archive · Preprocessing and evaluation: course-provided scripts.",
              0.60, 4.87, 8.75, 0.25, 10)],
          "See README.md, final_submission_summary.md, relational_study_summary.md, gnn_study_summary.md.", 55,
          "最後整理四個重點。第一，我們用八張表的外鍵建立可重現的使用者與互動特徵，而且逐時間切片避免未來資訊。第二，時間特徵與關聯特徵都有實作並做消融；有些較晚週會改善，但四折平均仍由較簡單的 User 模型勝出。第三，資料品質是本作業的重要限制，尤其 Push 在多個完整日期沒有觀測列，因此我分開處理並保留不確定性。第四，現在還沒有正式測試週答案，絕不能把歷史驗證 F1 當作正式成績。請以課程規定的名稱輸出 PDF 和本人錄製的影片。"),
]

# Put the tested GNN comparison directly after the relational ablation.
# This supports the model-choice argument before the temporal and Q4 sections.
SLIDES.insert(8, SLIDES.pop(14))

# Speaking plan leaves room inside the 16–20 minute video limit.
PACING_SECONDS = [40, 55, 80, 55, 60, 75, 75, 75,
                  75, 70, 80, 80, 75, 80, 80, 50]
for spec, seconds in zip(SLIDES, PACING_SECONDS, strict=True):
    spec["seconds"] = seconds


def remove_template_samples(prs):
    for slide_id in list(prs.slides._sldIdLst):
        prs.part.drop_rel(slide_id.rId)
        prs.slides._sldIdLst.remove(slide_id)


def code_parts(token):
    """Return a code identifier as base and its final underscore suffix.

    open_pr becomes open + subscript pr; reviewer_id becomes reviewer +
    subscript id. Earlier underscores remain in compound identifier bases.
    """
    value = token[1:-1]
    if "_" not in value:
        return value, None
    return value.rsplit("_", 1)


def add_rich_runs(paragraph, line, size, bold=False, color=INK):
    for token in re.split(r"(`[^`]+`)", line):
        if not token:
            continue
        code = token.startswith("`") and token.endswith("`")
        if not code:
            run = paragraph.add_run()
            run.text = token
            run.font.name = FONT
            run.font.size = Pt(size)
            run.font.bold = bold
            run.font.color.rgb = color
            continue
        base, suffix = code_parts(token)
        run = paragraph.add_run()
        run.text = base
        run.font.name = CODE_FONT
        run.font.size = Pt(size)
        run.font.bold = bold
        run.font.color.rgb = color
        if suffix:
            sub = paragraph.add_run()
            sub.text = suffix
            sub.font.name = CODE_FONT
            sub.font.size = Pt(size * 0.73)
            sub.font.bold = bold
            sub.font.color.rgb = color
            sub._r.get_or_add_rPr().set("baseline", "-25000")


def set_text(shape, block):
    frame = shape.text_frame
    frame.clear()
    frame.word_wrap = True
    frame.margin_left = 0
    frame.margin_right = 0
    frame.margin_top = 0
    frame.margin_bottom = 0
    frame.vertical_anchor = MSO_ANCHOR.TOP
    for index, line in enumerate(block["text"].split("\n")):
        paragraph = frame.paragraphs[0] if index == 0 else frame.add_paragraph()
        paragraph.clear()
        paragraph.space_after = Pt(2)
        paragraph.alignment = (PP_ALIGN.CENTER if block["align"] == "center"
                               else PP_ALIGN.LEFT)
        add_rich_runs(paragraph, line, block["size"], block["bold"],
                      INK if block["color"] == "black" else GRAY)


def ppt_block(slide_obj, block):
    x, y, w, h = [Inches(block[k]) for k in ("x", "y", "w", "h")]
    if block["kind"] == "text":
        shape = slide_obj.shapes.add_textbox(x, y, w, h)
        set_text(shape, block)
    elif block["kind"] == "image":
        path = ASSETS / block["name"]
        with Image.open(path) as picture:
            ratio = picture.width / picture.height
        box_ratio = block["w"] / block["h"]
        if ratio > box_ratio:
            height = block["w"] / ratio
            slide_obj.shapes.add_picture(str(path), x, Inches(block["y"] + (block["h"] - height) / 2),
                                         width=w)
        else:
            width = block["h"] * ratio
            slide_obj.shapes.add_picture(str(path), Inches(block["x"] + (block["w"] - width) / 2), y,
                                         height=h)
    elif block["kind"] == "table":
        rows = len(block["rows"]) + 1
        table = slide_obj.shapes.add_table(rows, len(block["headers"]), x, y, w, h).table
        for col, width in enumerate(block["widths"]):
            table.columns[col].width = Inches(width)
        for i, values in enumerate([block["headers"]] + block["rows"]):
            for j, value in enumerate(values):
                cell = table.cell(i, j)
                cell.text = ""
                cell.margin_left = Inches(0.08)
                cell.margin_right = Inches(0.05)
                cell.margin_top = Inches(0.02)
                cell.margin_bottom = Inches(0.01)
                cell.fill.solid()
                cell.fill.fore_color.rgb = RGBColor(242, 242, 242) if i == 0 else RGBColor(255, 255, 255)
                for line_no, line in enumerate(value.split("\n")):
                    paragraph = (cell.text_frame.paragraphs[0] if line_no == 0
                                 else cell.text_frame.add_paragraph())
                    add_rich_runs(paragraph, line, block["size"], i == 0)


def ppt():
    prs = Presentation(TEMPLATE)
    remove_template_samples(prs)
    assert abs(prs.slide_width / 914400 - W) < 0.001
    assert abs(prs.slide_height / 914400 - H) < 0.001
    blank = prs.slide_layouts[10]
    for number, spec in enumerate(SLIDES, 1):
        slide_obj = prs.slides.add_slide(blank)
        slide_obj.background.fill.solid()
        slide_obj.background.fill.fore_color.rgb = RGBColor(255, 255, 255)
        title = tx(spec["title"], 0.48, 0.22, 9.0, 0.62, 23, True)
        ppt_block(slide_obj, title)
        line = slide_obj.shapes.add_shape(MSO_SHAPE.RECTANGLE,
                                          Inches(0.49), Inches(0.86), Inches(9.0), Inches(0.012))
        line.fill.solid()
        line.fill.fore_color.rgb = INK
        line.line.fill.background()
        for block in spec["blocks"]:
            ppt_block(slide_obj, block)
        ppt_block(slide_obj, tx(spec["source"], 0.48, 5.22, 8.7, 0.25, 8, color="gray"))
        ppt_block(slide_obj, tx(f"{number} / {len(SLIDES)}", 9.05, 5.22, 0.45, 0.25,
                                8, color="gray", align="center"))
        try:
            slide_obj.notes_slide.notes_text_frame.text = spec["note"]
        except AttributeError:
            pass
    prs.save(PPTX)


PDF_FONT = "DejaVu"


def pdf_markup(text):
    parts = []
    for token in re.split(r"(`[^`]+`)", text):
        if not token:
            continue
        code = token.startswith("`") and token.endswith("`")
        if code:
            base, suffix = code_parts(token)
            value = escape(base)
            if suffix:
                value += f"<sub>{escape(suffix)}</sub>"
            parts.append(f'<font name="DejaVuMono">{value}</font>')
        else:
            parts.append(escape(token).replace("\n", "<br/>"))
    return "".join(parts)


def pdf_text(can, block):
    size = block["size"]
    font = PDF_FONT + ("-Bold" if block["bold"] else "")
    can.setFont(font, size)
    can.setFillColor(colors.black if block["color"] == "black" else colors.HexColor("#555555"))
    x = block["x"] * 72
    top = H * 72 - block["y"] * 72
    width = block["w"] * 72
    bottom = H * 72 - (block["y"] + block["h"]) * 72
    if "`" in block["text"]:
        style = ParagraphStyle(
            "inline-code", fontName=font, fontSize=size, leading=size * 1.22,
            textColor=colors.black if block["color"] == "black" else colors.HexColor("#555555"),
            alignment=TA_CENTER if block["align"] == "center" else TA_LEFT,
        )
        paragraph = Paragraph(pdf_markup(block["text"]), style)
        _, height = paragraph.wrap(width, 10000)
        paragraph.drawOn(can, x, top - height)
        if top - height < bottom - 2:
            print("WARN PDF code text may overflow:", block["text"][:65])
        return
    line_y = top - size * 0.95
    for part in block["text"].split("\n"):
        lines = simpleSplit(part, font, size, width) if part else [""]
        for line in lines:
            if block["align"] == "center":
                can.drawCentredString(x + width / 2, line_y, line)
            else:
                can.drawString(x, line_y, line)
            line_y -= size * 1.22
    if line_y < bottom - size * 0.7:
        print("WARN PDF text may overflow:", block["text"][:65])


def pdf_block(can, block):
    if block["kind"] == "text":
        pdf_text(can, block)
    elif block["kind"] == "image":
        path = ASSETS / block["name"]
        with Image.open(path) as picture:
            ratio = picture.width / picture.height
        box_ratio = block["w"] / block["h"]
        if ratio > box_ratio:
            width, height = block["w"], block["w"] / ratio
        else:
            width, height = block["h"] * ratio, block["h"]
        x = block["x"] + (block["w"] - width) / 2
        y = block["y"] + (block["h"] - height) / 2
        can.drawImage(ImageReader(str(path)), x * 72, (H - y - height) * 72,
                      width=width * 72, height=height * 72, preserveAspectRatio=True)
    elif block["kind"] == "table":
        row_h = block["h"] * 72 / (len(block["rows"]) + 1)
        x = block["x"] * 72
        top = H * 72 - block["y"] * 72
        values = [block["headers"]] + block["rows"]
        for i, row in enumerate(values):
            y = top - (i + 1) * row_h
            if i == 0:
                can.setFillColor(colors.HexColor("#f2f2f2"))
                can.rect(x, y, block["w"] * 72, row_h, fill=1, stroke=0)
            can.setStrokeColor(colors.HexColor("#888888"))
            can.setLineWidth(0.35)
            can.line(x, y, x + block["w"] * 72, y)
            col_x = x
            for j, value in enumerate(row):
                font = PDF_FONT + ("-Bold" if i == 0 else "")
                can.setFillColor(colors.black)
                can.setFont(font, block["size"])
                max_w = block["widths"][j] * 72 - 8
                if "`" in value:
                    style = ParagraphStyle("table-code", fontName=font,
                                           fontSize=block["size"], leading=block["size"] + 1)
                    paragraph = Paragraph(pdf_markup(value), style)
                    _, height = paragraph.wrap(max_w, 10000)
                    paragraph.drawOn(can, col_x + 5, y + row_h - height - 4)
                else:
                    lines = simpleSplit(value, font, block["size"], max_w)
                    for k, line in enumerate(lines[:2]):
                        can.drawString(col_x + 5, y + row_h - 14 - k * (block["size"] + 1), line)
                col_x += block["widths"][j] * 72


def pdf():
    font_dir = Path("/usr/share/fonts/truetype/dejavu")
    pdfmetrics.registerFont(TTFont(PDF_FONT, str(font_dir / "DejaVuSans.ttf")))
    pdfmetrics.registerFont(TTFont(PDF_FONT + "-Bold", str(font_dir / "DejaVuSans-Bold.ttf")))
    pdfmetrics.registerFont(TTFont("DejaVuMono", str(font_dir / "DejaVuSansMono.ttf")))
    can = canvas.Canvas(str(PDF), pagesize=(W * 72, H * 72))
    for number, spec in enumerate(SLIDES, 1):
        can.setFillColor(colors.white)
        can.rect(0, 0, W * 72, H * 72, fill=1, stroke=0)
        pdf_block(can, tx(spec["title"], 0.48, 0.22, 9.0, 0.62, 23, True))
        can.setLineWidth(0.9)
        can.setStrokeColor(colors.black)
        can.line(0.49 * 72, (H - 0.86) * 72, 9.49 * 72, (H - 0.86) * 72)
        for block in spec["blocks"]:
            pdf_block(can, block)
        pdf_block(can, tx(spec["source"], 0.48, 5.22, 8.7, 0.25, 8, color="gray"))
        pdf_block(can, tx(f"{number} / {len(SLIDES)}", 9.05, 5.22, 0.45, 0.25,
                          8, color="gray", align="center"))
        can.showPage()
    can.save()


def notes():
    lines = ["# Assignment 2 speaking notes (Traditional Chinese)", "",
             "The slides are in English. These notes support a 16–20 minute personal presentation.",
             f"Student ID: {STUDENT_ID}. Public repository: {PUBLIC_REPO}.", ""]
    elapsed = 0
    for i, spec in enumerate(SLIDES, 1):
        elapsed += spec["seconds"]
        lines += [f"## Slide {i} — {spec['title']} ({spec['seconds']} sec)", "",
                  spec["note"], ""]
    lines += [f"Planned speaking time: {elapsed // 60} min {elapsed % 60} sec, including natural transitions.", ""]
    NOTES.write_text("\n".join(lines))


if __name__ == "__main__":
    assert len(SLIDES) <= 16
    ppt()
    pdf()
    notes()
    manifest = {"slides": len(SLIDES), "seconds": sum(s["seconds"] for s in SLIDES),
                "template": str(TEMPLATE), "pptx": str(PPTX), "pdf": str(PDF),
                "repo_placeholder": False, "student_placeholder": False,
                "student_id": STUDENT_ID, "public_repo": PUBLIC_REPO}
    (DIR / "build_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
