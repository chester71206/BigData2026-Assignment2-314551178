# Assignment 2 speaking notes (Traditional Chinese)

The slides are in English. These notes support a 16–20 minute personal presentation.
Student ID: 314551178. Public repository: https://github.com/chester71206/BigData2026-Assignment2-314551178.

## Slide 1 — Predicting Seven-Day GitHub User Activity (40 sec)

這份報告說明我如何預測 GitHub 使用者在未來七天的六種活動。主方法是以老師提供的八張關聯式資料表，建立只有預測起點以前可見的使用者特徵，再訓練六個 LightGBM 分類器。我也實作了關聯特徵與 GNN 做對照。這份簡報的數字來自歷史週驗證；正式測試週還沒有答案。封面列出學號與個人公開 GitHub repository。

## Slide 2 — The prediction task (55 sec)

題目給我們四千四百六十五位正式測試使用者，每人要輸出六個零或一。六類分別是開 PR、開 Issue、Issue 或 PR 討論留言、PR Review、PR 程式碼留言，以及 Push。預測對象不是活動次數，也不是下一次活動的日期，而是接下來七天有沒有發生。正式測試週從台灣時間十一月二十二日零點開始，現在還沒有正式答案，所以後面的 F1 都是歷史週的回測結果。

## Slide 3 — Q1. Schema graph: eight tables, 20 foreign keys (80 sec)

這張 Schema Graph 畫出老師八張表與全部二十條外鍵邊。Users 和 Repositories 放在右側，其他活動表依 review、issue、PR、push 路徑由左向右排列。為了讓線條清楚，採用水平與垂直的直角連線；指向 Repositories 的邊改用虛線，方便和 Users 等其他邊區分。每個框內用等寬字體標示 primary key 與所有 foreign keys；每條箭頭由持有外鍵的表指向被參照表。欄位名稱放在節點內，不在二十條線上重複標字。例如 PRReviews.pr_id 指向 PullRequests.pr_id，PRReviewComments.pull_request_review_id 指向 PRReviews.review_id。左側自我迴圈是 in_reply_to_id。PullRequests.merged_by_id 也在 schema 裡，但它可能是事後才知道的資訊，建模時會排除。下一頁會示範一條實際 join 路徑。

## Slide 4 — Q1→Q2. One observable join path (55 sec)

第三頁交代完整 schema，這一頁用一條實際路徑示範怎麼做 join。從 PRReviews 的 reviewer_id 找到使用者，再以 pr_id 找到被審查的 PR，由 PR 的 repo_id 找到 Repository。若 PR 作者關係在當時已出現，還能查作者；PRReviewComments 則可以接回該次 review 的討論。這樣可統計使用者審查過幾個 PR、參與哪些 repo，以及是否有程式碼留言。每段路徑都必須在預測時間 t 以前可見。另有一個常見的 ID 陷阱：正式 test_input.csv 的 UserID 對應 Users.github_user_id，不是表的 user_id。

## Slide 5 — Q1. Dataset period and observed tables (60 sec)

這次實驗只用 GH Archive 官方小時資料和老師原始 preprocess.py。資料從九月一日零點 UTC 到十一月二十一日二點 UTC，共一千九百四十七個連續小時檔。老師的程式篩出十八萬八千六百八十八筆 cohort 事件，做成右圖所示的八張 Parquet 表。對五種非 Push 活動，我們可以使用從九月五日開始、十一月十四日結束的十個完整標籤週。十一月十四日以後的週在這次資料截止時尚未完整，不能拿來當負例，但仍可用已到手的事件建立提交候選的預測特徵。

## Slide 6 — Q2. Three levels of relation modeling (75 sec)

第二題研究三種不同深度的跨表關聯，而不是預先假設關聯越多越好。第一層把各事件表透過作者、reviewer、user_id 等外鍵歸到 Users，做十八個使用者歷史特徵；這就是最後選上的模型，仍然使用多表關聯。第二層經由 Repository 鄰域，計算最近三十天參與的 repo 有多少其他人的事件，並扣掉本人的事件避免重複計數。第三層沿 PR、Issue、Review 與 Comment 的外鍵建立互動統計，例如 PR 收到的 review 和不同 reviewer 人數。第二、三層並非沒有實作，而是要由後面的可觀測性分析與消融實驗決定是否留下。所有 join 都只用預測起點前可見的列。

## Slide 7 — Q2. Are rich relations actually observable? (75 sec)

關係特徵要有足夠觀測，否則再複雜的模型也學不到。這張表比較九月二十六日和十一月二十日的三十天視窗。近月 review 從九千多筆降到一千九百多筆；其中能連到已知 PR 作者的比例從百分之二十八點六降到百分之二點二。有兩位以上 reviewer 的 PR 在最新視窗只剩一個。Repository 聚合對兩千四百七十位測試者有值，但更細的互動特徵只有六百五十一位有值。所以真正可利用的訊號主要是 reviewer 與 repo 的重複活動，PR 作者及共同 reviewer 關係非常稀疏。

## Slide 8 — Q2. Relation-feature ablation (75 sec)

這張圖是方法比較的主要證據。所有折次都用同樣五種活動評分，排除已知資料缺漏的 Push。四個較晚的驗證週是九月二十六日、十月十日、十月十七日和十月三十一日；每折的訓練、調參和評估依時間排列。若四折等權平均，只用 User 特徵的模型是百分之四十點九五，加入完整關聯的 F 模型是四十點二三。但最新的十月三十一日折，F 反而以四十一點九八勝過 User 的四十點五七。這表示關聯特徵並非完全沒用，只是目前觀測稀疏、跨週效果不穩定，不能宣稱穩定提升。

## Slide 9 — Q2/Q3. GNN versus LightGBM (75 sec)

在關聯建模之後，我們也比較圖神經網路，看看明確的圖結構能否超過聚合特徵。兩層 GCN 加入時間特徵後，在三個評估週全部提升；區分節點和關係類型的 HeteroSAGE 又進一步提高。但是這三個週的最佳 LightGBM 仍然較高。因此只能說我們測試的 GNN 沒有打敗 LightGBM，不能推論所有 GNN 都不適合。可能限制包括 PR 作者邊覆蓋率低、近期互動觀測變少。這也支持我們的主軸：關聯越複雜不保證越有效，要看圖的觀測密度。這是較早三折研究，不能和最後的四折模型選擇當成同一次實驗。

## Slide 10 — Q3. A feature must exist at prediction time (70 sec)

第三題的核心是避免時間洩漏。每個樣本都有預測起點 t。只有事件發生時間早於 t，並且那筆資料首次出現在八表中的時間也早於 t，才能進入特徵。PR 與 Issue 尤其重要：某個 PR 可能較早建立，但老師資料是在較晚的 review 事件才第一次看到它。若只看 created_at，就會把當時根本不知道的 PR 放進特徵。標籤只看 t 到七天後的事件，且資料列在那週結束前可見。合併時間、關閉時間等事後狀態一律不當特徵。我們也刪掉截止後才可見的所有列重建特徵，確認結果完全不變。

## Slide 11 — Q3. Incremental temporal ablation (80 sec)

這頁是這次補跑的細分 Temporal 消融，所有數值都來自同樣四個時間回測與老師的 evaluate.py。第一列是十八個累積使用者特徵，平均四十點九五。只加最近七天各活動次數、總事件和活躍天數後，十月三個評估週都進步，但九月二十六日由四十五點二四降到三十七點三八，少七點八六個百分點；當週五個活動的 F1 都下降，因此平均反而變三十九點九二。再依序加十四天、三十天視窗，平均降到三十九點四四和三十八點八七。加入距離上次活動時間，平均從三十八點八七回升到三十九點六七；最後加最近週對前週的差值，只小幅升到三十九點六九。這是逐層加入特徵，不能把相鄰差值解釋成獨立的因果效應。最早週歷史視窗較短，可能加劇不穩定，但七天特徵也在那週大跌，所以不能只怪三十天視窗。三十天週頻率是三十天次數的線性換算，這次細分消融不重複放入。

## Slide 12 — Q3. Coverage gaps change the meaning of zero (80 sec)

時間資訊不只是做七天視窗，還要判斷這個視窗的標籤可不可信。我們檢查所有原始小時檔，發現它們雖然連續，但處理後 Pushes 有十五個完整 UTC 日完全零筆，而其他活動仍有資料。九月每日 Push 中位數是一千四百二十筆，十一月只有四十四筆。這不能完全證明其他類別沒有部分漏記，但 Push 的整天零列已是明確斷層。因此後期缺漏週不能把未看到 Push 的人當作負例。跨週比較統一只看五類；Push 模型只使用四個沒有整天零列的較早週，並在結論中標示外推風險。

## Slide 13 — Q3. Chronological validation and final tuning (75 sec)

這張表說明怎麼避免拿同一週反覆調參又宣稱是獨立驗證。每一折都把較早的週拿來訓練，下一週做早停和調每活動的分類門檻，再用更晚的一週評估。我們看四折五類平均分數選特徵組，不做 random split。選完方法後，再用最後一個完整週，也就是十一月七日至十四日，調整參數、樹的輪數和門檻。最後把這週併回所有已知的十個完整週，固定輪數重新訓練。正式預測則只需對測試 ID 建立最新合法特徵，不需要知道測試週答案。

## Slide 14 — Q4. Model architecture and training strategy (80 sec)

第四題這張改成真正的模型架構圖。最上方八張 Parquet 表先依預測時間切成可觀測快照，再形成三組候選特徵：累積 User History、短中期 Temporal，以及 Repository 與 PR/Issue/Review 互動。三組特徵都經四次時間回測的消融比較；四折等權平均選出 User History，所以只有這十八個特徵會進入最後的六個 LightGBM 二元分類器。六個分類器各輸出一種未來七天活動的機率，再用每類門檻轉成零或一。圖下方獨立列出訓練策略：滾動回測選方法、最後完整週定 rounds 和門檻、再用所有合格已知週重訓。這區分了模型如何預測和我們如何評估模型。

## Slide 15 — Q4. Final model, output, and cold start (80 sec)

實際訓練時，每種活動各有一個二元 LightGBM。只試兩組樹參數，學習率零點零五，最多兩百五十輪，二十五輪沒有改善就停。每類的最佳門檻不是固定零點五，而是在調參週找 F1 較好的值。四折等權平均選出累積 User 特徵組，所以五類模型最後用十個完整標籤週重訓。Push 因後期資料缺口，只用四個早期安全週，這是目前最大限制。輸出檔四千四百六十五列，有一百四十六位使用者在八表完全沒出現，就對六類輸出零。這份檔案使用截至十一月二十一日零三點 UTC 前可見的資料，仍是暫定候選，沒有正式分數。

## Slide 16 — Findings, limitations, and repository (50 sec)

最後整理四個重點。第一，我們用八張表的外鍵建立可重現的使用者與互動特徵，而且逐時間切片避免未來資訊。第二，時間特徵與關聯特徵都有實作並做消融；有些較晚週會改善，但四折平均仍由較簡單的 User 模型勝出。第三，資料品質是本作業的重要限制，尤其 Push 在多個完整日期沒有觀測列，因此我分開處理並保留不確定性。第四，現在還沒有正式測試週答案，絕不能把歷史驗證 F1 當作正式成績。請以課程規定的名稱輸出 PDF 和本人錄製的影片。

Planned speaking time: 18 min 25 sec, including natural transitions.
