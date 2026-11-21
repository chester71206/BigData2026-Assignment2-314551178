# Assignment 2 presentation

This directory contains the editable English PowerPoint, a matching PDF, and Traditional Chinese speaking notes paced for 18 minutes 25 seconds including transitions. The deck has 16 slides and answers all four required questions.

## Deliverables

- `Assignment2_report.pptx` — editable slides, built from the course's `Assignment2_report_template`.
- `Assignment2_report.pdf` — matching editable-source PDF; `Asg2_314551178.pdf` is the E3-named copy.
- `speaker_notes_zh_TW.md` — slide-by-slide narration and pacing. The same notes are embedded in PowerPoint.
- `assets/schema_graph_bw.dot` and `.png` — all eight tables and all 20 foreign keys, with right-angle arrows from FK to referenced PK; dashed edges target `Repositories`.
- `assets/model_architecture_bw.png`, `backtest_bw.png`, `push_quality_bw.png` — figures used in the deck.
- `temporal_ablation_summary.md` and `temporal_ablation/metrics/` — six incremental temporal feature sets evaluated on four historical holdouts.

## Submission identity

Student ID: **314551178**. The slides link to the student's [public GitHub repository](https://github.com/chester71206/BigData2026-Assignment2-314551178). The identity and repository URL are inserted by `build_report.py`. The historical validation results in the slides are not an official test score. The MP4 must be recorded by the student and uploaded to E3 separately; AI-generated videos receive zero points under the course rules.

## Rebuild

From the repository root, with Graphviz `dot` installed:

```bash
python -m pip install -r requirements.txt -r report/requirements.txt
python report/make_figures.py
python report/run_temporal_ablation.py  # regenerates the Q3 experiment; may take a few minutes
python report/build_report.py
```

The official course template is preserved at `assets/Assignment2_report_template.pptx`; its URL was extracted from the provided Assignment 2 PDF. The script does not modify any course-provided code.

Activity names and schema fields use math-style subscripts in both PPTX and PDF: for example, `open_pr` is displayed as open with `pr` as a subscript, and `reviewer_id` as reviewer with `id` as a subscript. Table names use monospaced type.

Sources for the reported results: `final_submission/metrics/backtest_scores.csv`, `final_submission/quality/daily_activity_quality.csv`, `final_submission/metrics/final_config.json`, `relational_study_summary.md`, `gnn_study/gnn_study_summary.md`, and the eight preprocessed Parquet tables. Historical validation excludes Push from the common five-class score because the archive has full-day gaps in that event type.
