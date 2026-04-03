# Training Runs Table Template

You can paste this into Google Docs, Notion, or your PDF report.

| Candidate | MLflow run link | Code version | Key hyperparams | Key model metrics | Key training cost metrics | Notes |
| --- | --- | --- | --- | --- | --- | --- |
| dummy_most_frequent | paste link | git sha | strategy=most_frequent | accuracy=..., f1=... | train_seconds=..., rows_per_second=... | lower-bound baseline |
| logistic_regression_baseline | paste link | git sha | C=1.0, max_iter=500, solver=liblinear | accuracy=..., f1=..., roc_auc=... | train_seconds=..., rows_per_second=... | simple and fast baseline |
| random_forest_v1 | paste link | git sha | n_estimators=300, max_depth=16 | accuracy=..., f1=..., roc_auc=... | train_seconds=..., rows_per_second=... | captures interactions |
| hist_gradient_boosting_v1 | paste link | git sha | learning_rate=0.08, max_depth=8, max_iter=250 | accuracy=..., f1=..., roc_auc=... | train_seconds=..., rows_per_second=... | promising if best quality per runtime |

## How To Mark Promising Candidates

Highlight rows that are strong tradeoffs.

Examples of good notes:

- best F1 overall, but slower than logistic regression
- almost the same quality as the best model, but trains much faster
- weaker accuracy, but easiest to serve and retrain

## Optional Automation

If you want the table in Markdown directly from MLflow, run:

```bash
python3 scripts/export_run_table.py \
  --experiment-name GemSpot-WillVisit \
  --tracking-uri http://YOUR_FLOATING_IP:5000 \
  --ui-base-url http://YOUR_FLOATING_IP:5000
```
