# Training Runs Table Template

You can paste this into Google Docs, Notion, or your PDF report.

| Candidate | MLflow run link | Code version | Key hyperparams | Key model metrics | Key training cost metrics | Notes |
| --- | --- | --- | --- | --- | --- | --- |
| baseline | paste link | git sha | strategy=most_frequent | accuracy=0.843, f1=0.915, roc_auc=0.500 | train_seconds=1.5, rows/s=198183 | Naive lower-bound. Always predicts majority class. |
| xgboost_v1 | paste link | git sha | lr=0.3, max_depth=6, n_estimators=100 | accuracy=0.861, f1=0.922, roc_auc=0.835, avg_prec=0.964 | train_seconds=2.4, rows/s=128387 | XGBoost with defaults. Proves architecture works. |
| **xgboost_v2** | paste link | git sha | lr=0.1, max_depth=8, n_estimators=200, scale_pos_weight=4.7 | accuracy=0.856, f1=0.921, roc_auc=0.836, avg_prec=0.965 | train_seconds=3.8, rows/s=83413 | **Best candidate: highest avg precision + 99.5% recall.** |

## How To Mark Promising Candidates

Bold the xgboost_v2 row. It is the best candidate because:

- highest average precision (0.965) — best metric for imbalanced classification
- highest recall (99.5%) — catches nearly every place a user would visit
- ROC-AUC (0.836) slightly better than v1
- handles class imbalance with `scale_pos_weight=4.7`
- reasonable training cost (3.8 seconds on 303k rows)

xgboost_v1 is still worth noting:

- higher precision (0.876 vs 0.858) — fewer false recommendations
- faster training (2.4s vs 3.8s)
- may be preferred if serving speed matters more than recall

## Automation

Generate the table with real MLflow links:

```bash
docker run --rm \
  -v "$(pwd):/app" \
  -e MLFLOW_TRACKING_URI=http://${PRIVATE_IP}:8000 \
  gemspot-train-proj10 \
  python scripts/export_run_table.py \
    --experiment-name GemSpot-WillVisit \
    --tracking-uri http://${PRIVATE_IP}:8000 \
    --ui-base-url http://YOUR_FLOATING_IP:8000
```
