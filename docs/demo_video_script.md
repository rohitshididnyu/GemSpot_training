# Demo Video Script

Keep the demo short and simple. Aim for 2 to 4 minutes.

## What To Show

### Part 1. Show MLflow is live

In a browser:

- open `http://YOUR_FLOATING_IP:8000`
- show the GemSpot-WillVisit experiment page

In terminal:

```bash
docker ps
```

Point out the MLflow container name: `gemspot-mlflow-proj10`.

### Part 2. Show the training container starting

On the training machine:

```bash
cd ~/gemspot
export PROJECT_SUFFIX=proj10
PRIVATE_IP=$(hostname -I | awk '{print $1}')
export MLFLOW_TRACKING_URI=http://${PRIVATE_IP}:8000
unset DOCKER_EXTRA_ARGS
bash scripts/run_training_container.sh
```

Say:

- this is running inside Docker on Chameleon
- the tracking URI points to the live MLflow service
- the script trains 3 candidates from one config file: baseline, xgboost_v1, xgboost_v2
- the data was split by time: training on Nov 2020 - Apr 2021, validating on May - Sep 2021

### Part 3. Show the runs appear in MLflow

Refresh the browser on the MLflow page.

Show:

- 3 new runs appearing (baseline, xgboost_v1, xgboost_v2)
- metrics for each candidate (accuracy, f1, roc_auc, average_precision)
- artifacts logged for a run (model files, run_summary.json)

### Part 4. Show the best candidate (xgboost_v2)

Open the xgboost_v2 run and point to:

- candidate name: xgboost_v2
- key hyperparameters: learning_rate=0.1, max_depth=8, scale_pos_weight=4.7
- main metrics: roc_auc=0.836, average_precision=0.965, recall=0.995
- training time: ~3.8 seconds on 303k rows
- system metrics tab: cpu usage, memory

Then say why it is promising:

- handles the 82/18 class imbalance with scale_pos_weight
- highest average precision and recall
- trained on 303k real Google Maps review rows

## What To Say In One Sentence

"For the GemSpot training subsystem, I trained a baseline and two XGBoost candidates on 337k real Google Maps reviews in Docker on Chameleon, used a time-based train/val split, tracked every run in MLflow, and selected xgboost_v2 as the best model for its superior ranking quality and recall."
