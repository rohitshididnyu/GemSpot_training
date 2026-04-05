# GemSpot Training

Training subsystem for the GemSpot destination recommendation project.

## What This Does

Trains XGBoost models to predict whether a user will visit a candidate place (`will_visit` = 0 or 1), using 337,698 real Google Maps review records. Features include category encodings (21-dim), destination vibe tags (20-dim), user preference scores (20-dim), and interaction features.

Three candidates are trained from one config file:

- **baseline** — naive dummy, always predicts majority class
- **xgboost_v1** — XGBoost with default params
- **xgboost_v2** — tuned XGBoost with class imbalance handling

All runs are tracked in MLflow with parameters, metrics, cost metrics, system metrics, and model artifacts.

## Repository Layout

```
├── configs/
│   └── candidates.yaml              # Model candidates and hyperparameters
├── data/demo/
│   └── initial_training_set.csv      # Full dataset (337k rows)
├── docs/
│   ├── training_documentation.md     # Full training documentation
│   ├── chameleon_steps.md            # Chameleon Cloud setup
│   ├── ml6_start_to_finish.md        # ML6 lab-aligned workflow
│   ├── demo_video_script.md          # Video recording guide
│   ├── run_table_template.md         # Run table template
│   └── submission_checklist.md       # Gradescope submission guide
├── interfaces/
│   ├── will_visit_input.sample.json  # Sample model input
│   └── will_visit_output.sample.json # Sample model output
├── scripts/
│   ├── split_dataset.py              # Time-based train/val split
│   ├── run_training_container.sh     # Docker training runner
│   ├── export_run_table.py           # Generate run table from MLflow
│   └── make_demo_dataset.py          # Synthetic data generator (fallback)
├── src/
│   ├── train.py                      # Main training entrypoint
│   └── gemspot_training/
│       ├── data.py                   # Data loading and feature engineering
│       ├── training.py               # Model building and metrics
│       └── utils.py                  # Environment info, git SHA
├── Dockerfile                        # Training container (Python 3.11-slim)
└── requirements.txt                  # Python dependencies
```

## Quick Start on Chameleon

```bash
# 1. Clone repo and upload data
git clone https://github.com/rohitshididnyu/GemSpot_training.git gemspot
cd gemspot
# scp initial_training_set.csv into data/demo/ if not in repo

# 2. Build Docker image
export PROJECT_SUFFIX=proj10
docker build -t gemspot-train-${PROJECT_SUFFIX} .

# 3. Split dataset by time
docker run --rm -v "$(pwd):/app" gemspot-train-${PROJECT_SUFFIX} \
  python scripts/split_dataset.py

# 4. Run training
PRIVATE_IP=$(hostname -I | awk '{print $1}')
export MLFLOW_TRACKING_URI=http://${PRIVATE_IP}:8000
unset DOCKER_EXTRA_ARGS
bash scripts/run_training_container.sh

# 5. View results at http://YOUR_FLOATING_IP:8000
```

## Data Pipeline

Raw CSV → drop IDs → explode list-encoded columns → create interaction features → 85 numeric features

See `docs/training_documentation.md` for full details on features, category mapping, and model explanations.

## Results (Time-Based Split: 303k train / 34k val)

| Candidate | Accuracy | F1 | ROC-AUC | Avg Precision | Train Time |
|-----------|----------|-----|---------|---------------|------------|
| baseline | 0.843 | 0.915 | 0.500 | 0.843 | 1.5s |
| xgboost_v1 | 0.861 | 0.922 | 0.835 | 0.964 | 2.4s |
| **xgboost_v2** | **0.856** | **0.921** | **0.836** | **0.965** | 3.8s |
