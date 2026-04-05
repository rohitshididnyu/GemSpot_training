# GemSpot Training Documentation

## 1. What Are We Predicting

GemSpot is a destination recommendation system. Given a **user** and a **candidate place** (restaurant, park, museum, etc.), the model predicts:

> **Will this user visit this place?** (`will_visit` = 1 for yes, 0 for no)

This is a **binary classification** problem. The model outputs a probability between 0 and 1. If the probability is above 0.5, we recommend the place to the user.

### Real-world usage

- The data team produces user-place pairs with features
- The training team builds a model that predicts `will_visit`
- The serving team uses the trained model to score new user-place pairs in real time
- The frontend shows the highest-scoring places as recommendations

---

## 2. The Dataset

### Source

The dataset comes from Google Maps reviews. Each row represents one user interacting with one place at a specific point in time.

**File:** `data/demo/initial_training_set.csv` (337,698 rows)

### Columns in the raw data

| Column | Type | Description |
|--------|------|-------------|
| `user_id` | float | Unique user identifier. **Dropped** before training (not a feature). |
| `gmap_id` | string | Google Maps place ID. **Dropped** before training (not a feature). |
| `time` | float | Timestamp in milliseconds (epoch). **Kept** as a numeric feature. |
| `category` | string | Raw category names like `['Restaurant', 'Breakfast restaurant']`. **Dropped** (redundant — already encoded in `category_encoded`). |
| `category_encoded` | string (list) | 21-element multi-hot vector. Each position maps to a category type. See mapping table below. |
| `avg_rating` | float | Average star rating of the place (1.0 to 5.0). |
| `location_popularity` | int | How popular the location is (higher = more popular). |
| `destination_vibe_tag` | string (list) | 20-element binary vector. Each position is 1 if the place has that vibe attribute, 0 otherwise. |
| `user_total_visits` | int | Total number of places this user has visited in the past. |
| `user_personal_preferences` | string (list) | 20-element float vector. Each position is the user's preference score (0.0 to 1.0) for the corresponding vibe dimension. |
| `will_visit` | int | **Target variable.** 1 = user will visit, 0 = user will not visit. |

### Category encoding mapping

The `category_encoded` column is a 21-element multi-hot vector. Each position corresponds to:

| Index | Category |
|-------|----------|
| 0 | Restaurant |
| 1 | Fast food restaurant |
| 2 | Grocery store |
| 3 | Gas station |
| 4 | Takeout Restaurant |
| 5 | American restaurant |
| 6 | Discount store |
| 7 | Tourist attraction |
| 8 | Sandwich shop |
| 9 | Hamburger restaurant |
| 10 | Convenience store |
| 11 | Auto repair shop |
| 12 | Mexican restaurant |
| 13 | Coffee shop |
| 14 | Pizza restaurant |
| 15 | Breakfast restaurant |
| 16 | Dollar store |
| 17 | Park |
| 18 | Clothing store |
| 19 | Tire shop |
| 20 | Other |

A place can belong to multiple categories. For example, a place that is both a Restaurant (0) and a Breakfast restaurant (15) would have `[1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0]`.

### Vibe tags and user preferences

The `destination_vibe_tag` (20 binary values) and `user_personal_preferences` (20 float scores) share the same 20 vibe dimensions. Each position in both vectors represents the same vibe attribute.

- `destination_vibe_tag[i] = 1` means the place has vibe attribute `i`
- `user_personal_preferences[i] = 0.25` means the user has a 0.25 preference score for vibe attribute `i`

The model can learn that when a user has a high preference score for a vibe that the place also has (i.e., `preference[i]` is high AND `vibe[i] = 1`), the user is more likely to visit.

### Class imbalance

The target is imbalanced:

- `will_visit = 1`: 278,225 rows (82.4%)
- `will_visit = 0`: 59,473 rows (17.6%)

This means a naive model that always predicts "yes" would get 82% accuracy. We handle this with `scale_pos_weight` in XGBoost and use ROC-AUC and average precision as primary metrics instead of raw accuracy.

---

## 3. How the Data is Prepared for Training

### Time-based train/validation split

We split the data **by time**, not randomly. This is more realistic because in production the model trains on past data and predicts future behavior.

- **Train set:** All rows before May 1, 2021 (303,117 rows)
- **Validation set:** All rows from May 1, 2021 onward (34,581 rows)

The split script is `scripts/split_dataset.py`.

### Feature engineering pipeline

The data preparation code (`src/gemspot_training/data.py`) transforms the raw CSV into model-ready numeric features in these steps:

**Step 1: Drop non-features**
- Drop `user_id`, `gmap_id` (identifiers, not predictive)
- Drop `category` (redundant, already encoded in `category_encoded`)

**Step 2: Explode list-encoded columns**
- `category_encoded` (string like `"[0,0,1,...]"`) → 21 individual columns: `cat_enc_0`, `cat_enc_1`, ..., `cat_enc_20`
- `destination_vibe_tag` (string like `"[1,1,0,...]"`) → 20 individual columns: `vibe_0`, `vibe_1`, ..., `vibe_19`
- `user_personal_preferences` (string like `"[0.18, 0.25,...]"`) → 20 individual columns: `user_pref_0`, `user_pref_1`, ..., `user_pref_19`

**Step 3: Create interaction features**
- For each of the 20 vibe dimensions: `pref_x_vibe_i = user_pref_i * vibe_i`
- This captures when a user's preference aligns with a place's vibe attribute
- 20 interaction columns: `pref_x_vibe_0`, `pref_x_vibe_1`, ..., `pref_x_vibe_19`

**Step 4: Keep scalar features**
- `avg_rating` (float)
- `location_popularity` (int)
- `user_total_visits` (int)
- `time` (float, epoch milliseconds)

**Step 5: Preprocessing pipeline**
- Impute missing values with median
- Standard-scale all features (zero mean, unit variance)

### Final feature count

| Feature group | Count | Description |
|---------------|-------|-------------|
| Scalar numeric | 4 | avg_rating, location_popularity, user_total_visits, time |
| Category encoding | 21 | cat_enc_0 through cat_enc_20 (multi-hot) |
| Destination vibe tags | 20 | vibe_0 through vibe_19 (binary) |
| User preferences | 20 | user_pref_0 through user_pref_19 (float) |
| Preference x vibe interactions | 20 | pref_x_vibe_0 through pref_x_vibe_19 (float) |
| **Total** | **85** | All numeric features |

---

## 4. The Models

We train three model candidates from a single configurable training script (`src/train.py`). All models and hyperparameters are defined in `configs/candidates.yaml`.

### Candidate 1: baseline (DummyClassifier)

- **What it does:** Always predicts the majority class (`will_visit = 1`)
- **Why we include it:** Required by the rubric. Establishes a lower bound that every real model must beat. Any model with ROC-AUC = 0.5 is no better than random.
- **Params:** `strategy = most_frequent`

### Candidate 2: xgboost_v1 (XGBoost with defaults)

- **What it does:** Gradient-boosted decision tree ensemble with default/standard parameters
- **Why we include it:** Shows that XGBoost works on this data with no special tuning. This is the "starting point" to justify the tuning done in v2.
- **Key params:**
  - `learning_rate = 0.3` (default, fast convergence)
  - `max_depth = 6` (default tree depth)
  - `n_estimators = 100` (100 boosting rounds)
  - `scale_pos_weight = 1.0` (no imbalance correction)

### Candidate 3: xgboost_v2 (Tuned XGBoost)

- **What it does:** XGBoost with hyperparameters tuned for this specific dataset, including class imbalance handling
- **Why we include it:** Improves on v1 by addressing the 82/18 class imbalance and using regularization to prevent overfitting.
- **Key params:**
  - `learning_rate = 0.1` (slower, more precise convergence)
  - `max_depth = 8` (deeper trees to capture complex interactions)
  - `n_estimators = 200` (more boosting rounds)
  - `scale_pos_weight = 4.7` (ratio of negative to positive class, corrects imbalance)
  - `subsample = 0.9` (row subsampling for regularization)
  - `colsample_bytree = 0.8` (column subsampling for regularization)

### How XGBoost works (simplified)

XGBoost is a gradient-boosted decision tree ensemble:

1. Start with a simple prediction (e.g., the average of `will_visit`)
2. Build a decision tree that corrects the errors from step 1
3. Add that tree's predictions (scaled by `learning_rate`) to the current prediction
4. Build another tree that corrects the remaining errors
5. Repeat for `n_estimators` rounds

Each tree asks questions like "Is `avg_rating > 4.2`?" and "Is `user_pref_5 > 0.3`?" to split the data. The final prediction is the sum of all trees' outputs, passed through a sigmoid function to get a probability between 0 and 1.

**Why XGBoost for this task:**
- The dataset is structured/tabular (not images or text) — XGBoost excels at tabular data
- It handles the mix of continuous features (ratings, preferences) and binary features (category encoding, vibe tags) naturally
- Built-in handling of class imbalance via `scale_pos_weight`
- Fast training even on 300k+ rows

---

## 5. How Training Works End-to-End

### Single command

```bash
export PROJECT_SUFFIX=proj10
export MLFLOW_TRACKING_URI=http://YOUR_MLFLOW_IP:8000
bash scripts/run_training_container.sh
```

### What happens under the hood

1. **Docker builds** the training image from `Dockerfile` (Python 3.11-slim + all dependencies)
2. **Data loading:** `data.py` reads `gemspot_train.csv` and `gemspot_val.csv`, parses list-encoded columns, creates 85 numeric features
3. **For each candidate** in `candidates.yaml`:
   a. Build the sklearn Pipeline (imputer → scaler → model)
   b. Start an MLflow run
   c. Log hyperparameters, environment info, and GPU info to MLflow
   d. Fit the pipeline on training data and measure wall-clock time
   e. Predict on validation data
   f. Compute metrics (accuracy, precision, recall, F1, ROC-AUC, average precision)
   g. Log all metrics to MLflow
   h. Save the trained model as a `.joblib` artifact
   i. Log the model artifact to MLflow
4. **All runs** appear in the MLflow experiment `GemSpot-WillVisit`

### Metrics we track

| Metric | What it measures | Why it matters |
|--------|-----------------|----------------|
| accuracy | % of correct predictions | Basic measure, but misleading with class imbalance |
| precision | Of predicted positives, how many are correct | Avoid recommending places users won't visit |
| recall | Of actual positives, how many did we catch | Don't miss places users would visit |
| f1 | Harmonic mean of precision and recall | Balances both concerns |
| roc_auc | Area under ROC curve | Overall ranking quality, independent of threshold |
| average_precision | Area under precision-recall curve | Best metric for imbalanced data |
| train_seconds | Wall-clock training time | Cost metric for retraining decisions |
| rows_per_second | Training throughput | Scalability metric |

**Primary metrics:** `roc_auc` and `average_precision` — these are threshold-independent and handle class imbalance well.

### Cost metrics

| Metric | What it measures | Why it matters |
|--------|-----------------|----------------|
| train_seconds | Total training wall time | How long to wait for a retrain |
| rows_per_second | Data throughput | Will it scale to 10x more data? |
| system/cpu_utilization_percentage | CPU usage during training | Resource efficiency |
| system/system_memory_usage_megabytes | RAM usage | Will it fit on a smaller instance? |

---

## 6. How to Reproduce a Training Run

### Prerequisites

- Chameleon Cloud instance with Docker installed
- MLflow service running on Chameleon
- This repository cloned on the instance

### Steps

```bash
# 1. Clone and enter repo
git clone https://github.com/rohitshididnyu/GemSpot_training.git gemspot
cd gemspot

# 2. Build Docker image
export PROJECT_SUFFIX=proj10
docker build -t gemspot-train-${PROJECT_SUFFIX} .

# 3. Make sure initial_training_set.csv is in data/demo/
ls data/demo/initial_training_set.csv

# 4. Split into train/val (time-based)
docker run --rm -v "$(pwd):/app" gemspot-train-${PROJECT_SUFFIX} \
  python scripts/split_dataset.py

# 5. Set MLflow tracking URI
PRIVATE_IP=$(hostname -I | awk '{print $1}')
export MLFLOW_TRACKING_URI=http://${PRIVATE_IP}:8000

# 6. Run training
unset DOCKER_EXTRA_ARGS
bash scripts/run_training_container.sh

# 7. View results
# Open http://YOUR_FLOATING_IP:8000 in browser
# Click experiment "GemSpot-WillVisit"
# Compare the 3 runs
```

### Changing hyperparameters

Edit `configs/candidates.yaml` and rerun. No code changes needed. Example — to add an xgboost_v3 with even more trees:

```yaml
  - name: xgboost_v3
    kind: xgboost
    notes: More boosting rounds with early stopping potential.
    params:
      learning_rate: 0.05
      max_depth: 6
      n_estimators: 500
      scale_pos_weight: 4.7
      subsample: 0.8
      colsample_bytree: 0.7
      random_state: 42
      n_jobs: -1
```

---

## 7. Repository Structure

```
.
├── configs/
│   └── candidates.yaml          # Model candidates and hyperparameters
├── data/
│   └── demo/
│       ├── initial_training_set.csv  # Raw full dataset (337k rows)
│       ├── gemspot_train.csv         # Train split (303k rows, before May 2021)
│       └── gemspot_val.csv           # Val split (34k rows, May 2021 onward)
├── docs/
│   └── training_documentation.md    # This file
├── interfaces/
│   ├── will_visit_input.sample.json  # Sample model input
│   └── will_visit_output.sample.json # Sample model output
├── scripts/
│   ├── split_dataset.py             # Time-based train/val split
│   ├── run_training_container.sh    # Docker training runner
│   ├── export_run_table.py          # Generate run table from MLflow
│   └── make_demo_dataset.py         # Synthetic data generator (fallback)
├── src/
│   ├── train.py                     # Main training entrypoint
│   └── gemspot_training/
│       ├── data.py                  # Data loading and feature engineering
│       ├── training.py              # Model building and metrics
│       └── utils.py                 # Environment info, git SHA
├── Dockerfile                       # Training container
└── requirements.txt                 # Python dependencies
```

---

## 8. Results Summary

Results from local testing (time-based split, 303k train / 34k val, 85 features):

| Candidate | Accuracy | F1 | ROC-AUC | Avg Precision | Train Time | Notes |
|-----------|----------|-----|---------|---------------|------------|-------|
| baseline | 0.843 | 0.915 | 0.500 | 0.843 | 1.6s | Lower bound — just predicts majority class |
| xgboost_v1 | 0.861 | 0.922 | 0.835 | 0.964 | 2.4s | Default params — proves model architecture works |
| **xgboost_v2** | **0.856** | **0.921** | **0.836** | **0.965** | 3.8s | **Tuned — best avg precision + 99.5% recall** |

### Why xgboost_v2 is the best candidate

- Highest **average precision** (0.965) — the best metric for imbalanced classification
- Highest **recall** (99.5%) — catches nearly every place a user would visit
- **ROC-AUC** (0.836) slightly better than v1 — better overall ranking quality
- Training cost is modest (3.8 seconds on 303k rows) — can retrain frequently
- `scale_pos_weight=4.7` correctly addresses the 82/18 class imbalance

### Why xgboost_v1 is still worth considering

- Trains faster (2.4s vs 3.8s)
- Higher precision (0.876 vs 0.858) — fewer false recommendations
- May be preferred if serving speed matters more than recall

### Why baseline exists

- Proves that XGBoost models are actually learning (ROC-AUC jumps from 0.50 to 0.84)
- Required by the course rubric as a simple baseline
