# Submission Checklist

Use this checklist before you submit anything to Gradescope.

## A. Joint Team Artifacts

These are shared with your teammates.

- `interfaces/will_visit_input.sample.json`
- `interfaces/will_visit_output.sample.json`

If your team is officially presenting two separate models at this stage, also include:

- `interfaces/vibe_tag_input.sample.json`
- `interfaces/vibe_tag_output.sample.json`

What these files do:

- the data teammate produces something shaped like the input JSON
- the training teammate trains on data derived from that schema
- the inference teammate serves predictions shaped like the output JSON

## B. Your Training Deliverables

### 1. Written document

Create a PDF called something like `gemspot_training_report.pdf`.

Include:

- your name and role: training
- short project description
- MLflow URL
- training runs table
- which candidates are promising and why
- next experiment you would try

### 2. Repository artifacts

Make sure these exist in your repo:

- `Dockerfile`
- `requirements.txt`
- `src/train.py`
- `src/gemspot_training/data.py`
- `src/gemspot_training/training.py`
- `src/gemspot_training/utils.py`
- `configs/candidates.yaml`

Optional but helpful:

- `scripts/make_demo_dataset.py`
- `scripts/start_mlflow_server.sh`
- `scripts/run_training_container.sh`
- `scripts/export_run_table.py`

### 3. Demo video

Record a short screen capture showing:

- the training container starting on Chameleon
- MLflow already running on Chameleon
- a full training run completing
- the new runs appearing in MLflow

### 4. Live service

Keep the MLflow server live on Chameleon so course staff can open it.

What course staff should be able to do:

- open MLflow in a browser
- inspect runs
- compare metrics
- download artifacts

## B2. Bonus Items If You Use The ML6.2 Path

If you plan to claim bonus credit for the Ray path, prepare evidence for these:

- Ray dashboard screenshot or video segment
- checkpoint files visible in object storage or MinIO
- job submission command and successful completion logs
- if demonstrating fault tolerance, evidence that a resumed job continued after worker interruption
- short explanation of how Ray improved robustness or tuning efficiency for GemSpot

## C. What Must Be True To Get Credit

Make sure all of these are true:

- all training runs were executed on Chameleon
- training ran inside Docker containers
- every run is in MLflow
- you used one configurable training script, not separate scripts for every candidate
- you logged params, metrics, cost metrics, and environment info
- you included at least one simple baseline and at least one stronger candidate

## D. Suggested Gradescope Packaging

Because course forms vary, prepare these items in advance:

- PDF report
- video link
- repo link plus commit SHA
- MLflow URL

If there is a free-text explanation box, paste:

- the public MLflow URL
- the repo URL
- the exact commit SHA you want graded
- a sentence explaining which run or runs are your best candidates
- if claiming bonus, a sentence explaining which Ray run demonstrates robustness or early-stopping benefits

## E. Final 5-Minute Sanity Check

Right before submission, verify:

```bash
docker ps
```

You should see your MLflow container.

Open the MLflow URL in a browser and confirm:

- the page loads
- the experiment is visible
- runs are not empty
- artifact links work

Then check your repo tree:

```bash
find . -maxdepth 3 -type f | sort
```
