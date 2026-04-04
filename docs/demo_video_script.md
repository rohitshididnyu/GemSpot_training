# Demo Video Script

Keep the demo short and simple. Aim for 2 to 4 minutes.

## What To Show

### Part 1. Show MLflow is live

In a browser:

- open `http://YOUR_FLOATING_IP:5000`
- show the GemSpot experiment page

In terminal:

```bash
docker ps
```

Point out the MLflow container name, for example `gemspot-mlflow-proj99`.

### Part 2. Show the training container starting

On the training machine:

```bash
cd gemspot
export PROJECT_SUFFIX=proj99
export MLFLOW_TRACKING_URI=http://YOUR_FLOATING_IP:5000
bash scripts/run_training_container.sh
```

Say:

- this is running inside Docker on Chameleon
- the tracking URI points to the live MLflow service
- the script is training multiple candidates from one config file

### Part 3. Show the runs appear in MLflow

Refresh the browser on the MLflow page.

Show:

- new runs appearing
- metrics for each candidate
- artifacts logged for a run

### Part 4. Show one promising run

Open one run and point to:

- candidate name
- key hyperparameters
- main metrics
- training time

Then say why you think it is promising.

## Optional Bonus Clip

If you use the ML6.2 bonus path, add a short extra clip showing:

- the Ray dashboard
- the Ray Tune job output with multiple trials
- checkpoint files appearing in storage
- if applicable, a worker/container restart followed by resumed progress

## What To Say In One Sentence

"For the GemSpot training subsystem, I trained multiple recommendation candidates in Docker on Chameleon, tracked every run in MLflow, and compared quality and cost so the team can choose a model instead of shipping a single untracked model file."
