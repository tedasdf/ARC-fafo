# Training

Install the shared training dependencies from the repository root:

~~~powershell
python -m pip install -r src/requirements.txt
~~~

The standard trainer, Modal image, and scoring defaults use files under `src`. The deferred `compressarc.analysis.solve_task` worker still needs the legacy folder.

From the repository root, run the standard trainer locally:

~~~powershell
python src/train.py --task <task-id> --set training.iterations=500
~~~

Omit --task to train every task in the configured split. The defaults are in src/config/default.yaml; use --config for another YAML file and repeat --set KEY=VALUE for one-off overrides.

## Run on Modal

Install the Modal CLI in the local Python environment and authenticate once:

~~~powershell
pip install modal
modal setup
~~~

Then use the same trainer and config with the Modal backend:

~~~powershell
python src/train.py --backend modal --task <task-id> --set training.iterations=500
~~~

Omit --task to run the full configured split remotely. Modal uses a T4 GPU by default; choose another supported GPU with --modal-gpu, for example --modal-gpu A100.

To save checkpoints, add --save-checkpoints. The remote run returns checkpoint files to the local --output-dir (default: outputs/training).

Use model.multitensor_constraints=strict (the default) or model.multitensor_constraints=relaxed to select the preprocessing policy. When W&B is enabled in the config, the Modal run expects a Modal secret named wandb-secret containing WANDB_API_KEY:

~~~powershell
modal secret create wandb-secret WANDB_API_KEY=<your-wandb-api-key>
~~~

The Modal backend runs the resolved configuration remotely; local execution remains the default.