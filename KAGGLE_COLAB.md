# Run the GroupARC ablation on Kaggle or Google Colab

This guide runs the same three strict-model conditions as the Modal experiment:

- `original`
- `projected_conv`: separate input/output projections around the tied convolution
- `pure_conv`: the tied group convolution without projections or a residual

The eight tasks are the original strict CompressARC runs that achieved pass@2:

```text
0dfd9992 22eb0ac0 239be575 50cb2852 54d82841 623ea044 c1d99e64 ec883f72
```

Each condition runs for 1,500 steps and writes to the W&B project
`arc_agi/grouparc`. Runs record parameter counts, kernel values, first-solve
step, stable-solve step, wall-clock-to-solve, and estimated FLOPs-to-solve.
There is no early stopping.

## Before uploading the code

The upstream `iliao2345/CompressARC` repository does not contain the local
group-convolution changes yet. Use this modified checkout. Either push it to a
Git branch you can clone from the notebook, or zip the complete `CompressARC`
folder and upload it to Kaggle/Colab. Confirm the uploaded copy contains:

```text
analyze_example_wandb.py
tied_convolution.py
arc_compressor.py
layers.py
dataset/
```

Do not upload a W&B API key in the archive.

## Kaggle

Create a notebook, enable a GPU under **Settings > Accelerator**, and disable
Internet only if the code and required packages are already supplied as Kaggle
datasets.

Upload the modified repository as a Kaggle Dataset. Because `/kaggle/input` is
read-only, copy it into the working directory. Replace the placeholder dataset
path with the path shown in the notebook's Input panel:

```python
from pathlib import Path
import shutil

source = Path("/kaggle/input/YOUR-DATASET/CompressARC")
destination = Path("/kaggle/working/CompressARC")
if destination.exists():
    shutil.rmtree(destination)
shutil.copytree(source, destination)
%cd /kaggle/working/CompressARC
```

Alternatively, clone a branch containing these changes:

```python
!git clone --branch YOUR_BRANCH https://github.com/YOUR_ACCOUNT/YOUR_REPOSITORY.git /kaggle/working/CompressARC
%cd /kaggle/working/CompressARC
```

Add `WANDB_API_KEY` under **Add-ons > Secrets**, enable it for the notebook,
then load it without printing it:

```python
import os
from kaggle_secrets import UserSecretsClient

os.environ["WANDB_API_KEY"] = UserSecretsClient().get_secret("WANDB_API_KEY")
```

Kaggle already supplies a CUDA build of PyTorch. Keep that build and install
only the tracking/runtime packages:

```python
!python -m pip install -q wandb==0.28.1 tqdm==4.66.6
```

## Google Colab

Create a notebook and select **Runtime > Change runtime type > GPU**. Upload the
modified repository archive and extract it, mount a Drive folder containing the
checkout, or clone your own branch. For a Drive checkout:

```python
from google.colab import drive
drive.mount("/content/drive")
%cd /content/drive/MyDrive/CompressARC
```

For a Git branch containing the changes:

```python
!git clone --branch YOUR_BRANCH https://github.com/YOUR_ACCOUNT/YOUR_REPOSITORY.git /content/CompressARC
%cd /content/CompressARC
```

Add `WANDB_API_KEY` in Colab's **Secrets** panel, grant notebook access, and
load it without printing it:

```python
import os
from google.colab import userdata

os.environ["WANDB_API_KEY"] = userdata.get("WANDB_API_KEY")
```

Keep Colab's CUDA-enabled PyTorch installation:

```python
!python -m pip install -q wandb==0.28.1 tqdm==4.66.6
```

## Verify the runtime

Run this cell on either platform:

```python
from pathlib import Path
import torch

assert Path("analyze_example_wandb.py").exists(), "Change into the CompressARC directory"
assert Path("tied_convolution.py").exists(), "This is not the modified GroupARC checkout"
assert torch.cuda.is_available(), "Enable a GPU accelerator before training"
print(torch.cuda.get_device_name(0))
```

## Run all tasks and variants

This launches one condition at a time in a fresh Python process. Completed runs
are uploaded immediately to W&B, so if the hosted session ends, remove the
completed task/variant pairs from the lists and rerun the cell.

```python
import os
import subprocess
import sys

TASKS = [
    "0dfd9992", "22eb0ac0", "239be575", "50cb2852",
    "54d82841", "623ea044", "c1d99e64", "ec883f72",
]
VARIANTS = ["original", "projected_conv", "pure_conv"]
PLATFORM_TAG = "kaggle"  # Change to "colab" on Google Colab.

assert os.environ.get("WANDB_API_KEY"), "Configure WANDB_API_KEY first"

for task_id in TASKS:
    for variant in VARIANTS:
        command = [
            sys.executable,
            "analyze_example_wandb.py",
            "--split", "training",
            "--task", task_id,
            "--iterations", "1500",
            "--multitensor-constraints", "strict",
            "--shift-variant", variant,
            "--wandb",
            "--wandb-entity", "arc_agi",
            "--wandb-project", "grouparc",
            "--wandb-tags", "group-convolution", "convolution-ablation", PLATFORM_TAG,
        ]
        print(f"Starting {task_id} / {variant}", flush=True)
        subprocess.run(command, check=True)
```

To run a smaller batch, shorten `TASKS` or `VARIANTS`. To reduce post-training
work, add `"--skip-pca"` to `command`; this does not change training or the
solve-time metrics.

Results appear at <https://wandb.ai/arc_agi/grouparc>. Filter with the `kaggle`
or `colab` tag to distinguish hosted runs from Modal runs.

## Hosted-runtime limits

Twenty-four 1,500-step conditions may exceed a single Kaggle or Colab session.
Run subsets across sessions and use W&B as the durable record. Keep the browser
connected when the provider requires it, and do not assume a notebook cell will
survive a runtime reset. Re-running a completed pair creates another W&B run
with the same display name but a different run ID.
