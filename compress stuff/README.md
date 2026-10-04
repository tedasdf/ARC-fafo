<a href="https://iliao2345.github.io/blog_posts/arc_agi_without_pretraining/arc_agi_without_pretraining.html"><img src="teaser_figure_w_title.png"></a>
This is the code base for the ARC-AGI Without Pretraining project. The Kaggle competition template version is available from the project page.

The new model and trainer live under src/. The old inference, scoring, and Modal batch scripts remain here; see LEGACY_TOOLS.md for their dependencies.

# Installation

For the current training workflow, install the dependencies and commands documented in ../src/TRAINING.md.

The legacy tools in this folder use requirements.txt:

~~~powershell
python -m pip install -r "compress stuff/requirements.txt"
~~~

# Train with the current model

See ../src/TRAINING.md for configuration, W&B, local training, and the Modal backend.

Example:

~~~powershell
python src/train.py --task 272f95fa --set training.iterations=500
~~~

# Run the legacy Modal strict-vs-relaxed ablation

The retained Modal batch runner now invokes src/train.py for each task and constraint setting. Install and authenticate the Modal CLI, then create the W&B secret:

~~~powershell
pip install modal
modal setup
modal secret create wandb-secret WANDB_API_KEY=<your-wandb-api-key>
~~~

Launch the default experiment:

~~~powershell
modal run "compress stuff/modal_runner.py"
~~~

For a short smoke test, run two tasks for 20 iterations:

~~~powershell
modal run "compress stuff/modal_runner.py" --number-of-tasks 2 --iterations 20
~~~

# Analysis utilities

Analysis modules live in src/compressarc/analysis. Run them from the src directory:

~~~powershell
cd src
python -m compressarc.analysis.plot_problems --split evaluation
~~~

list_solved_puzzles and plot_accuracy preserve compatibility with old predictions_*.npz histories written by the legacy solution_selection module. The new trainer does not currently create those history files.

~~~powershell
python -m compressarc.analysis.list_solved_puzzles "../compress stuff/results_for_the_blog_post/predictions_training.npz" training 2000
python -m compressarc.analysis.plot_accuracy "../compress stuff/results_for_the_blog_post/predictions_training.npz" training
~~~

# Legacy source

The remaining old inference and scoring tools are documented in LEGACY_TOOLS.md. Current model layers, configuration, initialization, and training code are under src/compressarc.

# Citation

If you'd like to cite this blog post, use the following entry:
```
@online{liao2025arcagiwithoutpretraining,
	author = {Isaac Liao and Albert Gu},
	title = {ARC-AGI Without Pretraining},
	year = {2025},
	url = {https://iliao2345.github.io/blog_posts/arc_agi_without_pretraining/arc_agi_without_pretraining.html},
}
```
