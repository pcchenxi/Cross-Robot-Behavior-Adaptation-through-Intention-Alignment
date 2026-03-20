
## Sim Picking Task

Standalone code for the IAIL simulated picking task used in the paper.

### Directory structure

```text
sim-picking-task/
├── iail_sim_picking/
├── scripts/
├── pyproject.toml
├── setup.py
└── README.md
```

### Install

This environment is adapted from [CLIPort](https://github.com/cliport/cliport). For dependency setup, simulator prerequisites, and the general runtime environment, please follow the CLIPort installation and usage instructions.

### How to Run

Run the pipeline in the following order:

1. collect dataset
2. train motion generator
3. train intention extractor
4. run online evaluation

#### 1. Collect dataset

Collect demonstration data first. The dataset should cover all three tasks: `ur5f`, `ur5l`, and `ur5r`. Each task should include both `train` and `test` splits, written under `./data/<task>-<mode>/`.

In the paper setting, we collected:

- 10,000 episodes per task for training
- 500 episodes per task for testing

```bash
cd sim-picking-task
python scripts/collect_picking_dataset.py n=10000 mode=train task=ur5f
python scripts/collect_picking_dataset.py n=500 mode=test task=ur5f
python scripts/collect_picking_dataset.py n=10000 mode=train task=ur5l
python scripts/collect_picking_dataset.py n=500 mode=test task=ur5l
python scripts/collect_picking_dataset.py n=10000 mode=train task=ur5r
python scripts/collect_picking_dataset.py n=500 mode=test task=ur5r
```

#### 2. Train motion generator

Train the motion generator after dataset collection. You need to train a separate motion generator for each task: `ur5f`, `ur5l`, and `ur5r`.

```bash
cd sim-picking-task
python scripts/train_motion_generator.py --data-root data --task-name ur5f --device cuda:0
python scripts/train_motion_generator.py --data-root data --task-name ur5l --device cuda:0
python scripts/train_motion_generator.py --data-root data --task-name ur5r --device cuda:0
```

#### 3. Train intention extractor

After the three task-specific motion generators have been trained, train the intention extractor jointly across `ur5f`, `ur5l`, and `ur5r` with a shared annotation encoder. This stage contains three task-specific image-action encoders plus one shared text/annotation encoder, and all four components are saved into a single checkpoint.

Pay attention to the batch size. In our experiments, an NVIDIA RTX 4090 could use `--batch-size 50`.

```bash
cd sim-picking-task
python scripts/train_intention_extractor.py --data-root data --batch-size 50 --device cuda:0
```

#### 4. Run online evaluation

Run online evaluation after all the models have been trained. We sample `--num-episodes` demonstrations from the demo task and ask the learner to imitate them. Each imitation attempt consists of multiple rounds of action sampling.

In each round, the learner samples `--candidate-batch-size` actions from the motion generator and evaluates them against the demonstration signal. The system keeps the valid candidates that pass the filtering criteria until it collects `--num-candidates` qualified samples, or until it reaches 10 rounds.

If the motion generator quality is high, the number of sampled actions can be reduced.

```bash
cd sim-picking-task
python scripts/run_evaluation_online.py --demo-task-name ur5f --learner-task-name ur5l --num-episodes 500 --num-candidates 50 --candidate-batch-size 500
```

Evaluation results will be saved under `results/evaluation/` with filenames in the format `learnertask-demotask.csv`.

### Maintenance

Ongoing maintenance for the simulated picking module will continue in the main repository:

https://github.com/pcchenxi/Cross-Robot-Behavior-Adaptation-through-Intention-Alignment

For any questions, please contact pcchenxi@gmail.com.
