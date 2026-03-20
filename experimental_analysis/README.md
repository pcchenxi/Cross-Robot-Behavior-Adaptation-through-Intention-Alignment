# Experimental Analysis

This directory preserves the paper artifacts and analysis scripts for:

> “Cross-Robot Behavior Adaptation through Intention Alignment”

## Directory structure

```text
experimental_analysis/
├── realrobot-result/
│   └── real-result.xlsx
├── latent-result/
├── simulation-result/
├── print_latent_analysis.py
├── print_simulation_stats.py
└── README.md
```

## Real-world experiments

`realrobot-result/real-result.xlsx` records 30 real-world imitation scenarios and is used for Tables 1 and 2 in the paper.

## Latent intention-space analysis

`latent-result/` contains the latent embeddings used for Table 3.

```bash
python print_latent_analysis.py
```

## Simulation experiments (`simulation-result/`)

The folder `simulation-result/` stores evaluation results for two simulation tasks: monitoring and item picking.

The script `print_simulation_stats.py` reads the precomputed results and prints the test statistics used to construct Tables S3-S6.

### Table S3

```bash
python print_simulation_stats.py --task navigation --method usc
```

### Table S4

```bash
python print_simulation_stats.py --task navigation --method language
```

### Table S5

```bash
python print_simulation_stats.py --task mp1 --method usc
```

### Table S6

```bash
python print_simulation_stats.py --task mp1 --method language
```
