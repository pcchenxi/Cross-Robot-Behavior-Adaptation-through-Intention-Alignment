# Cross-Robot-Behavior-Adaptation-through-Intention-Alignment


This repository contains the **experimental data and analysis scripts** used in the paper:

> **“Cross-Robot Behavior Adaptation through Intention Alignment”**

---

## Repository structure

```text
.
├── realrobot-result/
│   └── real-result.xlsx         # 30 real-world scenarios and imitation outcomes (Tables 1 & 2)
├── latent-result/                 # Latent embeddings for intention-space analysis (Table 3)
├── simulation-result/           # Simulation evaluation logs (Tables 6–9)
├── print_simulation_stats.py    # Script to summarize test statistics for simulation tasks
├── paper_latent_distance_table.py # Script to perform latent space analysis
├── LICENSE
└── README.md
```

---

## Real-world experiments (`realrobot-result/`)

The folder `realrobot-result/` contains:

- **`real-result.xlsx`**  
  This file records **30 real-world imitation scenarios**, including:

  - Motions performed by the **three demonstrator robots** in the team  
  - The **scenario type**  
  - The **learner robots** presented in each scenario  
  - The **motions performed by the learner robots** given the demonstration  
  - The **imitation result** (success / outcome)

These data are used to construct **Table 1** and **Table 2** in the paper.


---

## Latent intention-space analysis (`latent-result/`)

The folder `latent-result/` contains the **latent embeddings of 120 samples** used to create **Table 3**, which analyzes the structure of the learned intention space.

To compute:

- **Intra-class distance**
- **Cross-embodiment error**
- **Global inter-class distance**

run:

```bash
python print_latent_analysis.py
```

---

## Simulation experiments (`simulation-result/`)

The folder `simulation-result/` stores evaluation results for two simulation tasks: monitoring and item picking

The script `print_simulation_stats.py` reads the precomputed results and prints the **test statistics** used to construct **Tables 6–9**.


### Table 6

```bash
python print_simulation_stats.py --task navigation --method usc
```

### Table 7

```bash
python print_simulation_stats.py --task navigation --method language
```

### Table 8

```bash
python print_simulation_stats.py --task mp1 --method usc
```

### Table 9

```bash
python print_simulation_stats.py --task mp1 --method language
```