#!/bin/bash

python scripts/train_motion_generator.py --data-root data_paper --task-name ur5f --action-dim 3 --device cuda:0 --persistent-workers
python scripts/train_motion_generator.py --data-root data_paper --task-name ur5l --action-dim 3 --device cuda:0 --persistent-workers
python scripts/train_motion_generator.py --data-root data_paper --task-name ur5r --action-dim 3 --device cuda:0 --persistent-workers
python scripts/train_intention_extractor.py --persistent-workers