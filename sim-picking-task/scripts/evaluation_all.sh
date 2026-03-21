#!/bin/bash

python scripts/run_evaluation_online.py --demo-task-name ur5f --learner-task-name ur5f --learner-mode test --num-episodes 500 --num-candidates 50 --candidate-batch-size 500 --device cuda
python scripts/run_evaluation_online.py --demo-task-name ur5f --learner-task-name ur5l --learner-mode test --num-episodes 500 --num-candidates 50 --candidate-batch-size 500 --device cuda
python scripts/run_evaluation_online.py --demo-task-name ur5f --learner-task-name ur5r --learner-mode test --num-episodes 500 --num-candidates 50 --candidate-batch-size 500 --device cuda
python scripts/run_evaluation_online.py --demo-task-name ur5r --learner-task-name ur5f --learner-mode test --num-episodes 500 --num-candidates 50 --candidate-batch-size 500 --device cuda
python scripts/run_evaluation_online.py --demo-task-name ur5r --learner-task-name ur5l --learner-mode test --num-episodes 500 --num-candidates 50 --candidate-batch-size 500 --device cuda
python scripts/run_evaluation_online.py --demo-task-name ur5r --learner-task-name ur5r --learner-mode test --num-episodes 500 --num-candidates 50 --candidate-batch-size 500 --device cuda
python scripts/run_evaluation_online.py --demo-task-name ur5l --learner-task-name ur5f --learner-mode test --num-episodes 500 --num-candidates 50 --candidate-batch-size 300 --device cuda
python scripts/run_evaluation_online.py --demo-task-name ur5l --learner-task-name ur5l --learner-mode test --num-episodes 500 --num-candidates 50 --candidate-batch-size 300 --device cuda
python scripts/run_evaluation_online.py --demo-task-name ur5l --learner-task-name ur5r --learner-mode test --num-episodes 500 --num-candidates 50 --candidate-batch-size 300 --device cuda