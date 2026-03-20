#!/bin/bash

python scripts/run_evaluation_online.py --demo-task-name ur5f --learner-task-name ur5f --learner-mode test --num-episodes 100 --num-candidates 10 --candidate-batch-size 200 --device cuda
python scripts/run_evaluation_online.py --demo-task-name ur5f --learner-task-name ur5l --learner-mode test --num-episodes 100 --num-candidates 10 --candidate-batch-size 200 --device cuda
python scripts/run_evaluation_online.py --demo-task-name ur5f --learner-task-name ur5r --learner-mode test --num-episodes 100 --num-candidates 10 --candidate-batch-size 200 --device cuda
python scripts/run_evaluation_online.py --demo-task-name ur5r --learner-task-name ur5f --learner-mode test --num-episodes 100 --num-candidates 10 --candidate-batch-size 200 --device cuda
python scripts/run_evaluation_online.py --demo-task-name ur5r --learner-task-name ur5l --learner-mode test --num-episodes 100 --num-candidates 10 --candidate-batch-size 200 --device cuda
python scripts/run_evaluation_online.py --demo-task-name ur5r --learner-task-name ur5r --learner-mode test --num-episodes 100 --num-candidates 10 --candidate-batch-size 200 --device cuda
python scripts/run_evaluation_online.py --demo-task-name ur5l --learner-task-name ur5f --learner-mode test --num-episodes 100 --num-candidates 10 --candidate-batch-size 200 --device cuda
python scripts/run_evaluation_online.py --demo-task-name ur5l --learner-task-name ur5l --learner-mode test --num-episodes 100 --num-candidates 10 --candidate-batch-size 200 --device cuda
python scripts/run_evaluation_online.py --demo-task-name ur5l --learner-task-name ur5r --learner-mode test --num-episodes 100 --num-candidates 10 --candidate-batch-size 200 --device cuda