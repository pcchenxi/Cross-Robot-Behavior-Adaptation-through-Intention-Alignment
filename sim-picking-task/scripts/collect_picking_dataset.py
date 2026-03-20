"""Data collection script for IAIL simulated picking."""

import os
import random
from pathlib import Path

import hydra
import numpy as np

from iail_sim_picking import tasks
from iail_sim_picking.dataset import RavensDataset
from iail_sim_picking.envs import Environment


@hydra.main(config_path="../iail_sim_picking/configs", config_name="data")
def main(cfg):
    root_dir = Path(__file__).resolve().parents[1]
    assets_root = root_dir / "iail_sim_picking" / "assets"

    env = Environment(
        str(assets_root),
        disp=False,
        shared_memory=cfg["shared_memory"],
        hz=480,
        record_cfg=cfg["record"],
    )
    if cfg["disp"]:
        env.enable_live_display(interval=30)
    task = tasks.names[cfg["task"]]()
    task.mode = cfg["mode"]

    data_path = os.path.join("./data", f"{cfg['task']}-{task.mode}")
    dataset = RavensDataset(data_path, cfg, n_demos=0, augment=False)
    agent = task.oracle(env)
    record = cfg["record"]["save_video"]

    print(f"Saving to: {data_path}")
    print(f"Mode: {task.mode}")

    seed = dataset.max_seed
    if seed < 0:
        if task.mode == "train":
            seed = -2
        elif task.mode == "val":
            seed = -1
        elif task.mode == "test":
            seed = 14999
        else:
            raise ValueError("Invalid mode. Valid options: train, val, test")

    count = 0
    while dataset.n_episodes < cfg["n"]:
        episode, total_reward = [], 0
        if count % 4 == 0:
            seed += 2
        count += 1

        np.random.seed(seed)
        random.seed(seed)
        print(f"Oracle demo: {dataset.n_episodes + 1}/{cfg['n']} | Seed: {seed}")

        env.set_task(task)
        obs, info = env.reset()
        reward = 0

        if task.mode == "val" and seed > 9999:
            raise ValueError("Val seeds overlap with test seeds.")

        if record:
            env.start_rec(f"{dataset.n_episodes + 1:06d}")

        for _ in range(task.max_steps):
            act = agent.act(obs, info)
            if act is None:
                break
            episode.append((obs, act, reward, info))
            lang_goal = info["lang_goal"]
            obs, reward, done, info = env.step(act)
            total_reward += reward
            print(f"Total Reward: {total_reward:.3f} | Done: {done} | Goal: {lang_goal}")
            break

        episode.append((obs, None, reward, info))
        
        if record:
            env.end_rec()
        if cfg["save_data"] and total_reward > 0:
            dataset.add(seed, episode)

    if cfg["disp"]:
        env.disable_live_display()


if __name__ == "__main__":
    main()

# python scripts/collect_picking_dataset.py n=100 mode=train task=ur5f disp=True