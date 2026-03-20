"""Online evaluation runner for runtime association."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from iail_sim_picking import tasks
from iail_sim_picking.dataset.object_metadata import object_class
from iail_sim_picking.envs import Environment
from iail_sim_picking.runtime_associator.common import DEFAULT_ASSETS_ROOT
from iail_sim_picking.runtime_associator.types import EpisodeResult

DEFAULT_RECORD_CFG = {
    "save_video": False,
    "save_video_path": "./data/videos",
    "add_text": False,
    "fps": 20,
    "video_height": 640,
    "video_width": 720,
}
PREPICK_Z_OFFSET = np.float32(0.32)


def _trace(message):
    print(f"[Runner] {message}", flush=True)


def _trace_banner(message):
    print(f"{'=' * 24} {message} {'=' * 24}", flush=True)


def _goal_class(label):
    if label in object_class and object_class[label]:
        return object_class[label][0]
    return None


@dataclass
class SuccessResult:
    success: bool
    match_type: str
    goal_obj: str
    picked_obj: str


class ImitationSuccessScorer:
    def __init__(self, action_threshold=0.0):
        self.action_threshold = float(action_threshold)

    def _task_type(self, goal_obj, all_obj):
        if goal_obj in all_obj:
            return "exact"

        goal_obj_class = _goal_class(goal_obj)
        for obj in all_obj:
            if _goal_class(obj) == goal_obj_class:
                return "same_class"
        return "absent"

    def score(self, lang_goal, picked_obj, all_obj, decision):
        picked_obj = "none" if picked_obj is None else str(picked_obj)
        goal_obj = str(lang_goal)
        all_obj = [str(obj) for obj in all_obj]
        task_type = self._task_type(goal_obj, all_obj)

        success = False
        match_type = "wrong_object"
        if task_type == "exact":
            success = (
                decision.demo_similarity >= self.action_threshold
                and picked_obj == goal_obj
            )
            match_type = "exact" if success else "wrong_object"
        elif task_type == "same_class":
            success = (
                decision.demo_similarity >= self.action_threshold
                and _goal_class(picked_obj) is not None
                and _goal_class(picked_obj) == _goal_class(goal_obj)
            )
            match_type = "same_class" if success else "wrong_object"
        else:
            success = not decision.accepted
            match_type = "rejected" if success else "wrong_object"

        return SuccessResult(
            success=bool(success),
            match_type=match_type,
            goal_obj=goal_obj,
            picked_obj=picked_obj,
        )


class Runner:
    def __init__(
        self,
        *,
        demonstrator,
        learner,
        assets_root=DEFAULT_ASSETS_ROOT,
        environment=None,
        scorer=None,
        hz=480,
        learner_mode="test",
        env_display=False,
        record_cfg=None,
        live_display_interval=20,
        save_learner_image_dir=None,
    ):
        self.demonstrator = demonstrator
        self.learner = learner
        self.learner_mode = learner_mode
        self.scorer = ImitationSuccessScorer() if scorer is None else scorer
        self.task_cls = tasks.names[learner.task_name]
        effective_record_cfg = {**DEFAULT_RECORD_CFG, **(record_cfg or {})}
        if environment is None:
            environment = Environment(
                str(assets_root),
                disp=False,
                shared_memory=False,
                hz=hz,
                record_cfg=effective_record_cfg,
            )
        elif getattr(environment, "record_cfg", None) is None:
            environment.record_cfg = effective_record_cfg
        if env_display:
            environment.enable_live_display(interval=live_display_interval)
        self.environment = environment
        self.save_learner_image_dir = (
            Path(save_learner_image_dir) if save_learner_image_dir is not None else None
        )

    def _reset_environment(self):
        task = self.task_cls()
        task.mode = self.learner_mode
        self.environment.set_task(task)
        _trace(f"environment task set, mode={task.mode}; calling reset()")
        return self.environment.reset()

    def _object_relation(self, goal_obj, candidate_obj):
        goal_obj = str(goal_obj)
        candidate_obj = "none" if candidate_obj is None else str(candidate_obj)
        if candidate_obj == goal_obj:
            return "same item"

        goal_obj_class = _goal_class(goal_obj)
        candidate_obj_class = _goal_class(candidate_obj)
        if goal_obj_class is not None and candidate_obj_class == goal_obj_class:
            return "same class item"
        return "no match"

    def _episode_result(self, goal_obj, picked_obj, accepted):
        if not accepted:
            return "no match"

        object_relation = self._object_relation(goal_obj, picked_obj)
        if object_relation in {"same item", "same class item"}:
            return object_relation
        return "failed"

    def _scene_relation(self, goal_obj, scene_objects):
        scene_objects = [str(obj) for obj in scene_objects]
        if str(goal_obj) in scene_objects:
            return "same item"

        goal_obj_class = _goal_class(goal_obj)
        if goal_obj_class is None:
            return "no match"

        for scene_obj in scene_objects:
            if _goal_class(scene_obj) == goal_obj_class:
                return "same class item"
        return "no match"

    def _relation_score(self, relation):
        if relation == "same item":
            return 1.0
        if relation == "same class item":
            return 0.5
        if relation == "no match":
            return 0.0
        return -1.0

    def _trace_running_score_summary(self, episode_results):
        total = len(episode_results)
        average_episode_score = sum(
            result.episode_score for result in episode_results
        ) / max(total, 1)
        average_optimal_score = sum(
            result.optimal_score for result in episode_results
        ) / max(total, 1)
        _trace(
            f"overall average episode score={average_episode_score:.4f}, "
            f"overall average optimal score={average_optimal_score:.4f}"
        )

    def _trace_episode_type_match_summary(self, episode_results):
        def _match_ratio(target_type, target_result):
            matched_type_results = [
                result for result in episode_results if result.episode_type == target_type
            ]
            total = len(matched_type_results)
            ratio = sum(
                result.episode_result == target_result
                for result in matched_type_results
            ) / max(total, 1)
            return total, ratio

        same_item_count, same_item_ratio = _match_ratio("same item", "same item")
        same_class_count, same_class_ratio = _match_ratio(
            "same class item", "same class item"
        )
        no_match_count, no_match_ratio = _match_ratio("no match", "no match")
        _trace(
            f"type-match stats "
            f"same item: {same_item_ratio:.4f} ({same_item_count}), "
            f"same class item: {same_class_ratio:.4f} ({same_class_count}), "
            f"no match: {no_match_ratio:.4f} ({no_match_count}) "
        )

    def _build_env_action(self, action_denormalized):
        action_xyz = action_denormalized.view(-1).cpu().numpy().astype(np.float32)
        if action_xyz.shape != (3,):
            raise ValueError(
                f"Expected denormalized learner action to have shape (3,), got {action_xyz.shape}"
            )
        action_xyz = action_xyz.copy()
        action_xyz[2] -= PREPICK_Z_OFFSET
        return {
            "pose0": (
                action_xyz,
                np.asarray((0.0, 0.0, 0.0, 1.0), dtype=np.float32),
            )
        }

    def _build_learner_image_paths(self, episode_label):
        if self.save_learner_image_dir is None:
            return None, None

        base_dir = self.save_learner_image_dir
        base_dir.mkdir(parents=True, exist_ok=True)
        return (
            base_dir / f"{episode_label}_raw.png",
            base_dir / f"{episode_label}_learner_input.png",
        )

    def run_episode(
        self,
        *,
        index=None,
        num_candidates=50,
        candidate_batch_size=250,
        episode_label=None,
    ):
        signal = self.demonstrator.sample_demonstration(index=index)
        obs, info = self._reset_environment()
        scene_objects = [str(obj) for obj in info.get("all_obj", [])]
        _trace(f"learner scene objects: {scene_objects}")
        save_raw_path, save_processed_path = self._build_learner_image_paths(
            "episode" if episode_label is None else episode_label
        )
        decision = self.learner.imitate(
            obs,
            signal,
            num_candidates=num_candidates,
            candidate_batch_size=candidate_batch_size,
            save_raw_path=save_raw_path,
            save_processed_path=save_processed_path,
        )

        step_info = dict(info)
        if decision.accepted:
            env_action = self._build_env_action(decision.action_denormalized)
            _, _, _, step_info = self.environment.step(env_action)
        else:
            step_info["picked_obj"] = "none"

        success_result = self.scorer.score(
            signal.lang_goal,
            step_info.get("picked_obj", "none"),
            step_info.get("all_obj", []),
            decision,
        )
        episode_type = self._scene_relation(signal.lang_goal, scene_objects)
        episode_result = self._episode_result(
            signal.lang_goal,
            success_result.picked_obj,
            decision.accepted,
        )
        episode_score = self._relation_score(episode_result)
        optimal_score = self._relation_score(episode_type)
        _trace("Summary ------")
        _trace(
            f"episode finished index={index}, demo_obj={signal.lang_goal!r}, "
            f"picked_obj={success_result.picked_obj!r}"
        )
        _trace(f"learner scene objects: {scene_objects}")
        _trace(f"episode type={episode_type!r}")
        _trace(f"episode result={episode_result!r}")
        _trace(f"episode score={episode_score:g}, optimal score={optimal_score:g}")
        return EpisodeResult(
            index=index,
            learner_task_name=self.learner.task_name,
            demo_task_name=self.demonstrator.task_name,
            lang_goal=signal.lang_goal,
            accepted=decision.accepted,
            success=success_result.success,
            match_type=success_result.match_type,
            episode_type=episode_type,
            episode_result=episode_result,
            picked_obj=success_result.picked_obj,
            episode_score=episode_score,
            optimal_score=optimal_score,
            score=decision.score,
            failed_similarity=decision.failed_similarity,
            demo_similarity=decision.demo_similarity,
        )

    def run(
        self,
        num_episodes,
        *,
        index=None,
        num_candidates=50,
        candidate_batch_size=250,
        episode_callback=None,
    ):
        if num_episodes <= 0:
            raise ValueError(f"num_episodes must be positive, got {num_episodes}")

        episode_results = []
        if index is not None:
            _trace_banner(f"Evaluation 1/1 | index={index}")
            episode_result = self.run_episode(
                index=index,
                num_candidates=num_candidates,
                candidate_batch_size=candidate_batch_size,
                episode_label=f"episode0001_idx{index:04d}",
            )
            episode_results.append(episode_result)
            if episode_callback is not None:
                episode_callback(asdict(episode_result))
            self._trace_running_score_summary(episode_results)
            self._trace_episode_type_match_summary(episode_results)
        else:
            for episode_index in range(num_episodes):
                _trace_banner(
                    f"Evaluation {episode_index + 1}/{num_episodes} | index={episode_index}"
                )
                episode_result = self.run_episode(
                    index=episode_index,
                    num_candidates=num_candidates,
                    candidate_batch_size=candidate_batch_size,
                    episode_label=f"episode{episode_index + 1:04d}_idx{episode_index:04d}",
                )
                episode_results.append(episode_result)
                if episode_callback is not None:
                    episode_callback(asdict(episode_result))
                self._trace_running_score_summary(episode_results)
                self._trace_episode_type_match_summary(episode_results)

        total = len(episode_results)
        accepted = sum(result.accepted for result in episode_results)
        success = sum(result.success for result in episode_results)
        exact = sum(result.match_type == "exact" for result in episode_results)
        same_class = sum(result.match_type == "same_class" for result in episode_results)
        rejected = sum(result.match_type == "rejected" for result in episode_results)
        average_episode_score = sum(
            result.episode_score for result in episode_results
        ) / max(total, 1)
        average_optimal_score = sum(
            result.optimal_score for result in episode_results
        ) / max(total, 1)
        pair_key = f"{self.demonstrator.task_name}->{self.learner.task_name}"
        return {
            "num_episodes": total,
            "accepted_rate": accepted / max(total, 1),
            "execution_success_rate": success / max(total, 1),
            "exact_match_rate": exact / max(total, 1),
            "same_class_rate": same_class / max(total, 1),
            "rejection_count": rejected,
            "average_episode_score": average_episode_score,
            "average_optimal_score": average_optimal_score,
            "pair_stats": {
                pair_key: {
                    "accepted_rate": accepted / max(total, 1),
                    "execution_success_rate": success / max(total, 1),
                }
            },
            "episodes": [asdict(result) for result in episode_results],
        }
