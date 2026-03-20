"""Learner role for runtime association."""

from __future__ import annotations

from pathlib import Path

try:
    import imageio.v2 as imageio
except ImportError:
    import imageio
import numpy as np
import torch
from transformers import DistilBertTokenizer

from iail_sim_picking.runtime_associator.common import (
    DEFAULT_RESULTS_DIR,
    TASK_CAMERA_INDEX,
    add_batch_dim,
    ensure_supported_task_name,
    load_action_stats,
    load_intention_extractor,
    load_motion_generator,
    to_device_tensor,
)
from iail_sim_picking.runtime_associator.data_loader import build_runtime_transform
from iail_sim_picking.runtime_associator.types import (
    LearnerDecision,
    LearnerObservation,
)


def _trace(message):
    print(f"[Learner] {message}", flush=True)


class Learner:
    def __init__(
        self,
        *,
        task_name,
        intention_checkpoint,
        motion_checkpoint=None,
        action_dim=3,
        latent_dim=6,
        valid_threshold=0.5,
        aligned_threshold=0.32,
        device="cpu",
        results_dir=DEFAULT_RESULTS_DIR,
        text_encoder_model="distilbert-base-uncased",
        projection_dim=256,
    ):
        ensure_supported_task_name(task_name, "task_name")
        self.task_name = task_name
        self.device = torch.device(device)
        self.valid_threshold = float(valid_threshold)
        self.aligned_threshold = float(aligned_threshold)
        self.image_transform = build_runtime_transform()

        results_dir = Path(results_dir)
        action_mean, action_std = load_action_stats(task_name, results_dir=results_dir)
        action_mean = np.asarray(action_mean, dtype=np.float32)
        action_std = np.asarray(action_std, dtype=np.float32)
        if action_mean.shape != (action_dim,):
            raise ValueError(
                f"Action stats for {task_name} imply action_dim={action_mean.shape[0]}, "
                f"but action_dim={action_dim} was requested."
            )

        self.action_mean = torch.as_tensor(
            action_mean,
            device=self.device,
        ).view(1, -1)
        self.action_std = torch.as_tensor(
            action_std,
            device=self.device,
        ).view(1, -1)
        self.action_dim = int(self.action_mean.shape[1])

        self.intention_extractor = load_intention_extractor(
            intention_checkpoint,
            action_dim=action_dim,
            device=self.device,
            text_encoder_model=text_encoder_model,
            projection_dim=projection_dim,
        )

        if motion_checkpoint is None:
            motion_checkpoint = (
                results_dir / "motion_generator" / f"cvae_{task_name}_latest.pth"
            )
        self.motion_generator = load_motion_generator(
            motion_checkpoint,
            action_dim=action_dim,
            latent_dim=latent_dim,
            device=self.device,
        )
        self.actor_vae = self.motion_generator.actor_vae

        self.intention_extractor.eval()
        self.motion_generator.eval()
        self.actor_vae.eval()
        self.failed_embedding = self._build_failed_embedding(text_encoder_model)

    def denormalize_action(self, action_normalized):
        return action_normalized * (self.action_std + 1e-8) + self.action_mean

    def _prepare_text(self, input_ids, attention_mask):
        input_ids = add_batch_dim(to_device_tensor(input_ids, self.device)).long()
        attention_mask = add_batch_dim(
            to_device_tensor(attention_mask, self.device)
        ).long()
        return input_ids, attention_mask

    def _build_failed_embedding(self, text_encoder_model):
        tokenizer = DistilBertTokenizer.from_pretrained(text_encoder_model)
        token = tokenizer(
            "failed",
            padding="max_length",
            truncation=True,
            max_length=64,
        )
        input_ids = torch.tensor(token["input_ids"], dtype=torch.long)
        attention_mask = torch.tensor(token["attention_mask"], dtype=torch.long)
        input_ids, attention_mask = self._prepare_text(input_ids, attention_mask)
        with torch.no_grad():
            return self.intention_extractor.get_text_embeddings(
                input_ids,
                attention_mask,
            )

    def _prepare_image(self, image):
        image = add_batch_dim(to_device_tensor(image, self.device)).float()
        if image.ndim != 4:
            raise ValueError(f"Expected image tensor [B, C, H, W], got {tuple(image.shape)}")
        return image

    def _save_raw_image(self, image, path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        imageio.imwrite(path, np.asarray(image, dtype=np.uint8))

    def _save_processed_image(self, image, path):
        if not torch.is_tensor(image):
            raise TypeError("Processed learner image must be a torch.Tensor.")

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        image_uint8 = (
            image.detach()
            .cpu()
            .clamp(0.0, 1.0)
            .permute(1, 2, 0)
            .mul(255.0)
            .round()
            .to(torch.uint8)
            .numpy()
        )
        imageio.imwrite(path, image_uint8)

    def observe(self, env_obs, *, save_raw_path=None, save_processed_path=None):
        camera_index = TASK_CAMERA_INDEX[self.task_name]
        image = env_obs["color"][camera_index]
        if save_raw_path is not None:
            self._save_raw_image(image, save_raw_path)
        state = self.image_transform(image)
        if save_processed_path is not None:
            self._save_processed_image(state, save_processed_path)
        return LearnerObservation(
            state=self._prepare_image(state),
            raw_obs=env_obs,
            task_name=self.task_name,
        )

    @torch.no_grad()
    def sample_candidate_actions(self, state, batch_size):
        state = self._prepare_image(state)
        if batch_size <= 0:
            raise ValueError(f"batch_size must be positive, got {batch_size}")

        state_batch = state.expand(batch_size, -1, -1, -1)
        return self.actor_vae.decode(state_batch).detach()

    @torch.no_grad()
    def score_candidate_actions(self, state, sampled_actions, demo_embedding):
        state = self._prepare_image(state)
        state_batch = state.expand(sampled_actions.shape[0], -1, -1, -1)
        candidate_embeddings = self.intention_extractor.get_image_embeddings(
            state_batch,
            sampled_actions,
            self.task_name,
        )
        failed_similarity = torch.sum(
            candidate_embeddings * self.failed_embedding,
            dim=1,
        )
        demo_similarity = torch.sum(
            candidate_embeddings * demo_embedding,
            dim=1,
        )
        score = demo_similarity - failed_similarity

        return score, failed_similarity, demo_similarity

    @torch.no_grad()
    def match_and_select(
        self,
        observation,
        demonstration_signal,
        *,
        num_candidates=50,
        candidate_batch_size=250,
    ):
        _trace(
            f"match_and_select start num_candidates={num_candidates}, "
            f"candidate_batch_size={candidate_batch_size}"
        )
        if num_candidates <= 0:
            raise ValueError(f"num_candidates must be positive, got {num_candidates}")
        if candidate_batch_size <= 0:
            raise ValueError(
                f"candidate_batch_size must be positive, got {candidate_batch_size}"
            )

        state = self._prepare_image(observation.state)
        collected_actions = []
        collected_scores = []
        collected_failed_similarity = []
        collected_demo_similarity = []
        total_sampled = 0

        selected_action = None
        selected_score = None
        selected_failed_similarity = None
        selected_demo_similarity = None

        # Sample up to 10 batches, keep valid aligned candidates, and then
        # choose the highest-scoring retained action.
        for batch_idx in range(10):
            _trace(
                f"candidate sampling round {batch_idx + 1}/10, "
                f"lang_goal={demonstration_signal.lang_goal!r}"
            )
            sampled_actions = self.sample_candidate_actions(
                state,
                batch_size=candidate_batch_size,
            )
            total_sampled += int(sampled_actions.shape[0])
            score, failed_similarity, demo_similarity = self.score_candidate_actions(
                state,
                sampled_actions,
                demonstration_signal.demo_embedding,
            )
            best_demo_idx = int(torch.argmax(demo_similarity).item())
            max_demo_similarity = demo_similarity[best_demo_idx].item()
            failed_similarity_at_max_demo = failed_similarity[best_demo_idx].item()

            selected_action = sampled_actions[0:1]
            selected_score = score[0].item()
            selected_failed_similarity = failed_similarity[0].item()
            selected_demo_similarity = demo_similarity[0].item()

            valid_mask = (failed_similarity < self.valid_threshold) & (
                demo_similarity > self.aligned_threshold)

            valid_indices = torch.nonzero(valid_mask, as_tuple=False).view(-1)
            action_variance = self.denormalize_action(sampled_actions).var(
                dim=0,
                unbiased=False,
            )
            _trace(
                f"round {batch_idx + 1}: valid_in_batch={valid_indices.numel()}, "
                f"total_sampled={total_sampled}, collected_valid={len(collected_actions)}, "
                f"max_demo_sim={max_demo_similarity:.4f}, "
                f"failed_sim={failed_similarity_at_max_demo:.4f}"
            )
            _trace(
                "action variance="
                f"[{action_variance[0].item():.6f}, "
                f"{action_variance[1].item():.6f}, "
                f"{action_variance[2].item():.6f}]"
            )
            for valid_idx in valid_indices.tolist():
                collected_actions.append(sampled_actions[valid_idx : valid_idx + 1])
                collected_scores.append(score[valid_idx].item())
                collected_failed_similarity.append(failed_similarity[valid_idx].item())
                collected_demo_similarity.append(demo_similarity[valid_idx].item())
            if len(collected_actions) >= num_candidates:
                break

        if collected_actions:
            best_idx = int(np.argmax(collected_scores))
            selected_action = collected_actions[best_idx]
            selected_score = collected_scores[best_idx]
            selected_failed_similarity = collected_failed_similarity[best_idx]
            selected_demo_similarity = collected_demo_similarity[best_idx]
            _trace(
                f"selected best valid candidate best_idx={best_idx}, "
                f"score={selected_score:.4f}, demo_similarity={selected_demo_similarity:.4f}, "
                f"failed_similarity={selected_failed_similarity:.4f}"
            )

        accepted = len(collected_actions) > 0
        _trace(
            f"decision accepted={accepted}, total_sampled={total_sampled}, "
            f"valid_candidate_count={len(collected_actions)}"
        )
        return LearnerDecision(
            action_normalized=selected_action.detach().cpu(),
            action_denormalized=self.denormalize_action(selected_action).detach().cpu(),
            failed_similarity=selected_failed_similarity,
            demo_similarity=selected_demo_similarity,
            score=selected_score,
            accepted=bool(accepted),
            candidate_count=total_sampled,
            valid_candidate_count=len(collected_actions),
            matched_lang_goal=demonstration_signal.lang_goal,
        )

    def imitate(
        self,
        env_obs,
        demonstration_signal,
        *,
        save_raw_path=None,
        save_processed_path=None,
        **kwargs,
    ):
        observation = self.observe(
            env_obs,
            save_raw_path=save_raw_path,
            save_processed_path=save_processed_path,
        )
        decision = self.match_and_select(observation, demonstration_signal, **kwargs)
        return decision
