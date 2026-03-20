"""Demonstrator role for runtime association."""

from __future__ import annotations

import torch

from iail_sim_picking.runtime_associator.common import (
    ensure_supported_task_name,
    load_intention_extractor,
    to_device_tensor,
    add_batch_dim,
)
from iail_sim_picking.runtime_associator.data_loader import RuntimeAssociatorDataset
from iail_sim_picking.runtime_associator.types import (
    DemonstrationBatch,
    DemonstrationSignal,
)


def _trace(message):
    print(f"[Demonstrator] {message}", flush=True)


class Demonstrator:
    def __init__(
        self,
        *,
        data_dir,
        task_name,
        intention_checkpoint,
        action_dim=3,
        device="cpu",
        text_encoder_model="distilbert-base-uncased",
        projection_dim=256,
    ):
        ensure_supported_task_name(task_name, "task_name")
        self.task_name = task_name
        self.device = torch.device(device)
        self.dataset = RuntimeAssociatorDataset(path=str(data_dir), task_name=task_name)
        self.intention_extractor = load_intention_extractor(
            intention_checkpoint,
            action_dim=action_dim,
            device=self.device,
            text_encoder_model=text_encoder_model,
            projection_dim=projection_dim,
        )
        self.intention_extractor.eval()

    def _prepare_image(self, image):
        image = add_batch_dim(to_device_tensor(image, self.device)).float()
        if image.ndim != 4:
            raise ValueError(f"Expected image tensor [B, C, H, W], got {tuple(image.shape)}")
        return image

    def _prepare_action(self, action):
        action = add_batch_dim(to_device_tensor(action, self.device)).float()
        if action.ndim != 2:
            raise ValueError(f"Expected action tensor [B, D], got {tuple(action.shape)}")
        return action

    def _build_demonstration_batch(self, sample):
        return DemonstrationBatch(
            index=int(sample["index"]),
            image=self._prepare_image(sample["image"]),
            action=self._prepare_action(sample["action"]),
            lang_goal=str(sample["caption"]),
            task_name=self.task_name,
        )

    def _sample_dataset_item(self, index=None):
        return self.dataset.sample_item(index=index)

    def sample(self, index=None):
        sample = self._sample_dataset_item(index=index)
        return self._build_demonstration_batch(sample)

    @torch.no_grad()
    def encode(self, demo_batch):
        _trace(
            f"encoding demonstration index={demo_batch.index}, "
            f"lang_goal={demo_batch.lang_goal!r}"
        )
        demo_embedding = self.intention_extractor.get_image_embeddings(
            demo_batch.image,
            demo_batch.action,
            self.task_name,
        )
        return DemonstrationSignal(
            lang_goal=demo_batch.lang_goal,
            demo_embedding=demo_embedding,
            task_name=self.task_name,
        )

    @torch.no_grad()
    def sample_demonstration(self, index=None):
        return self.encode(self.sample(index=index))
