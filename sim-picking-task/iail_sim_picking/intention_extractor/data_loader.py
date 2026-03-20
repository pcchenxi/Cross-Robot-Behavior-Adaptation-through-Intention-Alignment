"""Dataset for the intention extractor."""

from pathlib import Path

import numpy as np
import torch
from torchvision import transforms
from transformers import DistilBertTokenizer

from iail_sim_picking.dataset.training_dataset import (
    BaseTrainingDataset,
    create_training_dataloader,
    crop_img,
)
from iail_sim_picking.dataset.object_metadata import object_labels
from iail_sim_picking.motion_generator.motion_generator_trainer import MotionGenerator


class IntentionExtractorDataset(BaseTrainingDataset):
    def __init__(
        self,
        path,
        transform=None,
        task_name='ur5f',
        init_stats=False,
        action_dim=3,
        anotate_prob=0.1,
        latent_dim=8,
    ):
        self.action_dim = action_dim
        self.anotate_prob = anotate_prob
        self.latent_dim = latent_dim

        super().__init__(
            path=path,
            transform=transform,
            task_name=task_name,
            init_stats=init_stats,
        )

        self.tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
        self.motion_generator = self._load_motion_generator()

    def _get_motion_generator_path(self, task_name):
        return (
            Path(__file__).resolve().parents[2]
            / 'results'
            / 'motion_generator'
            / f'cvae_{task_name}_latest.pth'
        )

    def _load_motion_generator(self):
        checkpoint_path = self._get_motion_generator_path(self.task_name)
        if not checkpoint_path.exists():
            raise FileNotFoundError(
                f'Motion generator checkpoint does not exist: {checkpoint_path}'
            )

        motion_generator = MotionGenerator(
            action_dim=self.action_dim,
            latent_dim=self.latent_dim,
            device=torch.device('cpu'),
        )
        try:
            state_dict = torch.load(
                checkpoint_path,
                map_location=torch.device('cpu'),
                weights_only=True,
            )
            motion_generator.actor_vae.load_state_dict(state_dict)
        except RuntimeError as exc:
            raise RuntimeError(
                f'Failed to load motion generator checkpoint {checkpoint_path}. '
                f'Check action_dim={self.action_dim} and latent_dim={self.latent_dim}.'
            ) from exc

        motion_generator.actor_vae.eval()
        return motion_generator

    def _annotate_caption(self, lang_goal):
        label = lang_goal.replace('_', ' ')
        label_annotated = object_labels.get(label)
        if label_annotated:
            select_idx = torch.randint(len(label_annotated), size=(1,)).item()
            return label_annotated[select_idx]
        return label

    def _tokenize_caption(self, caption):
        token = self.tokenizer(
            caption,
            padding='max_length',
            truncation=True,
            max_length=64,
        )
        return (
            torch.tensor(token['input_ids'], dtype=torch.long),
            torch.tensor(token['attention_mask'], dtype=torch.long),
        )

    def _sample_failed_action(self, image):
        if not torch.is_tensor(image):
            raise TypeError(
                'Failed-sample motion generation requires transform to return a torch.Tensor.'
            )

        image_batch = image.unsqueeze(0)
        with torch.no_grad():
            action = self.motion_generator.actor_vae.decode(image_batch)
        return action.detach().cpu().numpy()[0].astype(np.float32)

    def get_item(self, idx):
        obs, action = self.load(idx)
        image = self._get_camera_image(obs['color'])

        if self.transform:
            image = self.transform(image)

        use_failed_sample = torch.rand(1).item() < self.anotate_prob
        if use_failed_sample:
            caption = 'failed'
            action = self._sample_failed_action(image)
        else:
            caption = self._annotate_caption(self._get_lang_goal(idx))

        input_ids, attention_mask = self._tokenize_caption(caption)
        return {
            'image': image,
            'action': action,
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'caption': caption,
        }


def create_dataloader(
    data_dir='/media/xi_dataset/cliport/',
    batch_size=16,
    num_workers=4,
    task_name='ur5f',
    epoch_step_num=100,
    init_stats=False,
    action_dim=3,
    anotate_prob=0.1,
    latent_dim=8,
    persistent_workers=False,
):
    size = 224
    img_transforms = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Lambda(crop_img),
            transforms.Resize((size, size)),
        ]
    )

    dataset = IntentionExtractorDataset(
        data_dir,
        transform=img_transforms,
        task_name=task_name,
        init_stats=init_stats,
        action_dim=action_dim,
        anotate_prob=anotate_prob,
        latent_dim=latent_dim,
    )

    return create_training_dataloader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        epoch_step_num=epoch_step_num,
        persistent_workers=persistent_workers,
    )
