"""Dataset for the runtime associator."""

import torch
from torchvision import transforms

from iail_sim_picking.dataset.training_dataset import (
    BaseTrainingDataset,
    create_training_dataloader,
    crop_img,
)


class RuntimeAssociatorDataset(BaseTrainingDataset):
    def __init__(
        self,
        path,
        transform=None,
        task_name='ur5f',
        init_stats=False,
    ):
        transform = build_runtime_transform() if transform is None else transform
        super().__init__(
            path=path,
            transform=transform,
            task_name=task_name,
            init_stats=init_stats,
        )

    def get_item(self, idx):
        obs, action = self.load(idx)
        image = self._get_camera_image(obs['color'])

        if self.transform:
            image = self.transform(image)

        caption = self._get_lang_goal(idx)

        return {
            'image': image,
            'action': torch.tensor(action, dtype=torch.float32),
            'caption': caption,
        }

    def sample_item(self, index=None):
        if len(self) == 0:
            raise ValueError('Demonstrator dataset must not be empty.')

        sample_index = index
        if sample_index is None:
            sample_index = torch.randint(len(self), size=(1,)).item()

        sample = dict(self.get_item(sample_index))
        sample['index'] = int(sample_index)
        return sample


def build_runtime_transform(size=224):
    return transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Lambda(crop_img),
            transforms.Resize((size, size)),
        ]
    )


def create_dataloader(
    data_dir='/media/xi_dataset/cliport/',
    batch_size=16,
    num_workers=4,
    task_name='ur5f',
    epoch_step_num=100,
    init_stats=False,
    persistent_workers=False,
):
    dataset = RuntimeAssociatorDataset(
        data_dir,
        task_name=task_name,
        init_stats=init_stats,
    )

    return create_training_dataloader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        epoch_step_num=epoch_step_num,
        persistent_workers=persistent_workers,
    )
