"""Image dataset."""

from torchvision import transforms

from iail_sim_picking.dataset.training_dataset import (
    BaseTrainingDataset,
    create_training_dataloader,
    crop_img,
)


class MotionGeneratorDataset(BaseTrainingDataset):
    def __init__(self, path, transform=None, task_name='ur5f', init_stats=False):
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

        return {
            'image': image,
            'action': action,
        }


def create_dataloader(
    data_dir='/media/xi_dataset/cliport/',
    batch_size=16,
    num_workers=4,
    task_name='ur5f',
    epoch_step_num=100,
    init_stats=False,
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

    dataset = MotionGeneratorDataset(
        data_dir,
        transform=img_transforms,
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
