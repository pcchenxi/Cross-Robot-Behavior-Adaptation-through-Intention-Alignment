"""Trainer for the CLIP-style intention extractor."""

from pathlib import Path

import torch
import torch.nn as nn

from iail_sim_picking.intention_extractor.intention_extractor_network import (
    IntentionExtractorNetwork,
)


class IntentionExtractor(nn.Module):
    def __init__(
        self,
        action_dim,
        device,
        lr=1e-4,
        weight_decay=1e-6,
        text_encoder_model="distilbert-base-uncased",
        pretrained=True,
        trainable=True,
        projection_dim=256,
        dropout=0.1,
    ):
        super().__init__()
        self.network = IntentionExtractorNetwork(
            action_dim=action_dim,
            text_encoder_model=text_encoder_model,
            pretrained=pretrained,
            trainable=trainable,
            projection_dim=projection_dim,
            dropout=dropout,
            device=device,
        ).to(device)
        self.ce_loss = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.AdamW(
            self.network.parameters(),
            lr=lr,
            weight_decay=weight_decay,
        )
        self.device = device

    def get_ground_truth(self, logits):
        return torch.arange(
            logits.shape[0],
            dtype=torch.long,
            device=self.device,
        )

    def compute_loss(self, batch_by_task):
        outputs = self.network(batch_by_task)
        logits_per_image = outputs["logits_per_image"]
        logits_per_text = outputs["logits_per_text"]
        ground_truth = self.get_ground_truth(logits_per_image)

        loss_img = self.ce_loss(logits_per_image, ground_truth)
        loss_text = self.ce_loss(logits_per_text, ground_truth)
        loss = (loss_img + loss_text) / 2

        return {
            "loss_img": loss_img,
            "loss_text": loss_text,
            "loss": loss,
            **outputs,
        }

    def train_once(self, batch_by_task):
        self.network.train()
        outputs = self.compute_loss(batch_by_task)

        self.optimizer.zero_grad()
        outputs["loss"].backward()
        self.optimizer.step()

        return {
            "loss_img": outputs["loss_img"].item(),
            "loss_text": outputs["loss_text"].item(),
            "loss": outputs["loss"].item(),
        }

    def save(self, filename, directory):
        checkpoint_path = Path(directory) / f"{filename}.pth"
        torch.save(self.network.state_dict(), checkpoint_path)
