"""Network definitions for the intention extractor."""

import numpy as np
import torch
from torch import nn
from transformers import DistilBertConfig, DistilBertModel

from iail_sim_picking.intention_extractor.resnet import ResNet50


class ProjectionHead(nn.Module):
    def __init__(self, embedding_dim, projection_dim, dropout):
        super().__init__()
        self.projection = nn.Linear(embedding_dim, projection_dim)
        self.gelu = nn.GELU()
        self.fc = nn.Linear(projection_dim, projection_dim)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(projection_dim)

    def forward(self, x):
        projected = self.projection(x)
        x = self.gelu(projected)
        x = self.fc(x)
        x = self.dropout(x)
        x = x + projected
        return self.layer_norm(x)


class FiLMBlock(nn.Module):
    def forward(self, x, gamma, beta):
        beta = beta.view(x.size(0), x.size(1), 1, 1)
        gamma = gamma.view(x.size(0), x.size(1), 1, 1)
        return gamma * x + beta


class ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 1, 1, 0)
        self.norm1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, 1, 1)
        self.norm2 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.film = FiLMBlock()

    def forward(self, x, beta, gamma):
        x = self.conv1(x)
        x = self.norm1(x)
        x = self.relu(x)
        identity = x.clone()

        x = self.conv2(x)
        x = self.norm2(x)
        x = self.film(x, beta, gamma)
        x = self.relu(x)
        return x + identity, x


class ImageEncoder(nn.Module):
    """Encode an image-action pair into a fixed-size embedding."""

    def __init__(
        self,
        action_dim,
        projection_dim,
        dropout,
        n_channels=256,
        n_res_blocks=4,
        device=None,
    ):
        super().__init__()
        self.n_channels = n_channels
        self.n_res_blocks = n_res_blocks
        self.device = device
        self.conv = nn.Conv2d(n_channels, n_channels, 3, 1, 0)
        self.global_pool = nn.AdaptiveMaxPool2d((1, 1))
        self.projection_head = ProjectionHead(
            embedding_dim=n_channels,
            projection_dim=projection_dim,
            dropout=dropout,
        )
        self.film_generator = nn.Sequential(
            nn.Linear(action_dim, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 2 * n_res_blocks * n_channels),
        )
        self.task_feature_extractor = ResNet50()
        self.task_res_blocks = nn.ModuleList(
            [ResBlock(n_channels + 2, n_channels) for _ in range(n_res_blocks)]
        )

    def get_filmed_feature(self, x, res_blocks, film_vector):
        batch_size = x.size(0)
        spatial_dim = x.size(2)
        coordinate = torch.arange(
            -1,
            1 + 0.00001,
            2 / (spatial_dim - 1),
            device=x.device,
        )
        coordinate_x = coordinate.expand(batch_size, 1, spatial_dim, spatial_dim)
        coordinate_y = coordinate.view(spatial_dim, 1).expand(
            batch_size,
            1,
            spatial_dim,
            spatial_dim,
        )

        x_res = None
        for i, res_block in enumerate(res_blocks):
            beta = film_vector[:, i, 0, :]
            gamma = film_vector[:, i, 1, :]
            x = torch.cat([x, coordinate_x, coordinate_y], 1)
            x, x_res = res_block(x, beta, gamma)

        return x, x_res

    def process(self, x, res_blocks, film_vector):
        x_filmed, x_res = self.get_filmed_feature(x, res_blocks, film_vector)
        x_out = self.conv(x_filmed)
        x_pooled = self.global_pool(x_out)
        x_pooled = x_pooled.view(x_pooled.size(0), x_pooled.size(1))
        return x_pooled, x_out, x_res

    def forward(self, x, action):
        x = self.task_feature_extractor(x)
        batch_size = x.size(0)
        film_vector = self.film_generator(action.float()).view(
            batch_size,
            self.n_res_blocks,
            2,
            self.n_channels,
        )
        x_pooled, _, _ = self.process(x, self.task_res_blocks, film_vector)
        return self.projection_head(x_pooled)


class TextEncoderP(nn.Module):
    def __init__(
        self,
        model_name,
        pretrained,
        trainable,
        projection_dim,
        dropout,
    ):
        super().__init__()
        if pretrained:
            self.model = DistilBertModel.from_pretrained(model_name)
        else:
            self.model = DistilBertModel(config=DistilBertConfig())

        for param in self.model.parameters():
            param.requires_grad = trainable

        self.target_token_idx = 0
        self.projection_head = ProjectionHead(
            embedding_dim=768,
            projection_dim=projection_dim,
            dropout=dropout,
        )

    def forward(self, input_ids, attention_mask):
        output = self.model(input_ids=input_ids, attention_mask=attention_mask)
        last_hidden_state = output.last_hidden_state
        cls_embedding = last_hidden_state[:, self.target_token_idx, :]
        return self.projection_head(cls_embedding)


class IntentionExtractorNetwork(nn.Module):
    TASK_NAMES = ("ur5f", "ur5l", "ur5r")

    def __init__(
        self,
        task_names=TASK_NAMES,
        action_dim=3,
        text_encoder_model="distilbert-base-uncased",
        pretrained=True,
        trainable=True,
        projection_dim=256,
        dropout=0.1,
        device=None,
    ):
        super().__init__()
        if tuple(task_names) != self.TASK_NAMES:
            raise ValueError(
                f"task_names must be exactly {self.TASK_NAMES}, got {tuple(task_names)}"
            )

        self.task_names = tuple(task_names)
        self.text_encoder = TextEncoderP(
            model_name=text_encoder_model,
            pretrained=pretrained,
            trainable=trainable,
            projection_dim=projection_dim,
            dropout=dropout,
        )
        self.image_encoders = nn.ModuleDict(
            {
                task_name: ImageEncoder(
                    action_dim=action_dim,
                    projection_dim=projection_dim,
                    dropout=dropout,
                    device=device,
                )
                for task_name in self.task_names
            }
        )
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))

    def _normalize(self, embeddings):
        return embeddings / embeddings.norm(dim=1, keepdim=True)

    def get_image_embeddings(self, image, action, task_name):
        if task_name not in self.image_encoders:
            raise ValueError(f"Unsupported task_name: {task_name}")
        image_features = self.image_encoders[task_name](image, action)
        return self._normalize(image_features)

    def get_text_embeddings(self, input_ids, attention_mask):
        text_embeddings = self.text_encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )
        return self._normalize(text_embeddings)

    def encode_batch(self, batch, task_name):
        image_embeddings = self.get_image_embeddings(
            batch["image"],
            batch["action"],
            task_name,
        )
        text_embeddings = self.get_text_embeddings(
            batch["input_ids"],
            batch["attention_mask"],
        )
        return image_embeddings, text_embeddings

    def compute_logits(self, image_embeddings, text_embeddings):
        logit_scale = self.logit_scale.exp()
        logits_per_image = image_embeddings @ text_embeddings.t() * logit_scale
        return logits_per_image, logits_per_image.t()

    def forward(self, batch_by_task):
        missing_tasks = [task for task in self.task_names if task not in batch_by_task]
        if missing_tasks:
            raise KeyError(f"Missing batches for tasks: {missing_tasks}")

        image_embeddings_all = []
        text_embeddings_all = []
        for task_name in self.task_names:
            image_embeddings, text_embeddings = self.encode_batch(
                batch_by_task[task_name],
                task_name,
            )
            image_embeddings_all.append(image_embeddings)
            text_embeddings_all.append(text_embeddings)

        image_embeddings_all = torch.cat(image_embeddings_all, dim=0)
        text_embeddings_all = torch.cat(text_embeddings_all, dim=0)
        logits_per_image, logits_per_text = self.compute_logits(
            image_embeddings_all,
            text_embeddings_all,
        )

        return {
            "image_embeddings": image_embeddings_all,
            "text_embeddings": text_embeddings_all,
            "logits_per_image": logits_per_image,
            "logits_per_text": logits_per_text,
        }
