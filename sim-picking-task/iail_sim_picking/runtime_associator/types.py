"""Runtime associator public data types."""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass
class DemonstrationBatch:
    index: int
    image: torch.Tensor
    action: torch.Tensor
    lang_goal: str
    task_name: str


@dataclass
class DemonstrationSignal:
    lang_goal: str
    demo_embedding: torch.Tensor
    task_name: str


@dataclass
class LearnerObservation:
    state: torch.Tensor
    raw_obs: dict
    task_name: str


@dataclass
class LearnerDecision:
    action_normalized: torch.Tensor
    action_denormalized: torch.Tensor
    failed_similarity: float
    demo_similarity: float
    score: float
    accepted: bool
    candidate_count: int
    valid_candidate_count: int
    matched_lang_goal: str


@dataclass
class EpisodeResult:
    index: int
    learner_task_name: str
    demo_task_name: str
    lang_goal: str
    accepted: bool
    success: bool
    match_type: str
    episode_type: str
    episode_result: str
    picked_obj: str
    episode_score: float
    optimal_score: float
    score: float
    failed_similarity: float
    demo_similarity: float
