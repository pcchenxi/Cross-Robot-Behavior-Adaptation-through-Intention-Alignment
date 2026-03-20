import unittest
from unittest import mock

import torch

from iail_sim_picking.runtime_associator.data_loader import (
    RuntimeAssociatorDataset,
    build_runtime_transform,
)
from iail_sim_picking.runtime_associator.demonstrator import Demonstrator
from iail_sim_picking.runtime_associator.learner import Learner


class RuntimeAssociatorDatasetSampleItemTests(unittest.TestCase):
    def _make_dataset(self, length=3):
        dataset = object.__new__(RuntimeAssociatorDataset)
        dataset.n_episodes = length

        def get_item(idx):
            return {
                "image": torch.full((3, 4, 4), fill_value=float(idx)),
                "action": torch.tensor([idx, idx + 1, idx + 2], dtype=torch.float32),
                "caption": f"goal-{idx}",
            }

        dataset.get_item = get_item
        return dataset

    def test_sample_item_with_index_returns_same_payload_plus_index(self):
        dataset = self._make_dataset(length=5)

        sample = dataset.sample_item(index=2)
        expected = dataset.get_item(2)

        self.assertEqual(sample["index"], 2)
        self.assertEqual(sample["caption"], expected["caption"])
        self.assertTrue(torch.equal(sample["image"], expected["image"]))
        self.assertTrue(torch.equal(sample["action"], expected["action"]))

    def test_sample_item_uses_default_random_selection(self):
        dataset = self._make_dataset(length=7)
        with mock.patch(
            "iail_sim_picking.runtime_associator.data_loader.random.randrange",
            return_value=4,
        ) as randrange:
            sample = dataset.sample_item(index=None)

        randrange.assert_called_once_with(len(dataset))
        self.assertEqual(sample["index"], 4)
        self.assertEqual(sample["caption"], "goal-4")

    def test_sample_item_rejects_empty_dataset(self):
        dataset = self._make_dataset(length=0)

        with self.assertRaisesRegex(
            ValueError, "Demonstrator dataset must not be empty."
        ):
            dataset.sample_item()

    def test_init_uses_runtime_default_transform_when_unspecified(self):
        with mock.patch(
            "iail_sim_picking.runtime_associator.data_loader.BaseTrainingDataset.__init__",
            return_value=None,
        ) as base_init:
            RuntimeAssociatorDataset(path="/tmp/demo", transform=None, task_name="ur5f")

        transform = base_init.call_args.kwargs["transform"]
        expected = build_runtime_transform()
        self.assertIsNotNone(transform)
        self.assertEqual(type(transform).__name__, type(expected).__name__)
        self.assertEqual(
            [type(step).__name__ for step in transform.transforms],
            [type(step).__name__ for step in expected.transforms],
        )


class _FakeDataset:
    def __init__(self, sample):
        self.sample = sample
        self.calls = []

    def sample_item(self, index=None):
        self.calls.append(index)
        return dict(self.sample)


class _FakeIntentionExtractor:
    def __init__(self):
        self.eval_called = False

    def eval(self):
        self.eval_called = True
        return self

    def get_text_embeddings(self, input_ids, attention_mask):
        return torch.ones(input_ids.shape[0], 3)

    def get_image_embeddings(self, image, action, task_name):
        return torch.full((image.shape[0], 3), 2.0)


class _FakeActorVAE:
    def __init__(self):
        self.eval_called = False

    def eval(self):
        self.eval_called = True
        return self


class _FakeMotionGenerator:
    def __init__(self):
        self.eval_called = False
        self.actor_vae = _FakeActorVAE()

    def eval(self):
        self.eval_called = True
        return self


class DemonstratorSampleTests(unittest.TestCase):
    def test_init_builds_dataset_and_intention_extractor_from_paths(self):
        dataset = _FakeDataset({})
        extractor = _FakeIntentionExtractor()
        with mock.patch(
            "iail_sim_picking.runtime_associator.demonstrator.RuntimeAssociatorDataset",
            return_value=dataset,
        ) as dataset_cls, mock.patch(
            "iail_sim_picking.runtime_associator.demonstrator.load_intention_extractor",
            return_value=extractor,
        ) as load_extractor:
            demonstrator = Demonstrator(
                data_dir="/tmp/demo-data",
                task_name="ur5f",
                intention_checkpoint="/tmp/demo-model.pth",
                device="cpu",
            )

        dataset_cls.assert_called_once_with(path="/tmp/demo-data", task_name="ur5f")
        load_extractor.assert_called_once()
        self.assertNotIn("dropout", load_extractor.call_args.kwargs)
        self.assertIs(demonstrator.dataset, dataset)
        self.assertIs(demonstrator.intention_extractor, extractor)
        self.assertTrue(extractor.eval_called)

    def test_sample_builds_demonstration_batch_from_dataset_sample(self):
        sample = {
            "index": 3,
            "image": torch.ones(3, 4, 4),
            "action": torch.tensor([0.1, 0.2, 0.3], dtype=torch.float32),
            "caption": "pick-red-block",
        }
        dataset = _FakeDataset(sample)
        with mock.patch(
            "iail_sim_picking.runtime_associator.demonstrator.RuntimeAssociatorDataset",
            return_value=dataset,
        ), mock.patch(
            "iail_sim_picking.runtime_associator.demonstrator.load_intention_extractor",
            return_value=_FakeIntentionExtractor(),
        ):
            demonstrator = Demonstrator(
                data_dir="/tmp/demo-data",
                task_name="ur5f",
                intention_checkpoint="/tmp/demo-model.pth",
                device="cpu",
            )

        batch = demonstrator.sample(index=3)

        self.assertEqual(len(dataset.calls), 1)
        self.assertEqual(dataset.calls[0], 3)
        self.assertEqual(batch.index, 3)
        self.assertEqual(batch.lang_goal, "pick-red-block")
        self.assertEqual(batch.task_name, "ur5f")
        self.assertEqual(tuple(batch.image.shape), (1, 3, 4, 4))
        self.assertEqual(tuple(batch.action.shape), (1, 3))
        self.assertEqual(batch.image.device.type, "cpu")
        self.assertEqual(batch.action.device.type, "cpu")

    def test_encode_returns_failed_and_demo_embeddings_only(self):
        sample = {
            "index": 1,
            "image": torch.ones(3, 4, 4),
            "action": torch.tensor([0.1, 0.2, 0.3], dtype=torch.float32),
            "caption": "pick-red-block",
        }
        extractor = _FakeIntentionExtractor()
        with mock.patch(
            "iail_sim_picking.runtime_associator.demonstrator.RuntimeAssociatorDataset",
            return_value=_FakeDataset(sample),
        ), mock.patch(
            "iail_sim_picking.runtime_associator.demonstrator.load_intention_extractor",
            return_value=extractor,
        ):
            demonstrator = Demonstrator(
                data_dir="/tmp/demo-data",
                task_name="ur5f",
                intention_checkpoint="/tmp/demo-model.pth",
                device="cpu",
            )

        signal = demonstrator.sample_demonstration(index=1)

        self.assertFalse(hasattr(signal, "task_embedding"))
        self.assertFalse(hasattr(signal, "failed_embedding"))
        self.assertEqual(signal.lang_goal, "pick-red-block")
        self.assertEqual(tuple(signal.demo_embedding.shape), (1, 3))


class LearnerInitTests(unittest.TestCase):
    def test_init_builds_internal_dependencies_from_paths(self):
        extractor = _FakeIntentionExtractor()
        motion_generator = _FakeMotionGenerator()
        image_transform = mock.Mock(return_value=torch.ones(3, 4, 4))
        tokenizer = mock.Mock()
        tokenizer.return_value = {
            "input_ids": [1, 2, 3],
            "attention_mask": [1, 1, 1],
        }
        with mock.patch(
            "iail_sim_picking.runtime_associator.learner.build_runtime_transform",
            return_value=image_transform,
        ) as build_transform, mock.patch(
            "iail_sim_picking.runtime_associator.learner.load_action_stats",
            return_value=([1.0, 2.0, 3.0], [0.1, 0.2, 0.3]),
        ) as load_stats, mock.patch(
            "iail_sim_picking.runtime_associator.learner.load_intention_extractor",
            return_value=extractor,
        ) as load_extractor, mock.patch(
            "iail_sim_picking.runtime_associator.learner.load_motion_generator",
            return_value=motion_generator,
        ) as load_motion, mock.patch(
            "iail_sim_picking.runtime_associator.learner.DistilBertTokenizer.from_pretrained",
            return_value=tokenizer,
        ) as load_tokenizer:
            learner = Learner(
                task_name="ur5f",
                intention_checkpoint="/tmp/intention.pth",
                results_dir="/tmp/results",
                device="cpu",
            )

        build_transform.assert_called_once_with()
        load_stats.assert_called_once()
        load_extractor.assert_called_once()
        load_motion.assert_called_once()
        load_tokenizer.assert_called_once_with("distilbert-base-uncased")
        self.assertNotIn("dropout", load_extractor.call_args.kwargs)
        self.assertEqual(load_stats.call_args.kwargs["results_dir"].as_posix(), "/tmp/results")
        self.assertEqual(
            load_motion.call_args.args[0].as_posix(),
            "/tmp/results/motion_generator/cvae_ur5f_latest.pth",
        )
        self.assertIs(learner.image_transform, image_transform)
        self.assertIs(learner.intention_extractor, extractor)
        self.assertIs(learner.motion_generator, motion_generator)
        self.assertEqual(tuple(learner.failed_embedding.shape), (1, 3))
        self.assertTrue(extractor.eval_called)
        self.assertTrue(motion_generator.eval_called)
        self.assertTrue(motion_generator.actor_vae.eval_called)

    def test_observe_uses_default_transform(self):
        extractor = _FakeIntentionExtractor()
        motion_generator = _FakeMotionGenerator()
        image_transform = mock.Mock(return_value=torch.ones(3, 4, 4))
        tokenizer = mock.Mock()
        tokenizer.return_value = {
            "input_ids": [1, 2, 3],
            "attention_mask": [1, 1, 1],
        }
        with mock.patch(
            "iail_sim_picking.runtime_associator.learner.build_runtime_transform",
            return_value=image_transform,
        ), mock.patch(
            "iail_sim_picking.runtime_associator.learner.load_action_stats",
            return_value=([1.0, 2.0, 3.0], [0.1, 0.2, 0.3]),
        ), mock.patch(
            "iail_sim_picking.runtime_associator.learner.load_intention_extractor",
            return_value=extractor,
        ), mock.patch(
            "iail_sim_picking.runtime_associator.learner.load_motion_generator",
            return_value=motion_generator,
        ), mock.patch(
            "iail_sim_picking.runtime_associator.learner.DistilBertTokenizer.from_pretrained",
            return_value=tokenizer,
        ):
            learner = Learner(
                task_name="ur5f",
                intention_checkpoint="/tmp/intention.pth",
                motion_checkpoint="/tmp/motion.pth",
                device="cpu",
            )

        env_obs = {"color": ["front-image", "left-image", "right-image"]}
        observation = learner.observe(env_obs)

        image_transform.assert_called_once_with("front-image")
        self.assertEqual(tuple(observation.state.shape), (1, 3, 4, 4))
        self.assertEqual(observation.task_name, "ur5f")
        self.assertIs(observation.raw_obs, env_obs)

    def test_match_and_select_filters_invalid_candidates_with_failed_branch(self):
        learner = object.__new__(Learner)
        learner.task_name = "ur5f"
        learner.device = torch.device("cpu")
        learner.failed_sim_weight = 1.0
        learner.valid_threshold = 0.5
        learner.aligned_threshold = 0.0
        learner.action_mean = torch.zeros(1, 3)
        learner.action_std = torch.ones(1, 3)

        class _FilteringExtractor:
            def get_image_embeddings(self, image, action, task_name):
                return torch.tensor(
                    [
                        [0.2, 0.1],
                        [0.9, 0.7],
                        [0.8, 0.2],
                    ],
                    dtype=torch.float32,
                )

        learner.intention_extractor = _FilteringExtractor()
        learner.sample_candidate_actions = mock.Mock(
            return_value=torch.tensor(
                [
                    [0.1, 0.0, 0.0],
                    [0.2, 0.0, 0.0],
                    [0.3, 0.0, 0.0],
                ],
                dtype=torch.float32,
            )
        )

        observation = type(
            "Obs",
            (),
            {"state": torch.ones(1, 3, 4, 4), "raw_obs": {}, "task_name": "ur5f"},
        )()
        demonstration_signal = type(
            "Signal",
            (),
            {
                "lang_goal": "pick-red-block",
                "demo_embedding": torch.tensor([[1.0, 0.0]], dtype=torch.float32),
                "task_name": "ur5f",
            },
        )()
        learner.failed_embedding = torch.tensor([[0.0, 1.0]], dtype=torch.float32)

        decision = learner.match_and_select(
            observation,
            demonstration_signal,
            num_candidates=3,
            candidate_batch_size=3,
        )

        self.assertEqual(decision.candidate_count, 3)
        self.assertEqual(decision.valid_candidate_count, 2)
        self.assertAlmostEqual(decision.demo_similarity, 0.8)
        self.assertAlmostEqual(decision.failed_similarity, 0.2)
        self.assertAlmostEqual(decision.score, 0.6)
        self.assertTrue(decision.accepted)


if __name__ == "__main__":
    unittest.main()
