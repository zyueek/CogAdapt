"""Tiny synthetic unit fixtures; these tests do not measure model performance."""

import importlib.util
import json
import unittest
from pathlib import Path

import numpy as np

from cogadapt import selection_glm, selection_qwen
from cogadapt.alignment import length_controlled_spearman, rank_correlation, residualize
from cogadapt.weighting import _hotspot_token_flags, completion_weights, task_weights, token_category

ROOT = Path(__file__).resolve().parents[1]
HAS_TORCH = importlib.util.find_spec("torch") is not None


class SelectionTests(unittest.TestCase):
    def test_main_width_boundaries(self):
        for module in [selection_qwen, selection_glm]:
            self.assertEqual([module.active_block_count(x, hard_blocks=10)
                              for x in [-0.6, -0.5, 0, 0.5, 0.6]], [4, 6, 6, 6, 10])

    def test_hard_anchor_replaces_one_block(self):
        for module, n, anchor in [(selection_qwen, 48, 39), (selection_glm, 47, 38)]:
            scores = -np.arange(n, dtype=float)
            mask = module.select_active_blocks(scores, 0.8, hard_blocks=10)
            self.assertEqual(len(mask), 10)
            self.assertIn(anchor, mask)
            self.assertEqual(mask, sorted(set(mask)))

    def test_ties_are_stable(self):
        self.assertEqual(selection_qwen.select_active_blocks([0.0] * 48, 0), list(range(42, 48)))

    def test_glm_missing_block_is_imputed(self):
        result = selection_glm._standardize(np.array([np.nan, 2., 4.]))
        np.testing.assert_allclose(result, [0, -np.sqrt(1.5), np.sqrt(1.5)])

    def test_real_masks(self):
        for row in json.loads((ROOT / "results/rq3/selection_examples.json").read_text()):
            module = selection_qwen if row["model"] == "qwen" else selection_glm
            metrics = {k: np.asarray(v, dtype=float) for k, v in row["metrics"].items()}
            scores = module.combine_metrics(metrics)
            np.testing.assert_allclose(scores, row["sensitivity_scores"])
            self.assertEqual(module.select_active_blocks(scores, row["difficulty_z"], hard_blocks=10),
                             row["active_blocks"])


class WeightAndAlignmentTests(unittest.TestCase):
    def test_task_weight_normalization_and_control(self):
        eeg, q = [-1., 0., 1.], [.75, 1., 1.25]
        expected = np.clip(np.exp(.8 * np.array(eeg)), .4, 2.) * q
        np.testing.assert_allclose(task_weights(eeg, q), expected / expected.mean())
        np.testing.assert_allclose(task_weights(eeg, q, False), q)

    def test_tokens_and_hotspots(self):
        self.assertEqual(token_category("Ġreturn"), "keyword:return")
        self.assertEqual(token_category("foo"), "identifier")
        solution = "x = 1\nreturn x\n"
        self.assertEqual(_hotspot_token_flags([(0, 1), (6, 12)], solution), [True, True])
        result = completion_weights(["return", "foo"], [True, False], {"keyword:return": 1.})
        expected = np.array([min(np.exp(.45), 1.5) * 1.35, 1.])
        np.testing.assert_allclose(result, expected / expected.mean())
        np.testing.assert_allclose(completion_weights(["return", "foo"], [True, False], {}, False), [1, 1])

    def test_empty_completion_and_invalid_shapes(self):
        self.assertEqual(len(completion_weights([], [], {})), 0)
        with self.assertRaises(ValueError):
            task_weights([0], [1, 2])

    def test_spearman_ties(self):
        self.assertAlmostEqual(rank_correlation([1, 1, 2, 3], [2, 2, 4, 5]), 1.)
        self.assertTrue(np.isnan(rank_correlation([1, 1, 1], [1, 2, 3])))

    def test_residual_controls(self):
        rng = np.random.default_rng(24)
        controls = rng.normal(size=(32, 2))
        x, y = rng.normal(size=(2, 32))
        residual = residualize(x, controls)
        design = np.column_stack([np.ones(32), controls])
        np.testing.assert_allclose(design.T @ residual, 0, atol=1e-10)
        actual = length_controlled_spearman(x, y, controls)
        expected = rank_correlation(residual, residualize(y, controls))
        self.assertAlmostEqual(actual, expected)


@unittest.skipUnless(HAS_TORCH, "Optional PyTorch not installed")
class TorchExcerptTests(unittest.TestCase):
    def test_weighted_loss_masks_prompt(self):
        import torch
        from cogadapt.torch_helpers import weighted_causal_loss
        logits = torch.zeros(1, 4, 3, requires_grad=True)
        labels = torch.tensor([[-100, -100, 1, 2]])
        weights = torch.tensor([[0., 0., .5, 1.5]])
        loss = weighted_causal_loss(logits, labels, weights, torch.tensor([2.]))
        self.assertAlmostEqual(loss.item(), 2 * np.log(3), places=6)
        loss.backward()
        self.assertTrue(torch.all(logits.grad[:, 0] == 0))

    def test_accumulation_updates_union_not_last_mask(self):
        import torch
        from cogadapt.torch_helpers import _mask_trainable
        params = [torch.nn.Parameter(torch.tensor(1.)) for _ in range(3)]
        by_block = {i: [p] for i, p in enumerate(params)}
        optimizer = torch.optim.AdamW(params, lr=.1, weight_decay=.01)
        optimizer.zero_grad(set_to_none=True)
        for mask in [{0}, {1}]:
            _mask_trainable(by_block, mask)
            sum(p.square() for p in params).backward()
        self.assertIsNotNone(params[0].grad)
        self.assertIsNotNone(params[1].grad)
        self.assertIsNone(params[2].grad)
        _mask_trainable(by_block, set(by_block))
        optimizer.step()
        self.assertLess(params[0].item(), 1.)
        self.assertLess(params[1].item(), 1.)
        self.assertEqual(params[2].item(), 1.)

    def test_inference_activation_restores_base(self):
        import torch
        from cogadapt.torch_helpers import _activate_blocks

        class Adapter:
            def __init__(self):
                self.scaling = {"default": 1.}
            def set_scale(self, name, value):
                self.scaling[name] = value

        adapters = {i: [Adapter()] for i in range(2)}
        routers = {i: torch.nn.Parameter(torch.zeros(1)) for i in range(2)}
        base = {i: torch.tensor([1.]) for i in range(2)}
        learned = {i: torch.tensor([2.]) for i in range(2)}
        _activate_blocks({1}, routers, base, learned, adapters)
        self.assertEqual([routers[i].item() for i in range(2)], [1., 2.])
        self.assertEqual([adapters[i][0].scaling["default"] for i in range(2)], [0., 1.])

    def test_linear_cka_identity_and_bad_shapes(self):
        import torch
        from cogadapt.torch_helpers import linear_cka
        x = torch.arange(18.).reshape(6, 3)
        self.assertAlmostEqual(linear_cka(x, x), 1., places=6)
        self.assertAlmostEqual(linear_cka(x, x * 3 + 8), 1., places=6)
        with self.assertRaises(ValueError):
            linear_cka(x, x[:3])


if __name__ == "__main__":
    unittest.main()
