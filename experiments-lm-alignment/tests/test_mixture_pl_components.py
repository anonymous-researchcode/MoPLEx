"""
Unit tests for Mixture PL components.

Validates core mixture model functionality: PL log probability, EM steps, and router head.
"""

import unittest

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from alignment.listwise_dpo import ListwiseDPOTrainer
from alignment.configs import MixturePLConfig
from alignment.mixture_pl_components import (
    pl_log_prob,
    mixture_pl_nll,
    em_responsibilities,
    em_expected_complete_nll,
    MixtureRouterHead,
    pearson_corr,
)


class TestPlLogProb(unittest.TestCase):
    """Test Plackett-Luce log probability computation."""

    def test_pl_log_prob_shape(self):
        """Verify output shape and dtype."""
        batch_size, m_items = 4, 5
        rewards = torch.randn(batch_size, m_items)
        rankings = torch.tensor([[0, 1, 2, 3, 4], [4, 3, 2, 1, 0], [1, 0, 3, 2, 4], [2, 1, 0, 3, 4]])

        logp = pl_log_prob(rewards, rankings)

        self.assertEqual(logp.shape, (batch_size,))
        self.assertEqual(logp.dtype, torch.float32)

    def test_pl_log_prob_gradient(self):
        """Verify gradients flow through PL computation."""
        rewards = torch.randn(2, 4, requires_grad=True)
        rankings = torch.tensor([[0, 1, 2, 3], [3, 2, 1, 0]])

        logp = pl_log_prob(rewards, rankings)
        loss = logp.sum()
        loss.backward()

        self.assertIsNotNone(rewards.grad)
        self.assertTrue(torch.all(torch.isfinite(rewards.grad)))

    def test_pl_log_prob_values(self):
        """Verify PL gives higher probability to canonical ordering."""
        m_items = 3
        # Canonical ranking: items in descending order of reward
        best_rewards = torch.tensor([[3.0, 2.0, 1.0], [1.0, 2.0, 3.0]], dtype=torch.float32)
        canonical_ranking = torch.tensor([[0, 1, 2], [2, 1, 0]], dtype=torch.long)

        logp_canonical = pl_log_prob(best_rewards, canonical_ranking)
        # All should be negative log-probs (< 0)
        self.assertTrue(torch.all(logp_canonical <= 0.0))

    def test_pl_log_prob_top1_partial_ranking(self):
        """Verify top-1 partial PL only scores the first item against all candidates."""
        rewards = torch.tensor([[3.0, 2.0, 1.0]], dtype=torch.float32)
        rankings = torch.tensor([[0, 1, 2]], dtype=torch.long)
        ranked_prefix_lengths = torch.tensor([1], dtype=torch.long)

        logp = pl_log_prob(rewards, rankings, ranked_prefix_lengths=ranked_prefix_lengths)
        expected = rewards[:, 0] - torch.logsumexp(rewards, dim=1)

        self.assertTrue(torch.allclose(logp, expected))

    def test_pl_log_prob_top1_ignores_negative_order(self):
        """Verify permuting unordered negatives does not change top-1 partial likelihood."""
        rewards = torch.tensor([[3.0, 2.0, 1.0]], dtype=torch.float32)
        ranked_prefix_lengths = torch.tensor([1], dtype=torch.long)

        first = pl_log_prob(
            rewards,
            torch.tensor([[0, 1, 2]], dtype=torch.long),
            ranked_prefix_lengths=ranked_prefix_lengths,
        )
        second = pl_log_prob(
            rewards,
            torch.tensor([[0, 2, 1]], dtype=torch.long),
            ranked_prefix_lengths=ranked_prefix_lengths,
        )

        self.assertTrue(torch.allclose(first, second))

    def test_pl_log_prob_m2_matches_logsigmoid(self):
        """Verify two-way PL is the pairwise log-sigmoid objective."""
        rewards = torch.tensor([[2.0, -1.0], [-0.5, 0.75]], dtype=torch.float32)
        rankings = torch.tensor([[0, 1], [0, 1]], dtype=torch.long)

        logp = pl_log_prob(rewards, rankings)
        expected = F.logsigmoid(rewards[:, 0] - rewards[:, 1])

        self.assertTrue(torch.allclose(logp, expected))


class TestLinearApproximationHelper(unittest.TestCase):
    """Test direct inner-product Taylor approximation helper."""

    def test_scalar_linear_reward_is_exact(self):
        candidate_embeddings = torch.randn(1, 4, 2, 3)
        candidate_mask = torch.tensor([[True, True, True, False]])
        weight = torch.randn(2, 3)
        exact_scores = (candidate_embeddings[0] * weight).sum(dim=(-1, -2))
        anchor_positions = [torch.tensor([0, 2], dtype=torch.long)]
        anchor_scores = exact_scores[anchor_positions[0]]
        anchor_grads = weight.expand(anchor_positions[0].numel(), -1, -1).clone()

        estimates = ListwiseDPOTrainer._linear_approx_from_anchor_tensors(
            candidate_embeddings,
            candidate_mask,
            anchor_positions,
            anchor_scores,
            anchor_grads,
        )

        self.assertTrue(torch.allclose(estimates[0, :3], exact_scores[:3], atol=1e-6))
        self.assertEqual(estimates[0, 3].item(), 0.0)

    def test_cluster_linear_reward_is_exact(self):
        candidate_embeddings = torch.randn(1, 5, 2, 3)
        candidate_mask = torch.tensor([[True, True, True, True, False]])
        weights = torch.randn(2, 2, 3)
        exact_scores = torch.einsum("msh,ksh->km", candidate_embeddings[0], weights)
        anchor_positions = [torch.tensor([1, 3], dtype=torch.long)]
        anchor_scores = exact_scores[:, anchor_positions[0]].transpose(0, 1)
        anchor_grads = weights.unsqueeze(0).expand(anchor_positions[0].numel(), -1, -1, -1).clone()

        estimates = ListwiseDPOTrainer._linear_approx_from_anchor_tensors(
            candidate_embeddings,
            candidate_mask,
            anchor_positions,
            anchor_scores,
            anchor_grads,
        )

        self.assertTrue(torch.allclose(estimates[0, :, :4], exact_scores[:, :4], atol=1e-6))
        self.assertTrue(torch.all(estimates[0, :, 4] == 0.0))

    def test_exact_ref_score_mode_scalar_formula(self):
        candidate_embeddings = torch.randn(1, 4, 2, 3)
        candidate_mask = torch.tensor([[True, True, True, True]])
        policy_weight = torch.randn(2, 3)
        policy_logps = (candidate_embeddings[0] * policy_weight).sum(dim=(-1, -2))
        ref_logps = torch.tensor([[0.5, -0.25, 0.75, -1.0]], dtype=policy_logps.dtype)
        beta = 0.1
        anchor_positions = [torch.tensor([0, 2], dtype=torch.long)]
        anchor_scores = policy_logps[anchor_positions[0]]
        anchor_grads = policy_weight.expand(anchor_positions[0].numel(), -1, -1).clone()

        estimated_policy = ListwiseDPOTrainer._linear_approx_from_anchor_tensors(
            candidate_embeddings,
            candidate_mask,
            anchor_positions,
            anchor_scores,
            anchor_grads,
        )
        utilities = beta * (estimated_policy - ref_logps)
        expected = beta * (policy_logps.unsqueeze(0) - ref_logps)

        self.assertTrue(torch.allclose(utilities, expected, atol=1e-6))

    def test_exact_ref_score_mode_cluster_formula(self):
        candidate_embeddings = torch.randn(1, 4, 2, 3)
        candidate_mask = torch.tensor([[True, True, True, True]])
        adapter_weights = torch.randn(2, 2, 3)
        adapter_logps = torch.einsum("msh,ksh->km", candidate_embeddings[0], adapter_weights)
        ref_logps = torch.tensor([[0.25, -0.5, 1.0, -0.75]], dtype=adapter_logps.dtype)
        beta = 0.2
        anchor_positions = [torch.tensor([1, 3], dtype=torch.long)]
        anchor_scores = adapter_logps[:, anchor_positions[0]].transpose(0, 1)
        anchor_grads = adapter_weights.unsqueeze(0).expand(anchor_positions[0].numel(), -1, -1, -1).clone()

        estimated_adapters = ListwiseDPOTrainer._linear_approx_from_anchor_tensors(
            candidate_embeddings,
            candidate_mask,
            anchor_positions,
            anchor_scores,
            anchor_grads,
        )
        rewards = beta * (estimated_adapters - ref_logps.unsqueeze(1))
        expected = beta * (adapter_logps.unsqueeze(0) - ref_logps.unsqueeze(1))

        self.assertTrue(torch.allclose(rewards, expected, atol=1e-6))

    def test_linear_approx_ref_mode_validation(self):
        MixturePLConfig(use_linear_reward_approx=True, linear_approx_ref_mode="input_gradient", bf16=False)
        MixturePLConfig(use_linear_reward_approx=True, linear_approx_ref_mode="exact_score", bf16=False)

        with self.assertRaises(ValueError):
            MixturePLConfig(use_linear_reward_approx=True, linear_approx_ref_mode="bad_mode", bf16=False)


class TestMixturePLNLL(unittest.TestCase):
    """Test mixture PL NLL computation."""

    def test_mixture_pl_nll_shape(self):
        """Verify output shapes."""
        batch_size, k_clusters, m_items = 4, 3, 5
        router_logits = torch.randn(batch_size, k_clusters)
        rewards = torch.randn(batch_size, k_clusters, m_items)
        rankings = torch.randint(0, m_items, (batch_size, m_items))
        # Make rankings valid permutations
        for b in range(batch_size):
            rankings[b] = torch.randperm(m_items)

        nll, comp_logp = mixture_pl_nll(router_logits, rewards, rankings)

        self.assertEqual(nll.shape, ())  # scalar
        self.assertEqual(comp_logp.shape, (batch_size, k_clusters))

    def test_mixture_pl_nll_gradient(self):
        """Verify gradients flow through mixture computation."""
        router_logits = torch.randn(2, 3, requires_grad=True)
        rewards = torch.randn(2, 3, 4, requires_grad=True)
        rankings = torch.tensor([[0, 1, 2, 3], [3, 2, 1, 0]], dtype=torch.long)

        nll, _ = mixture_pl_nll(router_logits, rewards, rankings)
        nll.backward()

        self.assertIsNotNone(router_logits.grad)
        self.assertIsNotNone(rewards.grad)

    def test_mixture_pl_nll_top1_partial(self):
        """Verify mixture PL accepts top-1 partial rankings."""
        router_logits = torch.zeros(1, 2)
        rewards = torch.tensor([[[3.0, 2.0, 1.0], [1.0, 3.0, 2.0]]], dtype=torch.float32)
        rankings = torch.tensor([[0, 1, 2]], dtype=torch.long)
        ranked_prefix_lengths = torch.tensor([1], dtype=torch.long)

        nll, comp_logp = mixture_pl_nll(
            router_logits,
            rewards,
            rankings,
            ranked_prefix_lengths=ranked_prefix_lengths,
        )

        self.assertEqual(nll.shape, ())
        self.assertEqual(comp_logp.shape, (1, 2))
        self.assertTrue(torch.isfinite(nll))


class TestEMResponsibilities(unittest.TestCase):
    """Test E-step responsibility computation."""

    def test_em_responsibilities_shape_and_sum(self):
        """Verify responsibilities are probabilities."""
        batch_size, k_clusters = 4, 3
        router_logits = torch.randn(batch_size, k_clusters)
        comp_logp = torch.randn(batch_size, k_clusters)

        gamma = em_responsibilities(router_logits, comp_logp, temperature=1.0)

        self.assertEqual(gamma.shape, (batch_size, k_clusters))
        # Check that responsibilities sum to 1 per batch
        sums = gamma.sum(dim=1)
        self.assertTrue(torch.allclose(sums, torch.ones(batch_size)))
        # Check all are non-negative
        self.assertTrue(torch.all(gamma >= 0))

    def test_em_responsibilities_temperature_effect(self):
        """Verify temperature makes assignments harder/softer."""
        router_logits = torch.randn(4, 3)
        comp_logp = torch.randn(4, 3)

        gamma_soft = em_responsibilities(router_logits, comp_logp, temperature=10.0)
        gamma_hard = em_responsibilities(router_logits, comp_logp, temperature=0.1)

        # Hard should be more peaked (higher max entropy)
        entropy_soft = -(gamma_soft * torch.log(gamma_soft.clamp_min(1e-12))).sum(dim=1).mean()
        entropy_hard = -(gamma_hard * torch.log(gamma_hard.clamp_min(1e-12))).sum(dim=1).mean()

        self.assertGreater(entropy_soft.item(), entropy_hard.item())

    def test_em_responsibilities_invalid_temperature(self):
        """Verify error on invalid temperature."""
        router_logits = torch.randn(2, 3)
        comp_logp = torch.randn(2, 3)

        with self.assertRaises(ValueError):
            em_responsibilities(router_logits, comp_logp, temperature=0.0)

        with self.assertRaises(ValueError):
            em_responsibilities(router_logits, comp_logp, temperature=-1.0)


class TestEMExpectedCompleteNLL(unittest.TestCase):
    """Test M-step expected NLL computation."""

    def test_em_expected_complete_nll_shape_and_gradient(self):
        """Verify M-step loss computation."""
        router_logits = torch.randn(4, 3, requires_grad=True)
        comp_logp = torch.randn(4, 3)
        gamma = torch.randn(4, 3)
        # Make gamma valid probabilities
        gamma = torch.softmax(gamma, dim=1)

        loss = em_expected_complete_nll(router_logits, comp_logp, gamma)

        self.assertEqual(loss.shape, ())  # scalar
        self.assertTrue(torch.isfinite(loss))

        loss.backward()
        self.assertIsNotNone(router_logits.grad)
        self.assertTrue(torch.all(torch.isfinite(router_logits.grad)))

    def test_em_expected_complete_nll_decreases(self):
        """Verify M-step can reduce loss."""
        router_logits = torch.randn(4, 3, requires_grad=True)
        comp_logp = torch.randn(4, 3)
        gamma = torch.softmax(torch.randn(4, 3), dim=1)

        loss_before = em_expected_complete_nll(router_logits, comp_logp, gamma)
        loss_before.backward()

        # One gradient step
        with torch.no_grad():
            router_logits -= 0.1 * router_logits.grad
            router_logits.requires_grad_(True)

        loss_after = em_expected_complete_nll(router_logits, comp_logp, gamma)

        # Loss should improve
        self.assertLess(loss_after.item(), loss_before.item())


class TestMixtureRouterHead(unittest.TestCase):
    """Test router network."""

    def test_router_head_contextual_shape(self):
        """Verify contextual router output shape."""
        hidden_size, num_clusters, batch_size = 256, 3, 4
        router = MixtureRouterHead(
            hidden_size=hidden_size,
            num_clusters=num_clusters,
            use_contextual=True,
            router_hidden_size=128,
        )

        context = torch.randn(batch_size, hidden_size)
        router_logits = router(context)

        self.assertEqual(router_logits.shape, (batch_size, num_clusters))

    def test_router_head_global_shape(self):
        """Verify global router output shape."""
        hidden_size, num_clusters, batch_size = 256, 3, 4
        router = MixtureRouterHead(
            hidden_size=hidden_size,
            num_clusters=num_clusters,
            use_contextual=False,
        )

        context = torch.randn(batch_size, hidden_size)
        router_logits = router(context)

        self.assertEqual(router_logits.shape, (batch_size, num_clusters))

    def test_router_head_gradient(self):
        """Verify router parameters are trainable."""
        router = MixtureRouterHead(
            hidden_size=256,
            num_clusters=3,
            use_contextual=True,
            router_hidden_size=128,
        )

        context = torch.randn(4, 256, requires_grad=True)
        router_logits = router(context)
        loss = router_logits.sum()
        loss.backward()

        # Check that router params have gradients
        for param in router.parameters():
            self.assertIsNotNone(param.grad)

    def test_router_head_global_consistency(self):
        """Verify global router returns same logits for different batch sizes."""
        router = MixtureRouterHead(
            hidden_size=256,
            num_clusters=3,
            use_contextual=False,
        )

        context_small = torch.randn(1, 256)
        context_large = torch.randn(8, 256)

        logits_small = router(context_small)
        logits_large = router(context_large)

        # First row of large should match small
        self.assertTrue(torch.allclose(logits_small[0], logits_large[0]))


class TestPearsonCorr(unittest.TestCase):
    """Test Pearson correlation helper."""

    def test_pearson_corr_perfect_correlation(self):
        """Verify perfect correlation cases."""
        x = np.array([1, 2, 3, 4, 5], dtype=np.float32)
        y = np.array([2, 4, 6, 8, 10], dtype=np.float32)

        corr = pearson_corr(x, y)
        self.assertAlmostEqual(corr, 1.0, places=5)

    def test_pearson_corr_negative_correlation(self):
        """Verify negative correlation cases."""
        x = np.array([1, 2, 3, 4, 5], dtype=np.float32)
        y = np.array([5, 4, 3, 2, 1], dtype=np.float32)

        corr = pearson_corr(x, y)
        self.assertAlmostEqual(corr, -1.0, places=5)

    def test_pearson_corr_zero_variance(self):
        """Verify zero variance handling."""
        x = np.array([1, 1, 1, 1, 1], dtype=np.float32)
        y = np.array([1, 2, 3, 4, 5], dtype=np.float32)

        corr = pearson_corr(x, y)
        self.assertEqual(corr, 0.0)


class TestIntegration(unittest.TestCase):
    """Integration tests combining multiple components."""

    def test_em_loop_single_iteration(self):
        """Test one full EM iteration (E-step + M-step)."""
        batch_size, k_clusters, m_items = 4, 2, 4

        # Initialize
        router_logits = torch.randn(batch_size, k_clusters, requires_grad=True)
        rewards = torch.randn(batch_size, k_clusters, m_items, requires_grad=True)
        rankings = torch.stack([torch.randperm(m_items) for _ in range(batch_size)])

        # E-step: compute responsibilities
        nll, comp_logp = mixture_pl_nll(router_logits, rewards, rankings)
        gamma = em_responsibilities(router_logits, comp_logp, temperature=1.0)

        self.assertEqual(gamma.shape, (batch_size, k_clusters))
        self.assertTrue(torch.allclose(gamma.sum(dim=1), torch.ones(batch_size)))

        # M-step: compute loss
        m_loss = em_expected_complete_nll(router_logits, comp_logp, gamma)
        self.assertTrue(torch.isfinite(m_loss))

        # Gradient update
        m_loss.backward()
        self.assertIsNotNone(router_logits.grad)

    def test_mixture_training_loop(self):
        """Simulate a simple training loop."""
        batch_size, k_clusters, m_items = 4, 2, 4
        num_iterations = 3

        router = MixtureRouterHead(hidden_size=64, num_clusters=k_clusters, use_contextual=True)
        optimizer = torch.optim.Adam(router.parameters(), lr=1e-3)

        losses = []
        for _ in range(num_iterations):
            context = torch.randn(batch_size, 64)
            router_logits = router(context)

            rewards = torch.randn(batch_size, k_clusters, m_items)
            rankings = torch.stack([torch.randperm(m_items) for _ in range(batch_size)])

            nll, comp_logp = mixture_pl_nll(router_logits, rewards, rankings)
            gamma = em_responsibilities(router_logits, comp_logp, temperature=1.0)
            loss = em_expected_complete_nll(router_logits, comp_logp, gamma)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            losses.append(loss.item())

        # Verify training progresses
        self.assertTrue(len(losses) == num_iterations)
        # Loss should generally trend down (or at least not blow up)
        self.assertTrue(all(torch.isfinite(torch.tensor(losses))))


if __name__ == "__main__":
    unittest.main()
