"""
Simple integration test for Mixture DPO components.
Tests imports and basic functionality without full trainer instantiation.
"""

import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../src"))

import torch


def test_imports():
    """Test that all mixture modules can be imported."""
    print("=" * 80)
    print("TEST 1: Module Imports")
    print("=" * 80)
    
    try:
        from alignment import MixturePLConfig, MixtureDPOTrainer
        print("✓ MixturePLConfig imported from alignment")
        print("✓ MixtureDPOTrainer imported from alignment")
        
        from alignment.mixture_pl_components import (
            pl_log_prob,
            mixture_pl_nll,
            em_responsibilities,
            em_expected_complete_nll,
            MixtureRouterHead,
            compute_mixture_eval_metrics,
        )
        print("✓ All mixture components imported")
        
        return True
    except Exception as e:
        print(f"✗ Import failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_pl_log_prob():
    """Test pl_log_prob computation."""
    print("\n" + "=" * 80)
    print("TEST 2: PL Log Probability")
    print("=" * 80)
    
    try:
        from alignment.mixture_pl_components import pl_log_prob
        
        batch_size, num_items = 4, 5
        rewards = torch.randn(batch_size, num_items, requires_grad=True)
        rankings = torch.stack([torch.randperm(num_items) for _ in range(batch_size)])
        
        logp = pl_log_prob(rewards, rankings)
        
        assert logp.shape == (batch_size,), f"Expected shape ({batch_size},), got {logp.shape}"
        assert logp.dtype == torch.float32
        print(f"✓ pl_log_prob output shape: {logp.shape}")
        
        # Test backward pass
        loss = logp.sum()
        loss.backward()
        assert rewards.grad is not None
        print(f"✓ Gradient computed successfully")
        
        return True
    except Exception as e:
        print(f"✗ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_mixture_nll():
    """Test mixture_pl_nll computation."""
    print("\n" + "=" * 80)
    print("TEST 3: Mixture NLL")
    print("=" * 80)
    
    try:
        from alignment.mixture_pl_components import mixture_pl_nll
        
        batch_size, num_clusters, num_items = 4, 3, 5
        router_logits = torch.randn(batch_size, num_clusters, requires_grad=True)
        component_rewards = torch.randn(batch_size, num_clusters, num_items, requires_grad=True)
        rankings = torch.stack([torch.randperm(num_items) for _ in range(batch_size)])
        
        nll, comp_logp = mixture_pl_nll(router_logits, component_rewards, rankings)
        
        assert nll.shape == (), f"Expected scalar, got {nll.shape}"
        assert comp_logp.shape == (batch_size, num_clusters), f"Expected ({batch_size}, {num_clusters}), got {comp_logp.shape}"
        print(f"✓ nll is scalar: {nll.item():.4f}")
        print(f"✓ comp_logp shape: {comp_logp.shape}")
        
        # Test backward pass
        nll.backward()
        assert router_logits.grad is not None
        assert component_rewards.grad is not None
        print(f"✓ Gradients computed for both router and rewards")
        
        return True
    except Exception as e:
        print(f"✗ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_em_responsibilities():
    """Test EM responsibility computation."""
    print("\n" + "=" * 80)
    print("TEST 4: EM Responsibilities")
    print("=" * 80)
    
    try:
        from alignment.mixture_pl_components import em_responsibilities
        
        batch_size, num_clusters = 4, 3
        router_logits = torch.randn(batch_size, num_clusters)
        comp_logp = torch.randn(batch_size, num_clusters)
        
        # Test different temperatures
        for temperature in [0.1, 1.0, 10.0]:
            gamma = em_responsibilities(router_logits, comp_logp, temperature=temperature)
            
            assert gamma.shape == (batch_size, num_clusters)
            # Check that responsibilities sum to 1
            sums = gamma.sum(dim=1)
            assert torch.allclose(sums, torch.ones(batch_size), atol=1e-5)
            print(f"✓ Temperature={temperature}: shape {gamma.shape}, sums to 1")
        
        return True
    except Exception as e:
        print(f"✗ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_router_head():
    """Test MixtureRouterHead."""
    print("\n" + "=" * 80)
    print("TEST 5: MixtureRouterHead")
    print("=" * 80)
    
    try:
        from alignment.mixture_pl_components import MixtureRouterHead
        
        batch_size, hidden_size, num_clusters = 4, 256, 3
        
        # Test contextual router
        router_contextual = MixtureRouterHead(
            hidden_size=hidden_size,
            num_clusters=num_clusters,
            use_contextual=True,
            router_hidden_size=128,
        )
        context = torch.randn(batch_size, hidden_size, requires_grad=True)
        logits = router_contextual(context)
        
        assert logits.shape == (batch_size, num_clusters)
        print(f"✓ Contextual router: input {context.shape} -> output {logits.shape}")
        
        # Test global router
        router_global = MixtureRouterHead(
            hidden_size=hidden_size,
            num_clusters=num_clusters,
            use_contextual=False,
        )
        logits_global = router_global(context)
        
        assert logits_global.shape == (batch_size, num_clusters)
        # Global router should give same output regardless of input
        logits_global_2 = router_global(torch.randn(batch_size, hidden_size))
        assert torch.allclose(logits_global, logits_global_2)
        print(f"✓ Global router: consistent output {logits_global.shape}")
        
        # Test backward pass
        loss = logits.sum() + logits_global.sum()
        loss.backward()
        assert context.grad is not None
        print(f"✓ Gradients computed successfully")
        
        return True
    except Exception as e:
        print(f"✗ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_mixture_config():
    """Test MixturePLConfig."""
    print("\n" + "=" * 80)
    print("TEST 6: MixturePLConfig")
    print("=" * 80)
    
    try:
        from alignment import MixturePLConfig
        
        config = MixturePLConfig(
            output_dir="/tmp/test_mixture_dpo",
            bf16=False,
            use_mixture=True,
            num_clusters=2,
            mixture_nll_weight=0.1,
            use_contextual_router=True,
            em_temperature=1.0,
        )
        
        assert config.use_mixture == True
        assert config.num_clusters == 2
        assert config.mixture_nll_weight == 0.1
        assert config.use_contextual_router == True
        print(f"✓ MixturePLConfig created with:")
        print(f"  - use_mixture: {config.use_mixture}")
        print(f"  - num_clusters: {config.num_clusters}")
        print(f"  - mixture_nll_weight: {config.mixture_nll_weight}")
        print(f"  - use_contextual_router: {config.use_contextual_router}")
        
        return True
    except Exception as e:
        print(f"✗ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_cluster_metrics_by_dimension_aligns_permutation():
    """Test per-dimension clustering metrics use Hungarian-aligned labels."""
    from alignment.ranking_eval import _cluster_alignment, _cluster_metrics_by_dimension

    true_labels = [0, 0, 1, 1]
    pred_labels = [1, 1, 0, 0]
    dimensions = ["helpfulness", "helpfulness", "truthfulness", "truthfulness"]
    alignment = _cluster_alignment(true_labels, pred_labels, num_clusters=2)

    metrics = _cluster_metrics_by_dimension(true_labels, pred_labels, dimensions, alignment)

    assert metrics["mixture/by_dimension/helpfulness/num_examples"] == 2.0
    assert metrics["mixture/by_dimension/helpfulness/cluster_acc_raw"] == 0.0
    assert metrics["mixture/by_dimension/helpfulness/cluster_acc"] == 1.0
    assert metrics["mixture/by_dimension/truthfulness/num_examples"] == 2.0
    assert metrics["mixture/by_dimension/truthfulness/cluster_acc_raw"] == 0.0
    assert metrics["mixture/by_dimension/truthfulness/cluster_acc"] == 1.0


if __name__ == "__main__":
    print("\n" + "=" * 80)
    print("MIXTURE DPO - SIMPLIFIED INTEGRATION TESTS")
    print("=" * 80 + "\n")
    
    tests = [
        ("Imports", test_imports),
        ("PL Log Prob", test_pl_log_prob),
        ("Mixture NLL", test_mixture_nll),
        ("EM Responsibilities", test_em_responsibilities),
        ("Router Head", test_router_head),
        ("Config", test_mixture_config),
    ]
    
    results = {}
    for test_name, test_func in tests:
        results[test_name] = test_func()
    
    print("\n" + "=" * 80)
    print("TEST SUMMARY")
    print("=" * 80)
    passed_count = sum(1 for v in results.values() if v)
    total_count = len(results)
    
    for test_name, passed in results.items():
        status = "✓" if passed else "✗"
        print(f"{status} {test_name}")
    
    print("=" * 80)
    print(f"Results: {passed_count}/{total_count} tests passed")
    
    if all(results.values()):
        print("✓ ALL TESTS PASSED")
        sys.exit(0)
    else:
        print("✗ SOME TESTS FAILED")
        sys.exit(1)
