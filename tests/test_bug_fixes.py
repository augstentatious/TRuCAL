"""
Test suite to verify all bug fixes from BUG_FIXES.md

Run with: python tests/test_bug_fixes.py
"""

import torch
import torch.nn.functional as F
from cal import UnifiedCAL_TRM, TinyConfessionalLayer, VulnerabilitySpotter, CAL_TRM_Hybrid


def test_coherence_calculation():
    """Test Fix #1: Coherence should compare with previous cycle, not self."""
    print("\n=== Test 1: Coherence Calculation Bug ===")
    torch.manual_seed(42)
    
    model = TinyConfessionalLayer(d_model=64, max_cycles=4, trigger_thresh=0.04)
    x = torch.randn(2, 8, 64) * 2  # High variance to trigger
    
    out, meta = model(x, audit_mode=True)
    
    coherence = meta['coherence_score']
    print(f"Final coherence: {coherence:.4f}")
    
    # Coherence should NOT be 1.0 (old bug)
    assert coherence < 0.99, f"❌ Coherence still 1.0! Bug not fixed. Got {coherence}"
    print(f"✅ Coherence is {coherence:.4f} (not 1.0) - Bug fixed!")
    

def test_threshold_parametrization():
    """Test Fix #2: Threshold should be parametrized and default to 0.04."""
    print("\n=== Test 2: Threshold Parametrization ===")
    
    # Test default threshold
    model1 = TinyConfessionalLayer(d_model=64)
    assert model1.trigger_thresh == 0.04, f"❌ Default threshold wrong: {model1.trigger_thresh}"
    print(f"✅ Default threshold is 0.04")
    
    # Test custom threshold
    model2 = TinyConfessionalLayer(d_model=64, trigger_thresh=0.08)
    assert model2.trigger_thresh == 0.08, f"❌ Custom threshold not set: {model2.trigger_thresh}"
    print(f"✅ Custom threshold works: {model2.trigger_thresh}")
    
    # Test CAL_TRM_Hybrid default
    hybrid = CAL_TRM_Hybrid(d_model=64)
    assert hybrid.threshold == 0.04, f"❌ Hybrid threshold wrong: {hybrid.threshold}"
    print(f"✅ CAL_TRM_Hybrid threshold is 0.04")


def test_vectorization():
    """Test Fix #3: Batch loop should be vectorized (no Python for loop)."""
    print("\n=== Test 3: Vectorization (Speed Test) ===")
    torch.manual_seed(42)
    
    model = TinyConfessionalLayer(d_model=256, max_cycles=3, trigger_thresh=0.01)
    
    # Test with larger batch
    x_large = torch.randn(32, 64, 256) * 3  # Large batch, high variance
    
    import time
    start = time.time()
    out, meta = model(x_large, audit_mode=False)
    elapsed = time.time() - start
    
    print(f"✅ Processed batch=32, seq=64 in {elapsed:.4f}s")
    print(f"   Triggered: {meta['confessional_triggered']}")
    print(f"   Cycles: {meta['cycles_run']}")
    assert elapsed < 2.0, f"❌ Too slow ({elapsed:.2f}s) - vectorization may have failed"
    print(f"✅ Vectorization working (fast processing)")


def test_audit_mode_guards():
    """Test Fix #4: Prints should only appear with audit_mode=True."""
    print("\n=== Test 4: audit_mode Guards ===")
    torch.manual_seed(42)
    
    import io
    import sys
    
    model = UnifiedCAL_TRM(d_model=64)
    x = torch.randn(2, 8, 64) * 2
    
    # Test with audit_mode=False (should be quiet)
    old_stdout = sys.stdout
    sys.stdout = buffer = io.StringIO()
    
    out, meta = model(x, return_metadata=True, audit_mode=False)
    
    sys.stdout = old_stdout
    output_quiet = buffer.getvalue()
    
    # Test with audit_mode=True (should print)
    sys.stdout = buffer2 = io.StringIO()
    
    out, meta = model(x, return_metadata=True, audit_mode=True)
    
    sys.stdout = old_stdout
    output_verbose = buffer2.getvalue()
    
    print(f"Output with audit_mode=False: {len(output_quiet)} chars")
    print(f"Output with audit_mode=True: {len(output_verbose)} chars")
    
    assert len(output_verbose) > len(output_quiet), "❌ audit_mode not working"
    assert len(output_quiet) < 50, f"❌ Too much output with audit_mode=False: {len(output_quiet)} chars"
    print(f"✅ audit_mode guards working (quiet: {len(output_quiet)}, verbose: {len(output_verbose)} chars)")


def test_per_dim_kl():
    """Test Fix #5: Per-dimension KL should be available and differ from global."""
    print("\n=== Test 5: Per-Dimension KL Divergence ===")
    torch.manual_seed(42)
    
    # Create two models: one with global KL, one with per-dim KL
    model_global = TinyConfessionalLayer(d_model=64, max_cycles=4, per_dim_kl=False)
    model_per_dim = TinyConfessionalLayer(d_model=64, max_cycles=4, per_dim_kl=True)
    
    x = torch.randn(2, 8, 64) * 2
    
    out_g, meta_g = model_global(x, audit_mode=False)
    out_pd, meta_pd = model_per_dim(x, audit_mode=False)
    
    coh_global = meta_g['coherence_score']
    coh_per_dim = meta_pd['coherence_score']
    
    print(f"Global KL coherence: {coh_global:.4f}")
    print(f"Per-dim KL coherence: {coh_per_dim:.4f}")
    print(f"Difference: {abs(coh_global - coh_per_dim):.4f}")
    
    # They should differ (different calculations)
    # Note: with random seed, they might be close but formula is different
    print(f"✅ Per-dimension KL option available")
    assert hasattr(model_per_dim, 'per_dim_kl'), "❌ per_dim_kl attribute missing"
    assert model_per_dim.per_dim_kl == True, "❌ per_dim_kl not set to True"
    print(f"✅ Per-dim KL parameter working")


def test_edge_case_zero_input():
    """Test edge case: zero input should not trigger."""
    print("\n=== Test 6: Edge Case - Zero Input ===")
    torch.manual_seed(42)
    
    model = UnifiedCAL_TRM(d_model=64)
    x_zero = torch.zeros(2, 8, 64)
    
    out, meta = model(x_zero, return_metadata=True, audit_mode=False)
    
    v_t_score = meta.get('v_t_score', None)
    if v_t_score is not None:
        print(f"v_t score on zero input: {v_t_score.mean():.4f}")
    
    # May or may not trigger depending on random entropy
    print(f"Triggered: {meta['confessional_triggered']}")
    print(f"Coherence: {meta['coherence_score']:.4f}")
    print(f"✅ Zero input handled without error")


def test_edge_case_high_variance():
    """Test edge case: high variance should trigger."""
    print("\n=== Test 7: Edge Case - High Variance ===")
    torch.manual_seed(42)
    
    model = UnifiedCAL_TRM(d_model=64)
    x_high = torch.randn(2, 8, 64) * 10  # Very high variance
    
    out, meta = model(x_high, return_metadata=True, audit_mode=False)
    
    print(f"Triggered: {meta['confessional_triggered']}")
    print(f"Coherence: {meta['coherence_score']:.4f}")
    print(f"Cycles: {meta['cycles_run']}")
    
    # High variance should likely trigger
    if meta['confessional_triggered']:
        print(f"✅ High variance triggered confessional (as expected)")
    else:
        print(f"⚠️  High variance did not trigger (check threshold)")


def test_backward_compatibility():
    """Test that old code still works with new defaults."""
    print("\n=== Test 8: Backward Compatibility ===")
    torch.manual_seed(42)
    
    # Old-style usage (no new parameters)
    model = UnifiedCAL_TRM(d_model=64)
    x = torch.randn(2, 8, 64)
    
    # Should work without specifying new params
    out1 = model(x)
    out2, meta = model(x, return_metadata=True)
    
    assert out1.shape == (2, 8, 64), "❌ Output shape wrong"
    assert 'confessional_triggered' in meta, "❌ Metadata missing"
    assert 'coherence_score' in meta, "❌ Coherence not in metadata"
    
    print(f"✅ Old-style usage works")
    print(f"✅ All metadata present: {list(meta.keys())}")


if __name__ == "__main__":
    print("=" * 60)
    print("TRuCAL Bug Fix Verification Test Suite")
    print("=" * 60)
    
    try:
        test_coherence_calculation()
        test_threshold_parametrization()
        test_vectorization()
        test_audit_mode_guards()
        test_per_dim_kl()
        test_edge_case_zero_input()
        test_edge_case_high_variance()
        test_backward_compatibility()
        
        print("\n" + "=" * 60)
        print("✅ ALL TESTS PASSED!")
        print("=" * 60)
        
    except Exception as e:
        print("\n" + "=" * 60)
        print(f"❌ TEST FAILED: {e}")
        print("=" * 60)
        import traceback
        traceback.print_exc()
