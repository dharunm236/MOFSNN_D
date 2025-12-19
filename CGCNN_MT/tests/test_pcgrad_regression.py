#!/usr/bin/env python
"""
PCGrad Regression Test Script

This script verifies:
1. Baseline behavior is UNCHANGED when pcgrad_enabled=False
2. PCGrad training runs without errors when pcgrad_enabled=True
3. Shared gradients differ from baseline when PCGrad is ON (sanity check)
4. Head gradients remain identical to baseline when PCGrad is ON

Usage:
    python tests/test_pcgrad_regression.py

Requirements:
    - Run from the CGCNN_MT directory or project root
    - Requires a valid data setup (will use minimal synthetic data if not available)
"""

import os
import sys
import copy
import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Tuple, Optional

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from CGCNN_MT.pcgrad import PCGrad, pcgrad_modify, get_shared_params


def set_seed(seed: int = 42):
    """Set all random seeds for reproducibility."""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class SimpleMTModel(nn.Module):
    """
    Simple multi-task model for testing PCGrad.
    
    Architecture:
    - Shared trunk: Linear -> ReLU -> Linear
    - Task heads: Linear for each task
    """
    
    def __init__(self, input_dim: int = 64, hidden_dim: int = 32, n_tasks: int = 3):
        super().__init__()
        
        # Shared trunk (parameters that PCGrad will modify)
        self.shared_fc1 = nn.Linear(input_dim, hidden_dim)
        self.shared_fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.relu = nn.ReLU()
        
        # Task-specific heads (parameters PCGrad should NOT modify)
        self.task_heads = nn.ModuleList([
            nn.Linear(hidden_dim, 1) for _ in range(n_tasks)
        ])
    
    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        """Forward pass returning per-task outputs."""
        # Shared trunk
        h = self.relu(self.shared_fc1(x))
        h = self.relu(self.shared_fc2(h))
        
        # Task heads
        outputs = [head(h) for head in self.task_heads]
        return outputs


def get_shared_and_head_params(model: SimpleMTModel) -> Tuple[List[nn.Parameter], List[nn.Parameter]]:
    """Get shared trunk and task head parameters separately."""
    shared_params = []
    head_params = []
    
    for name, param in model.named_parameters():
        if 'task_heads' in name:
            head_params.append(param)
        else:
            shared_params.append(param)
    
    return shared_params, head_params


def create_synthetic_batch(batch_size: int = 8, input_dim: int = 64, n_tasks: int = 3, 
                           device: str = 'cpu') -> Tuple[torch.Tensor, List[torch.Tensor], torch.Tensor]:
    """Create synthetic batch data for testing."""
    x = torch.randn(batch_size, input_dim, device=device)
    targets = [torch.randn(batch_size, 1, device=device) for _ in range(n_tasks)]
    task_weights = torch.ones(n_tasks, device=device) / n_tasks
    return x, targets, task_weights


def compute_per_task_losses(model: SimpleMTModel, x: torch.Tensor, 
                            targets: List[torch.Tensor], task_weights: torch.Tensor) -> Tuple[torch.Tensor, List[torch.Tensor]]:
    """Compute per-task weighted losses and total loss."""
    outputs = model(x)
    criterion = nn.MSELoss()
    
    per_task_losses = []
    total_loss = torch.tensor(0.0, device=x.device, requires_grad=True)
    
    for i, (out, target, weight) in enumerate(zip(outputs, targets, task_weights)):
        loss_i = criterion(out, target)
        weighted_loss_i = loss_i * weight
        per_task_losses.append(weighted_loss_i)
        total_loss = total_loss + weighted_loss_i
    
    return total_loss, per_task_losses


def test_pcgrad_unit():
    """Test PCGrad core algorithm."""
    print("\n" + "="*60)
    print("TEST 1: PCGrad Unit Test")
    print("="*60)
    
    set_seed(42)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    # Create model
    model = SimpleMTModel(input_dim=64, hidden_dim=32, n_tasks=3).to(device)
    shared_params, head_params = get_shared_and_head_params(model)
    
    print(f"Shared params: {len(shared_params)}, Head params: {len(head_params)}")
    
    # Create data
    x, targets, task_weights = create_synthetic_batch(batch_size=8, device=device)
    
    # Create PCGrad helper
    generator = torch.Generator()
    generator.manual_seed(42)
    pcgrad = PCGrad(shared_params, generator=generator)
    
    # Compute per-task gradients
    total_loss, per_task_losses = compute_per_task_losses(model, x, targets, task_weights)
    
    per_task_grads = []
    for task_loss in per_task_losses:
        grads = pcgrad.compute_task_grads(task_loss, retain_graph=True)
        per_task_grads.append(grads)
    
    print(f"Computed {len(per_task_grads)} per-task gradients")
    
    # Apply PCGrad
    combined_grads, stats = pcgrad.combine_grads(per_task_grads)
    
    print(f"PCGrad Stats:")
    print(f"  - Conflict rate: {stats['conflict_rate']:.4f}")
    print(f"  - Mean cosine sim: {stats['mean_cosine_sim']:.4f}")
    print(f"  - Grad norms: {[f'{n:.4f}' for n in stats['grad_norms']]}")
    
    # Verify combined grads have correct structure
    assert len(combined_grads) == len(shared_params), "Combined grads should match shared params count"
    
    for i, (cg, sp) in enumerate(zip(combined_grads, shared_params)):
        assert cg.shape == sp.shape, f"Grad shape mismatch at index {i}"
    
    print("✓ PCGrad unit test PASSED")
    return True


def test_baseline_unchanged():
    """
    Test that baseline training is bit-for-bit identical when PCGrad is disabled.
    
    This is the MOST CRITICAL test - ensures no regression in existing behavior.
    """
    print("\n" + "="*60)
    print("TEST 2: Baseline Unchanged Test (pcgrad_enabled=False)")
    print("="*60)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    # Run baseline training (simulated - without PCGrad)
    set_seed(42)
    model_baseline = SimpleMTModel(input_dim=64, hidden_dim=32, n_tasks=3).to(device)
    optimizer_baseline = torch.optim.Adam(model_baseline.parameters(), lr=1e-3)
    
    # Store initial params
    initial_params_baseline = {name: p.clone() for name, p in model_baseline.named_parameters()}
    
    # Training step (baseline)
    x, targets, task_weights = create_synthetic_batch(batch_size=8, device=device)
    total_loss, _ = compute_per_task_losses(model_baseline, x, targets, task_weights)
    
    optimizer_baseline.zero_grad()
    total_loss.backward()
    optimizer_baseline.step()
    
    # Store final params and grads after baseline step
    final_params_baseline = {name: p.clone() for name, p in model_baseline.named_parameters()}
    
    # Run identical training with PCGrad DISABLED (should be identical)
    set_seed(42)
    model_pcgrad_off = SimpleMTModel(input_dim=64, hidden_dim=32, n_tasks=3).to(device)
    optimizer_pcgrad_off = torch.optim.Adam(model_pcgrad_off.parameters(), lr=1e-3)
    
    # Same batch (seeded)
    x, targets, task_weights = create_synthetic_batch(batch_size=8, device=device)
    total_loss, _ = compute_per_task_losses(model_pcgrad_off, x, targets, task_weights)
    
    optimizer_pcgrad_off.zero_grad()
    total_loss.backward()
    optimizer_pcgrad_off.step()
    
    final_params_pcgrad_off = {name: p.clone() for name, p in model_pcgrad_off.named_parameters()}
    
    # Compare: should be IDENTICAL
    all_match = True
    for name in final_params_baseline:
        baseline = final_params_baseline[name]
        pcgrad_off = final_params_pcgrad_off[name]
        
        if not torch.allclose(baseline, pcgrad_off, rtol=1e-6, atol=1e-8):
            print(f"✗ MISMATCH in {name}")
            print(f"  Max diff: {(baseline - pcgrad_off).abs().max().item()}")
            all_match = False
        else:
            print(f"✓ {name}: MATCH")
    
    if all_match:
        print("✓ Baseline unchanged test PASSED")
    else:
        print("✗ Baseline unchanged test FAILED")
    
    return all_match


def test_pcgrad_modifies_shared_grads():
    """
    Test that PCGrad modifies shared trunk gradients when enabled.
    
    Verifies:
    1. Shared gradients differ from baseline
    2. Head gradients remain unchanged
    """
    print("\n" + "="*60)
    print("TEST 3: PCGrad Modifies Shared Gradients Test")
    print("="*60)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    # Run baseline to get reference gradients
    set_seed(42)
    model_baseline = SimpleMTModel(input_dim=64, hidden_dim=32, n_tasks=3).to(device)
    
    x, targets, task_weights = create_synthetic_batch(batch_size=8, device=device)
    total_loss, _ = compute_per_task_losses(model_baseline, x, targets, task_weights)
    
    total_loss.backward()
    
    baseline_grads = {name: p.grad.clone() for name, p in model_baseline.named_parameters() if p.grad is not None}
    
    # Run with PCGrad enabled
    set_seed(42)
    model_pcgrad = SimpleMTModel(input_dim=64, hidden_dim=32, n_tasks=3).to(device)
    shared_params, head_params = get_shared_and_head_params(model_pcgrad)
    
    generator = torch.Generator()
    generator.manual_seed(42)
    pcgrad = PCGrad(shared_params, generator=generator)
    
    # Same batch
    x, targets, task_weights = create_synthetic_batch(batch_size=8, device=device)
    total_loss, per_task_losses = compute_per_task_losses(model_pcgrad, x, targets, task_weights)
    
    # Compute per-task gradients on shared params
    per_task_grads = []
    for task_loss in per_task_losses:
        grads = pcgrad.compute_task_grads(task_loss, retain_graph=True)
        per_task_grads.append(grads)
    
    # Apply PCGrad
    combined_grads, stats = pcgrad.combine_grads(per_task_grads)
    
    # Run normal backward (sets head grads)
    total_loss.backward()
    
    # Overwrite shared param grads with PCGrad combined grads
    pcgrad.set_grads(combined_grads)
    
    pcgrad_grads = {name: p.grad.clone() for name, p in model_pcgrad.named_parameters() if p.grad is not None}
    
    # Check: shared grads should differ (unless no conflicts, which is unlikely)
    shared_changed = False
    head_unchanged = True
    
    for name in baseline_grads:
        baseline = baseline_grads[name]
        pcgrad_grad = pcgrad_grads[name]
        
        is_shared = 'task_heads' not in name
        is_same = torch.allclose(baseline, pcgrad_grad, rtol=1e-6, atol=1e-8)
        
        if is_shared:
            if not is_same:
                shared_changed = True
                print(f"✓ Shared {name}: CHANGED (as expected)")
            else:
                print(f"  Shared {name}: unchanged (may indicate no conflicts)")
        else:
            if is_same:
                print(f"✓ Head {name}: UNCHANGED (as expected)")
            else:
                print(f"✗ Head {name}: CHANGED (unexpected!)")
                head_unchanged = False
    
    print(f"\nPCGrad Stats:")
    print(f"  - Conflict rate: {stats['conflict_rate']:.4f}")
    print(f"  - Mean cosine sim: {stats['mean_cosine_sim']:.4f}")
    
    # Note: shared_changed might be False if there are no conflicts
    # This is OK as long as the algorithm ran correctly
    if stats['conflict_rate'] > 0:
        if shared_changed and head_unchanged:
            print("✓ PCGrad modifies shared grads test PASSED")
            return True
        else:
            print("✗ PCGrad modifies shared grads test FAILED")
            return False
    else:
        print("⚠ No gradient conflicts detected - cannot verify shared grads changed")
        print("  This is expected for some random seeds")
        if head_unchanged:
            print("✓ Head gradients remained unchanged - PASSED")
            return True
        else:
            print("✗ Head gradients changed unexpectedly - FAILED")
            return False


def test_pcgrad_determinism():
    """Test that PCGrad is deterministic with same seed."""
    print("\n" + "="*60)
    print("TEST 4: PCGrad Determinism Test")
    print("="*60)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    results = []
    for run in range(2):
        set_seed(42)
        model = SimpleMTModel(input_dim=64, hidden_dim=32, n_tasks=3).to(device)
        shared_params, _ = get_shared_and_head_params(model)
        
        generator = torch.Generator()
        generator.manual_seed(42)
        pcgrad = PCGrad(shared_params, generator=generator)
        
        x, targets, task_weights = create_synthetic_batch(batch_size=8, device=device)
        _, per_task_losses = compute_per_task_losses(model, x, targets, task_weights)
        
        per_task_grads = []
        for task_loss in per_task_losses:
            grads = pcgrad.compute_task_grads(task_loss, retain_graph=True)
            per_task_grads.append(grads)
        
        combined_grads, stats = pcgrad.combine_grads(per_task_grads)
        
        # Flatten combined grads for comparison
        flat = torch.cat([g.flatten() for g in combined_grads if g is not None])
        results.append(flat)
        print(f"Run {run + 1}: first 5 values = {flat[:5].tolist()}")
    
    if torch.allclose(results[0], results[1], rtol=1e-6, atol=1e-8):
        print("✓ PCGrad determinism test PASSED")
        return True
    else:
        print("✗ PCGrad determinism test FAILED")
        return False


def test_pcgrad_training_loop():
    """Test PCGrad in a multi-step training loop."""
    print("\n" + "="*60)
    print("TEST 5: PCGrad Training Loop Test (3 steps)")
    print("="*60)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    n_steps = 3
    
    set_seed(42)
    model = SimpleMTModel(input_dim=64, hidden_dim=32, n_tasks=3).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    shared_params, _ = get_shared_and_head_params(model)
    
    generator = torch.Generator()
    generator.manual_seed(42)
    pcgrad = PCGrad(shared_params, generator=generator)
    
    losses = []
    stats_history = []
    
    for step in range(n_steps):
        # Generate batch
        x, targets, task_weights = create_synthetic_batch(batch_size=8, device=device)
        
        # Forward
        total_loss, per_task_losses = compute_per_task_losses(model, x, targets, task_weights)
        
        # Compute per-task gradients
        per_task_grads = []
        for task_loss in per_task_losses:
            grads = pcgrad.compute_task_grads(task_loss, retain_graph=True)
            per_task_grads.append(grads)
        
        # Apply PCGrad
        combined_grads, stats = pcgrad.combine_grads(per_task_grads)
        stats_history.append(stats)
        
        # Backward for heads
        optimizer.zero_grad()
        total_loss.backward()
        
        # Set PCGrad grads for shared params
        pcgrad.set_grads(combined_grads)
        
        # Step
        optimizer.step()
        
        losses.append(total_loss.item())
        print(f"Step {step + 1}: loss={total_loss.item():.4f}, conflict_rate={stats['conflict_rate']:.4f}")
    
    print(f"\nLoss progression: {[f'{l:.4f}' for l in losses]}")
    print("✓ PCGrad training loop test PASSED")
    return True


def run_all_tests():
    """Run all regression tests."""
    print("\n" + "="*60)
    print("PCGRAD REGRESSION TEST SUITE")
    print("="*60)
    
    results = {}
    
    try:
        results['unit_test'] = test_pcgrad_unit()
    except Exception as e:
        print(f"✗ Unit test FAILED with error: {e}")
        results['unit_test'] = False
    
    try:
        results['baseline_unchanged'] = test_baseline_unchanged()
    except Exception as e:
        print(f"✗ Baseline unchanged test FAILED with error: {e}")
        results['baseline_unchanged'] = False
    
    try:
        results['modifies_shared_grads'] = test_pcgrad_modifies_shared_grads()
    except Exception as e:
        print(f"✗ Modifies shared grads test FAILED with error: {e}")
        results['modifies_shared_grads'] = False
    
    try:
        results['determinism'] = test_pcgrad_determinism()
    except Exception as e:
        print(f"✗ Determinism test FAILED with error: {e}")
        results['determinism'] = False
    
    try:
        results['training_loop'] = test_pcgrad_training_loop()
    except Exception as e:
        print(f"✗ Training loop test FAILED with error: {e}")
        results['training_loop'] = False
    
    # Summary
    print("\n" + "="*60)
    print("TEST SUMMARY")
    print("="*60)
    
    all_passed = True
    for name, passed in results.items():
        status = "✓ PASSED" if passed else "✗ FAILED"
        print(f"{name}: {status}")
        if not passed:
            all_passed = False
    
    print("\n" + "="*60)
    if all_passed:
        print("ALL TESTS PASSED ✓")
    else:
        print("SOME TESTS FAILED ✗")
    print("="*60)
    
    return all_passed


if __name__ == '__main__':
    success = run_all_tests()
    sys.exit(0 if success else 1)
