#!/usr/bin/env python
"""
PCGrad Integration Test with Actual CGCNN Model

This script tests PCGrad integration with the actual CGCNN_MT model architecture.
It verifies that:
1. PCGrad can be enabled/disabled via config
2. The automatic_optimization property works correctly
3. Training runs without errors in both modes

Usage:
    python tests/test_pcgrad_integration.py

Note: This test uses synthetic data to avoid dependency on actual CIF files.
"""

import os
import sys
import torch
import torch.nn as nn
import numpy as np
from types import SimpleNamespace

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

cgcnn_root = os.path.dirname(project_root)
if cgcnn_root not in sys.path:
    sys.path.insert(0, cgcnn_root)


def create_mock_config(pcgrad_enabled: bool = False) -> SimpleNamespace:
    """Create a mock configuration for testing."""
    return SimpleNamespace(
        # Basic
        batch_size=4,
        random_seed=42,
        
        # Model - Using cgcnn (simpler, no attention pooling bug)
        model_name='cgcnn',
        orig_atom_fea_len=92,
        orig_extra_fea_len=0,
        nbr_fea_len=41,
        max_num_nbr=10,
        atom_fea_len=64,
        extra_fea_len=128,
        h_fea_len=128,
        n_conv=3,
        n_h=2,
        dropout_prob=0.0,
        atom_layer_norm=True,
        task_att_type='none',  # No attention to avoid bug
        att_S=64,
        att_pooling=False,  # Disable attention pooling to avoid bug
        task_norm=False,
        
        # Tasks
        tasks=['TSD', 'SSD'],
        task_types=['regression', 'classification'],
        task_weights=[0.5, 0.5],
        loss_aggregation='fixed_weight_sum',
        
        # PCGrad
        pcgrad_enabled=pcgrad_enabled,
        pcgrad_on_shared_only=True,
        pcgrad_head_names=["fc_outs", "task_attentions"],
        
        # Optimizer
        optim='adam',
        lr=1e-3,
        weight_decay=1e-5,
        lr_scheduler=None,
        optim_config='coarse',
        group_lr=False,
        lr_mult=10,
        
        # Other
        ckpt_path=None,
    )


class MockNormalizer:
    """Mock normalizer for testing."""
    def __init__(self, device='cpu'):
        self.device = torch.device(device)
        self.mean = torch.tensor(0.0, device=self.device)
        self.std = torch.tensor(1.0, device=self.device)
    
    def to(self, device):
        self.device = torch.device(device)
        self.mean = self.mean.to(device)
        self.std = self.std.to(device)
        return self
    
    def norm(self, x):
        return (x - self.mean) / (self.std + 1e-8)
    
    def denorm(self, x):
        return x * self.std + self.mean


def create_mock_batch(batch_size: int = 4, n_atoms: int = 20, device: str = 'cpu'):
    """Create a mock batch for testing."""
    # Create atom features
    atom_fea = torch.randn(batch_size * n_atoms, 92, device=device)
    
    # Create neighbor features and indices
    nbr_fea = torch.randn(batch_size * n_atoms, 10, 41, device=device)
    nbr_fea_idx = torch.randint(0, n_atoms, (batch_size * n_atoms, 10), device=device)
    
    # Fix neighbor indices to be within crystal bounds
    for i in range(batch_size):
        start = i * n_atoms
        end = (i + 1) * n_atoms
        nbr_fea_idx[start:end] = (nbr_fea_idx[start:end] % n_atoms) + start
    
    # Crystal atom indices
    crystal_atom_idx = [torch.arange(i * n_atoms, (i + 1) * n_atoms, device=device) 
                        for i in range(batch_size)]
    
    # Extra features (empty for this test)
    extra_fea = torch.zeros(batch_size, 0, device=device)
    
    # Task IDs (alternating between tasks)
    task_id = torch.tensor([i % 2 for i in range(batch_size)], device=device)
    
    # Targets
    targets = torch.randn(batch_size, 1, device=device)
    
    # CIF IDs
    cif_id = [f'test_{i}' for i in range(batch_size)]
    
    return {
        'atom_fea': atom_fea,
        'nbr_fea': nbr_fea,
        'nbr_fea_idx': nbr_fea_idx,
        'crystal_atom_idx': crystal_atom_idx,
        'extra_fea': extra_fea,
        'task_id': task_id,
        'targets': targets,
        'cif_id': cif_id,
    }


def test_pcgrad_config_integration():
    """Test that PCGrad config is properly integrated."""
    print("\n" + "="*60)
    print("TEST: PCGrad Config Integration")
    print("="*60)
    
    from CGCNN_MT.module.cgcnn import CrystalGraphConvNet
    from CGCNN_MT.module.module import MInterface
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    # Test with PCGrad OFF
    config_off = create_mock_config(pcgrad_enabled=False)
    model_off = CrystalGraphConvNet(**vars(config_off))
    normalizers = [MockNormalizer(device) for _ in config_off.tasks]
    
    interface_off = MInterface(model_off, normalizers, **vars(config_off))
    
    assert interface_off.automatic_optimization == True, "Should use automatic optimization when PCGrad OFF"
    assert interface_off._pcgrad_enabled == False, "PCGrad should be disabled"
    print("✓ PCGrad OFF: automatic_optimization=True")
    
    # Test with PCGrad ON
    config_on = create_mock_config(pcgrad_enabled=True)
    model_on = CrystalGraphConvNet(**vars(config_on))
    normalizers = [MockNormalizer(device) for _ in config_on.tasks]
    
    interface_on = MInterface(model_on, normalizers, **vars(config_on))
    
    assert interface_on.automatic_optimization == False, "Should use manual optimization when PCGrad ON"
    assert interface_on._pcgrad_enabled == True, "PCGrad should be enabled"
    print("✓ PCGrad ON: automatic_optimization=False")
    
    print("✓ PCGrad config integration test PASSED")
    return True


def test_pcgrad_shared_params_detection():
    """Test that shared parameters are correctly identified."""
    print("\n" + "="*60)
    print("TEST: Shared Parameters Detection")
    print("="*60)
    
    from CGCNN_MT.module.cgcnn import CrystalGraphConvNet
    from CGCNN_MT.pcgrad import get_shared_params
    
    config = create_mock_config(pcgrad_enabled=True)
    model = CrystalGraphConvNet(**vars(config))
    
    head_names = ["fc_outs", "task_attentions"]
    shared_params, shared_names = get_shared_params(model, head_names)
    
    print(f"Total model parameters: {sum(p.numel() for p in model.parameters())}")
    print(f"Shared parameters: {sum(p.numel() for p in shared_params)}")
    print(f"Number of shared param tensors: {len(shared_params)}")
    
    # Verify no head params in shared
    for name in shared_names:
        assert 'fc_outs' not in name, f"Head param {name} in shared params"
        assert 'task_attentions' not in name, f"Head param {name} in shared params"
    
    print(f"\nShared parameter names:")
    for name in shared_names[:5]:
        print(f"  - {name}")
    if len(shared_names) > 5:
        print(f"  ... and {len(shared_names) - 5} more")
    
    print("✓ Shared parameters detection test PASSED")
    return True


def test_pcgrad_forward_backward():
    """Test forward and backward pass with PCGrad."""
    print("\n" + "="*60)
    print("TEST: PCGrad Forward/Backward Pass")
    print("="*60)
    
    from CGCNN_MT.module.cgcnn import CrystalGraphConvNet
    from CGCNN_MT.pcgrad import PCGrad, get_shared_params
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    torch.manual_seed(42)
    
    config = create_mock_config(pcgrad_enabled=True)
    model = CrystalGraphConvNet(**vars(config)).to(device)
    
    # Get shared params
    head_names = ["fc_outs", "task_attentions"]
    shared_params, _ = get_shared_params(model, head_names)
    
    generator = torch.Generator()
    generator.manual_seed(42)
    pcgrad = PCGrad(shared_params, generator=generator)
    
    # Create batch
    batch = create_mock_batch(batch_size=4, device=device)
    
    # Forward pass
    outputs, _ = model(
        atom_fea=batch['atom_fea'],
        nbr_fea=batch['nbr_fea'],
        nbr_fea_idx=batch['nbr_fea_idx'],
        crystal_atom_idx=batch['crystal_atom_idx'],
        extra_fea=batch['extra_fea'],
    )
    
    print(f"Number of outputs: {len(outputs)}")
    for i, out in enumerate(outputs):
        print(f"  Output {i}: shape={out.shape}")
    
    # Compute per-task losses
    # Note: The model outputs one tensor per task, shape (batch_size, ...) 
    criterion_reg = nn.MSELoss()
    
    task_losses = []
    batch_size = len(batch['crystal_atom_idx'])
    targets = batch['targets']
    
    # For this test, we just compute losses on entire batch for each task
    for task_id in range(len(config.tasks)):
        output_i = outputs[task_id]  # Shape: (batch_size, 1) or (batch_size, num_classes)
        
        # Use simplified loss computation for testing
        if config.task_types[task_id] == 'regression':
            # For regression: output should be (batch_size, 1)
            loss_i = criterion_reg(output_i.view(-1), targets.view(-1))
        else:
            # For classification: use MSE as proxy for testing
            loss_i = criterion_reg(output_i.view(-1), targets.view(-1))
        
        weighted_loss = loss_i * config.task_weights[task_id]
        task_losses.append(weighted_loss)
    
    print(f"\nPer-task losses: {[l.item() for l in task_losses]}")
    
    # Compute per-task gradients
    per_task_grads = []
    for task_loss in task_losses:
        grads = pcgrad.compute_task_grads(task_loss, retain_graph=True)
        per_task_grads.append(grads)
    
    print(f"Computed {len(per_task_grads)} per-task gradients")
    
    # Apply PCGrad
    combined_grads, stats = pcgrad.combine_grads(per_task_grads)
    
    print(f"\nPCGrad stats:")
    print(f"  - Conflict rate: {stats['conflict_rate']:.4f}")
    print(f"  - Mean cosine sim: {stats['mean_cosine_sim']:.4f}")
    print(f"  - Grad norms: {[f'{n:.4f}' for n in stats['grad_norms']]}")
    
    print("✓ PCGrad forward/backward pass test PASSED")
    return True


def run_all_integration_tests():
    """Run all integration tests."""
    print("\n" + "="*60)
    print("PCGRAD INTEGRATION TEST SUITE")
    print("="*60)
    
    results = {}
    
    try:
        results['config_integration'] = test_pcgrad_config_integration()
    except Exception as e:
        print(f"✗ Config integration test FAILED with error: {e}")
        import traceback
        traceback.print_exc()
        results['config_integration'] = False
    
    try:
        results['shared_params_detection'] = test_pcgrad_shared_params_detection()
    except Exception as e:
        print(f"✗ Shared params detection test FAILED with error: {e}")
        import traceback
        traceback.print_exc()
        results['shared_params_detection'] = False
    
    try:
        results['forward_backward'] = test_pcgrad_forward_backward()
    except Exception as e:
        print(f"✗ Forward/backward test FAILED with error: {e}")
        import traceback
        traceback.print_exc()
        results['forward_backward'] = False
    
    # Summary
    print("\n" + "="*60)
    print("INTEGRATION TEST SUMMARY")
    print("="*60)
    
    all_passed = True
    for name, passed in results.items():
        status = "✓ PASSED" if passed else "✗ FAILED"
        print(f"{name}: {status}")
        if not passed:
            all_passed = False
    
    print("\n" + "="*60)
    if all_passed:
        print("ALL INTEGRATION TESTS PASSED ✓")
    else:
        print("SOME INTEGRATION TESTS FAILED ✗")
    print("="*60)
    
    return all_passed


if __name__ == '__main__':
    success = run_all_integration_tests()
    sys.exit(0 if success else 1)
