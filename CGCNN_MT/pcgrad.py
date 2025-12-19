"""
PCGrad (Projected Conflicting Gradients) implementation for multi-task learning.

Reference: 
    Yu, T., Kumar, S., Gupta, A., Levine, S., Hausman, K., & Finn, C. (2020).
    Gradient surgery for multi-task learning. NeurIPS 2020.
    https://arxiv.org/abs/2001.06782

This implementation follows the paper's algorithm:
1. For each task gradient g_i, iterate over other tasks g_j in random order
2. If conflict (g_i · g_j < 0), project g_i onto the normal plane of g_j:
   g_i = g_i - (g_i · g_j / ||g_j||^2) * g_j
3. Sum modified gradients across all tasks

Key design decisions:
- Option A (default): Compute per-task gradients of WEIGHTED losses (w_i * L_i)
  so PCGrad respects existing task weighting exactly.
- Only modifies shared trunk gradients; task head gradients remain unchanged.
"""

from typing import List, Dict, Tuple, Optional, Union
import torch
import torch.nn as nn
from torch import Tensor
import math


class PCGrad:
    """
    PCGrad helper class to compute and apply PCGrad-modified gradients.
    
    Usage:
        pcgrad = PCGrad(shared_params, generator=rng)
        
        # Compute per-task gradients (without modifying .grad buffers)
        per_task_grads = []
        for task_loss in task_losses:
            grads = pcgrad.compute_task_grads(task_loss, retain_graph=True)
            per_task_grads.append(grads)
        
        # Apply PCGrad algorithm
        combined_grads, stats = pcgrad.combine_grads(per_task_grads)
        
        # Set the combined gradients to shared params
        pcgrad.set_grads(combined_grads)
    """
    
    def __init__(
        self, 
        shared_params: List[nn.Parameter], 
        generator: Optional[torch.Generator] = None
    ):
        """
        Initialize PCGrad helper.
        
        Args:
            shared_params: List of shared (trunk) parameters to apply PCGrad on.
            generator: Optional torch.Generator for reproducible random task ordering.
        """
        self.shared_params = list(shared_params)
        self.generator = generator
        
        # Pre-compute total number of parameters for efficiency
        self._total_params = sum(p.numel() for p in self.shared_params if p.requires_grad)
        
    def compute_task_grads(
        self, 
        loss: Tensor, 
        retain_graph: bool = True,
        create_graph: bool = False
    ) -> List[Optional[Tensor]]:
        """
        Compute gradients of loss w.r.t. shared parameters without modifying .grad buffers.
        
        Args:
            loss: Scalar loss tensor for a single task.
            retain_graph: If True, keep computation graph for subsequent backward calls.
            create_graph: If True, create graph for higher-order derivatives.
            
        Returns:
            List of gradient tensors (or None for params without grad).
        """
        grads = torch.autograd.grad(
            outputs=loss,
            inputs=self.shared_params,
            retain_graph=retain_graph,
            create_graph=create_graph,
            allow_unused=True
        )
        # Convert to list and handle None gradients
        return [g.clone() if g is not None else None for g in grads]
    
    def combine_grads(
        self, 
        per_task_grads: List[List[Optional[Tensor]]]
    ) -> Tuple[List[Optional[Tensor]], Dict[str, float]]:
        """
        Apply PCGrad algorithm to combine per-task gradients.
        
        Args:
            per_task_grads: List of per-task gradients, where each element is
                           a list of gradients for shared parameters.
                           
        Returns:
            combined_grads: List of combined gradients for shared parameters.
            stats: Dictionary with PCGrad statistics (conflict_rate, mean_cosine_sim, etc.)
        """
        n_tasks = len(per_task_grads)
        
        if n_tasks == 0:
            return [], {'conflict_rate': 0.0, 'mean_cosine_sim': 0.0}
        
        if n_tasks == 1:
            # Single task: just return the gradients as-is
            return per_task_grads[0], {'conflict_rate': 0.0, 'mean_cosine_sim': 1.0}
        
        # Flatten gradients for easier manipulation
        flat_grads = []
        for task_grads in per_task_grads:
            flat = self._flatten_grads(task_grads)
            flat_grads.append(flat)
        
        # Compute statistics before modification
        stats = self._compute_stats(flat_grads)
        
        # Apply PCGrad algorithm
        modified_flat_grads = self._pcgrad_project(flat_grads)
        
        # Sum modified gradients
        combined_flat = torch.stack(modified_flat_grads).sum(dim=0)
        
        # Unflatten back to parameter shapes
        combined_grads = self._unflatten_grads(combined_flat, per_task_grads[0])
        
        return combined_grads, stats
    
    def set_grads(self, grads: List[Optional[Tensor]]) -> None:
        """
        Set the computed gradients to the shared parameters' .grad attributes.
        
        Args:
            grads: List of gradient tensors to set.
        """
        for param, grad in zip(self.shared_params, grads):
            if grad is not None and param.requires_grad:
                if param.grad is None:
                    param.grad = grad.clone()
                else:
                    param.grad.copy_(grad)
    
    def _flatten_grads(self, grads: List[Optional[Tensor]]) -> Tensor:
        """Flatten list of gradients into a single 1D tensor."""
        flat_parts = []
        for g in grads:
            if g is not None:
                flat_parts.append(g.flatten())
            else:
                # Find corresponding param to get shape
                idx = grads.index(g)
                param = self.shared_params[idx]
                flat_parts.append(torch.zeros(param.numel(), device=param.device, dtype=param.dtype))
        return torch.cat(flat_parts)
    
    def _unflatten_grads(
        self, 
        flat_grad: Tensor, 
        template_grads: List[Optional[Tensor]]
    ) -> List[Optional[Tensor]]:
        """Unflatten a 1D tensor back to list of gradients with original shapes."""
        unflat_grads = []
        offset = 0
        for i, template in enumerate(template_grads):
            param = self.shared_params[i]
            numel = param.numel()
            if template is not None or param.requires_grad:
                unflat_grads.append(flat_grad[offset:offset+numel].view(param.shape))
            else:
                unflat_grads.append(None)
            offset += numel
        return unflat_grads
    
    def _pcgrad_project(self, flat_grads: List[Tensor]) -> List[Tensor]:
        """
        Apply PCGrad projection algorithm.
        
        For each task gradient g_i, iterate other tasks g_j in random order.
        If conflict (g_i · g_j < 0), project:
            g_i = g_i - (g_i · g_j / ||g_j||^2) * g_j
        """
        n_tasks = len(flat_grads)
        device = flat_grads[0].device
        
        # Clone gradients for modification
        modified_grads = [g.clone() for g in flat_grads]
        
        for i in range(n_tasks):
            g_i = modified_grads[i]
            
            # Generate random order of other tasks
            other_tasks = list(range(n_tasks))
            other_tasks.remove(i)
            
            # Shuffle using the generator for reproducibility
            if self.generator is not None:
                # Use torch.randperm for shuffling
                perm = torch.randperm(len(other_tasks), generator=self.generator, device='cpu')
                other_tasks = [other_tasks[idx] for idx in perm.tolist()]
            else:
                # Use default random shuffle
                import random
                random.shuffle(other_tasks)
            
            # Project onto each conflicting gradient
            for j in other_tasks:
                g_j = flat_grads[j]  # Use original (unmodified) gradients
                
                dot_product = torch.dot(g_i, g_j)
                
                if dot_product < 0:
                    # Conflict detected: project g_i onto normal plane of g_j
                    g_j_norm_sq = torch.dot(g_j, g_j)
                    if g_j_norm_sq > 1e-12:  # Avoid division by zero
                        g_i = g_i - (dot_product / g_j_norm_sq) * g_j
            
            modified_grads[i] = g_i
        
        return modified_grads
    
    def _compute_stats(self, flat_grads: List[Tensor]) -> Dict[str, float]:
        """
        Compute statistics about gradient conflicts and similarities.
        
        Returns:
            Dictionary with:
            - conflict_rate: Fraction of task pairs with negative dot product
            - mean_cosine_sim: Mean cosine similarity between task pairs
            - grad_norms: List of L2 norms for each task's gradient
        """
        n_tasks = len(flat_grads)
        
        if n_tasks < 2:
            return {
                'conflict_rate': 0.0,
                'mean_cosine_sim': 1.0 if n_tasks == 1 else 0.0,
                'grad_norms': [g.norm().item() for g in flat_grads]
            }
        
        n_conflicts = 0
        total_pairs = 0
        cosine_sims = []
        grad_norms = [g.norm().item() for g in flat_grads]
        
        for i in range(n_tasks):
            for j in range(i + 1, n_tasks):
                g_i, g_j = flat_grads[i], flat_grads[j]
                dot_product = torch.dot(g_i, g_j)
                
                # Check for conflict
                if dot_product < 0:
                    n_conflicts += 1
                
                # Compute cosine similarity
                norm_i = g_i.norm()
                norm_j = g_j.norm()
                if norm_i > 1e-12 and norm_j > 1e-12:
                    cosine_sim = (dot_product / (norm_i * norm_j)).item()
                else:
                    cosine_sim = 0.0
                cosine_sims.append(cosine_sim)
                
                total_pairs += 1
        
        conflict_rate = n_conflicts / total_pairs if total_pairs > 0 else 0.0
        mean_cosine_sim = sum(cosine_sims) / len(cosine_sims) if cosine_sims else 0.0
        
        return {
            'conflict_rate': conflict_rate,
            'mean_cosine_sim': mean_cosine_sim,
            'grad_norms': grad_norms,
            'n_conflicts': n_conflicts,
            'total_pairs': total_pairs
        }


def pcgrad_modify(
    shared_params: List[nn.Parameter],
    per_task_grads: List[List[Optional[Tensor]]],
    generator: Optional[torch.Generator] = None
) -> Tuple[List[Optional[Tensor]], Dict[str, float]]:
    """
    Functional interface to apply PCGrad to per-task gradients.
    
    Args:
        shared_params: List of shared (trunk) parameters.
        per_task_grads: List of per-task gradients.
        generator: Optional torch.Generator for reproducibility.
        
    Returns:
        combined_grads: List of combined gradients.
        stats: Dictionary with PCGrad statistics.
    """
    pcgrad = PCGrad(shared_params, generator=generator)
    return pcgrad.combine_grads(per_task_grads)


def compute_pcgrad_stats(
    shared_params: List[nn.Parameter],
    per_task_grads: List[List[Optional[Tensor]]]
) -> Dict[str, float]:
    """
    Compute PCGrad statistics without modifying gradients.
    
    Args:
        shared_params: List of shared (trunk) parameters.
        per_task_grads: List of per-task gradients.
        
    Returns:
        stats: Dictionary with conflict_rate, mean_cosine_sim, grad_norms.
    """
    pcgrad = PCGrad(shared_params)
    flat_grads = [pcgrad._flatten_grads(grads) for grads in per_task_grads]
    return pcgrad._compute_stats(flat_grads)


def get_shared_params(
    model: nn.Module, 
    head_names: Optional[List[str]] = None
) -> Tuple[List[nn.Parameter], List[str]]:
    """
    Get shared (non-task-head) parameters from a model.
    
    Args:
        model: The model to extract parameters from.
        head_names: List of substrings that identify task head parameters.
                   Default: ["fc_outs", "task_attentions"]
                   
    Returns:
        shared_params: List of shared parameters.
        shared_param_names: List of parameter names for debugging.
    """
    if head_names is None:
        head_names = ["fc_outs", "task_attentions"]
    
    shared_params = []
    shared_param_names = []
    
    for name, param in model.named_parameters():
        if not any(h in name for h in head_names):
            shared_params.append(param)
            shared_param_names.append(name)
    
    return shared_params, shared_param_names
