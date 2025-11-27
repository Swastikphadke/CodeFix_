# attention_debugger.py
"""
Optional: heuristic analyzer for attention code.
Safe to import; does not affect validator.
(Scaffolding left intentionally for optional bonus work.)
"""

import inspect
import torch


class AttentionBugDetector:
    def analyze_code(self, module):
        bugs = []
        try:
            src = inspect.getsource(module)
        except Exception:
            return bugs
        if "self.scale" in src and "sqrt" not in src:
            bugs.append({
                'type': 'scaling_factor',
                'severity': 'high',
                'description': 'Potential incorrect scaling; should use sqrt(head_dim)'
            })
        return bugs


class AttentionRuntimeValidator:
    def __init__(self, tol=1e-6):
        self.tol = tol

    def validate_attention_weights(self, attn):
        sums = attn.sum(dim=-1)
        if not torch.allclose(sums, torch.ones_like(sums), atol=self.tol):
            return False, f"attention sums deviate max: {(sums - 1).abs().max().item():.6f}"
        return True, "OK"
