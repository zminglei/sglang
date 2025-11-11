# Test batch invariance for FLA LayerNorm/RMSNorm operators
import math
import unittest

import torch

from sglang.srt.layers.attention.fla.layernorm_gated import (
    LayerNorm,
    RMSNorm,
    layernorm_fn,
)
from sglang.test.test_utils import CustomTestCase

device_type = getattr(torch.accelerator.current_accelerator(), "type", "cpu")
torch.set_default_device(device_type)


class TestLayerNormBatchInvariance(CustomTestCase):
    def _create_test_inputs(self, num_rows, hidden_size, dtype, use_gating=False):
        """
        Create test inputs for layer norm.
        
        Args:
            num_rows: Number of rows (batch * seq_len)
            hidden_size: Hidden dimension
            dtype: Data type
            use_gating: Whether to include gating tensor z
            
        Returns:
            Dictionary with test inputs
        """
        torch.manual_seed(42)
        
        inputs = {
            'x': torch.randn(num_rows, hidden_size, dtype=dtype, device=device_type),
            'weight': torch.randn(hidden_size, dtype=dtype, device=device_type),
            'bias': torch.randn(hidden_size, dtype=dtype, device=device_type),
            'z': torch.randn(num_rows, hidden_size, dtype=dtype, device=device_type) if use_gating else None,
        }
        
        return inputs

    def _test_layernorm_batch_invariance(
        self, 
        num_rows, 
        hidden_size, 
        dtype, 
        use_gating=False,
        group_size=None,
        is_rms_norm=False,
        norm_before_gate=True,
    ):
        """
        Test that layernorm_fn produces identical results for the same input rows
        regardless of batch composition.
        
        LayerNorm operates on each row independently, so the same row should
        always produce the same output regardless of what other rows are in the batch.
        
        - Method 1: Process subset of rows in smaller batch
        - Method 2: Process all rows in larger batch, then extract subset
        
        The overlapping rows should produce identical outputs.
        """
        # Create full batch
        inputs_full = self._create_test_inputs(num_rows, hidden_size, dtype, use_gating)
        
        # Determine subset size (first N-1 rows or at least first 1)
        subset_size = max(1, num_rows - 1)
        
        # Method 1: Process subset
        inputs_subset = {
            'x': inputs_full['x'][:subset_size],
            'weight': inputs_full['weight'],
            'bias': inputs_full['bias'],
            'z': inputs_full['z'][:subset_size] if use_gating else None,
        }
        
        subset_out = layernorm_fn(
            x=inputs_subset['x'],
            weight=inputs_subset['weight'],
            bias=inputs_subset['bias'],
            z=inputs_subset['z'],
            eps=1e-5,
            group_size=group_size,
            norm_before_gate=norm_before_gate,
            is_rms_norm=is_rms_norm,
        )
        
        # Method 2: Process full batch
        full_out = layernorm_fn(
            x=inputs_full['x'],
            weight=inputs_full['weight'],
            bias=inputs_full['bias'],
            z=inputs_full['z'],
            eps=1e-5,
            group_size=group_size,
            norm_before_gate=norm_before_gate,
            is_rms_norm=is_rms_norm,
        )
        
        # Extract subset from full batch output
        full_out_subset = full_out[:subset_size]
        
        # Compare outputs - they should be identical
        output_diff = (subset_out - full_out_subset).abs().max().item()
        
        return output_diff

    def _test_layernorm_module_batch_invariance(
        self,
        num_rows,
        hidden_size,
        dtype,
        use_gating=False,
        group_size=None,
        is_rms_norm=False,
        norm_before_gate=True,
    ):
        """Test batch invariance using LayerNorm/RMSNorm module"""
        # Create module
        if is_rms_norm:
            norm_module = RMSNorm(
                hidden_size=hidden_size,
                eps=1e-5,
                group_size=group_size,
                norm_before_gate=norm_before_gate,
                device=device_type,
                dtype=dtype,
            )
        else:
            norm_module = LayerNorm(
                hidden_size=hidden_size,
                eps=1e-5,
                group_size=group_size,
                norm_before_gate=norm_before_gate,
                device=device_type,
                dtype=dtype,
            )
        
        # Create full batch
        inputs_full = self._create_test_inputs(num_rows, hidden_size, dtype, use_gating)
        
        # Determine subset size
        subset_size = max(1, num_rows - 1)
        
        # Method 1: Process subset
        with torch.no_grad():
            subset_out = norm_module(
                inputs_full['x'][:subset_size],
                z=inputs_full['z'][:subset_size] if use_gating else None,
            )
        
        # Method 2: Process full batch
        with torch.no_grad():
            full_out = norm_module(
                inputs_full['x'],
                z=inputs_full['z'] if use_gating else None,
            )
        
        # Extract subset from full batch output
        full_out_subset = full_out[:subset_size]
        
        # Compare outputs
        output_diff = (subset_out - full_out_subset).abs().max().item()
        
        return output_diff

    def _run_multiple_iterations(self, iters, test_func, **kwargs):
        """Run multiple iterations and collect diff statistics"""
        diffs = []
        for _ in range(iters):
            diff = test_func(**kwargs)
            diffs.append(diff)
        return diffs

    def _assert_batch_invariant_results(self, diffs, dtype, test_name):
        """
        Assert that outputs are batch-invariant.
        
        For LayerNorm, outputs should be EXACTLY identical since each row
        is processed independently with no cross-row dependencies.
        """
        max_diff = max(diffs)
        
        # Check for NaN values
        self.assertFalse(
            math.isnan(max_diff),
            f"{test_name}: max_diff is NaN for {dtype}"
        )
        
        # LayerNorm should be perfectly batch-invariant (within floating point precision)
        # We use a very small tolerance to account for potential numerical differences
        tolerance = 1e-6 if dtype == torch.float32 else 1e-4
        self.assertLessEqual(
            max_diff,
            tolerance,
            f"{test_name}: max_diff {max_diff} exceeds tolerance {tolerance} for {dtype}"
        )

    def test_layernorm_small(self):
        """Test LayerNorm batch invariance with small dimensions"""
        test_cases = [
            ("LN-Small-1", 8, 256),
            ("LN-Small-2", 16, 512),
            ("LN-Small-3", 4, 128),
        ]
        
        for name, num_rows, hidden_size in test_cases:
            with self.subTest(name=name, num_rows=num_rows, hidden_size=hidden_size):
                for dtype in [torch.bfloat16]:
                    with self.subTest(dtype=dtype):
                        diffs = self._run_multiple_iterations(
                            iters=3,
                            test_func=self._test_layernorm_batch_invariance,
                            num_rows=num_rows,
                            hidden_size=hidden_size,
                            dtype=dtype,
                            is_rms_norm=False,
                        )
                        self._assert_batch_invariant_results(diffs, dtype, name)

    def test_layernorm_medium(self):
        """Test LayerNorm batch invariance with medium dimensions"""
        test_cases = [
            ("LN-Medium-1", 32, 1024),
            ("LN-Medium-2", 64, 2048),
            ("LN-Medium-3", 24, 768),
        ]
        
        for name, num_rows, hidden_size in test_cases:
            with self.subTest(name=name, num_rows=num_rows, hidden_size=hidden_size):
                for dtype in [torch.bfloat16]:
                    with self.subTest(dtype=dtype):
                        diffs = self._run_multiple_iterations(
                            iters=3,
                            test_func=self._test_layernorm_batch_invariance,
                            num_rows=num_rows,
                            hidden_size=hidden_size,
                            dtype=dtype,
                            is_rms_norm=False,
                        )
                        self._assert_batch_invariant_results(diffs, dtype, name)

    def test_rmsnorm_small(self):
        """Test RMSNorm batch invariance with small dimensions"""
        test_cases = [
            ("RMS-Small-1", 8, 256),
            ("RMS-Small-2", 16, 512),
            ("RMS-Small-3", 4, 128),
        ]
        
        for name, num_rows, hidden_size in test_cases:
            with self.subTest(name=name, num_rows=num_rows, hidden_size=hidden_size):
                for dtype in [torch.bfloat16]:
                    with self.subTest(dtype=dtype):
                        diffs = self._run_multiple_iterations(
                            iters=3,
                            test_func=self._test_layernorm_batch_invariance,
                            num_rows=num_rows,
                            hidden_size=hidden_size,
                            dtype=dtype,
                            is_rms_norm=True,
                        )
                        self._assert_batch_invariant_results(diffs, dtype, name)

    def test_rmsnorm_medium(self):
        """Test RMSNorm batch invariance with medium dimensions"""
        test_cases = [
            ("RMS-Medium-1", 32, 1024),
            ("RMS-Medium-2", 64, 2048),
            ("RMS-Medium-3", 24, 768),
        ]
        
        for name, num_rows, hidden_size in test_cases:
            with self.subTest(name=name, num_rows=num_rows, hidden_size=hidden_size):
                for dtype in [torch.bfloat16]:
                    with self.subTest(dtype=dtype):
                        diffs = self._run_multiple_iterations(
                            iters=3,
                            test_func=self._test_layernorm_batch_invariance,
                            num_rows=num_rows,
                            hidden_size=hidden_size,
                            dtype=dtype,
                            is_rms_norm=True,
                        )
                        self._assert_batch_invariant_results(diffs, dtype, name)

    def test_layernorm_with_gating(self):
        """Test LayerNorm batch invariance with gating (z parameter)"""
        test_cases = [
            ("LN-Gate-1", 16, 512, True),  # norm_before_gate=True
            ("LN-Gate-2", 32, 1024, False),  # norm_before_gate=False
        ]
        
        for name, num_rows, hidden_size, norm_before_gate in test_cases:
            with self.subTest(name=name, num_rows=num_rows, hidden_size=hidden_size):
                for dtype in [torch.bfloat16]:
                    with self.subTest(dtype=dtype):
                        diffs = self._run_multiple_iterations(
                            iters=3,
                            test_func=self._test_layernorm_batch_invariance,
                            num_rows=num_rows,
                            hidden_size=hidden_size,
                            dtype=dtype,
                            use_gating=True,
                            is_rms_norm=False,
                            norm_before_gate=norm_before_gate,
                        )
                        self._assert_batch_invariant_results(diffs, dtype, name)

    def test_rmsnorm_with_gating(self):
        """Test RMSNorm batch invariance with gating (z parameter)"""
        test_cases = [
            ("RMS-Gate-1", 16, 512, True),  # norm_before_gate=True
            ("RMS-Gate-2", 32, 1024, False),  # norm_before_gate=False
        ]
        
        for name, num_rows, hidden_size, norm_before_gate in test_cases:
            with self.subTest(name=name, num_rows=num_rows, hidden_size=hidden_size):
                for dtype in [torch.bfloat16]:
                    with self.subTest(dtype=dtype):
                        diffs = self._run_multiple_iterations(
                            iters=3,
                            test_func=self._test_layernorm_batch_invariance,
                            num_rows=num_rows,
                            hidden_size=hidden_size,
                            dtype=dtype,
                            use_gating=True,
                            is_rms_norm=True,
                            norm_before_gate=norm_before_gate,
                        )
                        self._assert_batch_invariant_results(diffs, dtype, name)

    def test_grouped_layernorm(self):
        """Test GroupNorm (LayerNorm with group_size) batch invariance"""
        test_cases = [
            ("GroupLN-1", 16, 512, 128),  # 4 groups
            ("GroupLN-2", 32, 1024, 256),  # 4 groups
        ]
        
        for name, num_rows, hidden_size, group_size in test_cases:
            with self.subTest(name=name, num_rows=num_rows, hidden_size=hidden_size):
                for dtype in [torch.bfloat16]:
                    with self.subTest(dtype=dtype):
                        diffs = self._run_multiple_iterations(
                            iters=3,
                            test_func=self._test_layernorm_batch_invariance,
                            num_rows=num_rows,
                            hidden_size=hidden_size,
                            dtype=dtype,
                            group_size=group_size,
                            is_rms_norm=False,
                        )
                        self._assert_batch_invariant_results(diffs, dtype, name)

    def test_layernorm_module(self):
        """Test LayerNorm module batch invariance"""
        test_cases = [
            ("LN-Module-1", 16, 512),
            ("LN-Module-2", 32, 1024),
        ]
        
        for name, num_rows, hidden_size in test_cases:
            with self.subTest(name=name, num_rows=num_rows, hidden_size=hidden_size):
                for dtype in [torch.bfloat16]:
                    with self.subTest(dtype=dtype):
                        diffs = self._run_multiple_iterations(
                            iters=3,
                            test_func=self._test_layernorm_module_batch_invariance,
                            num_rows=num_rows,
                            hidden_size=hidden_size,
                            dtype=dtype,
                            is_rms_norm=False,
                        )
                        self._assert_batch_invariant_results(diffs, dtype, name)

    def test_rmsnorm_module(self):
        """Test RMSNorm module batch invariance"""
        test_cases = [
            ("RMS-Module-1", 16, 512),
            ("RMS-Module-2", 32, 1024),
        ]
        
        for name, num_rows, hidden_size in test_cases:
            with self.subTest(name=name, num_rows=num_rows, hidden_size=hidden_size):
                for dtype in [torch.bfloat16]:
                    with self.subTest(dtype=dtype):
                        diffs = self._run_multiple_iterations(
                            iters=3,
                            test_func=self._test_layernorm_module_batch_invariance,
                            num_rows=num_rows,
                            hidden_size=hidden_size,
                            dtype=dtype,
                            is_rms_norm=True,
                        )
                        self._assert_batch_invariant_results(diffs, dtype, name)


if __name__ == "__main__":
    unittest.main()

