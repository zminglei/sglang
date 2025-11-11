# Test batch invariance for FLA decode operations (single token generation)
import math
import unittest

import torch

from sglang.srt.layers.attention.fla.fused_sigmoid_gating_recurrent import (
    fused_sigmoid_gating_delta_rule_update,
)
from sglang.srt.layers.attention.mamba.causal_conv1d_triton import (
    causal_conv1d_update,
)
from sglang.srt.utils import is_cuda
from sglang.test.test_utils import CustomTestCase

device_type = getattr(torch.accelerator.current_accelerator(), "type", "cpu")
torch.set_default_device(device_type)


class TestFLADecodeBatchInvariance(CustomTestCase):
    def _create_conv1d_test_inputs(self, num_seqs, seq_len, dim, conv_width, dtype):
        """
        Create test inputs for causal_conv1d_update.
        
        Args:
            num_seqs: Number of sequences
            seq_len: Sequence length (usually 1 for decode)
            dim: Hidden dimension
            conv_width: Convolution window width
            dtype: Data type
        """
        # Create inputs for each sequence
        individual_seqs = []
        for i in range(num_seqs):
            torch.manual_seed(42 + i)
            seq = {
                'x': torch.randn(seq_len, dim, dtype=dtype, device=device_type),
                'conv_state': torch.randn(dim, conv_width, dtype=dtype, device=device_type),
                'weight': torch.randn(dim, conv_width, dtype=dtype, device=device_type),
                'bias': torch.randn(dim, dtype=dtype, device=device_type) if i % 2 == 0 else None,
            }
            individual_seqs.append(seq)
        
        # Stack into batched format
        batched = {
            'x': torch.cat([seq['x'] for seq in individual_seqs], dim=0),  # [num_seqs * seq_len, dim]
            'conv_state': torch.stack([seq['conv_state'] for seq in individual_seqs], dim=0),  # [num_seqs, dim, conv_width]
            'weight': individual_seqs[0]['weight'],  # Shared across sequences
            'bias': individual_seqs[0]['bias'],  # Shared across sequences (or None)
        }
        
        return individual_seqs, batched

    def _test_causal_conv1d_update_batch_invariance(self, num_seqs, dim, conv_width, dtype):
        """
        Test that causal_conv1d_update produces identical results for the same sequence
        regardless of batch composition.
        """
        if not is_cuda():
            self.skipTest("causal_conv1d_update only supports CUDA")
            
        seq_len = 1  # Decode mode: single token
        
        individual_seqs, batched_full = self._create_conv1d_test_inputs(
            num_seqs, seq_len, dim, conv_width, dtype
        )
        
        # Determine subset size
        subset_size = max(1, num_seqs - 1)
        
        # Method 1: Process subset in smaller batch
        subset_seqs = individual_seqs[:subset_size]
        batched_subset = {
            'x': torch.cat([seq['x'] for seq in subset_seqs], dim=0),
            'conv_state': torch.stack([seq['conv_state'] for seq in subset_seqs], dim=0),
            'weight': subset_seqs[0]['weight'],
            'bias': subset_seqs[0]['bias'],
        }
        
        # Create conv_state_indices for subset
        conv_state_indices_subset = torch.arange(subset_size, dtype=torch.int32, device=device_type)
        
        # Clone conv states to avoid in-place modification affecting comparison
        conv_state_subset_clone = batched_subset['conv_state'].clone()
        
        subset_out = causal_conv1d_update(
            batched_subset['x'],
            conv_state_subset_clone,
            batched_subset['weight'],
            batched_subset['bias'],
            activation='silu',
            conv_state_indices=conv_state_indices_subset,
        )
        
        # Method 2: Process all sequences in full batch
        conv_state_indices_full = torch.arange(num_seqs, dtype=torch.int32, device=device_type)
        conv_state_full_clone = batched_full['conv_state'].clone()
        
        full_out = causal_conv1d_update(
            batched_full['x'],
            conv_state_full_clone,
            batched_full['weight'],
            batched_full['bias'],
            activation='silu',
            conv_state_indices=conv_state_indices_full,
        )
        
        # Extract subset from full batch output
        full_out_subset = full_out[:subset_size]
        
        # Compare outputs
        output_diff = (subset_out - full_out_subset).abs().max().item()
        
        # Also compare conv states
        state_diff = (conv_state_subset_clone - conv_state_full_clone[:subset_size]).abs().max().item()
        
        return output_diff, state_diff

    def _create_gdn_test_inputs(self, num_seqs, H, K, V, dtype):
        """
        Create test inputs for fused_sigmoid_gating_delta_rule_update.
        """
        # Create inputs for each sequence (1 token per sequence in decode mode)
        individual_seqs = []
        for i in range(num_seqs):
            torch.manual_seed(42 + i)
            seq = {
                'q': torch.randn(1, 1, H, K, dtype=dtype, device=device_type),
                'k': torch.randn(1, 1, H, K, dtype=dtype, device=device_type),
                'v': torch.randn(1, 1, H, V, dtype=dtype, device=device_type),
                'a': torch.randn(1, 1, H, dtype=dtype, device=device_type) * 0.1,
                'b': torch.randn(1, 1, H, dtype=dtype, device=device_type),
                'A_log': torch.randn(H, dtype=dtype, device=device_type),
                'dt_bias': torch.randn(H, dtype=dtype, device=device_type),
                'initial_state': torch.randn(1, H, K, V, dtype=dtype, device=device_type) * 0.01,
            }
            individual_seqs.append(seq)
        
        # Concatenate into batched format [1, num_seqs, H, K/V]
        batched = {
            'q': torch.cat([seq['q'] for seq in individual_seqs], dim=1),
            'k': torch.cat([seq['k'] for seq in individual_seqs], dim=1),
            'v': torch.cat([seq['v'] for seq in individual_seqs], dim=1),
            'a': torch.cat([seq['a'] for seq in individual_seqs], dim=1),
            'b': torch.cat([seq['b'] for seq in individual_seqs], dim=1),
            'A_log': individual_seqs[0]['A_log'],  # Shared
            'dt_bias': individual_seqs[0]['dt_bias'],  # Shared
            'initial_state': torch.cat([seq['initial_state'] for seq in individual_seqs], dim=0),
        }
        
        return individual_seqs, batched

    def _test_gdn_update_batch_invariance(self, num_seqs, H, K, V, dtype):
        """
        Test that fused_sigmoid_gating_delta_rule_update produces identical results
        for the same sequence regardless of batch composition.
        """
        individual_seqs, batched_full = self._create_gdn_test_inputs(num_seqs, H, K, V, dtype)
        
        # Determine subset size
        subset_size = max(1, num_seqs - 1)
        
        # Method 1: Process subset in smaller batch
        subset_seqs = individual_seqs[:subset_size]
        batched_subset = {
            'q': torch.cat([seq['q'] for seq in subset_seqs], dim=1),
            'k': torch.cat([seq['k'] for seq in subset_seqs], dim=1),
            'v': torch.cat([seq['v'] for seq in subset_seqs], dim=1),
            'a': torch.cat([seq['a'] for seq in subset_seqs], dim=1),
            'b': torch.cat([seq['b'] for seq in subset_seqs], dim=1),
            'A_log': subset_seqs[0]['A_log'],
            'dt_bias': subset_seqs[0]['dt_bias'],
            'initial_state': torch.cat([seq['initial_state'] for seq in subset_seqs], dim=0),
        }
        
        # Create cu_seqlens for subset: [0, 1, 2, ..., subset_size]
        cu_seqlens_subset = torch.arange(
            0, subset_size + 1, dtype=torch.int32, device=device_type
        )
        initial_state_indices_subset = torch.arange(
            subset_size, dtype=torch.int32, device=device_type
        )
        
        # Clone initial states to create a source buffer
        initial_state_source_subset = batched_subset['initial_state'].clone()
        
        subset_out = fused_sigmoid_gating_delta_rule_update(
            A_log=batched_subset['A_log'],
            dt_bias=batched_subset['dt_bias'],
            q=batched_subset['q'],
            k=batched_subset['k'],
            v=batched_subset['v'],
            a=batched_subset['a'],
            b=batched_subset['b'],
            initial_state_source=initial_state_source_subset,
            initial_state_indices=initial_state_indices_subset,
            cu_seqlens=cu_seqlens_subset,
            use_qk_l2norm_in_kernel=True,
            softplus_beta=1.0,
            softplus_threshold=20.0,
        )
        
        # Method 2: Process all sequences in full batch
        cu_seqlens_full = torch.arange(
            0, num_seqs + 1, dtype=torch.int32, device=device_type
        )
        initial_state_indices_full = torch.arange(
            num_seqs, dtype=torch.int32, device=device_type
        )
        initial_state_source_full = batched_full['initial_state'].clone()
        
        full_out = fused_sigmoid_gating_delta_rule_update(
            A_log=batched_full['A_log'],
            dt_bias=batched_full['dt_bias'],
            q=batched_full['q'],
            k=batched_full['k'],
            v=batched_full['v'],
            a=batched_full['a'],
            b=batched_full['b'],
            initial_state_source=initial_state_source_full,
            initial_state_indices=initial_state_indices_full,
            cu_seqlens=cu_seqlens_full,
            use_qk_l2norm_in_kernel=True,
            softplus_beta=1.0,
            softplus_threshold=20.0,
        )
        
        # Extract subset from full batch output
        full_out_subset = full_out[:, :subset_size, :, :]
        
        # Compare outputs
        output_diff = (subset_out - full_out_subset).abs().max().item()
        
        return output_diff

    def _run_multiple_iterations(self, iters, test_func, **kwargs):
        """Run multiple iterations and collect diff statistics"""
        diffs = []
        for _ in range(iters):
            result = test_func(**kwargs)
            if isinstance(result, tuple):
                diffs.append(result[0])  # Take first diff (output diff)
            else:
                diffs.append(result)
        return diffs

    def _assert_batch_invariant_results(self, diffs, dtype, test_name, atol=1e-5):
        """
        Assert that outputs are batch-invariant.
        """
        max_diff = max(diffs)
        
        # Check for NaN values
        self.assertFalse(
            math.isnan(max_diff), 
            f"{test_name}: max_diff is NaN for {dtype}"
        )
        
        # Check that differences are within tolerance
        self.assertLessEqual(
            max_diff,
            atol,
            f"{test_name}: max_diff {max_diff} exceeds tolerance {atol} for {dtype}"
        )

    def test_causal_conv1d_update_small(self):
        """Test batch invariance for causal_conv1d_update with small dimensions"""
        if not is_cuda():
            self.skipTest("causal_conv1d_update only supports CUDA")
            
        test_cases = [
            ("Conv1D-Small-1", 4, 256, 4),
            ("Conv1D-Small-2", 8, 512, 4),
            ("Conv1D-Small-3", 6, 128, 4),
        ]
        
        for name, num_seqs, dim, conv_width in test_cases:
            with self.subTest(name=name, num_seqs=num_seqs, dim=dim):
                for dtype in [torch.bfloat16]:
                    with self.subTest(dtype=dtype):
                        diffs = self._run_multiple_iterations(
                            iters=3,
                            test_func=self._test_causal_conv1d_update_batch_invariance,
                            num_seqs=num_seqs,
                            dim=dim,
                            conv_width=conv_width,
                            dtype=dtype,
                        )
                        self._assert_batch_invariant_results(
                            diffs, dtype, name, atol=1e-4
                        )

    def test_causal_conv1d_update_medium(self):
        """Test batch invariance for causal_conv1d_update with medium dimensions"""
        if not is_cuda():
            self.skipTest("causal_conv1d_update only supports CUDA")
            
        test_cases = [
            ("Conv1D-Medium-1", 16, 1024, 4),
            ("Conv1D-Medium-2", 12, 2048, 4),
            ("Conv1D-Medium-3", 8, 768, 4),
        ]
        
        for name, num_seqs, dim, conv_width in test_cases:
            with self.subTest(name=name, num_seqs=num_seqs, dim=dim):
                for dtype in [torch.bfloat16]:
                    with self.subTest(dtype=dtype):
                        diffs = self._run_multiple_iterations(
                            iters=3,
                            test_func=self._test_causal_conv1d_update_batch_invariance,
                            num_seqs=num_seqs,
                            dim=dim,
                            conv_width=conv_width,
                            dtype=dtype,
                        )
                        self._assert_batch_invariant_results(
                            diffs, dtype, name, atol=1e-4
                        )

    def test_gdn_update_small(self):
        """Test batch invariance for GDN update with small dimensions"""
        test_cases = [
            ("GDN-Small-1", 4, 8, 64, 64),
            ("GDN-Small-2", 8, 16, 128, 128),
            ("GDN-Small-3", 6, 4, 64, 64),
        ]
        
        for name, num_seqs, H, K, V in test_cases:
            with self.subTest(name=name, num_seqs=num_seqs, H=H, K=K, V=V):
                for dtype in [torch.bfloat16]:
                    with self.subTest(dtype=dtype):
                        diffs = self._run_multiple_iterations(
                            iters=3,
                            test_func=self._test_gdn_update_batch_invariance,
                            num_seqs=num_seqs,
                            H=H,
                            K=K,
                            V=V,
                            dtype=dtype,
                        )
                        self._assert_batch_invariant_results(
                            diffs, dtype, name, atol=1e-4
                        )

    def test_gdn_update_medium(self):
        """Test batch invariance for GDN update with medium dimensions"""
        test_cases = [
            ("GDN-Medium-1", 16, 32, 128, 128),
            ("GDN-Medium-2", 12, 16, 128, 128),
            ("GDN-Medium-3", 8, 8, 128, 128),
        ]
        
        for name, num_seqs, H, K, V in test_cases:
            with self.subTest(name=name, num_seqs=num_seqs, H=H, K=K, V=V):
                for dtype in [torch.bfloat16]:
                    with self.subTest(dtype=dtype):
                        diffs = self._run_multiple_iterations(
                            iters=3,
                            test_func=self._test_gdn_update_batch_invariance,
                            num_seqs=num_seqs,
                            H=H,
                            K=K,
                            V=V,
                            dtype=dtype,
                        )
                        self._assert_batch_invariant_results(
                            diffs, dtype, name, atol=1e-4
                        )


if __name__ == "__main__":
    unittest.main()

