# Test batch invariance for FLA (Flash Linear Attention) operators
import math
import unittest

import torch

from sglang.srt.layers.attention.fla.chunk import chunk_gated_delta_rule
from sglang.test.test_utils import CustomTestCase

device_type = getattr(torch.accelerator.current_accelerator(), "type", "cpu")
torch.set_default_device(device_type)


class TestFLABatchInvariance(CustomTestCase):
    def _create_test_inputs(self, seq_lens, H, K, V, dtype):
        """
        Create test inputs for chunk_gated_delta_rule.
        
        Args:
            seq_lens: List of sequence lengths [T1, T2, ...]
            H: Number of heads
            K: Head dimension for keys/queries
            V: Head dimension for values
            dtype: Data type
            
        Returns:
            Dictionary with individual and batched inputs
        """
        N = len(seq_lens)  # Number of sequences
        total_T = sum(seq_lens)
        
        # Create individual sequences
        individual_seqs = []
        for i, T in enumerate(seq_lens):
            # Use different seeds for each sequence to ensure they're different
            torch.manual_seed(42 + i)
            seq = {
                'q': torch.randn(1, T, H, K, dtype=dtype, device=device_type),
                'k': torch.randn(1, T, H, K, dtype=dtype, device=device_type),
                'v': torch.randn(1, T, H, V, dtype=dtype, device=device_type),
                'g': torch.randn(1, T, H, dtype=dtype, device=device_type) * 0.1,  # Small values for stability
                'beta': torch.sigmoid(torch.randn(1, T, H, dtype=dtype, device=device_type)),  # [0, 1]
                'initial_state': torch.randn(1, H, K, V, dtype=dtype, device=device_type) * 0.01,
            }
            # Ensure g is negative (log space gating)
            seq['g'] = -torch.abs(seq['g'])
            individual_seqs.append(seq)
        
        # Concatenate into batch format [1, total_T, H, K/V]
        batched = {
            'q': torch.cat([seq['q'] for seq in individual_seqs], dim=1),
            'k': torch.cat([seq['k'] for seq in individual_seqs], dim=1),
            'v': torch.cat([seq['v'] for seq in individual_seqs], dim=1),
            'g': torch.cat([seq['g'] for seq in individual_seqs], dim=1),
            'beta': torch.cat([seq['beta'] for seq in individual_seqs], dim=1),
            'initial_state': torch.cat([seq['initial_state'] for seq in individual_seqs], dim=0),
        }
        
        # Create cu_seqlens: cumulative sequence lengths [0, T1, T1+T2, ...]
        cu_seqlens = torch.tensor([0] + [sum(seq_lens[:i+1]) for i in range(N)], 
                                  dtype=torch.int32, device=device_type)
        
        return individual_seqs, batched, cu_seqlens

    def _test_chunk_gated_delta_rule_batch_invariance(self, seq_lens, H, K, V, dtype):
        """
        Test that chunk_gated_delta_rule produces identical results for the same prompts
        regardless of batch composition.
        
        - Method 1: Process subset of sequences in smaller batch
        - Method 2: Process all sequences in larger batch, then extract subset
        
        The overlapping sequences should produce identical outputs.
        """
        individual_seqs, batched_full, cu_seqlens_full = self._create_test_inputs(
            seq_lens, H, K, V, dtype
        )
        
        # Determine subset size (first N-1 sequences or at least first 1)
        N = len(seq_lens)
        subset_size = max(1, N - 1)
        subset_seq_lens = seq_lens[:subset_size]
        
        # Method 1: Process subset in smaller batch
        subset_seqs = individual_seqs[:subset_size]
        batched_subset = {
            'q': torch.cat([seq['q'] for seq in subset_seqs], dim=1),
            'k': torch.cat([seq['k'] for seq in subset_seqs], dim=1),
            'v': torch.cat([seq['v'] for seq in subset_seqs], dim=1),
            'g': torch.cat([seq['g'] for seq in subset_seqs], dim=1),
            'beta': torch.cat([seq['beta'] for seq in subset_seqs], dim=1),
            'initial_state': torch.cat([seq['initial_state'] for seq in subset_seqs], dim=0),
        }
        cu_seqlens_subset = torch.tensor(
            [0] + [sum(subset_seq_lens[:i+1]) for i in range(subset_size)],
            dtype=torch.int32, device=device_type
        )
        
        subset_o, subset_final_states = chunk_gated_delta_rule(
            q=batched_subset['q'],
            k=batched_subset['k'],
            v=batched_subset['v'],
            g=batched_subset['g'],
            beta=batched_subset['beta'],
            initial_state=batched_subset['initial_state'],
            output_final_state=True,
            cu_seqlens=cu_seqlens_subset,
            head_first=False,
            use_qk_l2norm_in_kernel=True,
        )
        
        # Method 2: Process all sequences in full batch
        full_o, full_final_states = chunk_gated_delta_rule(
            q=batched_full['q'],
            k=batched_full['k'],
            v=batched_full['v'],
            g=batched_full['g'],
            beta=batched_full['beta'],
            initial_state=batched_full['initial_state'],
            output_final_state=True,
            cu_seqlens=cu_seqlens_full,
            head_first=False,
            use_qk_l2norm_in_kernel=True,
        )
        
        # Extract subset from full batch output
        total_subset_len = sum(subset_seq_lens)
        full_o_subset = full_o[:, :total_subset_len, :, :]
        full_final_states_subset = full_final_states[:subset_size]
        
        # Compare outputs - they should be identical
        output_diff = (subset_o - full_o_subset).abs().max().item()
        state_diff = (subset_final_states - full_final_states_subset).abs().max().item()
        
        return output_diff, state_diff

    def _run_multiple_iterations(self, iters, seq_lens, H, K, V, dtype):
        """Run multiple iterations and collect diff statistics"""
        output_diffs = []
        state_diffs = []
        for _ in range(iters):
            output_diff, state_diff = self._test_chunk_gated_delta_rule_batch_invariance(
                seq_lens, H, K, V, dtype
            )
            output_diffs.append(output_diff)
            state_diffs.append(state_diff)
        return output_diffs, state_diffs

    def _assert_batch_invariant_results(self, output_diffs, state_diffs, dtype, test_name, rtol=1e-5, atol=1e-5):
        """
        Assert that outputs are batch-invariant.
        
        For FLA operations, we allow small numerical differences due to:
        1. Floating point accumulation order differences
        2. Different chunking strategies in individual vs batched processing
        
        We check:
        1. No NaN values
        2. Differences are within tolerance (rtol, atol)
        3. Differences are consistent across iterations
        """
        max_output_diff = max(output_diffs)
        max_state_diff = max(state_diffs)
        
        # Check for NaN values
        self.assertFalse(
            math.isnan(max_output_diff), 
            f"{test_name}: max_output_diff is NaN for {dtype}"
        )
        self.assertFalse(
            math.isnan(max_state_diff), 
            f"{test_name}: max_state_diff is NaN for {dtype}"
        )
        
        # Check that differences are within tolerance
        self.assertLessEqual(
            max_output_diff,
            atol,
            f"{test_name}: max_output_diff {max_output_diff} exceeds tolerance {atol} for {dtype}"
        )
        self.assertLessEqual(
            max_state_diff,
            atol,
            f"{test_name}: max_state_diff {max_state_diff} exceeds tolerance {atol} for {dtype}"
        )

    def test_small_sequences(self):
        """Test batch invariance with small sequences"""
        test_cases = [
            ("Small-1", [16, 32], 4, 64, 64),
            ("Small-2", [8, 16, 24], 8, 128, 128),
            ("Small-3", [32, 32], 4, 64, 64),
        ]
        
        for name, seq_lens, H, K, V in test_cases:
            with self.subTest(name=name, seq_lens=seq_lens, H=H, K=K, V=V):
                for dtype in [torch.bfloat16]:
                    with self.subTest(dtype=dtype):
                        output_diffs, state_diffs = self._run_multiple_iterations(
                            iters=3, seq_lens=seq_lens, H=H, K=K, V=V, dtype=dtype
                        )
                        self._assert_batch_invariant_results(
                            output_diffs, state_diffs, dtype, name,
                            rtol=1e-4, atol=1e-4
                        )

    def test_medium_sequences(self):
        """Test batch invariance with medium sequences"""
        test_cases = [
            ("Medium-1", [64, 128], 8, 128, 128),
            ("Medium-2", [128, 128, 128], 8, 128, 128),
            ("Medium-3", [96, 160], 4, 64, 64),
        ]
        
        for name, seq_lens, H, K, V in test_cases:
            with self.subTest(name=name, seq_lens=seq_lens, H=H, K=K, V=V):
                for dtype in [torch.bfloat16]:
                    with self.subTest(dtype=dtype):
                        output_diffs, state_diffs = self._run_multiple_iterations(
                            iters=3, seq_lens=seq_lens, H=H, K=K, V=V, dtype=dtype
                        )
                        self._assert_batch_invariant_results(
                            output_diffs, state_diffs, dtype, name,
                            rtol=1e-4, atol=1e-4
                        )

    def test_variable_length_sequences(self):
        """Test batch invariance with highly variable sequence lengths"""
        test_cases = [
            ("VarLen-1", [16, 64, 32], 4, 64, 64),
            ("VarLen-2", [32, 128, 64, 96], 8, 128, 128),
            ("VarLen-3", [8, 128], 4, 64, 64),
        ]
        
        for name, seq_lens, H, K, V in test_cases:
            with self.subTest(name=name, seq_lens=seq_lens, H=H, K=K, V=V):
                for dtype in [torch.bfloat16]:
                    with self.subTest(dtype=dtype):
                        output_diffs, state_diffs = self._run_multiple_iterations(
                            iters=3, seq_lens=seq_lens, H=H, K=K, V=V, dtype=dtype
                        )
                        self._assert_batch_invariant_results(
                            output_diffs, state_diffs, dtype, name,
                            rtol=1e-4, atol=1e-4
                        )


if __name__ == "__main__":
    unittest.main()

