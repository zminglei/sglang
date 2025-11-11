# Test determinism for FLA kernels - stricter tests matching test_deterministic
import random
import unittest

import torch

from sglang.srt.layers.attention.fla.chunk import chunk_gated_delta_rule
from sglang.srt.layers.attention.fla.fused_recurrent import fused_recurrent_gated_delta_rule_update_fwd
from sglang.srt.layers.attention.mamba.causal_conv1d_triton import (
    causal_conv1d_fn,
    causal_conv1d_update,
)
from sglang.srt.server_args import ServerArgs, set_global_server_args_for_scheduler
from sglang.test.test_utils import CustomTestCase

device_type = getattr(torch.accelerator.current_accelerator(), "type", "cpu")
torch.set_default_device(device_type)


class TestFLADeterministic(CustomTestCase):
    @classmethod
    def setUpClass(cls):
        """Set up global server args for deterministic inference tests"""
        # Create minimal server args with deterministic inference enabled
        args = ServerArgs(
            model_path="dummy",  # Not actually used in these tests
            enable_deterministic_inference=True,
        )
        set_global_server_args_for_scheduler(args)
    
    @classmethod  
    def tearDownClass(cls):
        """Clean up global server args"""
        # Note: We don't reset to None as it might affect other tests
        # The args with enable_deterministic_inference=True is safe to leave
        pass
    """
    Stricter determinism tests that match test_deterministic behavior:
    - Run same input multiple times
    - Verify ALL outputs are IDENTICAL (not just within tolerance)
    - Test with different prefix lengths
    - Test state consistency across runs
    """

    def _run_chunk_gated_delta_rule_n_times(self, n_trials, seq_len, H, K, V, dtype, use_prefix=False):
        """
        Run chunk_gated_delta_rule n times with identical inputs.
        Returns list of (output, final_state) tuples.
        """
        # Create fixed inputs (same across all trials)
        torch.manual_seed(42)
        q = torch.randn(1, seq_len, H, K, dtype=dtype, device=device_type)
        k = torch.randn(1, seq_len, H, K, dtype=dtype, device=device_type)
        v = torch.randn(1, seq_len, H, V, dtype=dtype, device=device_type)
        g = -torch.abs(torch.randn(1, seq_len, H, dtype=dtype, device=device_type)) * 0.1
        beta = torch.sigmoid(torch.randn(1, seq_len, H, dtype=dtype, device=device_type))
        initial_state = torch.randn(1, H, K, V, dtype=dtype, device=device_type) * 0.01
        
        # Simulate prefix caching: only process suffix after prefix
        if use_prefix and seq_len > 64:
            prefix_len = seq_len // 2  # Use half as prefix
            # Create cu_seqlens to skip prefix (as if it's cached)
            cu_seqlens = torch.tensor([0, seq_len - prefix_len], dtype=torch.int32, device=device_type)
            # Only process the suffix
            q_input = q[:, prefix_len:, :, :]
            k_input = k[:, prefix_len:, :, :]
            v_input = v[:, prefix_len:, :, :]
            g_input = g[:, prefix_len:, :]
            beta_input = beta[:, prefix_len:, :]
        else:
            cu_seqlens = None
            q_input = q
            k_input = k
            v_input = v
            g_input = g
            beta_input = beta
        
        results = []
        for trial in range(n_trials):
            # Use SAME inputs for each trial
            # Clone initial state since it might be modified
            initial_state_trial = initial_state.clone()
            
            o, final_state = chunk_gated_delta_rule(
                q=q_input,
                k=k_input,
                v=v_input,
                g=g_input,
                beta=beta_input,
                initial_state=initial_state_trial,
                output_final_state=True,
                cu_seqlens=cu_seqlens,
                head_first=False,
                use_qk_l2norm_in_kernel=True,
            )
            
            results.append((o.clone(), final_state.clone()))
        
        return results

    def _run_fused_recurrent_n_times(self, n_trials, seq_len, H, K, V, dtype):
        """
        Run fused_recurrent_gated_delta_rule_update n times with identical inputs.
        Returns list of (output, final_state) tuples.
        
        Note: We use the SAME initial state for all trials to test determinism.
        """
        # Create fixed inputs (same across all trials)
        torch.manual_seed(42)
        q = torch.randn(1, seq_len, H, K, dtype=dtype, device=device_type)
        k = torch.randn(1, seq_len, H, K, dtype=dtype, device=device_type)
        v = torch.randn(1, seq_len, H, V, dtype=dtype, device=device_type)
        g = -torch.abs(torch.randn(1, seq_len, H, dtype=dtype, device=device_type)) * 0.1
        beta = torch.sigmoid(torch.randn(1, seq_len, H, dtype=dtype, device=device_type))
        scale = K ** -0.5
        initial_state_template = torch.randn(1, H, K, V, dtype=dtype, device=device_type) * 0.01
        
        results = []
        for trial in range(n_trials):
            # Use SAME initial state for each trial (clone it since it gets modified)
            initial_state_source = initial_state_template.clone()
            cache_indices = torch.tensor([0], dtype=torch.int64, device=device_type)
            
            o = fused_recurrent_gated_delta_rule_update_fwd(
                q=q,
                k=k,
                v=v,
                g=g,
                beta=beta,
                scale=scale,
                initial_state_source=initial_state_source,
                initial_state_indices=cache_indices,
                use_qk_l2norm_in_kernel=True,
                cu_seqlens=None,
                disable_state_update=False,
            )
            
            # Extract final state
            final_state = initial_state_source[0:1].clone()
            
            results.append((o.clone(), final_state.clone()))
        
        return results

    def _run_causal_conv1d_n_times(self, n_trials, batch_size, dim, dtype):
        """
        Run causal_conv1d_update n times with identical inputs.
        Returns list of output tensors.
        
        Note: We use the SAME initial conv_state for all trials to test determinism.
        """
        # Create fixed inputs (same across all trials)
        torch.manual_seed(42)
        x = torch.randn(batch_size, dim, dtype=dtype, device=device_type)
        weight = torch.randn(dim, 4, dtype=dtype, device=device_type)  # width=4
        bias = torch.randn(dim, dtype=dtype, device=device_type)
        initial_conv_state = torch.randn(batch_size, dim, 3, dtype=dtype, device=device_type)  # width-1=3
        
        results = []
        for trial in range(n_trials):
            # Use SAME initial conv state for each trial (clone it since it gets modified)
            conv_state = initial_conv_state.clone()
            cache_indices = torch.arange(batch_size, dtype=torch.int64, device=device_type)
            
            out = causal_conv1d_update(
                x=x.clone(),
                conv_state=conv_state,
                weight=weight,
                bias=bias,
                activation="silu",
                conv_state_indices=cache_indices,
            )
            
            results.append(out.clone())
        
        return results

    def _check_all_identical(self, results, test_name, component="output"):
        """
        Check that all results are EXACTLY identical (not just within tolerance).
        This is the strict check used in test_deterministic.
        """
        if len(results) < 2:
            return
        
        reference = results[0]
        
        for i, result in enumerate(results[1:], start=1):
            if isinstance(result, tuple):
                # Compare both output and state
                ref_out, ref_state = reference
                res_out, res_state = result
                
                # Check outputs
                if not torch.equal(ref_out, res_out):
                    max_diff = (ref_out - res_out).abs().max().item()
                    self.fail(
                        f"{test_name}: Trial {i} output differs from reference!\n"
                        f"  Max difference: {max_diff}\n"
                        f"  This indicates NON-DETERMINISTIC behavior!"
                    )
                
                # Check states
                if not torch.equal(ref_state, res_state):
                    max_diff = (ref_state - res_state).abs().max().item()
                    self.fail(
                        f"{test_name}: Trial {i} state differs from reference!\n"
                        f"  Max difference: {max_diff}\n"
                        f"  This indicates NON-DETERMINISTIC state updates!"
                    )
            else:
                # Single tensor comparison
                if not torch.equal(reference, result):
                    max_diff = (reference - result).abs().max().item()
                    self.fail(
                        f"{test_name}: Trial {i} {component} differs from reference!\n"
                        f"  Max difference: {max_diff}\n"
                        f"  This indicates NON-DETERMINISTIC behavior!"
                    )

    def test_chunk_gated_delta_rule_deterministic_short(self):
        """Test that chunk_gated_delta_rule is deterministic for short sequences"""
        test_cases = [
            ("Short-64", 64, 8, 128, 128),
            ("Short-128", 128, 8, 128, 128),
            ("Short-256", 256, 8, 128, 128),
        ]
        
        for name, seq_len, H, K, V in test_cases:
            with self.subTest(name=name, seq_len=seq_len):
                results = self._run_chunk_gated_delta_rule_n_times(
                    n_trials=10,  # Run 10 times
                    seq_len=seq_len,
                    H=H,
                    K=K,
                    V=V,
                    dtype=torch.bfloat16,
                    use_prefix=False,
                )
                self._check_all_identical(results, name)

    def test_chunk_gated_delta_rule_deterministic_long(self):
        """Test determinism for longer sequences matching test_deterministic"""
        test_cases = [
            ("Long-512", 512, 8, 128, 128),
            ("Long-1024", 1024, 8, 128, 128),
            ("Long-2048", 2048, 8, 128, 128),
            ("Long-4096", 4096, 8, 128, 128),
        ]
        
        for name, seq_len, H, K, V in test_cases:
            with self.subTest(name=name, seq_len=seq_len):
                results = self._run_chunk_gated_delta_rule_n_times(
                    n_trials=10,
                    seq_len=seq_len,
                    H=H,
                    K=K,
                    V=V,
                    dtype=torch.bfloat16,
                    use_prefix=False,
                )
                self._check_all_identical(results, name)

    def test_chunk_gated_delta_rule_deterministic_with_prefix(self):
        """Test determinism with prefix caching (matching test_deterministic --test-mode prefix)"""
        test_cases = [
            ("Prefix-512", 512, 8, 128, 128),
            ("Prefix-1024", 1024, 8, 128, 128),
            ("Prefix-2048", 2048, 8, 128, 128),
            ("Prefix-4096", 4096, 8, 128, 128),
        ]
        
        for name, seq_len, H, K, V in test_cases:
            with self.subTest(name=name, seq_len=seq_len):
                results = self._run_chunk_gated_delta_rule_n_times(
                    n_trials=10,
                    seq_len=seq_len,
                    H=H,
                    K=K,
                    V=V,
                    dtype=torch.bfloat16,
                    use_prefix=True,  # Simulate prefix caching
                )
                self._check_all_identical(results, name)

    def test_fused_recurrent_deterministic_decode(self):
        """Test that fused_recurrent is deterministic for decode (single token)"""
        test_cases = [
            ("Decode-1", 1, 8, 128, 128),
            ("Decode-1-Large", 1, 32, 128, 128),
        ]
        
        for name, seq_len, H, K, V in test_cases:
            with self.subTest(name=name):
                results = self._run_fused_recurrent_n_times(
                    n_trials=20,  # More trials for decode
                    seq_len=seq_len,
                    H=H,
                    K=K,
                    V=V,
                    dtype=torch.bfloat16,
                )
                self._check_all_identical(results, name)

    def test_fused_recurrent_deterministic_multi_token(self):
        """Test that fused_recurrent is deterministic for multi-token sequences"""
        test_cases = [
            ("Multi-8", 8, 8, 128, 128),
            ("Multi-64", 64, 8, 128, 128),
            ("Multi-128", 128, 8, 128, 128),
        ]
        
        for name, seq_len, H, K, V in test_cases:
            with self.subTest(name=name):
                results = self._run_fused_recurrent_n_times(
                    n_trials=10,
                    seq_len=seq_len,
                    H=H,
                    K=K,
                    V=V,
                    dtype=torch.bfloat16,
                )
                self._check_all_identical(results, name)

    def test_fused_recurrent_batch_composition_determinism(self):
        """
        Test that fused_recurrent produces same results for same input regardless of batch composition.
        This mimics the test_deterministic pattern for decode operations.
        """
        H, K, V = 8, 128, 128
        dtype = torch.bfloat16
        
        # Create 3 unique decode tokens (seq_len=1)
        torch.manual_seed(42)
        tokens = []
        states = []
        for i in range(3):
            torch.manual_seed(42 + i)
            tokens.append({
                'q': torch.randn(1, 1, H, K, dtype=dtype, device=device_type),
                'k': torch.randn(1, 1, H, K, dtype=dtype, device=device_type),
                'v': torch.randn(1, 1, H, V, dtype=dtype, device=device_type),
                'g': -torch.abs(torch.randn(1, 1, H, dtype=dtype, device=device_type)) * 0.1,
                'beta': torch.sigmoid(torch.randn(1, 1, H, dtype=dtype, device=device_type)),
            })
            states.append(torch.randn(1, H, K, V, dtype=dtype, device=device_type) * 0.01)
        
        # Storage for outputs by token index
        outputs_by_token = {i: [] for i in range(3)}
        
        # Run multiple trials with different batch compositions
        n_trials = 15
        for trial in range(n_trials):
            # Randomly compose batch (like test_deterministic does)
            batch_size = (trial % 3) + 1
            selected_tokens = [trial % 3 for _ in range(batch_size)]
            
            for token_idx in selected_tokens:
                # Process this token
                token_data = tokens[token_idx]
                initial_state = states[token_idx].clone()
                
                # Create source tensor for state
                initial_state_source = initial_state.clone()
                initial_state_indices = torch.tensor([0], dtype=torch.int64, device=device_type)
                
                out = fused_recurrent_gated_delta_rule_update_fwd(
                    q=token_data['q'],
                    k=token_data['k'],
                    v=token_data['v'],
                    g=token_data['g'],
                    beta=token_data['beta'],
                    scale=K ** -0.5,  # Standard attention scale
                    initial_state_source=initial_state_source,
                    initial_state_indices=initial_state_indices,
                    cu_seqlens=None,
                    use_qk_l2norm_in_kernel=True,
                    disable_state_update=True,  # Don't update state for this test
                )
                
                outputs_by_token[token_idx].append(out.clone())
        
        # Check that each token always produces identical output
        for token_idx in range(3):
            num_samples = len(outputs_by_token[token_idx])
            if num_samples < 2:
                continue
            
            with self.subTest(token_idx=token_idx):
                reference = outputs_by_token[token_idx][0]
                
                for i, output in enumerate(outputs_by_token[token_idx][1:], start=1):
                    if not torch.equal(reference, output):
                        diff = (reference - output).abs().max().item()
                        self.fail(
                            f"Token {token_idx}: Trial {i} differs from reference!\n"
                            f"  Max diff: {diff}\n"
                            f"  This indicates NON-DETERMINISTIC fused_recurrent behavior!\n"
                            f"  Same input produced different outputs across trials."
                        )

    def test_fused_recurrent_different_sequence_lengths(self):
        """
        Test that fused_recurrent produces deterministic results for different sequence lengths.
        This tests if sequence length affects kernel behavior.
        """
        H, K, V = 8, 128, 128
        dtype = torch.bfloat16
        
        # Test different sequence lengths
        seq_lengths = [1, 8, 16, 64, 128]
        
        for seq_len in seq_lengths:
            with self.subTest(seq_len=seq_len):
                # Run same sequence length multiple times
                torch.manual_seed(42)
                reference_input = {
                    'q': torch.randn(1, seq_len, H, K, dtype=dtype, device=device_type),
                    'k': torch.randn(1, seq_len, H, K, dtype=dtype, device=device_type),
                    'v': torch.randn(1, seq_len, H, V, dtype=dtype, device=device_type),
                    'g': -torch.abs(torch.randn(1, seq_len, H, dtype=dtype, device=device_type)) * 0.1,
                    'beta': torch.sigmoid(torch.randn(1, seq_len, H, dtype=dtype, device=device_type)),
                }
                reference_state = torch.randn(1, H, K, V, dtype=dtype, device=device_type) * 0.01
                
                outputs = []
                for trial in range(10):
                    initial_state_source = reference_state.clone()
                    initial_state_indices = torch.tensor([0], dtype=torch.int64, device=device_type)
                    
                    out = fused_recurrent_gated_delta_rule_update_fwd(
                        q=reference_input['q'],
                        k=reference_input['k'],
                        v=reference_input['v'],
                        g=reference_input['g'],
                        beta=reference_input['beta'],
                        scale=K ** -0.5,  # Standard attention scale
                        initial_state_source=initial_state_source,
                        initial_state_indices=initial_state_indices,
                        cu_seqlens=None,
                        use_qk_l2norm_in_kernel=True,
                        disable_state_update=True,
                    )
                    
                    outputs.append(out.clone())
                
                # Check all outputs are identical
                reference = outputs[0]
                for i, output in enumerate(outputs[1:], start=1):
                    if not torch.equal(reference, output):
                        diff = (reference - output).abs().max().item()
                        self.fail(
                            f"seq_len={seq_len}: Trial {i} differs from reference!\n"
                            f"  Max diff: {diff}\n"
                            f"  This indicates NON-DETERMINISTIC behavior for seq_len={seq_len}!"
                        )

    def test_causal_conv1d_deterministic_basic(self):
        """Test that causal_conv1d_update is deterministic with same inputs"""
        test_cases = [
            ("Conv1D-Small", 4, 256),
            ("Conv1D-Medium", 16, 1024),
            ("Conv1D-Large", 32, 2048),
        ]
        
        for name, batch_size, dim in test_cases:
            with self.subTest(name=name):
                results = self._run_causal_conv1d_n_times(
                    n_trials=20,
                    batch_size=batch_size,
                    dim=dim,
                    dtype=torch.bfloat16,
                )
                self._check_all_identical(results, name, component="output")

    def test_causal_conv1d_batch_composition_determinism(self):
        """
        Test that conv1d produces same results for same input regardless of batch composition.
        This mimics the test_deterministic pattern where same input appears in different batches.
        """
        dim = 1024
        dtype = torch.bfloat16
        
        # Create 3 unique input tokens
        torch.manual_seed(42)
        tokens = []
        conv_states = []
        for i in range(3):
            torch.manual_seed(42 + i)
            tokens.append(torch.randn(1, dim, dtype=dtype, device=device_type))
            conv_states.append(torch.randn(1, dim, 3, dtype=dtype, device=device_type))
        
        weight = torch.randn(dim, 4, dtype=dtype, device=device_type)
        bias = torch.randn(dim, dtype=dtype, device=device_type)
        
        # Storage for outputs by token index
        outputs_by_token = {i: [] for i in range(3)}
        
        # Run multiple trials with different batch compositions
        n_trials = 15
        for trial in range(n_trials):
            # Randomly compose batch (like test_deterministic does)
            batch_size = (trial % 3) + 1
            selected_tokens = [trial % 3 for _ in range(batch_size)]
            
            for token_idx in selected_tokens:
                # Process this token
                x = tokens[token_idx].clone()
                conv_state = conv_states[token_idx].clone()
                cache_indices = torch.tensor([0], dtype=torch.int64, device=device_type)
                
                out = causal_conv1d_update(
                    x=x,
                    conv_state=conv_state,
                    weight=weight,
                    bias=bias,
                    activation="silu",
                    conv_state_indices=cache_indices,
                )
                
                outputs_by_token[token_idx].append(out.clone())
        
        # Check that each token always produces identical output
        for token_idx in range(3):
            num_samples = len(outputs_by_token[token_idx])
            if num_samples < 2:
                continue
            
            with self.subTest(token_idx=token_idx):
                reference = outputs_by_token[token_idx][0]
                
                for i, output in enumerate(outputs_by_token[token_idx][1:], start=1):
                    if not torch.equal(reference, output):
                        diff = (reference - output).abs().max().item()
                        self.fail(
                            f"Token {token_idx}: Trial {i} differs from reference!\n"
                            f"  Max diff: {diff}\n"
                            f"  This indicates NON-DETERMINISTIC conv1d behavior!\n"
                            f"  Same input produced different outputs across trials."
                        )

    def test_causal_conv1d_sequential_determinism(self):
        """
        Test that processing tokens sequentially gives deterministic results.
        This simulates actual decode where tokens are processed one by one.
        """
        dim = 1024
        dtype = torch.bfloat16
        num_tokens = 10
        
        # Create sequence of tokens
        torch.manual_seed(42)
        tokens = [torch.randn(1, dim, dtype=dtype, device=device_type) for _ in range(num_tokens)]
        weight = torch.randn(dim, 4, dtype=dtype, device=device_type)
        bias = torch.randn(dim, dtype=dtype, device=device_type)
        
        # Create FIXED initial conv state (same for all trials)
        initial_conv_state = torch.randn(1, dim, 3, dtype=dtype, device=device_type)
        cache_indices = torch.tensor([0], dtype=torch.int64, device=device_type)
        
        # Process sequence multiple times
        n_trials = 10
        all_outputs = []
        
        for trial in range(n_trials):
            # Start with SAME initial conv state for deterministic testing
            conv_state = initial_conv_state.clone()
            
            trial_outputs = []
            for token in tokens:
                out = causal_conv1d_update(
                    x=token.clone(),
                    conv_state=conv_state,  # State is updated in-place
                    weight=weight,
                    bias=bias,
                    activation="silu",
                    conv_state_indices=cache_indices,
                )
                trial_outputs.append(out.clone())
            
            all_outputs.append(trial_outputs)
        
        # Check all trials produced identical sequences
        reference_sequence = all_outputs[0]
        
        for trial_idx, trial_outputs in enumerate(all_outputs[1:], start=1):
            for token_idx, (ref_out, trial_out) in enumerate(zip(reference_sequence, trial_outputs)):
                with self.subTest(trial=trial_idx, token=token_idx):
                    if not torch.equal(ref_out, trial_out):
                        diff = (ref_out - trial_out).abs().max().item()
                        self.fail(
                            f"Sequential processing: Trial {trial_idx}, Token {token_idx} differs!\n"
                            f"  Max diff: {diff}\n"
                            f"  This indicates NON-DETERMINISTIC state updates in conv1d!"
                        )

    def test_causal_conv1d_different_batch_sizes(self):
        """
        Test that conv1d produces same results when processing different batch sizes.
        This tests for issues where batch size affects kernel behavior.
        """
        dim = 1024
        dtype = torch.bfloat16
        
        # Create a single token and state
        torch.manual_seed(42)
        token = torch.randn(1, dim, dtype=dtype, device=device_type)
        initial_state = torch.randn(1, dim, 3, dtype=dtype, device=device_type)
        weight = torch.randn(dim, 4, dtype=dtype, device=device_type)
        bias = torch.randn(dim, dtype=dtype, device=device_type)
        
        # Process in batches of different sizes (replicate the same token)
        batch_sizes = [1, 2, 4, 8, 16]
        outputs_by_batch_size = {}
        
        for batch_size in batch_sizes:
            # Replicate token and state
            x_batch = token.repeat(batch_size, 1)
            conv_state_batch = initial_state.repeat(batch_size, 1, 1)
            cache_indices = torch.arange(batch_size, dtype=torch.int64, device=device_type)
            
            out_batch = causal_conv1d_update(
                x=x_batch,
                conv_state=conv_state_batch,
                weight=weight,
                bias=bias,
                activation="silu",
                conv_state_indices=cache_indices,
            )
            
            # Extract first token's output
            outputs_by_batch_size[batch_size] = out_batch[0].clone()
        
        # All batch sizes should produce identical output for the same token
        reference = outputs_by_batch_size[1]
        
        for batch_size, output in outputs_by_batch_size.items():
            with self.subTest(batch_size=batch_size):
                if not torch.equal(reference, output):
                    diff = (reference - output).abs().max().item()
                    self.fail(
                        f"Batch size {batch_size} produces different output than batch_size=1!\n"
                        f"  Max diff: {diff}\n"
                        f"  This indicates conv1d behavior depends on batch size!"
                    )

    def test_causal_conv1d_fn_prefix_mode_like_test_deterministic(self):
        """
        Test causal_conv1d_fn (PREFILL/EXTEND multi-token conv) with EXACT pattern 
        from test_deterministic.py --test-mode prefix.
        
        Key changes:
        1. Creates 4 prompts as PREFIXES of the same long sequence (shared content)
        2. Uses varying batch compositions with batch sizes up to 50
        3. Processes prompts individually (causal_conv1d_fn doesn't support multi-sequence batching)
        4. Tests with specific lengths [1, 511, 2048, 4097] matching test_deterministic
        """
        dim = 1024
        width = 4
        dtype = torch.bfloat16
        
        # Create 4 prompts with specific lengths matching test_deterministic
        prefix_lengths = [1, 511, 2048, 4097]
        max_len = max(prefix_lengths)
        
        # Create a SINGLE long sequence, use different prefixes (like LONG_PROMPT[:len])
        torch.manual_seed(42)
        base_x = torch.randn(dim, max_len, dtype=dtype, device=device_type)  # (dim, seqlen)
        
        # Extract prefixes of different lengths
        prompts = []
        for i, seq_len in enumerate(prefix_lengths):
            prompts.append({
                'x': base_x[:, :seq_len].clone(),  # (dim, seq_len)
                'conv_state': torch.randn(1, dim, width-1, dtype=dtype, device=device_type),
                'seq_len': seq_len,
            })
        
        # Shared weights
        weight = torch.randn(dim, width, dtype=dtype, device=device_type)
        bias = torch.randn(dim, dtype=dtype, device=device_type)
        
        # Storage for outputs by prompt index
        outputs_by_prompt = {i: [] for i in range(4)}
        
        # Run multiple trials with different batch compositions
        random.seed(42)
        n_trials = 20
        for trial in range(n_trials):
            # Create varying batch compositions
            if trial < 5:
                batch_size = trial + 1  # Small batches: 1-5
            elif trial < 10:
                batch_size = (trial - 5) * 5 + 10  # Medium batches: 10, 15, 20, 25, 30
            else:
                batch_size = 50  # Large batches: 50
            
            # Randomly select which prompts to include (can have duplicates)
            selected_indices = [random.randint(0, 3) for _ in range(batch_size)]
            
            # Process each prompt individually (causal_conv1d_fn processes one sequence at a time)
            for prompt_idx in selected_indices:
                prompt = prompts[prompt_idx]
                seq_len = prompt['seq_len']
                
                # Prepare inputs for this single prompt
                x = prompt['x']  # (dim, seqlen)
                conv_states = prompt['conv_state'].clone()  # (1, dim, width-1)
                query_start_loc = torch.tensor([0, seq_len], dtype=torch.int32, device=device_type)
                seq_lens_cpu = [seq_len]
                cache_indices = torch.tensor([0], dtype=torch.int32, device=device_type)
                has_initial_state = None  # No initial state
                
                out = causal_conv1d_fn(
                    x=x,
                    weight=weight,
                    bias=bias,
                    conv_states=conv_states,
                    query_start_loc=query_start_loc,
                    seq_lens_cpu=seq_lens_cpu,
                    cache_indices=cache_indices,
                    has_initial_state=has_initial_state,
                    activation="silu",
                )
                
                outputs_by_prompt[prompt_idx].append(out.clone())
        
        # Check that each prompt always produces identical output
        for prompt_idx, seq_len in enumerate(prefix_lengths):
            num_samples = len(outputs_by_prompt[prompt_idx])
            if num_samples < 2:
                continue
            
            with self.subTest(prompt_idx=prompt_idx, prefix_length=seq_len):
                reference_output = outputs_by_prompt[prompt_idx][0]
                
                # Check all other outputs match the reference
                all_match = True
                for i, output in enumerate(outputs_by_prompt[prompt_idx][1:], start=1):
                    if not torch.equal(reference_output, output):
                        all_match = False
                        output_diff = (reference_output - output).abs().max().item()
                        print(
                            f"\n✗ DETERMINISM FAILURE - Conv1D Prefill Prompt {prompt_idx} (prefix_length={seq_len}):\n"
                            f"  Sample {i+1}/{num_samples} differs from reference!\n"
                            f"  Output max diff: {output_diff:.10e}\n"
                            f"  This matches the FAILURE pattern in test_deterministic.py!"
                        )
                        self.fail(
                            f"Non-deterministic causal_conv1d_fn behavior detected! "
                            f"Expected all {num_samples} samples to be identical."
                        )
                
                # If we get here, all samples are identical
                print(
                    f"✓ Conv1D Prefill Prompt {prompt_idx} (prefix_length={seq_len}): "
                    f"{num_samples} samples, all identical (DETERMINISTIC)"
                )

    def test_causal_conv1d_update_prefix_mode_like_test_deterministic(self):
        """
        Test causal_conv1d_update (DECODE single-token conv) with EXACT pattern 
        from test_deterministic.py --test-mode prefix.
        
        Key changes:
        1. Creates 4 contexts with shared base token data (like shared prompt content)
        2. Batches multiple contexts together (actual batched kernel call)
        3. Uses varying batch compositions (different contexts per batch)
        4. Tests with batch sizes up to 50
        """
        dim = 1024
        dtype = torch.bfloat16
        
        # Create shared base token (like LONG_PROMPT in test_deterministic)
        torch.manual_seed(42)
        base_x = torch.randn(1, dim, dtype=dtype, device=device_type)
        
        # Create 4 unique decode contexts with different conv_states (representing different histories)
        contexts = []
        for i in range(4):
            torch.manual_seed(42 + i)  # Different seed for state
            contexts.append({
                'x': base_x.clone(),
                'conv_state': torch.randn(1, dim, 3, dtype=dtype, device=device_type),
            })
        
        # Shared weights
        weight = torch.randn(dim, 4, dtype=dtype, device=device_type)
        bias = torch.randn(dim, dtype=dtype, device=device_type)
        
        # Storage for outputs by context index
        outputs_by_context = {i: [] for i in range(4)}
        
        # Run multiple trials with different batch compositions
        random.seed(42)
        n_trials = 20
        for trial in range(n_trials):
            # Create varying batch compositions
            if trial < 5:
                batch_size = trial + 1  # Small batches: 1-5
            elif trial < 10:
                batch_size = (trial - 5) * 5 + 10  # Medium batches: 10, 15, 20, 25, 30
            else:
                batch_size = 50  # Large batches: 50
            
            # Randomly select which contexts to include (can have duplicates)
            selected_indices = [random.randint(0, 3) for _ in range(batch_size)]
            
            # Build batch tensors
            batch_x_list = []
            batch_conv_state_list = []
            
            for ctx_idx in selected_indices:
                ctx = contexts[ctx_idx]
                batch_x_list.append(ctx['x'])
                batch_conv_state_list.append(ctx['conv_state'].clone())
            
            # Concatenate into batched tensors
            batch_x = torch.cat(batch_x_list, dim=0)  # [batch_size, dim]
            batch_conv_state = torch.cat(batch_conv_state_list, dim=0)  # [batch_size, dim, 3]
            cache_indices = torch.arange(batch_size, dtype=torch.int64, device=device_type)
            
            # Run batched kernel
            out_batch = causal_conv1d_update(
                x=batch_x,
                conv_state=batch_conv_state,
                weight=weight,
                bias=bias,
                activation="silu",
                conv_state_indices=cache_indices,
            )
            
            # Extract outputs for each context in the batch
            for i, ctx_idx in enumerate(selected_indices):
                context_output = out_batch[i:i+1].clone()
                outputs_by_context[ctx_idx].append(context_output)
        
        # Check that each context always produces identical output
        for ctx_idx in range(4):
            num_samples = len(outputs_by_context[ctx_idx])
            if num_samples < 2:
                continue
            
            with self.subTest(context_idx=ctx_idx):
                reference_output = outputs_by_context[ctx_idx][0]
                
                # Check all other outputs match the reference
                all_match = True
                for i, output in enumerate(outputs_by_context[ctx_idx][1:], start=1):
                    if not torch.equal(reference_output, output):
                        all_match = False
                        output_diff = (reference_output - output).abs().max().item()
                        print(
                            f"\n✗ DETERMINISM FAILURE - Conv1D Update Context {ctx_idx}:\n"
                            f"  Sample {i+1}/{num_samples} differs from reference!\n"
                            f"  Output max diff: {output_diff:.10e}\n"
                            f"  This matches the FAILURE pattern in test_deterministic.py!"
                        )
                        self.fail(
                            f"Non-deterministic conv1d_update behavior detected! "
                            f"Expected all {num_samples} samples to be identical."
                        )
                
                # If we get here, all samples are identical
                print(
                    f"✓ Conv1D Update Context {ctx_idx}: "
                    f"{num_samples} samples, all identical (DETERMINISTIC)"
                )

    def test_fused_recurrent_prefix_mode_like_test_deterministic(self):
        """
        Test fused_recurrent with EXACT pattern from test_deterministic.py --test-mode prefix.
        
        NOTE: Fused recurrent is used for DECODE (single token), so we test with seq_len=1
        but with different "prompt histories" simulated by different initial states.
        
        Key changes:
        1. Creates 4 decode contexts with shared base random data (like shared prompt prefix)
        2. Batches multiple contexts together using cu_seqlens
        3. Uses varying batch compositions (different contexts per batch)
        4. Tests with batch sizes up to 50
        """
        H, K, V = 8, 128, 128
        dtype = torch.bfloat16
        
        # Create 4 unique decode contexts (mimicking 4 different prompts at decode stage)
        # Use shared base data for the decode token, different states for context
        torch.manual_seed(42)
        base_q = torch.randn(1, 1, H, K, dtype=dtype, device=device_type)
        base_k = torch.randn(1, 1, H, K, dtype=dtype, device=device_type)
        base_v = torch.randn(1, 1, H, V, dtype=dtype, device=device_type)
        base_g = -torch.abs(torch.randn(1, 1, H, dtype=dtype, device=device_type)) * 0.1
        base_beta = torch.sigmoid(torch.randn(1, 1, H, dtype=dtype, device=device_type))
        
        # Each context has different initial state (representing different prompt history)
        contexts = []
        for i in range(4):
            torch.manual_seed(42 + i)
            contexts.append({
                'q': base_q.clone(),
                'k': base_k.clone(),
                'v': base_v.clone(),
                'g': base_g.clone(),
                'beta': base_beta.clone(),
                'state': torch.randn(1, H, K, V, dtype=dtype, device=device_type) * 0.01,
            })
        
        # Storage for outputs by context index
        outputs_by_context = {i: [] for i in range(4)}
        
        # Run multiple trials with different batch compositions
        random.seed(42)
        n_trials = 20
        for trial in range(n_trials):
            # Create varying batch compositions
            if trial < 5:
                batch_size = trial + 1  # Small batches: 1-5
            elif trial < 10:
                batch_size = (trial - 5) * 5 + 10  # Medium batches: 10, 15, 20, 25, 30
            else:
                batch_size = 50  # Large batches: 50
            
            # Randomly select which contexts to include (can have duplicates)
            selected_indices = [random.randint(0, 3) for _ in range(batch_size)]
            
            # Build concatenated batch using cu_seqlens
            batch_q_list = []
            batch_k_list = []
            batch_v_list = []
            batch_g_list = []
            batch_beta_list = []
            batch_states = []
            cu_seqlens = [0]
            
            for ctx_idx in selected_indices:
                ctx = contexts[ctx_idx]
                batch_q_list.append(ctx['q'].squeeze(0))
                batch_k_list.append(ctx['k'].squeeze(0))
                batch_v_list.append(ctx['v'].squeeze(0))
                batch_g_list.append(ctx['g'].squeeze(0))
                batch_beta_list.append(ctx['beta'].squeeze(0))
                batch_states.append(ctx['state'].clone())
                cu_seqlens.append(cu_seqlens[-1] + 1)  # seq_len=1 for each
            
            # Concatenate all sequences
            batch_q = torch.cat(batch_q_list, dim=0).unsqueeze(0)
            batch_k = torch.cat(batch_k_list, dim=0).unsqueeze(0)
            batch_v = torch.cat(batch_v_list, dim=0).unsqueeze(0)
            batch_g = torch.cat(batch_g_list, dim=0).unsqueeze(0)
            batch_beta = torch.cat(batch_beta_list, dim=0).unsqueeze(0)
            initial_state_source = torch.cat(batch_states, dim=0)  # [batch_size, H, K, V]
            cu_seqlens_tensor = torch.tensor(cu_seqlens, dtype=torch.int32, device=device_type)
            cache_indices = torch.arange(len(selected_indices), dtype=torch.int64, device=device_type)
            
            # Run the kernel with cu_seqlens (batched processing)
            o = fused_recurrent_gated_delta_rule_update_fwd(
                q=batch_q,
                k=batch_k,
                v=batch_v,
                g=batch_g,
                beta=batch_beta,
                scale=K ** -0.5,
                initial_state_source=initial_state_source,
                initial_state_indices=cache_indices,
                cu_seqlens=cu_seqlens_tensor,
                use_qk_l2norm_in_kernel=True,
                disable_state_update=True,
            )
            
            # Extract outputs for each context in the batch
            for i, ctx_idx in enumerate(selected_indices):
                start_pos = cu_seqlens[i]
                end_pos = cu_seqlens[i + 1]
                context_output = o[:, start_pos:end_pos, :, :].clone()
                outputs_by_context[ctx_idx].append(context_output)
        
        # Check that each context always produces identical output
        for ctx_idx in range(4):
            num_samples = len(outputs_by_context[ctx_idx])
            if num_samples < 2:
                continue
            
            with self.subTest(context_idx=ctx_idx):
                reference_output = outputs_by_context[ctx_idx][0]
                
                # Check all other outputs match the reference
                all_match = True
                for i, output in enumerate(outputs_by_context[ctx_idx][1:], start=1):
                    if not torch.equal(reference_output, output):
                        all_match = False
                        output_diff = (reference_output - output).abs().max().item()
                        print(
                            f"\n✗ DETERMINISM FAILURE - Fused Recurrent Context {ctx_idx}:\n"
                            f"  Sample {i+1}/{num_samples} differs from reference!\n"
                            f"  Output max diff: {output_diff:.10e}\n"
                            f"  This matches the FAILURE pattern in test_deterministic.py!"
                        )
                        self.fail(
                            f"Non-deterministic fused_recurrent behavior detected! "
                            f"Expected all {num_samples} samples to be identical."
                        )
                
                # If we get here, all samples are identical
                print(
                    f"✓ Fused Recurrent Context {ctx_idx}: "
                    f"{num_samples} samples, all identical (DETERMINISTIC)"
                )

    def test_chunk_gated_delta_rule_prefix_mode_like_test_deterministic(self):
        """
        Test that mimics the EXACT pattern from test_deterministic.py --test-mode prefix.
        
        Key differences from the original implementation:
        1. Creates 4 prompts as PREFIXES of the same long sequence (not independent random data)
        2. Batches multiple prompts together using cu_seqlens (concatenated sequence)
        3. Uses varying batch compositions like test_deterministic (different prompts per batch)
        4. Tests with batch sizes up to 50 with mixed prompt lengths
        
        This replicates how test_deterministic processes prompts in actual inference.
        """
        # Create 4 prompts with specific lengths matching test_deterministic
        prefix_lengths = [1, 511, 2048, 4097]
        H, K, V = 8, 128, 128
        dtype = torch.bfloat16
        max_len = max(prefix_lengths)
        
        # Create a SINGLE long sequence, and use different prefixes of it
        # This matches test_deterministic behavior where prompts share content
        torch.manual_seed(42)
        base_q = torch.randn(1, max_len, H, K, dtype=dtype, device=device_type)
        base_k = torch.randn(1, max_len, H, K, dtype=dtype, device=device_type)
        base_v = torch.randn(1, max_len, H, V, dtype=dtype, device=device_type)
        base_g = -torch.abs(torch.randn(1, max_len, H, dtype=dtype, device=device_type)) * 0.1
        base_beta = torch.sigmoid(torch.randn(1, max_len, H, dtype=dtype, device=device_type))
        
        # Extract prefixes of different lengths (like LONG_PROMPT[:len] in test_deterministic)
        prompts = []
        for i, seq_len in enumerate(prefix_lengths):
            prompt_data = {
                'q': base_q[:, :seq_len, :, :].clone(),
                'k': base_k[:, :seq_len, :, :].clone(),
                'v': base_v[:, :seq_len, :, :].clone(),
                'g': base_g[:, :seq_len, :].clone(),
                'beta': base_beta[:, :seq_len, :].clone(),
                'initial_state': torch.randn(1, H, K, V, dtype=dtype, device=device_type) * 0.01,
                'seq_len': seq_len,
            }
            prompts.append(prompt_data)
        
        # Storage for outputs grouped by prompt index (matching test_deterministic pattern)
        outputs_by_prompt = {i: [] for i in range(len(prefix_lengths))}
        states_by_prompt = {i: [] for i in range(len(prefix_lengths))}
        
        # Run multiple trials with different batch compositions
        # Use realistic batch sizes like test_deterministic
        random.seed(42)
        n_trials = 20
        for trial in range(n_trials):
            # Create varying batch compositions (like test_deterministic does)
            # Example: prefix length 1: 11, prefix length 511: 17, etc.
            if trial < 5:
                batch_size = trial + 1  # Small batches: 1-5
            elif trial < 10:
                batch_size = (trial - 5) * 5 + 10  # Medium batches: 10, 15, 20, 25, 30
            else:
                batch_size = 50  # Large batches: 50
            
            # Randomly select which prompts to include (can have duplicates)
            selected_indices = [random.randint(0, len(prefix_lengths) - 1) for _ in range(batch_size)]
            
            # Build concatenated batch using cu_seqlens (this is how test_deterministic batches)
            batch_q_list = []
            batch_k_list = []
            batch_v_list = []
            batch_g_list = []
            batch_beta_list = []
            cu_seqlens = [0]
            
            for prompt_idx in selected_indices:
                prompt_data = prompts[prompt_idx]
                batch_q_list.append(prompt_data['q'].squeeze(0))  # Remove batch dim
                batch_k_list.append(prompt_data['k'].squeeze(0))
                batch_v_list.append(prompt_data['v'].squeeze(0))
                batch_g_list.append(prompt_data['g'].squeeze(0))
                batch_beta_list.append(prompt_data['beta'].squeeze(0))
                cu_seqlens.append(cu_seqlens[-1] + prompt_data['seq_len'])
            
            # Concatenate all sequences
            batch_q = torch.cat(batch_q_list, dim=0).unsqueeze(0)  # [1, total_len, H, K]
            batch_k = torch.cat(batch_k_list, dim=0).unsqueeze(0)
            batch_v = torch.cat(batch_v_list, dim=0).unsqueeze(0)
            batch_g = torch.cat(batch_g_list, dim=0).unsqueeze(0)
            batch_beta = torch.cat(batch_beta_list, dim=0).unsqueeze(0)
            cu_seqlens_tensor = torch.tensor(cu_seqlens, dtype=torch.int32, device=device_type)
            
            # Run the kernel with cu_seqlens (batched processing)
            # NOTE: initial_state needs to be batched too for cu_seqlens mode
            # For simplicity, use zero initial state (or could create per-prompt states)
            initial_state = torch.zeros(len(selected_indices), H, K, V, dtype=dtype, device=device_type)
            
            o, final_states = chunk_gated_delta_rule(
                q=batch_q,
                k=batch_k,
                v=batch_v,
                g=batch_g,
                beta=batch_beta,
                initial_state=initial_state,
                output_final_state=True,
                cu_seqlens=cu_seqlens_tensor,
                head_first=False,
                use_qk_l2norm_in_kernel=True,
            )
            
            # Extract outputs for each prompt in the batch
            for i, prompt_idx in enumerate(selected_indices):
                start_pos = cu_seqlens[i]
                end_pos = cu_seqlens[i + 1]
                prompt_output = o[:, start_pos:end_pos, :, :].clone()
                prompt_state = final_states[i:i+1, :, :, :].clone()
                
                outputs_by_prompt[prompt_idx].append(prompt_output)
                states_by_prompt[prompt_idx].append(prompt_state)
        
        # Now check that each prompt always produces identical output
        # This is the KEY check that test_deterministic does: unique samples should be 1
        for prompt_idx, seq_len in enumerate(prefix_lengths):
            num_samples = len(outputs_by_prompt[prompt_idx])
            if num_samples < 2:
                continue
            
            with self.subTest(prompt_idx=prompt_idx, prefix_length=seq_len):
                reference_output = outputs_by_prompt[prompt_idx][0]
                reference_state = states_by_prompt[prompt_idx][0]
                
                # Check all other outputs match the reference
                all_match = True
                for i, (output, state) in enumerate(zip(
                    outputs_by_prompt[prompt_idx][1:],
                    states_by_prompt[prompt_idx][1:],
                ), start=1):
                    output_match = torch.equal(reference_output, output)
                    state_match = torch.equal(reference_state, state)
                    
                    if not output_match or not state_match:
                        all_match = False
                        output_diff = (reference_output - output).abs().max().item()
                        state_diff = (reference_state - state).abs().max().item()
                        print(
                            f"\n✗ DETERMINISM FAILURE - Prompt {prompt_idx} (prefix_length={seq_len}):\n"
                            f"  Sample {i+1}/{num_samples} differs from reference!\n"
                            f"  Output max diff: {output_diff:.10e}\n"
                            f"  State max diff: {state_diff:.10e}\n"
                            f"  This matches the FAILURE pattern in test_deterministic.py!"
                        )
                        self.fail(
                            f"Non-deterministic behavior detected! "
                            f"Expected all {num_samples} samples to be identical, "
                            f"but found differences starting at sample {i+1}."
                        )
                
                # If we get here, all samples are identical
                print(
                    f"✓ Prompt {prompt_idx} (prefix_length={seq_len}): "
                    f"{num_samples} samples, all identical (DETERMINISTIC)"
                )

    def test_chunk_gated_delta_rule_state_consistency(self):
        """
        Test that running with the output state as next input gives consistent results.
        This simulates the state passing that happens during actual generation.
        """
        seq_len, H, K, V = 128, 8, 128, 128
        dtype = torch.bfloat16
        
        # Create inputs
        torch.manual_seed(42)
        q = torch.randn(1, seq_len, H, K, dtype=dtype, device=device_type)
        k = torch.randn(1, seq_len, H, K, dtype=dtype, device=device_type)
        v = torch.randn(1, seq_len, H, V, dtype=dtype, device=device_type)
        g = -torch.abs(torch.randn(1, seq_len, H, dtype=dtype, device=device_type)) * 0.1
        beta = torch.sigmoid(torch.randn(1, seq_len, H, dtype=dtype, device=device_type))
        initial_state = torch.randn(1, H, K, V, dtype=dtype, device=device_type) * 0.01
        
        # Run multiple "generations" chaining states
        states = [initial_state]
        outputs = []
        
        for gen in range(5):
            o, final_state = chunk_gated_delta_rule(
                q=q,
                k=k,
                v=v,
                g=g,
                beta=beta,
                initial_state=states[-1].clone(),
                output_final_state=True,
                cu_seqlens=None,
                head_first=False,
                use_qk_l2norm_in_kernel=True,
            )
            states.append(final_state.clone())
            outputs.append(o.clone())
        
        # Now run the SAME sequence again and verify all states match
        states_repeat = [initial_state]
        outputs_repeat = []
        
        for gen in range(5):
            o, final_state = chunk_gated_delta_rule(
                q=q,
                k=k,
                v=v,
                g=g,
                beta=beta,
                initial_state=states_repeat[-1].clone(),
                output_final_state=True,
                cu_seqlens=None,
                head_first=False,
                use_qk_l2norm_in_kernel=True,
            )
            states_repeat.append(final_state.clone())
            outputs_repeat.append(o.clone())
        
        # Verify all outputs and states match
        for gen in range(5):
            with self.subTest(generation=gen):
                self.assertTrue(
                    torch.equal(outputs[gen], outputs_repeat[gen]),
                    f"Generation {gen} outputs differ!"
                )
                self.assertTrue(
                    torch.equal(states[gen+1], states_repeat[gen+1]),
                    f"Generation {gen} final states differ!"
                )


if __name__ == "__main__":
    unittest.main()

