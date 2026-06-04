# Copyright Contributors to the OpenVDB Project
# SPDX-License-Identifier: Apache-2.0
#
"""
Test the mathematical ground truth for scaled dot-product attention (SDPA).

These tests validate that SDPA behaves as we expect from its definition:

    Attention(Q, K, V) = softmax(Q @ K^T * scale) @ V

Each test constructs inputs where the correct output is known analytically,
not by running a reference implementation.  This establishes baseline
understanding of:
  - Softmax concentration and uniformity
  - Scale parameter effects
  - One-hot selection via orthogonal keys (the attention "impulse response")
  - Convex combination property of the output
  - Batch independence through the jagged path
  - Gradient flow through V and Q
  - Linearity in V and permutation symmetries

This file does NOT test fVDB sparse attention in isolation -- it validates
our understanding of SDPA semantics that we rely on as ground truth when
testing fvdb.scaled_dot_product_attention against expected behavior.
"""

import unittest

import torch
import torch.nn.attention

import fvdb


def _sdpa(q_list, k_list, v_list, scale):
    """Run fvdb SDPA on lists of per-batch tensors using the MATH backend."""
    with torch.nn.attention.sdpa_kernel([torch.nn.attention.SDPBackend.MATH]):
        return fvdb.scaled_dot_product_attention(
            fvdb.JaggedTensor(q_list),
            fvdb.JaggedTensor(k_list),
            fvdb.JaggedTensor(v_list),
            scale=scale,
        )


class TestSDPAGroundTruth(unittest.TestCase):
    """
    Validate mathematical properties of scaled dot-product attention.

    These tests verify our understanding of:
    - Basic forward properties (single token, uniform attention, one-hot selection)
    - Scale effects (concentration and uniformity)
    - Jagged batch properties (independence, convex combination)
    - Backward / gradient properties (grad w.r.t. V and Q)
    - Structural invariances (linearity in V, permutation symmetries, head independence)

    Dimension key (per batch element):
        Lq    = query sequence length (varies per batch element)
        Skv   = key/value sequence length (varies per batch element)
        H     = number of attention heads
        E     = embedding dimension for queries and keys
        V     = value dimension (may differ from E except for Flash Attention)

    Tensor shapes:
        query = (Lq,  H, E)
        key   = (Skv, H, E)
        value = (Skv, H, V)
    """

    DEVICE = "cuda"

    def setUp(self):
        torch.random.manual_seed(42)

    # =========================================================================
    # Section 1: Basic Forward Properties
    # =========================================================================

    def test_single_kv_token(self):
        """
        The simplest possible SDPA.

        With a single key-value token, softmax([q . k * scale]) = [1.0]
        regardless of the actual dot product.  So the output must always
        equal v, no matter what q, k, or scale are.
        """
        H, E, V = 2, 8, 8
        scale = 0.5

        for Lq in [1, 5]:
            q = torch.randn(Lq, H, E, device=self.DEVICE, dtype=torch.float64)
            k = torch.randn(1, H, E, device=self.DEVICE, dtype=torch.float64)
            v = torch.randn(1, H, V, device=self.DEVICE, dtype=torch.float64)

            out = _sdpa([q], [k], [v], scale=scale)
            out_data = out[0].jdata  # (Lq, H, V)

            expected = v.expand(Lq, H, V)
            torch.testing.assert_close(out_data, expected, atol=1e-12, rtol=1e-10)

    def test_identical_keys_uniform_attention(self):
        """
        When all keys are identical, attention weights are uniform.

        If every key vector is the same, then q . k_i is the same for all i,
        so softmax produces weights [1/Skv, 1/Skv, ...].  The output for every
        query is therefore mean(V) regardless of the query content.
        """
        Lq, Skv, H, E, V = 3, 4, 2, 8, 8
        scale = 1.0

        q = torch.randn(Lq, H, E, device=self.DEVICE, dtype=torch.float64)
        single_k = torch.randn(1, H, E, device=self.DEVICE, dtype=torch.float64)
        k = single_k.expand(Skv, H, E).contiguous()
        v = torch.randn(Skv, H, V, device=self.DEVICE, dtype=torch.float64)

        out = _sdpa([q], [k], [v], scale=scale)
        out_data = out[0].jdata  # (Lq, H, V)

        expected = v.mean(dim=0, keepdim=True).expand(Lq, H, V)
        torch.testing.assert_close(out_data, expected, atol=1e-10, rtol=1e-10)

    def test_one_hot_selection(self):
        """
        Orthogonal keys allow exact value selection -- the attention "impulse".

        Construct Skv orthonormal key vectors.  Set the query to alpha * key[j]
        for a large alpha.  The dot product q . key[j] = alpha * ||key[j]||^2
        dominates, so softmax concentrates on index j and out ≈ V[j].

        This acts as a "selection impulse" -- a query that isolates exactly
        one value vector from the set, confirming pointwise retrieval.
        """
        Skv, H, E, V = 4, 1, 8, 8
        scale = 1.0
        alpha = 50.0

        # Orthonormal keys via QR decomposition (one set per head)
        random_matrix = torch.randn(H, E, Skv, device=self.DEVICE, dtype=torch.float64)
        ortho_basis, _ = torch.linalg.qr(random_matrix)  # (H, E, Skv)
        keys = ortho_basis[:, :, :Skv].permute(2, 0, 1).contiguous()  # (Skv, H, E)

        v = torch.randn(Skv, H, V, device=self.DEVICE, dtype=torch.float64)

        for j in range(Skv):
            q = (alpha * keys[j : j + 1]).contiguous()  # (1, H, E)

            out = _sdpa([q], [keys], [v], scale=scale)
            out_data = out[0].jdata  # (1, H, V)

            torch.testing.assert_close(out_data, v[j : j + 1], atol=1e-6, rtol=1e-6)

    # =========================================================================
    # Section 2: Scale Effects
    # =========================================================================

    def test_large_scale_concentrates(self):
        """
        Large scale makes attention approach argmax.

        With scale -> infinity, softmax(q @ K^T * scale) concentrates all
        weight on the key with the maximum dot product.  The output converges
        to the value at that index.
        """
        Skv, H, E, V = 5, 1, 8, 8
        scale = 1000.0

        q = torch.randn(1, H, E, device=self.DEVICE, dtype=torch.float64)
        k = torch.randn(Skv, H, E, device=self.DEVICE, dtype=torch.float64)
        v = torch.randn(Skv, H, V, device=self.DEVICE, dtype=torch.float64)

        # Find which key has the max dot product with q, per head
        dots = torch.einsum("qhe,she->hs", q, k)  # (H, Skv)
        best_idx = dots.argmax(dim=1)  # (H,)

        out = _sdpa([q], [k], [v], scale=scale)
        out_data = out[0].jdata.squeeze(0)  # (H, V)

        for h in range(H):
            torch.testing.assert_close(out_data[h], v[best_idx[h], h], atol=1e-6, rtol=1e-6)

    def test_zero_scale_uniform(self):
        """
        Near-zero scale makes attention uniform.

        When scale -> 0, all logits q . k * scale -> 0, so
        softmax -> [1/Skv, 1/Skv, ...].  The output approaches mean(V)
        for every query, regardless of query content.
        """
        Lq, Skv, H, E, V = 3, 6, 2, 8, 8
        scale = 1e-7

        q = torch.randn(Lq, H, E, device=self.DEVICE, dtype=torch.float64)
        k = torch.randn(Skv, H, E, device=self.DEVICE, dtype=torch.float64)
        v = torch.randn(Skv, H, V, device=self.DEVICE, dtype=torch.float64)

        out = _sdpa([q], [k], [v], scale=scale)
        out_data = out[0].jdata  # (Lq, H, V)

        expected = v.mean(dim=0, keepdim=True).expand(Lq, H, V)
        torch.testing.assert_close(out_data, expected, atol=1e-5, rtol=1e-5)

    # =========================================================================
    # Section 3: Jagged / Batch Properties
    # =========================================================================

    def test_batch_independence(self):
        """
        Each jagged batch element is processed independently.

        Create three batch elements where elements 0 and 1 have identical
        Q/K/V and element 2 has different data.  Verify:
        - Elements 0 and 1 produce identical outputs
        - Element 2 produces a different output
        """
        H, E, V = 2, 8, 8
        scale = 1.0

        Lq, Skv = 4, 6
        q_shared = torch.randn(Lq, H, E, device=self.DEVICE, dtype=torch.float64)
        k_shared = torch.randn(Skv, H, E, device=self.DEVICE, dtype=torch.float64)
        v_shared = torch.randn(Skv, H, V, device=self.DEVICE, dtype=torch.float64)
        q_diff = torch.randn(Lq, H, E, device=self.DEVICE, dtype=torch.float64)
        k_diff = torch.randn(Skv, H, E, device=self.DEVICE, dtype=torch.float64)
        v_diff = torch.randn(Skv, H, V, device=self.DEVICE, dtype=torch.float64)

        out = _sdpa(
            [q_shared, q_shared.clone(), q_diff],
            [k_shared, k_shared.clone(), k_diff],
            [v_shared, v_shared.clone(), v_diff],
            scale=scale,
        )

        out_0 = out[0].jdata
        out_1 = out[1].jdata
        out_2 = out[2].jdata

        torch.testing.assert_close(out_0, out_1, atol=1e-12, rtol=1e-10)
        self.assertFalse(torch.allclose(out_0, out_2, atol=1e-3))

    def test_output_is_convex_combination(self):
        """
        SDPA output is a convex combination of values.

        Since softmax produces non-negative weights that sum to 1, each output
        token is a weighted average of V.  Therefore, component-wise:

            min(V, dim=0) <= out[i] <= max(V, dim=0)

        for every query position i and every component.
        """
        Lq, Skv, H, E, V = 5, 8, 2, 16, 16
        scale = 1.0

        q = torch.randn(Lq, H, E, device=self.DEVICE, dtype=torch.float64)
        k = torch.randn(Skv, H, E, device=self.DEVICE, dtype=torch.float64)
        v = torch.randn(Skv, H, V, device=self.DEVICE, dtype=torch.float64)

        out = _sdpa([q], [k], [v], scale=scale)
        out_data = out[0].jdata  # (Lq, H, V)

        v_min = v.min(dim=0).values  # (H, V)
        v_max = v.max(dim=0).values  # (H, V)

        eps = 1e-10
        self.assertTrue(
            torch.all(out_data >= v_min - eps),
            "Output has components below min(V)",
        )
        self.assertTrue(
            torch.all(out_data <= v_max + eps),
            "Output has components above max(V)",
        )

    # =========================================================================
    # Section 4: Backward / Gradient Properties
    # =========================================================================
    #
    # SDPA is differentiable.  These tests verify gradient flow by constructing
    # inputs where d(loss)/d(input) is known analytically.
    #
    # Pattern: create tensors with requires_grad_(True), run _sdpa, compute
    # loss = out[0].jdata.sum(), call loss.backward(), assert on .grad.

    def test_grad_v_uniform_attention(self):
        """
        Gradient w.r.t. V under uniform attention weights.

        With identical keys, every attention weight is 1/Skv.  Each of the Lq
        query positions produces mean(V), so for loss = sum(out):

            d(loss)/d(v[j, h, d]) = Lq / Skv    for all j, h, d

        This is the simplest possible value gradient -- a uniform scalar.
        """
        Lq, Skv, H, E, V = 3, 4, 2, 8, 8
        scale = 1.0

        q = torch.randn(Lq, H, E, device=self.DEVICE, dtype=torch.float64)
        single_k = torch.randn(1, H, E, device=self.DEVICE, dtype=torch.float64)
        k = single_k.expand(Skv, H, E).contiguous()
        v = torch.randn(Skv, H, V, device=self.DEVICE, dtype=torch.float64, requires_grad=True)

        out = _sdpa([q], [k], [v], scale=scale)
        loss = out[0].jdata.sum()
        loss.backward()

        assert v.grad is not None
        expected_grad = torch.full_like(v, Lq / Skv)
        torch.testing.assert_close(v.grad, expected_grad, atol=1e-10, rtol=1e-10)

    def test_grad_v_one_hot_selection(self):
        """
        Gradient w.r.t. V under concentrated (one-hot) attention.

        With orthogonal keys and q = alpha * key[j] for large alpha, attention
        concentrates entirely on index j.  For loss = sum(out):

            d(loss)/d(v[j])  ≈ 1    (all weight on selected value)
            d(loss)/d(v[j']) ≈ 0    (no weight on other values)
        """
        Skv, H, E, V = 4, 1, 8, 8
        scale = 1.0
        alpha = 50.0

        random_matrix = torch.randn(H, E, Skv, device=self.DEVICE, dtype=torch.float64)
        ortho_basis, _ = torch.linalg.qr(random_matrix)
        keys = ortho_basis[:, :, :Skv].permute(2, 0, 1).contiguous()

        for j in range(Skv):
            v = torch.randn(Skv, H, V, device=self.DEVICE, dtype=torch.float64, requires_grad=True)
            q = (alpha * keys[j : j + 1]).contiguous()

            out = _sdpa([q], [keys], [v], scale=scale)
            loss = out[0].jdata.sum()
            loss.backward()

            assert v.grad is not None
            selected_grad = v.grad[j]
            torch.testing.assert_close(selected_grad, torch.ones_like(selected_grad), atol=1e-6, rtol=1e-6)

            for j_prime in range(Skv):
                if j_prime == j:
                    continue
                torch.testing.assert_close(
                    v.grad[j_prime],
                    torch.zeros_like(v.grad[j_prime]),
                    atol=1e-6,
                    rtol=1e-6,
                )

    def test_grad_q_single_kv_constant_output(self):
        """
        Gradient w.r.t. Q is zero when there is a single key-value token.

        With Skv = 1, softmax of a single logit is identically 1.0, so the
        output is v[0] regardless of Q.  The Jacobian of softmax([x]) w.r.t. x
        is zero (the output is constant), hence d(loss)/d(Q) = 0.
        """
        H, E, V = 2, 8, 8
        scale = 0.5

        for Lq in [1, 5]:
            q = torch.randn(Lq, H, E, device=self.DEVICE, dtype=torch.float64, requires_grad=True)
            k = torch.randn(1, H, E, device=self.DEVICE, dtype=torch.float64)
            v = torch.randn(1, H, V, device=self.DEVICE, dtype=torch.float64)

            out = _sdpa([q], [k], [v], scale=scale)
            loss = out[0].jdata.sum()
            loss.backward()

            assert q.grad is not None
            torch.testing.assert_close(q.grad, torch.zeros_like(q), atol=1e-12, rtol=1e-10)

    # =========================================================================
    # Section 5: Structural Invariances
    # =========================================================================
    #
    # These tests verify algebraic properties of SDPA that hold for any inputs,
    # without requiring analytically known outputs.

    def test_linearity_in_v(self):
        """
        SDPA is linear in V.

        Attention weights depend only on Q and K, so:

            SDPA(Q, K, a*V1 + b*V2) = a * SDPA(Q, K, V1) + b * SDPA(Q, K, V2)

        for any scalars a, b and value tensors V1, V2.
        """
        Lq, Skv, H, E, V = 4, 6, 2, 8, 8
        scale = 1.0
        a, b = 0.7, -0.3

        q = torch.randn(Lq, H, E, device=self.DEVICE, dtype=torch.float64)
        k = torch.randn(Skv, H, E, device=self.DEVICE, dtype=torch.float64)
        v1 = torch.randn(Skv, H, V, device=self.DEVICE, dtype=torch.float64)
        v2 = torch.randn(Skv, H, V, device=self.DEVICE, dtype=torch.float64)

        out_combined = _sdpa([q], [k], [a * v1 + b * v2], scale=scale)[0].jdata
        out_v1 = _sdpa([q], [k], [v1], scale=scale)[0].jdata
        out_v2 = _sdpa([q], [k], [v2], scale=scale)[0].jdata

        expected = a * out_v1 + b * out_v2
        torch.testing.assert_close(out_combined, expected, atol=1e-10, rtol=1e-10)

    def test_kv_permutation_invariance(self):
        """
        SDPA is invariant to key-value pair ordering.

        Softmax + weighted sum treats the KV set as unordered:

            SDPA(Q, K[perm], V[perm]) = SDPA(Q, K, V)

        for any permutation perm of [0, ..., Skv-1].
        """
        Lq, Skv, H, E, V = 4, 6, 2, 8, 8
        scale = 1.0

        q = torch.randn(Lq, H, E, device=self.DEVICE, dtype=torch.float64)
        k = torch.randn(Skv, H, E, device=self.DEVICE, dtype=torch.float64)
        v = torch.randn(Skv, H, V, device=self.DEVICE, dtype=torch.float64)

        perm = torch.randperm(Skv)

        out_original = _sdpa([q], [k], [v], scale=scale)[0].jdata
        out_permuted = _sdpa([q], [k[perm]], [v[perm]], scale=scale)[0].jdata

        torch.testing.assert_close(out_original, out_permuted, atol=1e-10, rtol=1e-10)

    def test_query_permutation_equivariance(self):
        """
        SDPA is equivariant to query ordering.

        Each query position is computed independently, so permuting queries
        permutes the output in the same way:

            SDPA(Q[perm], K, V)[i] = SDPA(Q, K, V)[perm[i]]
        """
        Lq, Skv, H, E, V = 5, 6, 2, 8, 8
        scale = 1.0

        q = torch.randn(Lq, H, E, device=self.DEVICE, dtype=torch.float64)
        k = torch.randn(Skv, H, E, device=self.DEVICE, dtype=torch.float64)
        v = torch.randn(Skv, H, V, device=self.DEVICE, dtype=torch.float64)

        perm = torch.randperm(Lq)

        out_original = _sdpa([q], [k], [v], scale=scale)[0].jdata
        out_permuted = _sdpa([q[perm]], [k], [v], scale=scale)[0].jdata

        torch.testing.assert_close(out_permuted, out_original[perm], atol=1e-10, rtol=1e-10)

    def test_head_independence(self):
        """
        Attention heads operate independently.

        Replacing head h's data in Q, K, V should only affect head h's output.
        All other heads must produce identical results.
        """
        Lq, Skv, H, E, V = 4, 6, 4, 8, 8
        scale = 1.0

        q = torch.randn(Lq, H, E, device=self.DEVICE, dtype=torch.float64)
        k = torch.randn(Skv, H, E, device=self.DEVICE, dtype=torch.float64)
        v = torch.randn(Skv, H, V, device=self.DEVICE, dtype=torch.float64)

        out_original = _sdpa([q], [k], [v], scale=scale)[0].jdata  # (Lq, H, V)

        modified_head = 0
        q2 = q.clone()
        k2 = k.clone()
        v2 = v.clone()
        q2[:, modified_head, :] = torch.randn(Lq, E, device=self.DEVICE, dtype=torch.float64)
        k2[:, modified_head, :] = torch.randn(Skv, E, device=self.DEVICE, dtype=torch.float64)
        v2[:, modified_head, :] = torch.randn(Skv, V, device=self.DEVICE, dtype=torch.float64)

        out_modified = _sdpa([q2], [k2], [v2], scale=scale)[0].jdata

        for h in range(H):
            if h == modified_head:
                self.assertFalse(
                    torch.allclose(out_original[:, h, :], out_modified[:, h, :], atol=1e-3),
                    f"Head {h} should have changed but didn't",
                )
            else:
                torch.testing.assert_close(
                    out_original[:, h, :],
                    out_modified[:, h, :],
                    atol=1e-12,
                    rtol=1e-10,
                )
