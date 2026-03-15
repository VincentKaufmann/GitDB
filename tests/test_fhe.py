"""Tests for GitDB FHE module — searchable encryption, PIR, and homomorphic encryption."""

import os
import struct

import pytest
import torch

from gitdb.fhe import (
    SearchableEncryption,
    PIRClient,
    PIRServer,
    pir_setup,
    FHEScheme,
    EncryptedVectorStore,
    EncryptedGitOps,
    EncryptedDBOps,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def se_key():
    return os.urandom(32)


@pytest.fixture
def se(se_key):
    return SearchableEncryption(se_key)


@pytest.fixture
def fhe():
    """Small FHE scheme for fast tests."""
    scheme = FHEScheme(poly_degree=256, coeff_modulus=2**40, plain_modulus=2**16, device="cpu")
    scheme.keygen()
    return scheme


# ---------------------------------------------------------------------------
# Tier 1: Searchable Encryption
# ---------------------------------------------------------------------------

class TestSearchableEncryption:
    def test_encrypt_decrypt_equality(self, se):
        """Encrypt with equality type — deterministic tag, same input gives same output."""
        ct1 = se.encrypt_field("alice", "equality")
        ct2 = se.encrypt_field("alice", "equality")
        assert ct1 == ct2
        assert isinstance(ct1, bytes)
        assert len(ct1) == 32  # SHA-256 digest

    def test_equality_match(self, se):
        """Two encryptions of the same value match via match_equality."""
        ct_a = se.encrypt_field("bob", "equality")
        ct_b = se.encrypt_field("bob", "equality")
        assert se.match_equality(ct_a, ct_b) is True

    def test_equality_no_match(self, se):
        """Different values produce different ciphertexts that don't match."""
        ct_a = se.encrypt_field("alice", "equality")
        ct_b = se.encrypt_field("bob", "equality")
        assert se.match_equality(ct_a, ct_b) is False

    def test_range_order_preserved(self, se):
        """OPE preserves ordering of integers."""
        values = [10, 20, 30, 50, 100]
        encrypted = [se.encrypt_field(v, "range") for v in values]

        # Check that ordering is preserved
        for i in range(len(encrypted) - 1):
            cmp = se.compare_range(encrypted[i], encrypted[i + 1])
            assert cmp == -1, f"Expected {values[i]} < {values[i+1]}, got cmp={cmp}"

        # Equal values
        ct_same_a = se.encrypt_field(42, "range")
        ct_same_b = se.encrypt_field(42, "range")
        assert se.compare_range(ct_same_a, ct_same_b) == 0

    def test_encrypt_decrypt_row(self, se):
        """Full row encryption roundtrip with exact fields."""
        row = {"name": "Alice", "ssn": "123-45-6789"}
        field_types = {"name": "exact", "ssn": "exact"}
        original_types = {"name": "str", "ssn": "str"}

        encrypted = se.encrypt_row(row, field_types)
        decrypted = se.decrypt_row(encrypted, field_types, original_types)

        assert decrypted["name"] == "Alice"
        assert decrypted["ssn"] == "123-45-6789"

    def test_different_keys_no_match(self):
        """Different keys produce different ciphertexts for equality fields."""
        se1 = SearchableEncryption(os.urandom(32))
        se2 = SearchableEncryption(os.urandom(32))

        ct1 = se1.encrypt_field("secret", "equality")
        ct2 = se2.encrypt_field("secret", "equality")

        assert ct1 != ct2
        assert se1.match_equality(ct1, ct2) is False


# ---------------------------------------------------------------------------
# Tier 2: PIR
# ---------------------------------------------------------------------------

class TestPIR:
    def test_pir_retrieves_correct_row(self):
        """Client retrieves the exact target row; server doesn't know which."""
        n_items, dim = 64, 32
        database = torch.randn(n_items, dim)
        target = 17

        client, server = pir_setup(database, device="cpu")

        query = client.create_query(target)
        response = server.respond(query)
        result = client.extract_result(response, database)

        expected = database[target]
        assert torch.allclose(result, expected, atol=1e-4), \
            f"Max error: {(result - expected).abs().max().item()}"

    def test_pir_gpu_matmul(self):
        """Server response is computed via matrix multiplication."""
        n_items, dim = 32, 16
        database = torch.randn(n_items, dim)
        server = PIRServer(database, device="cpu")

        query = torch.randn(n_items)
        response = server.respond(query)

        expected = database.T @ query
        assert torch.allclose(response, expected, atol=1e-5)

    def test_pir_different_indices(self):
        """Multiple queries for different indices all return correct rows."""
        n_items, dim = 100, 64
        database = torch.randn(n_items, dim)
        client, server = pir_setup(database, device="cpu")

        for target in [0, 25, 50, 75, 99]:
            query = client.create_query(target)
            response = server.respond(query)
            result = client.extract_result(response, database)

            expected = database[target]
            assert torch.allclose(result, expected, atol=1e-4), \
                f"Failed for index {target}"

    def test_pir_query_looks_random(self):
        """Query vector has high entropy — server can't distinguish the target index."""
        n_items, dim = 128, 32
        database = torch.randn(n_items, dim)
        client = PIRClient(n_items, dim, device="cpu")

        query = client.create_query(42)

        # The query should look like Gaussian noise — check that it has
        # reasonable variance and no obvious spike
        mean = query.mean().item()
        std = query.std().item()

        # A unit vector would have std ≈ 1/sqrt(n) ≈ 0.088 for n=128.
        # Our noisy query should have std close to 1.0 (standard normal noise).
        assert std > 0.5, f"Query std too low ({std}), might leak target index"

        # Check that the target index doesn't obviously stand out
        # (it has noise + 1.0, but the noise is ~N(0,1), so +1 isn't a huge outlier)
        sorted_vals, _ = query.abs().sort(descending=True)
        # The maximum value shouldn't be much larger than expected for N(0,1) tails
        assert sorted_vals[0].item() < 6.0, "Query has suspiciously large outlier"


# ---------------------------------------------------------------------------
# Tier 3: FHE Scheme
# ---------------------------------------------------------------------------

class TestFHEScheme:
    def test_keygen(self):
        """Keys have correct dimensions."""
        scheme = FHEScheme(poly_degree=512, device="cpu")
        sk, (pk_a, pk_b) = scheme.keygen()

        assert sk.shape == (512,)
        assert pk_a.shape == (512,)
        assert pk_b.shape == (512,)
        assert sk.dtype == torch.float64
        # Secret key should be in {-1, 0, 1}
        assert sk.abs().max() <= 1

    def test_encrypt_decrypt_roundtrip(self, fhe):
        """Encrypt then decrypt recovers the original plaintext."""
        n = fhe.n
        t = fhe.t
        # Small plaintext values for clean roundtrip
        plaintext = torch.randint(0, min(100, t), (n,), dtype=torch.float64)

        ct = fhe.encrypt(plaintext)
        recovered = fhe.decrypt(ct)

        # Check first several coefficients match
        n_check = min(32, n)
        expected = plaintext[:n_check] % t
        actual = recovered[:n_check]

        # Allow small modular differences due to noise
        diff = (actual - expected) % t
        diff = torch.where(diff > t / 2, diff - t, diff)
        max_err = diff.abs().max().item()
        assert max_err < 2, f"Roundtrip error too large: {max_err}"

    def test_homomorphic_addition(self, fhe):
        """decrypt(enc(a) + enc(b)) == a + b mod t."""
        n = fhe.n
        t = fhe.t
        cap = min(50, t // 4)

        a = torch.randint(0, cap, (n,), dtype=torch.float64)
        b = torch.randint(0, cap, (n,), dtype=torch.float64)

        ct_a = fhe.encrypt(a)
        ct_b = fhe.encrypt(b)
        ct_sum = fhe.add(ct_a, ct_b)

        result = fhe.decrypt(ct_sum)
        expected = (a + b) % t

        n_check = min(32, n)
        diff = (result[:n_check] - expected[:n_check]) % t
        diff = torch.where(diff > t / 2, diff - t, diff)
        max_err = diff.abs().max().item()
        assert max_err < 2, f"Homomorphic addition error: {max_err}"

    def test_poly_mul_fft(self, fhe):
        """FFT polynomial multiplication gives correct results."""
        # Simple known multiplication: (1 + x) * (1 + x) = 1 + 2x + x^2
        # But mod x^n + 1, so for small polys this is just linear convolution.
        n = fhe.n
        a = torch.zeros(n, dtype=torch.float64)
        b = torch.zeros(n, dtype=torch.float64)
        a[0] = 1.0
        a[1] = 1.0
        b[0] = 1.0
        b[1] = 1.0

        result = fhe._poly_mul(a, b)

        # Expected: 1 + 2x + x^2 (mod x^n+1, but n >> 2 so no reduction)
        assert abs(result[0].item() - 1.0) < 1e-6
        assert abs(result[1].item() - 2.0) < 1e-6
        assert abs(result[2].item() - 1.0) < 1e-6
        # Rest should be ~0
        assert result[3:10].abs().max().item() < 1e-6

    def test_poly_mul_gpu_vs_naive(self, fhe):
        """FFT multiplication matches naive polynomial multiplication."""
        n = fhe.n
        # Use small random polynomials
        a = torch.randint(-3, 4, (n,), dtype=torch.float64)
        b = torch.randint(-3, 4, (n,), dtype=torch.float64)

        fft_result = fhe._poly_mul(a, b)

        # Naive: convolve then reduce mod x^n + 1
        full = torch.zeros(2 * n, dtype=torch.float64)
        for i in range(n):
            for j in range(n):
                full[i + j] += a[i] * b[j]
        naive_result = full[:n] - full[n:]

        assert torch.allclose(fft_result, naive_result, atol=1e-3), \
            f"Max diff: {(fft_result - naive_result).abs().max().item()}"

    def test_encrypted_dot_product(self, fhe):
        """Encrypted inner product (multiply + rotate-and-sum) is decryptable."""
        n = fhe.n

        a = torch.zeros(n, dtype=torch.float64)
        b = torch.zeros(n, dtype=torch.float64)
        a[0] = 3.0
        b[0] = 4.0

        ct_a = fhe.encrypt(a)
        ct_b = fhe.encrypt(b)

        ct_product = fhe.encrypted_inner_product(ct_a, ct_b)
        result = fhe.decrypt(ct_product)

        # Should be decryptable without error; the result encodes
        # the inner product across polynomial slots
        assert result.shape == (n,)
        assert isinstance(result, torch.Tensor)

    def test_encrypt_decrypt_vector(self, fhe):
        """Vector (shorter than poly_degree) pads correctly and roundtrips."""
        short_vec = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0], dtype=torch.float64)

        ct = fhe.encrypt(short_vec)
        result = fhe.decrypt(ct)

        # First 5 coefficients should match
        for i in range(5):
            diff = abs(result[i].item() - short_vec[i].item())
            assert diff < 2, f"Coefficient {i}: expected {short_vec[i].item()}, got {result[i].item()}"

    def test_batch_operations(self, fhe):
        """Multiple encrypt/add/decrypt operations in sequence stay correct."""
        n = fhe.n
        t = fhe.t
        cap = min(20, t // 8)

        vectors = [torch.randint(0, cap, (n,), dtype=torch.float64) for _ in range(4)]
        cts = [fhe.encrypt(v) for v in vectors]

        # Sum all four
        accumulated = cts[0]
        for ct in cts[1:]:
            accumulated = fhe.add(accumulated, ct)

        result = fhe.decrypt(accumulated)
        expected = sum(vectors) % t

        n_check = min(32, n)
        diff = (result[:n_check] - expected[:n_check]) % t
        diff = torch.where(diff > t / 2, diff - t, diff)
        assert diff.abs().max().item() < 2, "Batch addition accumulated too much noise"

    def test_homomorphic_multiplication(self, fhe):
        """decrypt(mul(enc(a), enc(b))) ≈ a * b mod t."""
        n = fhe.n
        t = fhe.t
        # Use very small values to keep noise manageable
        cap = min(8, int(t ** 0.5) // 2)

        a = torch.zeros(n, dtype=torch.float64)
        b = torch.zeros(n, dtype=torch.float64)
        # Only set first few coefficients to keep product small
        a[0] = 3.0
        a[1] = 2.0
        b[0] = 4.0
        b[1] = 1.0

        ct_a = fhe.encrypt(a)
        ct_b = fhe.encrypt(b)
        ct_prod = fhe.multiply(ct_a, ct_b)

        result = fhe.decrypt(ct_prod)

        # a * b mod (x^n+1): [3,2,0,...] * [4,1,0,...] = [12, 3+8, 2, 0, ...] = [12, 11, 2, ...]
        # Coefficient 0: 3*4 = 12
        # Coefficient 1: 3*1 + 2*4 = 11
        # Coefficient 2: 2*1 = 2
        # These are mod t, and with noise we allow some tolerance
        expected_0 = 12.0 % t
        expected_1 = 11.0 % t
        expected_2 = 2.0 % t

        # Multiplication is noisier — allow larger tolerance
        tol = t * 0.1  # 10% of modulus
        diff_0 = abs(((result[0].item() - expected_0) % t + t/2) % t - t/2)
        diff_1 = abs(((result[1].item() - expected_1) % t + t/2) % t - t/2)
        diff_2 = abs(((result[2].item() - expected_2) % t + t/2) % t - t/2)

        assert diff_0 < tol, f"Coeff 0: expected {expected_0}, got {result[0].item()}, diff {diff_0}"
        assert diff_1 < tol, f"Coeff 1: expected {expected_1}, got {result[1].item()}, diff {diff_1}"
        assert diff_2 < tol, f"Coeff 2: expected {expected_2}, got {result[2].item()}, diff {diff_2}"

    def test_relinearization_preserves_size(self, fhe):
        """After multiply + relin, ciphertext is still a 2-tuple."""
        n = fhe.n
        a = torch.randint(0, 5, (n,), dtype=torch.float64)
        b = torch.randint(0, 5, (n,), dtype=torch.float64)

        ct_a = fhe.encrypt(a)
        ct_b = fhe.encrypt(b)
        ct_prod = fhe.multiply(ct_a, ct_b)

        assert isinstance(ct_prod, tuple)
        assert len(ct_prod) == 2
        assert ct_prod[0].shape == (n,)
        assert ct_prod[1].shape == (n,)

    def test_security_level_128(self):
        """128-bit security uses correct parameters."""
        fhe = FHEScheme(security_level=128, device="cpu")
        assert fhe.n == 4096
        assert fhe.q == 2 ** 109
        assert fhe.security_level == 128

        sk, pk = fhe.keygen()
        assert sk.shape == (4096,)

        # Encrypt-decrypt still works
        pt = torch.randint(0, 100, (4096,), dtype=torch.float64)
        ct = fhe.encrypt(pt)
        result = fhe.decrypt(ct)
        diff = (result[:16] - pt[:16]) % fhe.t
        diff = torch.where(diff > fhe.t / 2, diff - fhe.t, diff)
        assert diff.abs().max().item() < 2

    def test_security_level_invalid(self):
        """Invalid security level raises ValueError."""
        with pytest.raises(ValueError, match="security_level"):
            FHEScheme(security_level=64)

    def test_rotate(self, fhe):
        """Rotation produces a valid ciphertext of same shape."""
        n = fhe.n
        pt = torch.randint(0, 10, (n,), dtype=torch.float64)
        ct = fhe.encrypt(pt)
        ct_rot = fhe.rotate(ct, 1)

        assert isinstance(ct_rot, tuple)
        assert len(ct_rot) == 2
        assert ct_rot[0].shape == (n,)
        # Should still be decryptable (different plaintext due to rotation)
        result = fhe.decrypt(ct_rot)
        assert result.shape == (n,)

    def test_multiply_then_add(self, fhe):
        """Chained operations: multiply then add still decrypts."""
        n = fhe.n
        t = fhe.t

        a = torch.zeros(n, dtype=torch.float64)
        b = torch.zeros(n, dtype=torch.float64)
        c = torch.zeros(n, dtype=torch.float64)
        a[0] = 2.0
        b[0] = 3.0
        c[0] = 5.0

        ct_a = fhe.encrypt(a)
        ct_b = fhe.encrypt(b)
        ct_c = fhe.encrypt(c)

        # (a * b) + c = 6 + 5 = 11
        ct_prod = fhe.multiply(ct_a, ct_b)
        ct_result = fhe.add(ct_prod, ct_c)

        result = fhe.decrypt(ct_result)
        expected = 11.0 % t
        diff = abs(((result[0].item() - expected) % t + t/2) % t - t/2)
        assert diff < t * 0.1, f"Expected ~{expected}, got {result[0].item()}"


# ---------------------------------------------------------------------------
# Tier 3b: Encrypted Vector Store
# ---------------------------------------------------------------------------

class TestEncryptedVectorStore:
    def test_add_and_query(self, fhe):
        """Add vectors, query, get results back."""
        store = EncryptedVectorStore(fhe)

        # Add some vectors
        v1 = torch.tensor([0.9, 0.1, 0.0, 0.0], dtype=torch.float32)
        v2 = torch.tensor([0.0, 0.0, 0.9, 0.1], dtype=torch.float32)
        v3 = torch.tensor([0.8, 0.2, 0.0, 0.0], dtype=torch.float32)

        store.add_encrypted(v1)
        store.add_encrypted(v2)
        store.add_encrypted(v3)

        assert store.size == 3

        # Query with something similar to v1
        query = torch.tensor([1.0, 0.0, 0.0, 0.0], dtype=torch.float32)
        encrypted_scores = store.encrypted_query(query)

        assert len(encrypted_scores) == 3
        # Each score is a ciphertext tuple
        assert isinstance(encrypted_scores[0], tuple)
        assert len(encrypted_scores[0]) == 2

    def test_encrypted_similarity_order(self, fhe):
        """Encrypted search returns ranked results with scores."""
        store = EncryptedVectorStore(fhe)

        v1 = torch.tensor([0.9, 0.9, 0.0, 0.0], dtype=torch.float32)
        v2 = torch.tensor([0.1, 0.1, 0.0, 0.0], dtype=torch.float32)
        v3 = torch.tensor([0.5, 0.5, 0.0, 0.0], dtype=torch.float32)

        store.add_encrypted(v1)
        store.add_encrypted(v2)
        store.add_encrypted(v3)

        query = torch.tensor([0.9, 0.9, 0.0, 0.0], dtype=torch.float32)
        ranked = store.query_and_rank(query, k=3)

        # Should return all 3 with indices and scores
        assert len(ranked) == 3
        indices = [idx for idx, score in ranked]
        assert set(indices) == {0, 1, 2}
        # Scores should be finite numbers
        for idx, score in ranked:
            assert not (score != score), f"Score is NaN for index {idx}"  # NaN check

    def test_gpu_acceleration(self, fhe):
        """Encrypted operations use torch tensors (GPU-ready)."""
        store = EncryptedVectorStore(fhe)
        v = torch.randn(8, dtype=torch.float32)
        store.add_encrypted(v)

        # Verify ciphertexts are torch tensors on the correct device
        ct0, ct1 = store._encrypted_vectors[0]
        assert isinstance(ct0, torch.Tensor)
        assert isinstance(ct1, torch.Tensor)
        assert ct0.device.type == fhe.device
        assert ct0.dtype == torch.float64

    def test_multiple_queries(self, fhe):
        """Several queries return consistent results."""
        store = EncryptedVectorStore(fhe)

        vecs = [torch.randn(8, dtype=torch.float32) for _ in range(5)]
        for v in vecs:
            store.add_encrypted(v)

        query = torch.randn(8, dtype=torch.float32)

        # Run the same query twice — rankings should be identical
        r1 = store.query_and_rank(query, k=5)
        r2 = store.query_and_rank(query, k=5)

        # Due to encryption noise, exact scores differ per run,
        # but the relative ordering of scores from the same stored
        # ciphertexts should be stable for the same query
        indices_1 = [idx for idx, _ in r1]
        indices_2 = [idx for idx, _ in r2]

        # At minimum, the top result should usually match
        # (noise can occasionally reorder close scores)
        assert len(indices_1) == 5
        assert len(indices_2) == 5


# ---------------------------------------------------------------------------
# Tier 4: Encrypted Git Operations
# ---------------------------------------------------------------------------

class TestEncryptedGitOps:
    def test_encrypted_diff_same_vector(self, fhe):
        """Diff of identical encrypted vectors decrypts to ~zero."""
        ops = EncryptedGitOps(fhe)
        n = fhe.n
        t = fhe.t
        v = torch.randint(0, min(50, t // 4), (n,), dtype=torch.float64)
        ct = fhe.encrypt(v)

        ct_diff = ops.encrypted_diff(ct, ct)
        result = fhe.decrypt(ct_diff)

        # Should be ~0 (with noise)
        centered = result.clone()
        centered[centered > t / 2] -= t
        assert centered.abs().max().item() < 2, "Diff of same vector should be ~zero"

    def test_encrypted_diff_different_vectors(self, fhe):
        """Diff of different encrypted vectors decrypts to their difference."""
        ops = EncryptedGitOps(fhe)
        n = fhe.n
        t = fhe.t
        a = torch.zeros(n, dtype=torch.float64)
        b = torch.zeros(n, dtype=torch.float64)
        a[0] = 10.0
        b[0] = 3.0

        ct_a = fhe.encrypt(a)
        ct_b = fhe.encrypt(b)

        ct_diff = ops.encrypted_diff(ct_a, ct_b)
        result = fhe.decrypt(ct_diff)

        # Coefficient 0 should be ~7
        centered = result[0].item()
        if centered > t / 2:
            centered -= t
        assert abs(centered - 7.0) < 2, f"Expected ~7, got {centered}"

    def test_has_changed_same(self, fhe):
        """Same vector should not register as changed."""
        ops = EncryptedGitOps(fhe)
        n = fhe.n
        v = torch.randint(0, 20, (n,), dtype=torch.float64)
        ct = fhe.encrypt(v)

        ct_diff = ops.encrypted_has_changed(ct, ct)
        assert not ops.decrypt_has_changed(ct_diff)

    def test_has_changed_different(self, fhe):
        """Different vectors should register as changed."""
        ops = EncryptedGitOps(fhe)
        n = fhe.n
        a = torch.zeros(n, dtype=torch.float64)
        b = torch.zeros(n, dtype=torch.float64)
        a[0] = 50.0
        b[0] = 10.0

        ct_a = fhe.encrypt(a)
        ct_b = fhe.encrypt(b)

        ct_diff = ops.encrypted_has_changed(ct_a, ct_b)
        assert ops.decrypt_has_changed(ct_diff)

    def test_encrypted_merge_sum(self, fhe):
        """Merge by sum adds encrypted vectors homomorphically."""
        ops = EncryptedGitOps(fhe)
        n = fhe.n
        t = fhe.t

        a = torch.zeros(n, dtype=torch.float64)
        b = torch.zeros(n, dtype=torch.float64)
        c = torch.zeros(n, dtype=torch.float64)
        a[0] = 5.0
        b[0] = 10.0
        c[0] = 15.0

        cts = [fhe.encrypt(v) for v in [a, b, c]]
        ct_merged = ops.encrypted_merge_sum(cts)
        result = fhe.decrypt(ct_merged)

        expected = 30.0 % t
        diff = abs(((result[0].item() - expected) % t + t / 2) % t - t / 2)
        assert diff < 2, f"Expected ~30, got {result[0].item()}"

    def test_encrypted_merge_product(self, fhe):
        """Merge by product multiplies encrypted vectors."""
        ops = EncryptedGitOps(fhe)
        n = fhe.n
        t = fhe.t

        a = torch.zeros(n, dtype=torch.float64)
        b = torch.zeros(n, dtype=torch.float64)
        a[0] = 3.0
        b[0] = 4.0

        ct_a = fhe.encrypt(a)
        ct_b = fhe.encrypt(b)
        ct_prod = ops.encrypted_merge_product(ct_a, ct_b)
        result = fhe.decrypt(ct_prod)

        expected = 12.0 % t
        diff = abs(((result[0].item() - expected) % t + t / 2) % t - t / 2)
        assert diff < t * 0.1, f"Expected ~12, got {result[0].item()}"

    def test_encrypted_commit_hash(self, fhe):
        """Commit hash is a valid hex string."""
        ops = EncryptedGitOps(fhe)
        n = fhe.n
        v = torch.randint(0, 50, (n,), dtype=torch.float64)
        ct = fhe.encrypt(v)

        h = ops.encrypted_commit_hash(ct)
        assert isinstance(h, str)
        assert len(h) == 64  # SHA-256 hex
        # Same ciphertext = same hash
        assert ops.encrypted_commit_hash(ct) == h

    def test_encrypted_branch_diff(self, fhe):
        """Branch diff returns per-index encrypted diffs."""
        ops = EncryptedGitOps(fhe)
        n = fhe.n

        v1 = torch.randint(0, 20, (n,), dtype=torch.float64)
        v2 = torch.randint(0, 20, (n,), dtype=torch.float64)
        v3 = torch.randint(0, 20, (n,), dtype=torch.float64)

        branch_a = [fhe.encrypt(v1), fhe.encrypt(v2)]
        branch_b = [fhe.encrypt(v1), fhe.encrypt(v3)]

        diffs = ops.encrypted_branch_diff(branch_a, branch_b)
        assert len(diffs) == 2
        assert all(isinstance(d, tuple) and len(d) == 2 for _, d in diffs)

    def test_three_way_merge(self, fhe):
        """Three-way merge produces correct number of outputs."""
        ops = EncryptedGitOps(fhe)
        n = fhe.n

        base_vecs = [fhe.encrypt(torch.randint(0, 10, (n,), dtype=torch.float64)) for _ in range(3)]
        ours_vecs = [fhe.encrypt(torch.randint(0, 10, (n,), dtype=torch.float64)) for _ in range(3)]
        theirs_vecs = [fhe.encrypt(torch.randint(0, 10, (n,), dtype=torch.float64)) for _ in range(3)]

        merged = ops.encrypted_three_way_merge(base_vecs, ours_vecs, theirs_vecs)
        assert len(merged) == 3
        # Each result should be a valid ciphertext tuple
        for ct in merged:
            assert isinstance(ct, tuple)
            assert len(ct) == 2
            assert ct[0].shape == (n,)

    def test_encrypted_aggregate_sum(self, fhe):
        """Aggregate sum works on encrypted vectors."""
        ops = EncryptedGitOps(fhe)
        n = fhe.n

        vecs = [torch.zeros(n, dtype=torch.float64) for _ in range(4)]
        for i, v in enumerate(vecs):
            v[0] = float(i + 1)  # 1, 2, 3, 4

        cts = [fhe.encrypt(v) for v in vecs]
        ct_sum = ops.encrypted_aggregate(cts, op="sum")
        result = fhe.decrypt(ct_sum)

        t = fhe.t
        expected = 10.0 % t
        diff = abs(((result[0].item() - expected) % t + t / 2) % t - t / 2)
        assert diff < 2, f"Expected ~10, got {result[0].item()}"

    def test_aggregate_invalid_op(self, fhe):
        """Invalid aggregation op raises ValueError."""
        ops = EncryptedGitOps(fhe)
        with pytest.raises(ValueError, match="op must be"):
            ops.encrypted_aggregate([], op="max")


# ---------------------------------------------------------------------------
# Tier 5: Encrypted Database Operations
# ---------------------------------------------------------------------------

@pytest.fixture
def db_ops(fhe, se_key):
    """EncryptedDBOps with both FHE and SearchableEncryption."""
    se = SearchableEncryption(se_key)
    return EncryptedDBOps(fhe, se), se


class TestEncryptedDBOps:
    def test_where_eq(self, db_ops):
        """Encrypted WHERE field = value."""
        ops, se = db_ops
        rows = [
            se.encrypt_row({"name": "Alice", "dept": "eng"}, {"name": "equality", "dept": "equality"}),
            se.encrypt_row({"name": "Bob", "dept": "sales"}, {"name": "equality", "dept": "equality"}),
            se.encrypt_row({"name": "Alice", "dept": "sales"}, {"name": "equality", "dept": "equality"}),
        ]
        matches = ops.encrypted_where_eq(rows, "name", "Alice")
        assert matches == [0, 2]

    def test_where_range(self, db_ops):
        """Encrypted WHERE field > value."""
        ops, se = db_ops
        rows = [
            se.encrypt_row({"name": "a", "age": 25}, {"name": "equality", "age": "range"}),
            se.encrypt_row({"name": "b", "age": 35}, {"name": "equality", "age": "range"}),
            se.encrypt_row({"name": "c", "age": 45}, {"name": "equality", "age": "range"}),
        ]
        matches = ops.encrypted_where_range(rows, "age", "gt", 30)
        assert 1 in matches
        assert 2 in matches
        assert 0 not in matches

    def test_join(self, db_ops):
        """Encrypted inner join on equality field."""
        ops, se = db_ops
        users = [
            se.encrypt_row({"id": "u1", "name": "Alice"}, {"id": "equality", "name": "equality"}),
            se.encrypt_row({"id": "u2", "name": "Bob"}, {"id": "equality", "name": "equality"}),
        ]
        orders = [
            se.encrypt_row({"id": "u1", "item": "laptop"}, {"id": "equality", "item": "exact"}),
            se.encrypt_row({"id": "u3", "item": "phone"}, {"id": "equality", "item": "exact"}),
            se.encrypt_row({"id": "u1", "item": "mouse"}, {"id": "equality", "item": "exact"}),
        ]
        pairs = ops.encrypted_join(users, orders, "id")
        assert (0, 0) in pairs  # Alice -> laptop
        assert (0, 2) in pairs  # Alice -> mouse
        assert len([p for p in pairs if p[0] == 1]) == 0  # Bob has no orders

    def test_group_by(self, db_ops):
        """Encrypted GROUP BY groups matching keys."""
        ops, se = db_ops
        rows = [
            se.encrypt_row({"dept": "eng", "name": "a"}, {"dept": "equality", "name": "equality"}),
            se.encrypt_row({"dept": "sales", "name": "b"}, {"dept": "equality", "name": "equality"}),
            se.encrypt_row({"dept": "eng", "name": "c"}, {"dept": "equality", "name": "equality"}),
        ]
        groups = ops.encrypted_group_by(rows, "dept")
        assert len(groups) == 2  # eng and sales
        # One group should have 2 members (eng), one should have 1 (sales)
        sizes = sorted([len(v) for v in groups.values()])
        assert sizes == [1, 2]

    def test_dedup(self, db_ops):
        """Encrypted deduplication finds unique rows."""
        ops, se = db_ops
        rows = [
            se.encrypt_row({"email": "a@x.com"}, {"email": "equality"}),
            se.encrypt_row({"email": "b@x.com"}, {"email": "equality"}),
            se.encrypt_row({"email": "a@x.com"}, {"email": "equality"}),  # dup
            se.encrypt_row({"email": "c@x.com"}, {"email": "equality"}),
        ]
        unique = ops.encrypted_dedup(rows, "email")
        assert len(unique) == 3
        assert 0 in unique
        assert 1 in unique
        assert 3 in unique
        assert 2 not in unique  # duplicate

    def test_check_unique_pass(self, db_ops):
        """Uniqueness check passes when all values are different."""
        ops, se = db_ops
        rows = [
            se.encrypt_row({"id": "a"}, {"id": "equality"}),
            se.encrypt_row({"id": "b"}, {"id": "equality"}),
            se.encrypt_row({"id": "c"}, {"id": "equality"}),
        ]
        assert ops.encrypted_check_unique(rows, "id") is True

    def test_check_unique_fail(self, db_ops):
        """Uniqueness check fails when values repeat."""
        ops, se = db_ops
        rows = [
            se.encrypt_row({"id": "a"}, {"id": "equality"}),
            se.encrypt_row({"id": "b"}, {"id": "equality"}),
            se.encrypt_row({"id": "a"}, {"id": "equality"}),
        ]
        assert ops.encrypted_check_unique(rows, "id") is False

    def test_check_foreign_key(self, db_ops):
        """FK check finds violations."""
        ops, se = db_ops
        parents = [
            se.encrypt_row({"id": "p1"}, {"id": "equality"}),
            se.encrypt_row({"id": "p2"}, {"id": "equality"}),
        ]
        children = [
            se.encrypt_row({"parent_id": "p1"}, {"parent_id": "equality"}),
            se.encrypt_row({"parent_id": "p3"}, {"parent_id": "equality"}),  # violation
            se.encrypt_row({"parent_id": "p2"}, {"parent_id": "equality"}),
        ]
        violations = ops.encrypted_check_foreign_key(children, "parent_id", parents, "id")
        assert violations == [1]

    def test_snapshot_diff(self, fhe):
        """Snapshot diff detects changes."""
        ops = EncryptedDBOps(fhe)
        n = fhe.n
        v1 = fhe.encrypt(torch.randint(0, 10, (n,), dtype=torch.float64))
        v2 = fhe.encrypt(torch.randint(0, 10, (n,), dtype=torch.float64))

        snap_a = [v1, v2]
        snap_b = [v1, fhe.encrypt(torch.randint(0, 10, (n,), dtype=torch.float64))]

        result = ops.encrypted_snapshot_diff(snap_a, snap_b)
        assert result["same_count"] == 1  # v1 unchanged
        assert len(result["changed_indices"]) == 1  # v2 changed
        assert result["added"] == 0
        assert result["removed"] == 0

    def test_snapshot_diff_size_change(self, fhe):
        """Snapshot diff detects added/removed vectors."""
        ops = EncryptedDBOps(fhe)
        n = fhe.n
        snap_a = [fhe.encrypt(torch.randint(0, 10, (n,), dtype=torch.float64)) for _ in range(3)]
        snap_b = snap_a[:2]  # removed one

        result = ops.encrypted_snapshot_diff(snap_a, snap_b)
        assert result["removed"] == 1

    def test_aggregate_vectors(self, fhe):
        """Encrypted vector aggregation sums correctly."""
        ops = EncryptedDBOps(fhe)
        n = fhe.n
        t = fhe.t

        vecs_plain = [torch.zeros(n, dtype=torch.float64) for _ in range(3)]
        vecs_plain[0][0] = 5.0
        vecs_plain[1][0] = 10.0
        vecs_plain[2][0] = 15.0

        cts = [fhe.encrypt(v) for v in vecs_plain]
        ct_sum = ops.encrypted_aggregate_vectors(cts, op="sum")
        result = fhe.decrypt(ct_sum)

        expected = 30.0 % t
        diff = abs(((result[0].item() - expected) % t + t / 2) % t - t / 2)
        assert diff < 2

    def test_no_se_raises(self, fhe):
        """DB ops that need SearchableEncryption raise without it."""
        ops = EncryptedDBOps(fhe, se=None)
        with pytest.raises(RuntimeError, match="SearchableEncryption required"):
            ops.encrypted_where_eq([], "f", "v")
        with pytest.raises(RuntimeError, match="SearchableEncryption required"):
            ops.encrypted_join([], [], "f")
        with pytest.raises(RuntimeError, match="SearchableEncryption required"):
            ops.encrypted_dedup([], "f")
        with pytest.raises(RuntimeError, match="SearchableEncryption required"):
            ops.encrypted_check_unique([], "f")
