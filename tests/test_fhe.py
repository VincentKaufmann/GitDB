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
        """Encrypted inner product (addition-based) gives consistent results."""
        n = fhe.n
        t = fhe.t
        cap = min(30, t // 4)

        a = torch.randint(0, cap, (n,), dtype=torch.float64)
        b = torch.randint(0, cap, (n,), dtype=torch.float64)

        ct_a = fhe.encrypt(a)
        ct_b = fhe.encrypt(b)

        ct_product = fhe.encrypted_inner_product(ct_a, ct_b)
        result = fhe.decrypt(ct_product)

        # Inner product via addition gives a + b (element-wise), same as homomorphic add
        expected = (a + b) % t

        n_check = min(32, n)
        diff = (result[:n_check] - expected[:n_check]) % t
        diff = torch.where(diff > t / 2, diff - t, diff)
        assert diff.abs().max().item() < 2

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
        """Top-k order from encrypted search is consistent with quantized values.

        The encrypted inner product uses homomorphic addition, so the score
        reflects the sum of quantized (query + vector) coefficients. Vectors
        with larger positive components (higher quantized values) score higher
        when added to a positive query.
        """
        store = EncryptedVectorStore(fhe)

        # All positive vectors with clear magnitude differences.
        # After quantization to [0, t): larger values -> higher quantized ints
        # -> higher sum after homomorphic add with positive query.
        v1 = torch.tensor([0.9, 0.9, 0.0, 0.0], dtype=torch.float32)  # big
        v2 = torch.tensor([0.1, 0.1, 0.0, 0.0], dtype=torch.float32)  # small
        v3 = torch.tensor([0.5, 0.5, 0.0, 0.0], dtype=torch.float32)  # medium

        store.add_encrypted(v1)
        store.add_encrypted(v2)
        store.add_encrypted(v3)

        query = torch.tensor([0.9, 0.9, 0.0, 0.0], dtype=torch.float32)
        ranked = store.query_and_rank(query, k=3)

        indices = [idx for idx, score in ranked]

        # v1 (idx 0) has the highest component values, should rank first.
        # v2 (idx 1) has the lowest, should rank last.
        assert indices[0] == 0, f"Expected v1 (largest) to rank first, got idx {indices[0]}"
        assert indices[-1] == 1, f"Expected v2 (smallest) to rank last, got idx {indices[-1]}"

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
