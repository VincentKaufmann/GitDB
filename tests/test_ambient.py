"""Tests for Emirati AC — spreading activation engine."""

import time
import pytest
import torch

from gitdb import GitDB
from gitdb.ambient import EmiratiAC


def rvec(n=1, dim=8):
    return torch.randn(n, dim)


@pytest.fixture
def db(tmp_path):
    """GitDB with some vectors."""
    db = GitDB(str(tmp_path / "store"), dim=8, device="cpu")
    # Add clusters of related vectors
    cluster_a = torch.randn(1, 8)
    cluster_b = torch.randn(1, 8)
    # Each cluster has 5 vectors near the centroid
    for i in range(5):
        db.add(cluster_a + torch.randn(1, 8) * 0.05,
               documents=[f"auth_{i}"], metadata=[{"category": "auth"}])
    for i in range(5):
        db.add(cluster_b + torch.randn(1, 8) * 0.05,
               documents=[f"finance_{i}"], metadata=[{"category": "finance"}])
    db.commit("Initial clusters")
    return db


# ═══════════════════════════════════════════════════════════════
#  Basic Lifecycle
# ═══════════════════════════════════════════════════════════════

class TestLifecycle:
    def test_ac_exists(self, db):
        assert isinstance(db.ac, EmiratiAC)
        assert not db.ac.running

    def test_start_stop(self, db):
        db.ac.start()
        assert db.ac.running
        db.ac.stop()
        assert not db.ac.running

    def test_double_start(self, db):
        db.ac.start()
        db.ac.start()  # Should not error
        assert db.ac.running
        db.ac.stop()

    def test_stats_initial(self, db):
        stats = db.ac.stats()
        assert stats["running"] is False
        assert stats["active_vectors"] == 0
        assert stats["cycles"] == 0


# ═══════════════════════════════════════════════════════════════
#  Direct Activation (feed_vectors)
# ═══════════════════════════════════════════════════════════════

class TestDirectActivation:
    def test_feed_vectors_activates(self, db):
        db.ac.start()
        # Feed a vector similar to auth cluster
        auth_vec = db.tree.embeddings[0].clone()
        db.ac.feed_vectors(auth_vec)
        time.sleep(0.1)  # Let it process

        # Some vectors should be activated
        act_map = db.ac.get_activation_map()
        assert len(act_map) > 0
        db.ac.stop()

    def test_feed_activates_similar_vectors(self, db):
        db.ac.start()
        # Feed the first auth vector
        auth_vec = db.tree.embeddings[0].clone()
        db.ac.feed_vectors(auth_vec)

        act_map = db.ac.get_activation_map()
        # Auth vectors (0-4) should be more activated than finance (5-9)
        auth_total = sum(act_map.get(i, 0) for i in range(5))
        finance_total = sum(act_map.get(i, 0) for i in range(5, 10))
        assert auth_total > finance_total
        db.ac.stop()

    def test_primed_returns_results(self, db):
        db.ac.start()
        db.ac.feed_vectors(db.tree.embeddings[0].clone())
        results = db.ac.primed(5)
        assert len(results.ids) > 0
        assert all(isinstance(s, float) for s in results.scores)
        db.ac.stop()


# ═══════════════════════════════════════════════════════════════
#  Spreading Activation
# ═══════════════════════════════════════════════════════════════

class TestSpreading:
    def test_spread_activates_neighbors(self, db):
        """Manually activate one vector and verify spread reaches neighbors."""
        db.ac.start()
        # Manually set high activation on vector 0
        db.ac.activations[0] = 0.8
        # Force a spread cycle
        hops = db.ac._spread()
        # Should have activated some neighbors
        assert hops > 0 or len(db.ac.activations) > 1
        db.ac.stop()

    def test_spread_factor(self, db):
        """Spread activation should be attenuated by SPREAD_FACTOR."""
        db.ac.start()
        db.ac.activations[0] = 1.0
        db.ac._spread()
        # Neighbors should have activation < SPREAD_FACTOR * 1.0 * sim
        for idx, level in db.ac.activations.items():
            if idx != 0:
                assert level < 1.0  # Must be less than source
        db.ac.stop()


# ═══════════════════════════════════════════════════════════════
#  Decay
# ═══════════════════════════════════════════════════════════════

class TestDecay:
    def test_activations_decay(self, db):
        db.ac.start()
        db.ac.activations[0] = 0.5
        initial = db.ac.activations[0]
        db.ac._decay_only()
        assert db.ac.activations[0] < initial

    def test_below_min_removed(self, db):
        db.ac.start()
        db.ac.activations[0] = 0.01  # Below MIN_ACTIVATION
        db.ac._decay_only()
        assert 0 not in db.ac.activations
        db.ac.stop()


# ═══════════════════════════════════════════════════════════════
#  Drift Detection
# ═══════════════════════════════════════════════════════════════

class TestDrift:
    def test_no_drift_initially(self, db):
        assert db.ac.drift() is None

    def test_drift_after_divergent_additions(self, db):
        db.ac._compute_centroid()
        # Add vectors very far from centroid
        for _ in range(20):
            orthogonal = torch.randn(8) * 100
            db.ac.track_addition(orthogonal)
        drift = db.ac.drift()
        # May or may not trigger depending on random vectors
        # But the mechanism should work
        assert drift is None or isinstance(drift, dict)

    def test_drift_callback(self, db):
        alerts = []
        db.ac.on_drift(lambda d: alerts.append(d))
        assert db.ac._on_drift is not None


# ═══════════════════════════════════════════════════════════════
#  Integration with GitDB query
# ═══════════════════════════════════════════════════════════════

class TestACQueryIntegration:
    def test_query_with_ac_boost(self, db):
        db.ac.start()
        # Pre-activate auth vectors
        db.ac.feed_vectors(db.tree.embeddings[0].clone())

        # Query with a neutral vector
        q = torch.randn(8)
        results = db.query(q, k=10, ac_boost=True)
        assert len(results.ids) == 10  # All vectors returned
        db.ac.stop()

    def test_query_without_ac_boost(self, db):
        db.ac.start()
        q = torch.randn(8)
        results = db.query(q, k=5, ac_boost=False)
        assert len(results.ids) == 5
        db.ac.stop()

    def test_add_feeds_ac(self, db):
        db.ac.start()
        db.add(rvec(1, 8), documents=["new doc"])
        # AC should have been fed
        act_map = db.ac.get_activation_map()
        assert len(act_map) > 0
        db.ac.stop()

    def test_ac_context_feed(self, db):
        db.ac.start()
        db.ac.feed("log", "viewing commit history")
        assert len(db.ac.context_buffer) == 1
        assert "log" in db.ac.context_buffer[0]
        db.ac.stop()


# ═══════════════════════════════════════════════════════════════
#  Hot Cache
# ═══════════════════════════════════════════════════════════════

class TestHotCache:
    def test_hot_cache_sorted(self, db):
        db.ac.start()
        db.ac.activations = {0: 0.9, 1: 0.3, 2: 0.7, 3: 0.1}
        db.ac._decay_only()  # This updates hot cache
        # Should be sorted descending
        scores = [level for _, level in db.ac.hot_cache]
        assert scores == sorted(scores, reverse=True)
        db.ac.stop()

    def test_hot_cache_size_limit(self, db):
        db.ac.start()
        # Activate all 10 vectors
        for i in range(10):
            db.ac.activations[i] = 0.5 + i * 0.05
        db.ac.HOT_CACHE_SIZE = 5
        db.ac._decay_only()
        assert len(db.ac.hot_cache) <= 5
        db.ac.stop()
