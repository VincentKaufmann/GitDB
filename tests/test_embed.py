"""Tests for embedding integration — Arctic + NV-EmbedQA, Matryoshka, semantic ops."""

import pytest
import torch

from gitdb.embed import (
    MODELS, DEFAULT_MODEL, embed, embed_query, list_models,
    get_model_info, similarity, semantic_filter,
)


# ═══════════════════════════════════════════════════════════════
#  Model Registry
# ═══════════════════════════════════════════════════════════════

class TestRegistry:
    def test_default_model_is_arctic(self):
        assert DEFAULT_MODEL == "arctic"

    def test_arctic_in_registry(self):
        assert "arctic" in MODELS
        assert MODELS["arctic"]["dim"] == 1024
        assert MODELS["arctic"]["params"] == "335M"

    def test_nv_embed_qa_in_registry(self):
        assert "nv-embed-qa" in MODELS
        assert MODELS["nv-embed-qa"]["dim"] == 2048
        assert MODELS["nv-embed-qa"]["matryoshka"] is True
        assert 384 in MODELS["nv-embed-qa"]["matryoshka_dims"]
        assert 1024 in MODELS["nv-embed-qa"]["matryoshka_dims"]

    def test_list_models(self):
        models = list_models()
        assert len(models) >= 2
        assert "arctic" in models
        assert "nv-embed-qa" in models

    def test_get_model_info(self):
        info = get_model_info("arctic")
        assert "hf_name" in info
        assert "dim" in info

    def test_unknown_model_raises(self):
        with pytest.raises(ValueError, match="Unknown model"):
            get_model_info("nonexistent")


# ═══════════════════════════════════════════════════════════════
#  Similarity Utility
# ═══════════════════════════════════════════════════════════════

class TestSimilarity:
    def test_identical_vectors(self):
        a = torch.randn(1, 128)
        sim = similarity(a, a)
        assert sim.item() == pytest.approx(1.0, abs=1e-5)

    def test_orthogonal_vectors(self):
        a = torch.zeros(128)
        a[0] = 1.0
        b = torch.zeros(128)
        b[1] = 1.0
        sim = similarity(a, b)
        assert abs(sim.item()) < 1e-5

    def test_batch_similarity(self):
        a = torch.randn(3, 64)
        b = torch.randn(5, 64)
        sim = similarity(a, b)
        assert sim.shape == (3, 5)

    def test_1d_input(self):
        a = torch.randn(64)
        b = torch.randn(64)
        sim = similarity(a, b)
        assert sim.shape == (1, 1)


# ═══════════════════════════════════════════════════════════════
#  Semantic Filter (no model required — uses pre-computed embeddings)
# ═══════════════════════════════════════════════════════════════

class TestSemanticFilterUnit:
    """Tests that don't need a loaded model — mock the embed step."""

    def test_filter_by_similarity_threshold(self):
        """Given pre-computed embeddings, filter by cosine similarity."""
        dim = 64
        query_dir = torch.randn(dim)
        query_dir = query_dir / query_dir.norm()

        # Make first 3 nearly identical to query, rest random
        embeddings = torch.randn(10, dim)
        for i in range(3):
            embeddings[i] = query_dir + torch.randn(dim) * 0.01
            embeddings[i] = embeddings[i] / embeddings[i].norm()

        sims = similarity(query_dir, embeddings).squeeze(0)
        # First 3 should have very high similarity
        for i in range(3):
            assert sims[i].item() > 0.95


# ═══════════════════════════════════════════════════════════════
#  Integration Tests (require sentence-transformers + model download)
#  These are slow — skip in CI with: pytest -m "not slow"
# ═══════════════════════════════════════════════════════════════

try:
    import sentence_transformers
    HAS_ST = True
except ImportError:
    HAS_ST = False

needs_st = pytest.mark.skipif(not HAS_ST, reason="sentence-transformers not installed")
slow = pytest.mark.slow


@needs_st
@slow
class TestEmbedIntegration:
    def test_embed_single_text(self):
        result = embed("hello world", model_name="arctic")
        assert result.shape == (1, 1024)
        assert result.device.type == "cpu"

    def test_embed_batch(self):
        texts = ["first document", "second document", "third document"]
        result = embed(texts, model_name="arctic")
        assert result.shape == (3, 1024)

    def test_embed_query_returns_1d(self):
        result = embed_query("test query", model_name="arctic")
        assert result.shape == (1024,)

    def test_embed_normalized(self):
        result = embed("normalized test", model_name="arctic", normalize=True)
        norm = result.norm(dim=1)
        assert norm.item() == pytest.approx(1.0, abs=1e-4)

    def test_similar_texts_high_similarity(self):
        a = embed("the cat sat on the mat", model_name="arctic")
        b = embed("a cat was sitting on a mat", model_name="arctic")
        sim = similarity(a, b)
        assert sim.item() > 0.7  # Should be quite similar

    def test_dissimilar_texts_low_similarity(self):
        a = embed("quantum physics equations", model_name="arctic")
        b = embed("chocolate cake recipe", model_name="arctic")
        sim = similarity(a, b)
        assert sim.item() < 0.5  # Should be dissimilar

    def test_semantic_filter_integration(self):
        texts = [
            "machine learning model training",
            "deep neural network architecture",
            "chocolate chip cookie recipe",
            "gradient descent optimization",
            "flower garden maintenance",
        ]
        embeddings = embed(texts, model_name="arctic")
        indices = semantic_filter(
            embeddings, "AI and machine learning", threshold=0.4, model_name="arctic"
        )
        # ML-related texts (0, 1, 3) should match
        assert 0 in indices or 1 in indices or 3 in indices
        # Cookie recipe and gardening probably shouldn't
        assert 4 not in indices


@needs_st
@slow
class TestMatryoshka:
    """Test NV-EmbedQA Matryoshka dimension truncation.

    Note: These tests require the NV-EmbedQA model (~2.5GB download).
    Skip with: pytest -k "not Matryoshka"
    """

    def test_nv_embed_qa_default_dim(self):
        result = embed("test matryoshka", model_name="nv-embed-qa")
        assert result.shape == (1, 2048)

    def test_nv_embed_qa_1024d(self):
        result = embed("test matryoshka", model_name="nv-embed-qa", dim=1024)
        assert result.shape == (1, 1024)

    def test_nv_embed_qa_384d(self):
        result = embed("test matryoshka", model_name="nv-embed-qa", dim=384)
        assert result.shape == (1, 384)

    def test_invalid_matryoshka_dim(self):
        with pytest.raises(ValueError, match="Matryoshka dim"):
            embed("test", model_name="nv-embed-qa", dim=500)

    def test_truncated_still_normalized(self):
        result = embed("test norm", model_name="nv-embed-qa", dim=768)
        norm = result.norm(dim=1)
        assert norm.item() == pytest.approx(1.0, abs=1e-4)


# ═══════════════════════════════════════════════════════════════
#  GitDB Integration (text-based add/query)
# ═══════════════════════════════════════════════════════════════

@needs_st
@slow
class TestGitDBEmbed:
    def test_add_text(self, tmp_path):
        from gitdb import GitDB
        db = GitDB(str(tmp_path / "store"), dim=1024, device="cpu")
        indices = db.add(texts=["hello world", "foo bar baz"])
        assert len(indices) == 2
        assert db.tree.size == 2

    def test_add_text_autosets_documents(self, tmp_path):
        from gitdb import GitDB
        db = GitDB(str(tmp_path / "store"), dim=1024, device="cpu")
        db.add(texts=["my document text"])
        assert db.tree.metadata[0].document == "my document text"

    def test_query_text(self, tmp_path):
        from gitdb import GitDB
        db = GitDB(str(tmp_path / "store"), dim=1024, device="cpu")
        db.add(texts=[
            "machine learning optimization",
            "neural network training",
            "chocolate cake recipe",
        ])
        db.commit("initial")
        results = db.query_text("AI and deep learning", k=2)
        assert len(results.ids) == 2
        # ML-related should rank higher than cake
        docs = results.documents
        assert "chocolate" not in docs[0]

    def test_add_text_with_commit(self, tmp_path):
        from gitdb import GitDB
        db = GitDB(str(tmp_path / "store"), dim=1024, device="cpu")
        db.add(texts=["document one", "document two"])
        h = db.commit("Add text embeddings")
        assert h is not None
        assert db.tree.size == 2

    def test_re_embed(self, tmp_path):
        from gitdb import GitDB
        db = GitDB(str(tmp_path / "store"), dim=1024, device="cpu")
        # Add with random vectors but real documents
        db.add(
            embeddings=torch.randn(3, 1024),
            documents=["cat on mat", "dog in park", "fish in sea"],
        )
        db.commit("initial random vectors")
        # Re-embed from document text
        msg = db.re_embed(embed_model="arctic")
        assert "re-embedded 3" in msg

    def test_semantic_blame(self, tmp_path):
        from gitdb import GitDB
        db = GitDB(str(tmp_path / "store"), dim=1024, device="cpu")
        db.add(texts=["machine learning model"])
        db.commit("add ML")
        db.add(texts=["chocolate cake recipe"])
        db.commit("add recipe")
        results = db.semantic_blame("artificial intelligence", threshold=0.3)
        assert len(results) >= 1
        assert results[0]["document"] == "machine learning model"
