"""Tests for the BERT-based SentenceEncoder."""

from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from thesischeck.core.semantic.encoder import SentenceEncoder

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def mock_sentence_transformer():
    """Patch ``sentence_transformers.SentenceTransformer`` at source so the
    encoder's lazy import picks up the mock instead of the real package."""
    with patch("sentence_transformers.SentenceTransformer") as mock_cls:
        mock_instance = MagicMock()
        mock_instance.get_sentence_embedding_dimension.return_value = 384
        mock_instance.encode.return_value = np.random.rand(2, 384).astype(np.float32)
        mock_cls.return_value = mock_instance
        yield mock_cls, mock_instance


# ---------------------------------------------------------------------------
# Shape / basic encoding
# ---------------------------------------------------------------------------


class TestEncodingShape:
    """Verify the output shape of ``encode`` under various inputs."""

    def test_encode_returns_correct_shape(self, mock_sentence_transformer):
        """Two texts → shape (2, 384)."""
        mock_cls, mock_instance = mock_sentence_transformer
        encoder = SentenceEncoder()

        texts = ["Hello world.", "Test sentence here."]
        result = encoder.encode(texts)

        assert isinstance(result, np.ndarray)
        assert result.shape == (2, 384)
        mock_instance.encode.assert_called_once()

    def test_encode_empty_list(self, mock_sentence_transformer):
        """Empty list → shape (0, 384)."""
        mock_cls, mock_instance = mock_sentence_transformer
        encoder = SentenceEncoder()

        result = encoder.encode([])

        assert result.shape == (0, 384)
        # Model is still loaded (needed to get dim)
        mock_cls.assert_called_once()

    def test_encode_single_text(self, mock_sentence_transformer):
        """Single text → shape (1, 384)."""
        mock_cls, mock_instance = mock_sentence_transformer
        mock_instance.encode.return_value = np.random.rand(1, 384).astype(np.float32)

        encoder = SentenceEncoder()
        result = encoder.encode(["Just one sentence."])

        assert result.shape == (1, 384)


# ---------------------------------------------------------------------------
# Sentence splitting
# ---------------------------------------------------------------------------


class TestSentenceSplitting:
    """Verify the sentence-splitting heuristic."""

    @patch("sentence_transformers.SentenceTransformer")
    def test_encode_sentences_splits_correctly(self, mock_st):
        """Punctuation-based splitting produces the expected number of sentences."""
        mock_instance = MagicMock()
        mock_instance.get_sentence_embedding_dimension.return_value = 384
        mock_instance.encode.return_value = np.random.rand(3, 384).astype(np.float32)
        mock_st.return_value = mock_instance

        encoder = SentenceEncoder()
        text = "First sentence. Second sentence! Third sentence?"
        result = encoder.encode_sentences(text)

        assert len(result) == 3

    @patch("sentence_transformers.SentenceTransformer")
    def test_encode_sentences_returns_dict(self, mock_st):
        """Result is a dict mapping sentence → vector with correct shapes."""
        mock_instance = MagicMock()
        mock_instance.get_sentence_embedding_dimension.return_value = 384
        mock_instance.encode.return_value = np.random.rand(2, 384).astype(np.float32)
        mock_st.return_value = mock_instance

        encoder = SentenceEncoder()
        text = "First sentence. Second sentence."
        result = encoder.encode_sentences(text)

        assert isinstance(result, dict)
        assert len(result) == 2
        for sent, vec in result.items():
            assert isinstance(sent, str)
            assert isinstance(vec, np.ndarray)
            assert vec.shape == (384,)

    @patch("sentence_transformers.SentenceTransformer")
    def test_split_sentences_handles_chinese(self, mock_st):
        """Chinese punctuation is also treated as a sentence boundary."""
        mock_instance = MagicMock()
        mock_instance.get_sentence_embedding_dimension.return_value = 384
        mock_st.return_value = mock_instance

        encoder = SentenceEncoder()
        text = "这是第一句。这是第二句！这是第三句？"
        result = encoder.encode_sentences(text)

        assert len(result) == 3

    @patch("sentence_transformers.SentenceTransformer")
    def test_split_sentences_filters_empty(self, mock_st):
        """Empty / whitespace-only segments are dropped."""
        mock_instance = MagicMock()
        mock_instance.get_sentence_embedding_dimension.return_value = 384
        mock_st.return_value = mock_instance

        encoder = SentenceEncoder()
        text = "Hello.   . ! ? \nWorld."
        result = encoder.encode_sentences(text)

        # "Hello" and "World" — punctuation-only parts are dropped
        assert len(result) == 2

    @patch("sentence_transformers.SentenceTransformer")
    def test_split_sentences_empty_text(self, mock_st):
        """Empty input returns an empty dict."""
        mock_instance = MagicMock()
        mock_instance.get_sentence_embedding_dimension.return_value = 384
        mock_st.return_value = mock_instance

        encoder = SentenceEncoder()
        result = encoder.encode_sentences("")

        assert result == {}


# ---------------------------------------------------------------------------
# Caching
# ---------------------------------------------------------------------------


class TestCaching:
    """Verify that the optional VectorCache is consulted correctly."""

    @patch("sentence_transformers.SentenceTransformer")
    def test_encode_cache_hit(self, mock_st):
        """Cache hit → model not called, cached vector returned."""
        mock_instance = MagicMock()
        mock_instance.get_sentence_embedding_dimension.return_value = 384
        mock_st.return_value = mock_instance

        mock_cache = MagicMock()
        cached_vec = np.random.rand(384).astype(np.float32)
        mock_cache.get.return_value = cached_vec  # cache always hits

        encoder = SentenceEncoder(cache=mock_cache)
        texts = ["Cached sentence one.", "Cached sentence two."]
        result = encoder.encode(texts)

        # Cache checked for each text
        assert mock_cache.get.call_count == 2
        # Model encoding should NOT have been invoked
        mock_instance.encode.assert_not_called()
        assert result.shape == (2, 384)

    @patch("sentence_transformers.SentenceTransformer")
    def test_encode_cache_partial_miss(self, mock_st):
        """Partial cache miss: misses are encoded, hits are reused."""
        mock_instance = MagicMock()
        mock_instance.get_sentence_embedding_dimension.return_value = 384
        mock_instance.encode.return_value = np.random.rand(1, 384).astype(np.float32)
        mock_st.return_value = mock_instance

        mock_cache = MagicMock()
        # Build the expected cache keys using the same logic as the encoder
        import hashlib
        key_one = f"emb:all-MiniLM-L6-v2:{hashlib.md5('One sentence.'.encode('utf-8')).hexdigest()}"
        key_two = f"emb:all-MiniLM-L6-v2:{hashlib.md5('Another sentence.'.encode('utf-8')).hexdigest()}"  # noqa: F841
        cached_vec = np.random.rand(384).astype(np.float32)

        def cache_get(key):
            if key == key_one:
                return cached_vec
            return None

        mock_cache.get.side_effect = cache_get

        encoder = SentenceEncoder(cache=mock_cache)
        texts = ["One sentence.", "Another sentence."]
        result = encoder.encode(texts)

        assert result.shape == (2, 384)
        # Model must have been called once for the second text
        mock_instance.encode.assert_called_once()

    @patch("sentence_transformers.SentenceTransformer")
    def test_encode_no_cache(self, mock_st):
        """Without cache, model is always called."""
        mock_instance = MagicMock()
        mock_instance.get_sentence_embedding_dimension.return_value = 384
        mock_instance.encode.return_value = np.random.rand(2, 384).astype(np.float32)
        mock_st.return_value = mock_instance

        encoder = SentenceEncoder(cache=None)  # explicit no cache
        texts = ["First.", "Second."]
        encoder.encode(texts)

        mock_instance.encode.assert_called_once()


# ---------------------------------------------------------------------------
# Properties
# ---------------------------------------------------------------------------


class TestProperties:
    """Verify property accessors."""

    @patch("sentence_transformers.SentenceTransformer")
    def test_model_name_property(self, mock_st):
        """``model_name`` returns the value passed at construction."""
        mock_instance = MagicMock()
        mock_st.return_value = mock_instance

        encoder = SentenceEncoder(model_name="all-MiniLM-L6-v2")
        assert encoder.model_name == "all-MiniLM-L6-v2"

        encoder2 = SentenceEncoder(model_name="paraphrase-MiniLM-L3-v2")
        assert encoder2.model_name == "paraphrase-MiniLM-L3-v2"

    @patch("sentence_transformers.SentenceTransformer")
    def test_embedding_dim_property(self, mock_st):
        """``embedding_dim`` returns the model's reported dimension."""
        mock_instance = MagicMock()
        mock_instance.get_sentence_embedding_dimension.return_value = 768
        mock_st.return_value = mock_instance

        encoder = SentenceEncoder()
        dim = encoder.embedding_dim
        assert dim == 768

    @patch("sentence_transformers.SentenceTransformer")
    def test_lazy_loading(self, mock_st):
        """Model is NOT loaded during ``__init__`` — only on first ``encode`` call."""
        mock_st.return_value = MagicMock()

        encoder = SentenceEncoder()
        # Model should not be loaded yet
        assert encoder._model is None

        # Accessing embedding_dim triggers loading
        _ = encoder.embedding_dim
        mock_st.assert_called_once()
