"""Tests for ThoughtRepresentationLoader — text storage and get_group_data interface.

Covers:
- generated_texts_cache is populated (not generated_ids_cache)
- get_group_data() returns "thought_vec" and "generated_texts" keys
- get_group_data() must NOT return "generated_ids" or "other_vector"
- RANDOM_VECTOR: get_group_data() generates on-the-fly, still returns generated_texts
- LAST_INPUT_HIDDEN_STATE: loads from last_input_token file, slices to [-1:]
- SOFT_THINKING / SOFT_THINKING_NOISE / LATENT_THINKING: sliced to think_steps
- FileNotFoundError raised if cache files are missing
"""

import logging
import tempfile
from pathlib import Path

import pytest
import torch

from src.dataset.tr_loader import ThoughtRepresentationLoader
from src.utils.config import ThoughtRepresentation

VEC_DIM = 16
NUM_GROUPS = 5
NUM_BEAMS = 8

def make_logger():
    log = logging.getLogger("test_tr_loader")
    log.setLevel(logging.CRITICAL)
    return log

def write_generated_texts(tmp_dir: Path, split: str, num_groups: int = NUM_GROUPS):
    texts = {i: [f"group {i} beam {k} text" for k in range(NUM_BEAMS)] for i in range(num_groups)}
    torch.save(texts, tmp_dir / f"{split}_generated_texts.pt")
    return texts

def write_vectors(tmp_dir: Path, split: str, tr_type: ThoughtRepresentation,
                  num_groups: int = NUM_GROUPS, seq_len: int = 1):
    vectors = {i: torch.randn(seq_len, VEC_DIM) for i in range(num_groups)}
    key = (
        ThoughtRepresentation.LAST_INPUT_TOKEN.value
        if tr_type == ThoughtRepresentation.LAST_INPUT_HIDDEN_STATE
        else tr_type.value
    )
    torch.save(vectors, tmp_dir / f"{split}_other_vectors_{key}.pt")
    return vectors

def make_loader(tmp_dir: Path, tr_type: ThoughtRepresentation,
                split: str = "test", think_steps: int = 64):
    return ThoughtRepresentationLoader(
        logger=make_logger(),
        tr_type=tr_type,
        cached_data_dir=str(tmp_dir),
        split_name=split,
        vec_dim=VEC_DIM,
        think_steps=think_steps,
    )

class TestGeneratedTextsCache:

    def test_has_generated_texts_cache_attr(self):
        """Loader must expose generated_texts_cache (not generated_ids_cache)."""
        with tempfile.TemporaryDirectory() as tmp:
            d = Path(tmp)
            write_generated_texts(d, "test")
            write_vectors(d, "test", ThoughtRepresentation.LAST_INPUT_TOKEN, seq_len=33)
            loader = make_loader(d, ThoughtRepresentation.LAST_INPUT_TOKEN)
            assert hasattr(loader, "generated_texts_cache")
            assert not hasattr(loader, "generated_ids_cache")

    def test_generated_texts_cache_contains_strings(self):
        """generated_texts_cache values must be lists of strings."""
        with tempfile.TemporaryDirectory() as tmp:
            d = Path(tmp)
            write_generated_texts(d, "test")
            write_vectors(d, "test", ThoughtRepresentation.LAST_INPUT_TOKEN, seq_len=33)
            loader = make_loader(d, ThoughtRepresentation.LAST_INPUT_TOKEN)
            for g_idx, texts in loader.generated_texts_cache.items():
                assert isinstance(texts, list)
                assert all(isinstance(t, str) for t in texts), (
                    f"Group {g_idx}: expected list of str, got {[type(t) for t in texts]}"
                )

    def test_generated_texts_cache_correct_count(self):
        """generated_texts_cache must have the same number of groups as the cache file."""
        with tempfile.TemporaryDirectory() as tmp:
            d = Path(tmp)
            write_generated_texts(d, "test", num_groups=7)
            write_vectors(d, "test", ThoughtRepresentation.LAST_INPUT_TOKEN, seq_len=33, num_groups=7)
            loader = make_loader(d, ThoughtRepresentation.LAST_INPUT_TOKEN)
            assert len(loader.generated_texts_cache) == 7

class TestGetGroupDataKeys:

    def test_returns_thought_vec_key(self):
        """get_group_data() must return 'thought_vec'."""
        with tempfile.TemporaryDirectory() as tmp:
            d = Path(tmp)
            write_generated_texts(d, "test")
            write_vectors(d, "test", ThoughtRepresentation.LAST_INPUT_TOKEN, seq_len=33)
            loader = make_loader(d, ThoughtRepresentation.LAST_INPUT_TOKEN)
            data = loader.get_group_data(0)
            assert "thought_vec" in data

    def test_returns_generated_texts_key(self):
        """get_group_data() must return 'generated_texts' (list of strings)."""
        with tempfile.TemporaryDirectory() as tmp:
            d = Path(tmp)
            write_generated_texts(d, "test")
            write_vectors(d, "test", ThoughtRepresentation.LAST_INPUT_TOKEN, seq_len=33)
            loader = make_loader(d, ThoughtRepresentation.LAST_INPUT_TOKEN)
            data = loader.get_group_data(0)
            assert "generated_texts" in data
            assert isinstance(data["generated_texts"], list)
            assert all(isinstance(t, str) for t in data["generated_texts"])

    def test_does_not_return_generated_ids(self):
        """get_group_data() must NOT return 'generated_ids'."""
        with tempfile.TemporaryDirectory() as tmp:
            d = Path(tmp)
            write_generated_texts(d, "test")
            write_vectors(d, "test", ThoughtRepresentation.LAST_INPUT_TOKEN, seq_len=33)
            loader = make_loader(d, ThoughtRepresentation.LAST_INPUT_TOKEN)
            data = loader.get_group_data(0)
            assert "generated_ids" not in data

    def test_does_not_return_other_vector(self):
        """get_group_data() must NOT return 'other_vector' (old name)."""
        with tempfile.TemporaryDirectory() as tmp:
            d = Path(tmp)
            write_generated_texts(d, "test")
            write_vectors(d, "test", ThoughtRepresentation.EMBEDDING_POOLING)
            loader = make_loader(d, ThoughtRepresentation.EMBEDDING_POOLING)
            data = loader.get_group_data(0)
            assert "other_vector" not in data

class TestRandomVector:

    def test_random_vector_returns_generated_texts(self):
        """RANDOM_VECTOR loader must still expose generated_texts."""
        with tempfile.TemporaryDirectory() as tmp:
            d = Path(tmp)
            write_generated_texts(d, "test")
            loader = make_loader(d, ThoughtRepresentation.RANDOM_VECTOR)
            data = loader.get_group_data(0)
            assert "generated_texts" in data
            assert "generated_ids" not in data

    def test_random_vector_thought_vec_shape(self):
        """RANDOM_VECTOR: thought_vec shape = [K, 1, vec_dim] where K = len(generated_texts)."""
        with tempfile.TemporaryDirectory() as tmp:
            d = Path(tmp)
            write_generated_texts(d, "test")
            loader = make_loader(d, ThoughtRepresentation.RANDOM_VECTOR)
            data = loader.get_group_data(0)
            tv = data["thought_vec"]
            assert tv.dim() == 3
            assert tv.shape[1] == 1
            assert tv.shape[2] == VEC_DIM

class TestLastInputHiddenState:

    def test_uses_lit_file_not_lihs_file(self):
        """LIHS must load from last_input_token cache file (no separate file)."""
        with tempfile.TemporaryDirectory() as tmp:
            d = Path(tmp)
            write_generated_texts(d, "test")
            vectors = {i: torch.randn(33, VEC_DIM) for i in range(NUM_GROUPS)}
            torch.save(vectors, d / f"test_other_vectors_{ThoughtRepresentation.LAST_INPUT_TOKEN.value}.pt")
                                                    
            loader = make_loader(d, ThoughtRepresentation.LAST_INPUT_HIDDEN_STATE)
            data = loader.get_group_data(0)
            assert data["thought_vec"].shape == (1, VEC_DIM), (
                f"LIHS must slice to [1, VEC_DIM], got {data['thought_vec'].shape}"
            )

    def test_slices_to_last_layer(self):
        """LIHS slice = [-1:] → shape (1, vec_dim) regardless of full cache size."""
        with tempfile.TemporaryDirectory() as tmp:
            d = Path(tmp)
            write_generated_texts(d, "test")
            vectors = {i: torch.arange(33 * VEC_DIM, dtype=torch.float).reshape(33, VEC_DIM)
                       for i in range(NUM_GROUPS)}
            torch.save(vectors, d / f"test_other_vectors_{ThoughtRepresentation.LAST_INPUT_TOKEN.value}.pt")
            loader = make_loader(d, ThoughtRepresentation.LAST_INPUT_HIDDEN_STATE)
            data = loader.get_group_data(0)
            expected_last_row = vectors[0][-1:]
            assert torch.allclose(data["thought_vec"], expected_last_row)

@pytest.mark.parametrize("tr_type", [
    ThoughtRepresentation.SOFT_THINKING,
    ThoughtRepresentation.SOFT_THINKING_NOISE,
    ThoughtRepresentation.LATENT_THINKING,
])
class TestThinkingSlice:

    def test_sliced_to_think_steps(self, tr_type):
        """Thinking TR types must be sliced to think_steps from the full 128-step cache."""
        think_steps = 16
        with tempfile.TemporaryDirectory() as tmp:
            d = Path(tmp)
            write_generated_texts(d, "test")
            vectors = {i: torch.randn(128, VEC_DIM) for i in range(NUM_GROUPS)}
            torch.save(vectors, d / f"test_other_vectors_{tr_type.value}.pt")
            loader = make_loader(d, tr_type, think_steps=think_steps)
            data = loader.get_group_data(0)
            assert data["thought_vec"].shape == (think_steps, VEC_DIM), (
                f"{tr_type.value}: expected ({think_steps}, {VEC_DIM}), "
                f"got {data['thought_vec'].shape}"
            )

    def test_full_think_steps_returned_when_exact(self, tr_type):
        """When think_steps matches cache size, full vector is returned."""
        think_steps = 128
        with tempfile.TemporaryDirectory() as tmp:
            d = Path(tmp)
            write_generated_texts(d, "test")
            vectors = {i: torch.randn(128, VEC_DIM) for i in range(NUM_GROUPS)}
            torch.save(vectors, d / f"test_other_vectors_{tr_type.value}.pt")
            loader = make_loader(d, tr_type, think_steps=think_steps)
            data = loader.get_group_data(0)
            assert data["thought_vec"].shape == (128, VEC_DIM)

class TestLoaderErrors:

    def test_missing_generated_texts_file_raises(self):
        """FileNotFoundError if the generated_texts .pt file is absent."""
        with tempfile.TemporaryDirectory() as tmp:
            with pytest.raises(FileNotFoundError):
                make_loader(Path(tmp), ThoughtRepresentation.LAST_INPUT_TOKEN)

    def test_missing_vectors_file_raises(self):
        """FileNotFoundError if the thought_vecs .pt file is absent for a non-RV type."""
        with tempfile.TemporaryDirectory() as tmp:
            d = Path(tmp)
            write_generated_texts(d, "test")
                                      
            with pytest.raises(FileNotFoundError):
                make_loader(d, ThoughtRepresentation.LAST_INPUT_TOKEN)

    def test_unknown_group_raises_key_error(self):
        """get_group_data() raises KeyError for an out-of-range group index."""
        with tempfile.TemporaryDirectory() as tmp:
            d = Path(tmp)
            write_generated_texts(d, "test")
            write_vectors(d, "test", ThoughtRepresentation.LAST_INPUT_TOKEN, seq_len=33)
            loader = make_loader(d, ThoughtRepresentation.LAST_INPUT_TOKEN)
            with pytest.raises(KeyError):
                loader.get_group_data(9999)
