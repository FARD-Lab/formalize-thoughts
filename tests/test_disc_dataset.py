"""Tests for DiscriminatorDataset with text-based generated outputs.

Covers:
- __getitem__ returns {'text': str, 'thought_vec': Tensor, 'label': float}
- 'text' is a string (not a list of IDs)
- 'thought_vec' is 2D: [expand_dim, vec_dim]
- EMBEDDING_NO_POOLING and RANDOM_VECTOR: beam-indexed T
- All other TR types: same T for the group (input-derived)
- Correct label (0.0 or 1.0) from pairs file
"""

import json
import logging
import tempfile
from pathlib import Path

import pytest
import torch

from src.dataset.disc_dataset import DiscriminatorDataset
from src.utils.config import ThoughtRepresentation

VEC_DIM = 16
NUM_GROUPS = 10
NUM_BEAMS = 8
EXPAND_DIM = 4                        

def make_logger():
    log = logging.getLogger("test_disc_dataset")
    log.setLevel(logging.CRITICAL)
    return log

def write_generated_texts(tmp_dir: Path, split: str, num_groups: int = NUM_GROUPS):
    texts = {
        i: [f"Generated text for group {i} beam {k}" for k in range(NUM_BEAMS)]
        for i in range(num_groups)
    }
    torch.save(texts, tmp_dir / f"{split}_generated_texts.pt")
    return texts

def write_vectors(tmp_dir: Path, split: str, tr_type: ThoughtRepresentation,
                  num_groups: int = NUM_GROUPS):
    if tr_type == ThoughtRepresentation.LAST_INPUT_TOKEN:
        vectors = {i: torch.randn(33, VEC_DIM) for i in range(num_groups)}
    elif tr_type in (ThoughtRepresentation.EMBEDDING_POOLING,
                     ThoughtRepresentation.INPUT_EMBEDDING,
                     ThoughtRepresentation.LAST_INPUT_HIDDEN_STATE):
        vectors = {i: torch.randn(1, VEC_DIM) for i in range(num_groups)}
        if tr_type == ThoughtRepresentation.LAST_INPUT_HIDDEN_STATE:
            tr_type = ThoughtRepresentation.LAST_INPUT_TOKEN               
    elif tr_type == ThoughtRepresentation.EMBEDDING_NO_POOLING:
        vectors = {i: torch.randn(NUM_BEAMS, 1, VEC_DIM) for i in range(num_groups)}
    elif tr_type in (ThoughtRepresentation.SOFT_THINKING,
                     ThoughtRepresentation.SOFT_THINKING_NOISE,
                     ThoughtRepresentation.LATENT_THINKING):
        vectors = {i: torch.randn(128, VEC_DIM) for i in range(num_groups)}
    else:
        vectors = {i: torch.randn(1, VEC_DIM) for i in range(num_groups)}
    torch.save(vectors, tmp_dir / f"{split}_other_vectors_{tr_type.value}.pt")

def write_pairs(tmp_dir: Path, split: str, num_pairs: int = 10):
    pairs_file = tmp_dir / f"{split}_discriminator_pairs.jsonl"
    with open(pairs_file, "w") as f:
        for i in range(num_pairs):
            f.write(json.dumps({
                "group_1_idx": i % NUM_GROUPS,
                "group_2_idx": (i + 1) % NUM_GROUPS,
                "feat_1_idx": i % NUM_BEAMS,
                "feat_2_idx": (i + 1) % NUM_BEAMS,
                "label": i % 2,
            }) + "\n")
    return str(pairs_file)

def make_dataset(tmp_dir: Path, tr_type: ThoughtRepresentation) -> DiscriminatorDataset:
    write_generated_texts(tmp_dir, "train")
    write_vectors(tmp_dir, "train", tr_type)
    pairs_file = write_pairs(tmp_dir, "train")
    return DiscriminatorDataset(
        logger=make_logger(),
        pairs_file=pairs_file,
        tr_type=tr_type,
        cached_data_dir=str(tmp_dir),
        vec_dim=VEC_DIM,
        expand_dim=EXPAND_DIM,
    )

class TestItemKeys:

    def test_text_key_is_string(self):
        """__getitem__['text'] must be a Python string."""
        with tempfile.TemporaryDirectory() as tmp:
            ds = make_dataset(Path(tmp), ThoughtRepresentation.EMBEDDING_POOLING)
            item = ds[0]
            assert "text" in item
            assert isinstance(item["text"], str), (
                f"Expected str, got {type(item['text'])}"
            )

    def test_thought_vec_key_present(self):
        """'thought_vec' must be in the returned dict."""
        with tempfile.TemporaryDirectory() as tmp:
            ds = make_dataset(Path(tmp), ThoughtRepresentation.EMBEDDING_POOLING)
            item = ds[0]
            assert "thought_vec" in item
            assert "other_vector" not in item

    def test_label_key_is_float(self):
        """'label' must be a Python float."""
        with tempfile.TemporaryDirectory() as tmp:
            ds = make_dataset(Path(tmp), ThoughtRepresentation.EMBEDDING_POOLING)
            item = ds[0]
            assert "label" in item
            assert isinstance(item["label"], float)

    def test_text_is_nonempty(self):
        """'text' must be a non-empty string."""
        with tempfile.TemporaryDirectory() as tmp:
            ds = make_dataset(Path(tmp), ThoughtRepresentation.EMBEDDING_POOLING)
            item = ds[0]
            assert len(item["text"]) > 0

class TestThoughtVecShape:

    @pytest.mark.parametrize("tr_type", [
        ThoughtRepresentation.EMBEDDING_POOLING,
        ThoughtRepresentation.INPUT_EMBEDDING,
    ])
    def test_single_vec_expanded_to_expand_dim(self, tr_type):
        """Single-vector TR types ([1, d]) must be expanded to [expand_dim, d]."""
        with tempfile.TemporaryDirectory() as tmp:
            ds = make_dataset(Path(tmp), tr_type)
            item = ds[0]
            tv = item["thought_vec"]
            assert tv.dim() == 2, f"{tr_type.value}: expected 2D, got {tv.dim()}D"
            assert tv.shape == (EXPAND_DIM, VEC_DIM), (
                f"{tr_type.value}: expected ({EXPAND_DIM}, {VEC_DIM}), got {tv.shape}"
            )

    def test_last_input_token_shape(self):
        """LAST_INPUT_TOKEN [33, d] repeated to [expand_dim, d] (since 33 > EXPAND_DIM)."""
        with tempfile.TemporaryDirectory() as tmp:
                                                                          
            ds = make_dataset(Path(tmp), ThoughtRepresentation.LAST_INPUT_TOKEN)
            item = ds[0]
            tv = item["thought_vec"]
            assert tv.dim() == 2
            assert tv.shape[-1] == VEC_DIM

    def test_thought_vec_always_2d(self):
        """thought_vec must always be 2D regardless of TR type."""
        tr_types = [
            ThoughtRepresentation.EMBEDDING_POOLING,
            ThoughtRepresentation.LAST_INPUT_TOKEN,
        ]
        for tr_type in tr_types:
            with tempfile.TemporaryDirectory() as tmp:
                ds = make_dataset(Path(tmp), tr_type)
                for i in range(min(5, len(ds))):
                    item = ds[i]
                    assert item["thought_vec"].dim() == 2, (
                        f"{tr_type.value} item {i}: expected 2D thought_vec, "
                        f"got {item['thought_vec'].dim()}D"
                    )

class TestLabels:

    def test_label_is_zero_or_one(self):
        """Labels must be 0.0 or 1.0."""
        with tempfile.TemporaryDirectory() as tmp:
            ds = make_dataset(Path(tmp), ThoughtRepresentation.EMBEDDING_POOLING)
            for i in range(len(ds)):
                label = ds[i]["label"]
                assert label in (0.0, 1.0), f"Item {i}: unexpected label {label}"

    def test_alternating_labels(self):
        """Pairs file has alternating 0/1 labels — dataset must reflect this."""
        with tempfile.TemporaryDirectory() as tmp:
            ds = make_dataset(Path(tmp), ThoughtRepresentation.EMBEDDING_POOLING)
            labels = [ds[i]["label"] for i in range(len(ds))]
            assert 0.0 in labels and 1.0 in labels, (
                "Expected both positive and negative examples"
            )

class TestDatasetLength:

    def test_len_matches_pairs_file(self):
        """__len__ must equal the number of pairs in the JSONL file."""
        with tempfile.TemporaryDirectory() as tmp:
            d = Path(tmp)
            write_generated_texts(d, "train")
            write_vectors(d, "train", ThoughtRepresentation.EMBEDDING_POOLING)
            pairs_file = write_pairs(d, "train", num_pairs=7)
            ds = DiscriminatorDataset(
                logger=make_logger(),
                pairs_file=pairs_file,
                tr_type=ThoughtRepresentation.EMBEDDING_POOLING,
                cached_data_dir=str(d),
                vec_dim=VEC_DIM,
                expand_dim=EXPAND_DIM,
            )
            assert len(ds) == 7
