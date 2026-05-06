"""Unit tests for the minimality / output-reconstruction probe components.

Tests cover:
- MinimalityDataset shape correctness for all TR types (no GPU required)
- target_source="input" and target_source="output" branching
- Beam expansion: __len__ = num_examples * num_beams when target_source="output"
- Beam indexing: idx k maps to beam k % num_beams of example k // num_beams
- RANDOM_VECTOR shape bug fix ([8,1,4096] → [0] → [1,4096])
- EMBEDDING_NO_POOLING: [8,4096] as-is for target_source="input";
                         [1,4096] beam-indexed for target_source="output"
- EMBEDDING_POOLING treated as input-derived (single T, all 8 beams)
- MinimalityDataCollator batch assembly
- ThoughtDescriptor post_projection_norm device placement
"""

import logging
import tempfile
from pathlib import Path
from typing import List
from unittest.mock import MagicMock, patch

import pytest
import torch

from src.utils.config import ThoughtRepresentation

NUM_BEAMS = 8
VEC_DIM = 4096

def make_logger():
    log = logging.getLogger("test_minimality")
    log.setLevel(logging.CRITICAL)
    return log

def make_tokenizer(vocab_size: int = 32000):
    """Minimal mock tokenizer."""
    tok = MagicMock()
    tok.pad_token_id = 0
    tok.eos_token_id = 2
    tok.pad_token = "<pad>"
    tok.eos_token = "</s>"

    def tokenize(text, max_length=512, truncation=True, padding=False, return_tensors="pt"):
        length = min(len(text.split()), max_length)
        ids = torch.randint(3, vocab_size, (1, length))
        mask = torch.ones(1, length, dtype=torch.long)
        return {"input_ids": ids, "attention_mask": mask}

    tok.side_effect = tokenize
    tok.__call__ = tokenize
    return tok

def make_tr_loader(tr_type: ThoughtRepresentation, vec_dim: int = VEC_DIM, num_beams: int = NUM_BEAMS):
    """Mock ThoughtRepresentationLoader returning the correct shape for each TR type."""
    loader = MagicMock()
    loader.tr_type = tr_type

    generated_texts = [f"beam {k} output text" for k in range(num_beams)]

    def get_group_data(g_idx):
        if tr_type == ThoughtRepresentation.RANDOM_VECTOR:
            thought_vec = torch.randn(num_beams, 1, vec_dim)
        elif tr_type == ThoughtRepresentation.EMBEDDING_NO_POOLING:
            thought_vec = torch.randn(num_beams, 1, vec_dim)                          
        elif tr_type == ThoughtRepresentation.LAST_INPUT_TOKEN:
            thought_vec = torch.randn(33, vec_dim)
        elif tr_type in (
            ThoughtRepresentation.SOFT_THINKING,
            ThoughtRepresentation.SOFT_THINKING_NOISE,
            ThoughtRepresentation.LATENT_THINKING,
        ):
            thought_vec = torch.randn(16, vec_dim)                  
        else:
                                                                         
            thought_vec = torch.randn(1, vec_dim)

        return {"thought_vec": thought_vec, "generated_texts": generated_texts}

    loader.get_group_data.side_effect = get_group_data
    loader.generated_texts_cache = {0: generated_texts, 1: generated_texts}
    return loader

def make_example_loader(input_text: str = "What is 2+2?"):
    loader = MagicMock()
    loader.load_example.return_value = {"input_text": input_text}
    return loader

def _make_ds(tr_type, target_source="input", indices=None):
    """Build a MinimalityDataset via __new__ without disk I/O."""
    from src.dataset.minimality_dataset import MinimalityDataset

    ds = MinimalityDataset.__new__(MinimalityDataset)
    ds.logger = make_logger()
    ds.tokenizer = make_tokenizer()
    ds.split_name = "test"
    ds.max_input_length = 512
    ds.target_source = target_source
    ds.num_return_sequences = NUM_BEAMS
    ds.tr_loader = make_tr_loader(tr_type)
    ds.example_loader = make_example_loader()
    ds.indices = indices if indices is not None else [0, 1]
    return ds

@pytest.mark.parametrize("tr_type,expected_seq_len", [
    (ThoughtRepresentation.LAST_INPUT_TOKEN, 33),
    (ThoughtRepresentation.LAST_INPUT_HIDDEN_STATE, 1),
    (ThoughtRepresentation.EMBEDDING_POOLING, 1),
    (ThoughtRepresentation.EMBEDDING_NO_POOLING, 8),                            
    (ThoughtRepresentation.SOFT_THINKING, 16),
    (ThoughtRepresentation.INPUT_EMBEDDING, 1),
    (ThoughtRepresentation.RANDOM_VECTOR, 1),                                     
])
def test_dataset_getitem_input_shape(tr_type, expected_seq_len):
    """input_vecs must be 2D (seq_len, vec_dim) for every TR type."""
    ds = _make_ds(tr_type, target_source="input")
    item = ds[0]

    assert item["input_vecs"].dim() == 2, (
        f"{tr_type.value}: expected 2D input_vecs, got shape {item['input_vecs'].shape}"
    )
    assert item["input_vecs"].shape == (expected_seq_len, VEC_DIM), (
        f"{tr_type.value}: expected ({expected_seq_len}, {VEC_DIM}), "
        f"got {item['input_vecs'].shape}"
    )
    assert item["target_token_ids"].dim() == 1
    assert item["target_attention_mask"].dim() == 1

def test_dataset_output_len_is_expanded():
    """With target_source='output', __len__ == num_examples * num_beams."""
    ds = _make_ds(ThoughtRepresentation.LAST_INPUT_TOKEN, target_source="output", indices=[0, 1])
    assert len(ds) == 2 * NUM_BEAMS

def test_dataset_input_len_is_not_expanded():
    """With target_source='input', __len__ == num_examples."""
    ds = _make_ds(ThoughtRepresentation.LAST_INPUT_TOKEN, target_source="input", indices=[0, 1])
    assert len(ds) == 2

def test_dataset_output_beam_indexing():
    """idx k yields beam k % 8 from example k // 8 (text-based storage)."""
    tr_type = ThoughtRepresentation.LAST_INPUT_TOKEN
                                                                                     
    generated_texts = [f"beam {i} " + "word " * i for i in range(NUM_BEAMS)]
    loader = make_tr_loader(tr_type)
    loader.get_group_data.side_effect = lambda g: {
        "thought_vec": torch.randn(33, VEC_DIM),
        "generated_texts": generated_texts,
    }

    from src.dataset.minimality_dataset import MinimalityDataset
    ds = MinimalityDataset.__new__(MinimalityDataset)
    ds.logger = make_logger()
    ds.tokenizer = make_tokenizer()
    ds.split_name = "test"
    ds.max_input_length = 512
    ds.target_source = "output"
    ds.num_return_sequences = NUM_BEAMS
    ds.tr_loader = loader
    ds.example_loader = make_example_loader()
    ds.indices = [0]

    for beam_idx in range(1, NUM_BEAMS):
        item_0 = ds[0]
        item_k = ds[beam_idx]
                                                                       
        assert len(item_0["target_token_ids"]) != len(item_k["target_token_ids"]) or beam_idx == 1, (
            f"beam {beam_idx}: expected different token lengths to confirm distinct text used"
        )

def test_dataset_target_source_output_uses_first_beam_at_idx0():
    """target_source='output' at idx=0 uses beam 0 text; ExampleLoader NOT called."""
    ds = _make_ds(ThoughtRepresentation.LAST_INPUT_TOKEN, target_source="output", indices=[0])
    item = ds[0]

    assert item["target_token_ids"].dim() == 1
    assert item["target_attention_mask"].dim() == 1
    assert item["target_token_ids"].shape == item["target_attention_mask"].shape
    ds.example_loader.load_example.assert_not_called()

def test_dataset_target_source_output_truncates():
    """target_source='output' must truncate to max_input_length."""
    from src.dataset.minimality_dataset import MinimalityDataset

    long_text = " ".join(["word"] * 1000)
    loader = make_tr_loader(ThoughtRepresentation.LAST_INPUT_TOKEN)
    loader.get_group_data.side_effect = lambda g: {
        "thought_vec": torch.randn(33, VEC_DIM),
        "generated_texts": [long_text] * NUM_BEAMS,
    }

    ds = MinimalityDataset.__new__(MinimalityDataset)
    ds.logger = make_logger()
    ds.tokenizer = make_tokenizer()
    ds.split_name = "test"
    ds.max_input_length = 50
    ds.target_source = "output"
    ds.num_return_sequences = NUM_BEAMS
    ds.tr_loader = loader
    ds.example_loader = MagicMock()
    ds.indices = [0]

    item = ds[0]
    assert len(item["target_token_ids"]) == 50

def test_embedding_no_pooling_input_shape():
    """EMBEDDING_NO_POOLING + target_source='input': all 8 beams as [8, 4096] sequence."""
    ds = _make_ds(ThoughtRepresentation.EMBEDDING_NO_POOLING, target_source="input")
    item = ds[0]
    assert item["input_vecs"].shape == (8, VEC_DIM), (
        f"Expected (8, {VEC_DIM}), got {item['input_vecs'].shape}"
    )

def test_embedding_no_pooling_output_shape():
    """EMBEDDING_NO_POOLING + target_source='output': beam-indexed T_k → [1, 4096]."""
    ds = _make_ds(ThoughtRepresentation.EMBEDDING_NO_POOLING, target_source="output", indices=[0])
                                                              
    for beam_idx in range(NUM_BEAMS):
        item = ds[beam_idx]
        assert item["input_vecs"].shape == (1, VEC_DIM), (
            f"beam {beam_idx}: expected (1, {VEC_DIM}), got {item['input_vecs'].shape}"
        )

def test_embedding_no_pooling_output_beam_vector_differs():
    """Each beam_idx yields a different T_k (from thought_vec[beam_idx])."""
    from src.dataset.minimality_dataset import MinimalityDataset

    fixed_vectors = torch.stack([torch.full((1, VEC_DIM), float(k)) for k in range(NUM_BEAMS)])
    loader = make_tr_loader(ThoughtRepresentation.EMBEDDING_NO_POOLING)
    loader.get_group_data.side_effect = lambda g: {
        "thought_vec": fixed_vectors,
        "generated_texts": ["some text"] * NUM_BEAMS,
    }

    ds = MinimalityDataset.__new__(MinimalityDataset)
    ds.logger = make_logger()
    ds.tokenizer = make_tokenizer()
    ds.split_name = "test"
    ds.max_input_length = 512
    ds.target_source = "output"
    ds.num_return_sequences = NUM_BEAMS
    ds.tr_loader = loader
    ds.example_loader = make_example_loader()
    ds.indices = [0]

    for beam_idx in range(NUM_BEAMS):
        item = ds[beam_idx]
        expected_val = float(beam_idx)
        assert item["input_vecs"].squeeze(0).mean().item() == expected_val, (
            f"beam {beam_idx}: expected all {expected_val}, "
            f"got {item['input_vecs'].squeeze(0).mean().item()}"
        )

def test_embedding_pooling_output_same_vector_all_beams():
    """EMBEDDING_POOLING + target_source='output': same [1,4096] T for all 8 beams."""
    from src.dataset.minimality_dataset import MinimalityDataset

    fixed_vec = torch.randn(1, VEC_DIM)
    loader = make_tr_loader(ThoughtRepresentation.EMBEDDING_POOLING)
                                                                                 
    loader.get_group_data.side_effect = None
    loader.get_group_data.return_value = {
        "thought_vec": fixed_vec,
        "generated_texts": ["some output text"] * NUM_BEAMS,
    }

    ds = MinimalityDataset.__new__(MinimalityDataset)
    ds.logger = make_logger()
    ds.tokenizer = make_tokenizer()
    ds.split_name = "test"
    ds.max_input_length = 512
    ds.target_source = "output"
    ds.num_return_sequences = NUM_BEAMS
    ds.tr_loader = loader
    ds.example_loader = make_example_loader()
    ds.indices = [0]

    assert len(ds) == NUM_BEAMS
    vecs = [ds[k]["input_vecs"] for k in range(NUM_BEAMS)]
    for k in range(1, NUM_BEAMS):
        assert torch.equal(vecs[0], vecs[k]), f"beam {k} vector differs from beam 0"

def test_random_vector_shape_fix():
    """RANDOM_VECTOR: [8,1,4096][0] → [1,4096]."""
    ds = _make_ds(ThoughtRepresentation.RANDOM_VECTOR, target_source="input")
    item = ds[0]
    assert item["input_vecs"].shape == (1, VEC_DIM), (
        f"RANDOM_VECTOR: expected (1, {VEC_DIM}), got {item['input_vecs'].shape}"
    )

def test_data_collator_pads_correctly():
    """Collator must right-pad targets to the max length in the batch."""
    from scripts.minimality_trainer import MinimalityDataCollator

    tokenizer = make_tokenizer()
    collator = MinimalityDataCollator(tokenizer=tokenizer)

    features = [
        {
            "input_vecs": torch.randn(33, VEC_DIM),
            "target_token_ids": torch.tensor([1, 2, 3]),
            "target_attention_mask": torch.ones(3, dtype=torch.long),
        },
        {
            "input_vecs": torch.randn(33, VEC_DIM),
            "target_token_ids": torch.tensor([4, 5, 6, 7, 8]),
            "target_attention_mask": torch.ones(5, dtype=torch.long),
        },
    ]

    batch = collator(features)

    assert batch["vecs"].shape == (2, 33, VEC_DIM)
    assert batch["target_token_ids"].shape == (2, 5)
    assert batch["target_attention_mask"].shape == (2, 5)

    assert batch["target_token_ids"][0, 3].item() == 0
    assert batch["target_token_ids"][0, 4].item() == 0
    assert batch["target_attention_mask"][0, 3].item() == 0

def test_data_collator_variable_vec_lengths():
    """Collator stacks vecs — all items in a batch must have the same seq_len."""
    from scripts.minimality_trainer import MinimalityDataCollator

    tokenizer = make_tokenizer()
    collator = MinimalityDataCollator(tokenizer=tokenizer)

    features = [
        {
            "input_vecs": torch.randn(33, VEC_DIM),
            "target_token_ids": torch.tensor([1, 2]),
            "target_attention_mask": torch.ones(2, dtype=torch.long),
        },
        {
            "input_vecs": torch.randn(33, VEC_DIM),
            "target_token_ids": torch.tensor([3, 4]),
            "target_attention_mask": torch.ones(2, dtype=torch.long),
        },
    ]

    batch = collator(features)
    assert batch["vecs"].shape == (2, 33, VEC_DIM)

def test_thought_descriptor_post_projection_norm_device():
    """post_projection_norm must be on the same device as the projection after load()."""
    from src.model.minimality_probe import ThoughtDescriptor

    logger = make_logger()

    mock_llm = MagicMock()
    mock_llm.config.hidden_size = 2048
    mock_llm.config.use_cache = True

    with patch(
        "src.model.minimality_probe.AutoModelForCausalLM.from_pretrained",
        return_value=mock_llm,
    ):
        descriptor = ThoughtDescriptor(
            logger=logger,
            model_name="meta-llama/Llama-3.2-1B",
            vector_dim=VEC_DIM,
            device="cpu",
            torch_dtype=torch.float32,
            trust_remote_code=False,
            freeze_base_model=True,
        )
        descriptor.load()

    assert descriptor.shared_projection.weight.device.type == "cpu"
    assert descriptor.post_projection_norm.weight.device.type == "cpu"
    assert descriptor.input_norm.weight is None                

def test_thought_descriptor_projection_type_must_be_shared():
    """Unsupported projection types should fail fast to avoid silent config drift."""
    from src.model.minimality_probe import ThoughtDescriptor

    logger = make_logger()
    descriptor = ThoughtDescriptor(
        logger=logger,
        model_name="meta-llama/Llama-3.2-1B",
        vector_dim=VEC_DIM,
        device="cpu",
        torch_dtype=torch.float32,
        trust_remote_code=False,
        freeze_base_model=True,
        projection_type="none",
    )

    with pytest.raises(NotImplementedError):
        descriptor.load()

def test_thought_descriptor_trainable_params_include_post_projection_norm():
    """Optimizer parameter list must include post_projection_norm parameters."""
    from src.model.minimality_probe import ThoughtDescriptor

    logger = make_logger()
    mock_llm = MagicMock()
    mock_llm.config.hidden_size = 2048
    mock_llm.config.use_cache = True

    with patch(
        "src.model.minimality_probe.AutoModelForCausalLM.from_pretrained",
        return_value=mock_llm,
    ):
        descriptor = ThoughtDescriptor(
            logger=logger,
            model_name="meta-llama/Llama-3.2-1B",
            vector_dim=VEC_DIM,
            device="cpu",
            torch_dtype=torch.float32,
            trust_remote_code=False,
            freeze_base_model=True,
        )
        descriptor.load()

    trainable_params = list(descriptor.get_trainable_parameters())
    assert descriptor.post_projection_norm.weight in trainable_params
    assert descriptor.post_projection_norm.bias in trainable_params

def test_create_minimality_datasets_propagates_target_source():
    """target_source must be forwarded to each split dataset."""
    from src.dataset.minimality_dataset import MinimalityDataset, create_minimality_datasets

    created = []

    def capture_init(self, *args, **kwargs):
        created.append(kwargs.get("target_source", "input"))
        self.indices = [0]
        self.num_return_sequences = NUM_BEAMS
        self.tr_loader = MagicMock()
        self.tr_loader.generated_texts_cache = {0: ["text"] * NUM_BEAMS}
        self.example_loader = MagicMock()
        self.logger = make_logger()
        self.tokenizer = make_tokenizer()
        self.split_name = kwargs.get("split_name", "train")
        self.max_input_length = 512
        self.target_source = kwargs.get("target_source", "input")

    with patch.object(MinimalityDataset, "__init__", capture_init):
        create_minimality_datasets(
            logger=make_logger(),
            tokenizer=make_tokenizer(),
            llm_data_dir="/fake",
            tr_data_dir="/fake",
            tr_type=ThoughtRepresentation.LAST_INPUT_TOKEN,
            target_source="output",
        )

    assert all(ts == "output" for ts in created), (
        f"Expected all target_source='output', got {created}"
    )
    assert len(created) == 3                    
