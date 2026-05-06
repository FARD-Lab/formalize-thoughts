import gc
import json
import logging
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, call, patch

import pytest
import torch

from src.dataset.curator import DataLLM

@pytest.fixture
def mock_logger():
    logger = logging.getLogger("test_curator")
    logger.setLevel(logging.CRITICAL)
    return logger

def _make_curator(tmp_path, mock_logger, backend="vllm", return_hs=True):
    """Build a DataLLM with fully mocked dependencies."""
    base_llm = MagicMock()
    base_llm.inference_backend = backend
    base_llm.tokenizer = MagicMock()
    base_llm.tokenizer.eos_token_id = 2
    base_llm.tokenizer.decode = lambda ids, skip_special_tokens=True: "generated text"

    data_loader = MagicMock()
    data_loader.instruction = "Solve this."
    data_loader.get_text_field = lambda ex: ex["text"]
    data_loader.__iter__ = lambda s: iter([{"text": f"q{i}"} for i in range(3)])

    tok_out = MagicMock()
    tok_out.__getitem__ = lambda s, k: {"input_ids": [[1, 2, 3, 4, 5]]}[k]
    base_llm.tokenize = MagicMock(return_value=tok_out)

    curator = DataLLM(
        logger=mock_logger,
        base_llm=base_llm,
        data_loader=data_loader,
        output_dir=str(tmp_path),
        max_input_length=8192,
        max_new_tokens=512,
        batch_size=1,
        num_return_sequences=2,
        return_hidden_states=return_hs,
        return_logits=False,
        save_format="jsonl",
        shard_size=1024,
    )
    return curator, base_llm, data_loader

class TestTwoPassRouting:
    def test_two_pass_loads_transformers_first_then_vllm(self, tmp_path, mock_logger):
        """vllm + return_hidden_states=True → load_for_hidden_states then load."""
        curator, base_llm, _ = _make_curator(tmp_path, mock_logger, backend="vllm", return_hs=True)

        base_llm.forward_prefill = MagicMock(
            return_value=[torch.zeros(2, 4)]
        )
        generated_ids = torch.zeros(2, 5, dtype=torch.long)
        base_llm.generate = MagicMock(return_value={
            "generated_ids": generated_ids,
            "input_ids": torch.zeros(1, 5, dtype=torch.long),
        })

        curator.run()

        base_llm.load_for_hidden_states.assert_called_once()
        base_llm.unload_model.assert_called_once()
        base_llm.load.assert_called_once()             

    def test_single_pass_when_transformers_backend(self, tmp_path, mock_logger):
        """transformers backend → single-pass load, no load_for_hidden_states."""
        curator, base_llm, _ = _make_curator(
            tmp_path, mock_logger, backend="transformers", return_hs=True
        )

        generated_ids = torch.zeros(2, 5, dtype=torch.long)
        base_llm.generate = MagicMock(return_value={
            "generated_ids": generated_ids,
            "input_ids": torch.zeros(1, 5, dtype=torch.long),
            "hidden_states": [[torch.zeros(1, 5, 4)] * 2],            
        })

        curator.run()

        base_llm.load.assert_called_once()
        base_llm.load_for_hidden_states.assert_not_called()

    def test_single_pass_when_return_hidden_states_false(self, tmp_path, mock_logger):
        """vllm + return_hidden_states=False → single-pass vllm, no phase 1."""
        curator, base_llm, _ = _make_curator(
            tmp_path, mock_logger, backend="vllm", return_hs=False
        )

        generated_ids = torch.zeros(2, 5, dtype=torch.long)
        base_llm.generate = MagicMock(return_value={
            "generated_ids": generated_ids,
            "input_ids": torch.zeros(1, 5, dtype=torch.long),
        })

        curator.run()

        base_llm.load.assert_called_once()
        base_llm.load_for_hidden_states.assert_not_called()

class TestTwoPassOutput:
    def test_first_hs_files_written_for_each_group(self, tmp_path, mock_logger):
        """Phase 1 hidden states are saved to first_hs_group_*.pt files."""
        curator, base_llm, _ = _make_curator(tmp_path, mock_logger, backend="vllm", return_hs=True)

        hs_tensor = torch.ones(2, 4)
        base_llm.forward_prefill = MagicMock(return_value=[hs_tensor])

        generated_ids = torch.zeros(2, 5, dtype=torch.long)
        base_llm.generate = MagicMock(return_value={
            "generated_ids": generated_ids,
            "input_ids": torch.zeros(1, 5, dtype=torch.long),
        })

        curator.run()

        first_hs_files = list((tmp_path / "tensors" / "first_hs").glob("*.pt"))
        assert len(first_hs_files) == 3                                        

    def test_generations_shard_written(self, tmp_path, mock_logger):
        """Phase 2 generation outputs are written to the shard file."""
        curator, base_llm, _ = _make_curator(tmp_path, mock_logger, backend="vllm", return_hs=True)

        base_llm.forward_prefill = MagicMock(return_value=[torch.zeros(2, 4)])

        generated_ids = torch.zeros(2, 5, dtype=torch.long)
        base_llm.generate = MagicMock(return_value={
            "generated_ids": generated_ids,
            "input_ids": torch.zeros(1, 5, dtype=torch.long),
        })

        curator.run()

        shard = tmp_path / "generations_shard_0.jsonl"
        assert shard.exists()
        lines = [json.loads(l) for l in shard.read_text().splitlines() if l.strip()]
                                                   
        assert len(lines) == 6
