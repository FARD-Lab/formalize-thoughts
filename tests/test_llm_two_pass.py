import logging
from unittest.mock import MagicMock, call, patch

import pytest
import torch

from src.model.llm import BaseLLM
from src.utils.config import ModelType

@pytest.fixture
def mock_logger():
    logger = logging.getLogger("test")
    logger.setLevel(logging.CRITICAL)
    return logger

def _make_llm(logger, backend="vllm") -> BaseLLM:
    """Construct a BaseLLM without loading any model."""
    llm = BaseLLM(
        logger=logger,
        model_name="meta-llama/Llama-3.1-8B-Instruct",
        model_type=ModelType.LLAMA,
        inference_backend=backend,
    )
    return llm

class TestLoadForHiddenStates:
    def test_calls_load_with_transformers_backend(self, mock_logger):
        """load_for_hidden_states temporarily switches backend to transformers."""
        llm = _make_llm(mock_logger, backend="vllm")
        recorded = []

        def _fake_load(self_inner):
            recorded.append(self_inner.inference_backend)

        with patch.object(BaseLLM, "load", _fake_load):
            llm.load_for_hidden_states()

        assert recorded == ["transformers"], "load() must be called with transformers backend"

    def test_restores_original_backend_after_call(self, mock_logger):
        """Backend is vllm again after load_for_hidden_states returns."""
        llm = _make_llm(mock_logger, backend="vllm")

        with patch.object(BaseLLM, "load", lambda self: None):
            llm.load_for_hidden_states()

        assert llm.inference_backend == "vllm"

    def test_restores_backend_even_if_load_raises(self, mock_logger):
        """Backend is restored even when load() throws."""
        llm = _make_llm(mock_logger, backend="vllm")

        def _boom(self_inner):
            raise RuntimeError("load failed")

        with pytest.raises(RuntimeError, match="load failed"):
            with patch.object(BaseLLM, "load", _boom):
                llm.load_for_hidden_states()

        assert llm.inference_backend == "vllm"

class TestForwardPrefill:
    def _make_loaded_llm(self, mock_logger):
        """Return a BaseLLM whose .model and .tokenizer are mocked."""
        llm = _make_llm(mock_logger, backend="vllm")

        tok = MagicMock()
        tok.return_value = {
            "input_ids": torch.zeros(2, 3, dtype=torch.long),
            "attention_mask": torch.ones(2, 3, dtype=torch.long),
        }
                                                                   
        inputs_mock = MagicMock()
        inputs_mock.__getitem__ = lambda s, k: {
            "input_ids": torch.zeros(2, 3, dtype=torch.long),
            "attention_mask": torch.ones(2, 3, dtype=torch.long),
        }[k]
        tok.return_value = inputs_mock
        inputs_mock.to = lambda device: inputs_mock
        llm.tokenizer = tok

        llm._get_formatted = lambda prompts, instruction=None: list(prompts)

        hidden = [torch.randn(2, 3, 4) for _ in range(3)]            
        fwd_out = MagicMock()
        fwd_out.hidden_states = hidden

        model_mock = MagicMock()
        model_mock.return_value = fwd_out
        model_mock.__call__ = lambda *a, **kw: fwd_out
        llm.model = model_mock
        llm.device = "cpu"

        return llm, hidden

    def test_raises_if_model_not_loaded(self, mock_logger):
        llm = _make_llm(mock_logger)
        llm.model = None
        with pytest.raises(RuntimeError, match="load_for_hidden_states"):
            llm.forward_prefill(["hello"])

    def test_returns_one_tensor_per_prompt(self, mock_logger):
        llm, hidden = self._make_loaded_llm(mock_logger)
        results = llm.forward_prefill(["hello", "world"])
        assert len(results) == 2

    def test_output_shape_is_num_layers_times_hidden_size(self, mock_logger):
        llm, hidden = self._make_loaded_llm(mock_logger)
        results = llm.forward_prefill(["hello", "world"])
                                           
        assert results[0].shape == (3, 4)
        assert results[1].shape == (3, 4)

    def test_output_is_on_cpu(self, mock_logger):
        llm, _ = self._make_loaded_llm(mock_logger)
        results = llm.forward_prefill(["hello"])
        assert results[0].device.type == "cpu"

    def test_single_string_input_works(self, mock_logger):
        llm, hidden = self._make_loaded_llm(mock_logger)
                                                      
        llm._get_formatted = lambda prompts, instruction=None: (
            [prompts] if isinstance(prompts, str) else list(prompts)
        )
                                                                          
        single_hidden = [torch.randn(1, 3, 4) for _ in range(3)]
        fwd_out = MagicMock()
        fwd_out.hidden_states = single_hidden
        llm.model = MagicMock(return_value=fwd_out)
        llm.model.__call__ = lambda *a, **kw: fwd_out

        tok_mock = MagicMock()
        inputs_mock = MagicMock()
        inputs_mock.__getitem__ = lambda s, k: {
            "input_ids": torch.zeros(1, 3, dtype=torch.long),
        }[k]
        inputs_mock.to = lambda d: inputs_mock
        tok_mock.return_value = inputs_mock
        llm.tokenizer = tok_mock

        results = llm.forward_prefill("hello")
        assert len(results) == 1
        assert results[0].shape == (3, 4)
