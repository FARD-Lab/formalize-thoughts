"""Lightweight unit tests for scripts/causality_eval.py.

Tests cover (no GPU, no real model weights):
- project_thought: output shape, positional embedding tiling (seq_len > num_rep_spaces),
                   num_rep_spaces=1 path
- compute_kl: returns a non-negative float, KL(P||P) ≈ 0 when both arms have identical logits
- Beam vector selection: RANDOM_VECTOR, EMBEDDING_NO_POOLING, and all other TR types
- CausalityEvalConfig: num_rep_spaces derived from ThoughtRepresentation.num_features
- ThoughtRepresentation.is_embedding_based for all TR types
"""

import types
from unittest.mock import MagicMock

import pytest
import torch
import torch.nn as nn

from src.utils.config import ThoughtRepresentation, resolve_other_vector_dim

D_IN = 8                                       
D_MODEL = 16                                       
VOCAB = 32                     

def make_fake_discriminator(num_rep_spaces: int = 4, d_in: int = D_IN, d_model: int = D_MODEL):
    """Build a namespace mimicking LlamaDiscriminator's projection attributes."""
    disc = types.SimpleNamespace()
    disc.num_rep_spaces = num_rep_spaces
    disc.other_norm = nn.LayerNorm(d_in, elementwise_affine=False)
    disc.other_projection = nn.Linear(d_in, d_model, bias=True)
    disc.post_projection_norm = nn.LayerNorm(d_model, elementwise_affine=True)
    disc.other_type_emb = nn.Parameter(torch.randn(num_rep_spaces, d_model))
    disc.cls_token = nn.Parameter(torch.randn(d_model))
    return disc

def make_fake_llm(seq_vocab: int = VOCAB, d_model: int = D_MODEL):
    """Minimal LLM stub: returns fixed logits regardless of input."""
    llm = MagicMock()

    def forward_fn(*args, input_ids=None, inputs_embeds=None, **kwargs):
                                                                     
        if input_ids is not None:
            batch, seq = input_ids.shape
        else:
            batch, seq, _ = inputs_embeds.shape
        logits = torch.zeros(batch, seq, seq_vocab)
        out = MagicMock()
        out.logits = logits
        return out

    llm.side_effect = forward_fn
    llm.__call__ = forward_fn

    embed_table = nn.Embedding(seq_vocab, d_model)
    llm.get_input_embeddings = MagicMock(return_value=embed_table)
    return llm

class TestProjectThought:
    from scripts.causality_eval import project_thought

    def test_output_shape_single_token(self):
        """seq_len=1 (e.g. EMBEDDING_POOLING) → [1, 1, D_MODEL]."""
        from scripts.causality_eval import project_thought
        disc = make_fake_discriminator(num_rep_spaces=4)
        vecs = torch.randn(1, 1, D_IN)
        out = project_thought(disc, vecs)
        assert out.shape == (1, 1, D_MODEL)

    def test_output_shape_multi_token(self):
        """seq_len=4 (e.g. LAST_INPUT_TOKEN with 4 layers) → [1, 4, D_MODEL]."""
        from scripts.causality_eval import project_thought
        disc = make_fake_discriminator(num_rep_spaces=4)
        vecs = torch.randn(1, 4, D_IN)
        out = project_thought(disc, vecs)
        assert out.shape == (1, 4, D_MODEL)

    def test_positional_emb_tiled_when_seq_exceeds_num_rep_spaces(self):
        """seq_len > num_rep_spaces: other_type_emb must be tiled, not truncated."""
        from scripts.causality_eval import project_thought
        num_rep = 3
        seq_len = 7             
        disc = make_fake_discriminator(num_rep_spaces=num_rep)

        with torch.no_grad():
            disc.other_type_emb.copy_(torch.arange(num_rep * D_MODEL, dtype=torch.float)
                                      .reshape(num_rep, D_MODEL))

        vecs = torch.zeros(1, seq_len, D_IN)
                                                                      
        t = disc.post_projection_norm(disc.other_projection(disc.other_norm(vecs)))

        out = project_thought(disc, vecs)

        expected_pos = disc.other_type_emb.repeat(3, 1)[:seq_len]                      
        expected = t + expected_pos
        assert torch.allclose(out, expected, atol=1e-5), (
            "Tiled positional embeddings do not match expected values."
        )

    def test_num_rep_spaces_one_uses_slice(self):
        """num_rep_spaces=1: other_type_emb[:seq_len] path (no tiling)."""
        from scripts.causality_eval import project_thought
        disc = make_fake_discriminator(num_rep_spaces=1)
                                                                    
        vecs = torch.randn(1, 1, D_IN)
        out = project_thought(disc, vecs)
        assert out.shape == (1, 1, D_MODEL)

class TestComputeKL:

    def _make_disc_with_llm(self, llm):
        disc = types.SimpleNamespace()
        disc.model = llm
        return disc

    def test_returns_float(self):
        """compute_kl must return a Python float."""
        from scripts.causality_eval import compute_kl
        llm = make_fake_llm()
        disc = self._make_disc_with_llm(llm)
        projected_T = torch.randn(1, 3, D_MODEL)
        y_ids = torch.tensor([1, 2, 3])
        z_ids = torch.tensor([4, 5, 6, 7])
        result = compute_kl(disc, projected_T, y_ids, z_ids, "cpu")
        assert isinstance(result, float)

    def test_nonnegative(self):
        """KL divergence is always >= 0."""
        from scripts.causality_eval import compute_kl
        llm = make_fake_llm()
        disc = self._make_disc_with_llm(llm)
        projected_T = torch.randn(1, 2, D_MODEL)
        y_ids = torch.tensor([1, 2])
        z_ids = torch.tensor([3, 4, 5])
        result = compute_kl(disc, projected_T, y_ids, z_ids, "cpu")
        assert result >= 0.0, f"KL should be non-negative, got {result}"

    def test_identical_logits_gives_zero_kl(self):
        """KL(P || P) = 0: when both arms produce the same logits, KL must be ~0."""
        from scripts.causality_eval import compute_kl

        fixed_logits = torch.randn(1, 5, VOCAB)                                         

        llm = MagicMock()
        out = MagicMock()
        out.logits = fixed_logits
        llm.return_value = out
        llm.side_effect = None

        embed_table = nn.Embedding(VOCAB, D_MODEL)
        llm.get_input_embeddings = MagicMock(return_value=embed_table)

        disc = self._make_disc_with_llm(llm)
        projected_T = torch.randn(1, 2, D_MODEL)
        y_ids = torch.tensor([1, 2])
        z_ids = torch.tensor([3, 4, 5])

        result = compute_kl(disc, projected_T, y_ids, z_ids, "cpu")
        assert abs(result) < 1e-5, (
            f"KL(P||P) should be ~0 when logits are identical, got {result:.6f}"
        )

def select_t_vec(tr_type, thought_vec, beam_idx):
    """Replicate the t_vec selection logic from scripts/causality_eval.py main()."""
    if tr_type == ThoughtRepresentation.RANDOM_VECTOR:
        return thought_vec[beam_idx]
    elif tr_type == ThoughtRepresentation.EMBEDDING_NO_POOLING:
        return thought_vec[beam_idx]                         
    else:
        return thought_vec

class TestBeamVectorSelection:

    K = 8

    def test_random_vector_selects_beam(self):
        """RANDOM_VECTOR cache shape [K, 1, d]: beam k → [1, d]."""
        thought_vec = torch.randn(self.K, 1, D_IN)
        for beam_idx in range(self.K):
            t_vec = select_t_vec(ThoughtRepresentation.RANDOM_VECTOR, thought_vec, beam_idx)
            assert t_vec.shape == (1, D_IN)
            assert torch.equal(t_vec, thought_vec[beam_idx])

    def test_embedding_no_pooling_selects_beam(self):
        """EMBEDDING_NO_POOLING cache shape [K, 1, d]: beam k → [1, d]."""
        thought_vec = torch.randn(self.K, 1, D_IN)
        for beam_idx in range(self.K):
            t_vec = select_t_vec(ThoughtRepresentation.EMBEDDING_NO_POOLING, thought_vec, beam_idx)
            assert t_vec.shape == (1, D_IN)
            assert torch.equal(t_vec, thought_vec[beam_idx])

    @pytest.mark.parametrize("tr_type", [
        ThoughtRepresentation.LAST_INPUT_TOKEN,
        ThoughtRepresentation.LAST_INPUT_HIDDEN_STATE,
        ThoughtRepresentation.SOFT_THINKING,
        ThoughtRepresentation.SOFT_THINKING_NOISE,
        ThoughtRepresentation.LATENT_THINKING,
        ThoughtRepresentation.EMBEDDING_POOLING,
        ThoughtRepresentation.INPUT_EMBEDDING,
    ])
    def test_input_derived_types_return_same_vector_for_all_beams(self, tr_type):
        """Input-derived TR types: same T for every beam (identical object returned)."""
        thought_vec = torch.randn(33, D_IN)                                       
        vecs = [select_t_vec(tr_type, thought_vec, k) for k in range(self.K)]
        for k in range(1, self.K):
            assert torch.equal(vecs[0], vecs[k]), (
                f"{tr_type.value}: expected same T for all beams, but beam {k} differs"
            )

class TestNumRepSpaces:

    def test_last_input_token_has_33_features(self):
        """LAST_INPUT_TOKEN.num_features == 33 (one slot per LLaMA-8B layer)."""
        tr = ThoughtRepresentation.LAST_INPUT_TOKEN
        assert tr.num_features == 33

    @pytest.mark.parametrize("tr_type", [
        ThoughtRepresentation.LAST_INPUT_HIDDEN_STATE,
        ThoughtRepresentation.EMBEDDING_POOLING,
        ThoughtRepresentation.EMBEDDING_NO_POOLING,
        ThoughtRepresentation.SOFT_THINKING,
        ThoughtRepresentation.SOFT_THINKING_NOISE,
        ThoughtRepresentation.LATENT_THINKING,
        ThoughtRepresentation.INPUT_EMBEDDING,
        ThoughtRepresentation.RANDOM_VECTOR,
    ])
    def test_non_lit_types_have_one_feature(self, tr_type):
        """All non-LIT TR types have num_features=1 → num_rep_spaces=1 in discriminator."""
        assert tr_type.num_features == 1, (
            f"{tr_type.value}: expected num_features=1, got {tr_type.num_features}"
        )

class TestResolveNumRepSpaces:

    def test_last_input_token_uses_source_num_layers(self):
        from scripts.causality_eval import resolve_num_rep_spaces

        assert resolve_num_rep_spaces(ThoughtRepresentation.LAST_INPUT_TOKEN, 33) == 33
        assert resolve_num_rep_spaces(ThoughtRepresentation.LAST_INPUT_TOKEN, 81) == 81
        assert resolve_num_rep_spaces(ThoughtRepresentation.LAST_INPUT_TOKEN, 65) == 65

    @pytest.mark.parametrize("tr_type", [
        ThoughtRepresentation.LAST_INPUT_HIDDEN_STATE,
        ThoughtRepresentation.EMBEDDING_POOLING,
        ThoughtRepresentation.EMBEDDING_NO_POOLING,
        ThoughtRepresentation.SOFT_THINKING,
        ThoughtRepresentation.SOFT_THINKING_NOISE,
        ThoughtRepresentation.LATENT_THINKING,
        ThoughtRepresentation.INPUT_EMBEDDING,
        ThoughtRepresentation.RANDOM_VECTOR,
    ])
    def test_non_lit_types_ignore_source_num_layers(self, tr_type):
        from scripts.causality_eval import resolve_num_rep_spaces

        result_33 = resolve_num_rep_spaces(tr_type, 33)
        result_81 = resolve_num_rep_spaces(tr_type, 81)
        assert result_33 == tr_type.num_features
        assert result_81 == tr_type.num_features

class TestIsEmbeddingBased:

    @pytest.mark.parametrize("tr_type", [
        ThoughtRepresentation.EMBEDDING_POOLING,
        ThoughtRepresentation.EMBEDDING_NO_POOLING,
        ThoughtRepresentation.EMBEDDING_ALL,
        ThoughtRepresentation.INPUT_EMBEDDING,
    ])
    def test_embedding_based_types_are_flagged(self, tr_type):
        """Embedding-based TR types must have is_embedding_based=True."""
        assert tr_type.is_embedding_based is True, (
            f"{tr_type.value}: expected is_embedding_based=True"
        )

    @pytest.mark.parametrize("tr_type", [
        ThoughtRepresentation.LAST_INPUT_TOKEN,
        ThoughtRepresentation.LAST_INPUT_HIDDEN_STATE,
        ThoughtRepresentation.SOFT_THINKING,
        ThoughtRepresentation.SOFT_THINKING_NOISE,
        ThoughtRepresentation.LATENT_THINKING,
        ThoughtRepresentation.RANDOM_VECTOR,
    ])
    def test_llm_based_types_are_not_flagged(self, tr_type):
        """LLM hidden-state TR types must have is_embedding_based=False."""
        assert tr_type.is_embedding_based is False, (
            f"{tr_type.value}: expected is_embedding_based=False"
        )

class TestResolveOtherVectorDim:

    SOURCE_HIDDEN = 8192
    EMBEDDER_DIM = 4096

    @pytest.mark.parametrize("tr_type", [
        ThoughtRepresentation.EMBEDDING_POOLING,
        ThoughtRepresentation.EMBEDDING_NO_POOLING,
        ThoughtRepresentation.EMBEDDING_ALL,
        ThoughtRepresentation.INPUT_EMBEDDING,
    ])
    def test_embedding_based_uses_embedder_dim(self, tr_type):
        result = resolve_other_vector_dim(
            tr_type,
            source_hidden_size=self.SOURCE_HIDDEN,
            embedder_output_dim=self.EMBEDDER_DIM,
        )
        assert result == self.EMBEDDER_DIM, (
            f"{tr_type.value}: expected {self.EMBEDDER_DIM}, got {result}"
        )

    @pytest.mark.parametrize("tr_type", [
        ThoughtRepresentation.LAST_INPUT_TOKEN,
        ThoughtRepresentation.LAST_INPUT_HIDDEN_STATE,
        ThoughtRepresentation.SOFT_THINKING,
        ThoughtRepresentation.SOFT_THINKING_NOISE,
        ThoughtRepresentation.LATENT_THINKING,
        ThoughtRepresentation.RANDOM_VECTOR,
    ])
    def test_llm_based_uses_source_hidden_size(self, tr_type):
        result = resolve_other_vector_dim(
            tr_type,
            source_hidden_size=self.SOURCE_HIDDEN,
            embedder_output_dim=self.EMBEDDER_DIM,
        )
        assert result == self.SOURCE_HIDDEN, (
            f"{tr_type.value}: expected {self.SOURCE_HIDDEN}, got {result}"
        )

    def test_default_embedder_dim_is_4096(self):
        result = resolve_other_vector_dim(
            ThoughtRepresentation.EMBEDDING_POOLING, source_hidden_size=8192
        )
        assert result == 4096

    def test_8b_both_resolve_to_4096(self):
        """Default 8B model: source_hidden_size=4096 means all types return 4096."""
        for tr in ThoughtRepresentation:
            dim = resolve_other_vector_dim(tr, source_hidden_size=4096, embedder_output_dim=4096)
            assert dim == 4096, f"{tr.value}: expected 4096, got {dim}"

    def test_70b_distinguishes_embedding_vs_llm_types(self):
        """70B model: embedding-based stays 4096, LLM-based becomes 8192."""
        for tr in ThoughtRepresentation:
            dim = resolve_other_vector_dim(tr, source_hidden_size=8192, embedder_output_dim=4096)
            if tr.is_embedding_based:
                assert dim == 4096, f"{tr.value}: expected 4096 (embedding-based), got {dim}"
            else:
                assert dim == 8192, f"{tr.value}: expected 8192 (llm-based), got {dim}"
