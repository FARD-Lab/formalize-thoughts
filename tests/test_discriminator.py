"""Tests for LlamaDiscriminator architecture (CPU-only, no real weights).

Covers:
- forward() output shape and parameter wiring
- CLS token appended — no token_type_emb
- Has other_norm / other_projection (not input_norm / shared_projection)
- trainable_state_dict() excludes backbone (model.*) keys — non-trivially tested
  by using a real nn.Module backbone so model.* keys DO appear in full state_dict()
- trainable_state_dict() roundtrip: save → modify → reload → match
- load_trainable_state_dict() works with no backbone keys present
- num_trainable_parameters() > 0
"""

import copy
import logging
import types

import pytest
import torch
import torch.nn as nn

from src.model.discriminator import LlamaDiscriminator

D_IN = 8                         
D_MODEL = 16                         
VOCAB = 32
BATCH = 2
TOKEN_LEN = 10
VEC_LEN = 4

def make_logger():
    log = logging.getLogger("test_discriminator")
    log.setLevel(logging.CRITICAL)
    return log

class TinyFakeLLM(nn.Module):
    """Minimal nn.Module stand-in for AutoModelForCausalLM.

    Uses a real Embedding layer so that disc.model IS a proper nn.Module
    submodule — its weights appear as model.* keys in LlamaDiscriminator.state_dict().
    """

    def __init__(self, d_model: int = D_MODEL, vocab: int = VOCAB):
        super().__init__()
        self.config = types.SimpleNamespace(hidden_size=d_model, use_cache=True)
        self._embed = nn.Embedding(vocab, d_model)

    def get_input_embeddings(self) -> nn.Embedding:
        return self._embed

    def gradient_checkpointing_enable(self): pass
    def enable_input_require_grads(self): pass

    def forward(self, inputs_embeds=None, attention_mask=None,
                output_hidden_states=False, **kwargs):
        b, s, _ = inputs_embeds.shape
        d = self.config.hidden_size
        return types.SimpleNamespace(
            logits=torch.zeros(b, s, VOCAB),
            hidden_states=[torch.zeros(b, s, d)] * 3,
        )

    def eval(self):
        return super().eval()

def make_discriminator(num_rep_spaces: int = 4) -> LlamaDiscriminator:
    """Build LlamaDiscriminator bypassing load() — no disk I/O, CPU-only."""
    disc = LlamaDiscriminator.__new__(LlamaDiscriminator)
    nn.Module.__init__(disc)

    disc.logger = make_logger()
    disc.model_name = "fake"
    disc.other_vector_dim = D_IN
    disc.device = "cpu"
    disc.torch_dtype = torch.float32
    disc.freeze_base_model = True
    disc.dropout_rate = 0.0
    disc.num_rep_spaces = num_rep_spaces
    disc.model_embedding_dim = None
    disc.other_projection = None
    disc.other_norm = None
    disc.post_projection_norm = None
    disc.other_type_emb = None
    disc.separator_embedding = None
    disc.cls_token = None
    disc.classifier_head = None

    disc.model = TinyFakeLLM(D_MODEL, VOCAB)
    disc.model_embedding_dim = D_MODEL

    disc.other_norm = nn.LayerNorm(D_IN, elementwise_affine=False)
    disc.other_projection = nn.Linear(D_IN, D_MODEL, bias=True)
    disc.post_projection_norm = nn.LayerNorm(D_MODEL, elementwise_affine=True)
    disc.other_type_emb = nn.Parameter(torch.randn(num_rep_spaces, D_MODEL))
    disc.separator_embedding = nn.Parameter(torch.randn(D_MODEL))
    disc.cls_token = nn.Parameter(torch.randn(D_MODEL))
    disc.classifier_head = nn.Sequential(
        nn.Linear(D_MODEL, D_MODEL // 2),
        nn.LayerNorm(D_MODEL // 2),
        nn.ReLU(),
        nn.Dropout(0.0),
        nn.Linear(D_MODEL // 2, 1),
    )
    return disc

class TestDiscriminatorForward:

    def test_forward_output_shape(self):
        """forward() must return (batch_size, 1) logits."""
        disc = make_discriminator()
        token_ids = torch.randint(0, VOCAB, (BATCH, TOKEN_LEN))
        thought_vecs = torch.randn(BATCH, VEC_LEN, D_IN)
        logits = disc.forward(token_ids, thought_vecs)
        assert logits.shape == (BATCH, 1), f"Expected ({BATCH}, 1), got {logits.shape}"

    def test_forward_batch_size_one(self):
        """Batch size 1 must produce (1, 1) logits."""
        disc = make_discriminator()
        token_ids = torch.randint(0, VOCAB, (1, TOKEN_LEN))
        thought_vecs = torch.randn(1, VEC_LEN, D_IN)
        logits = disc.forward(token_ids, thought_vecs)
        assert logits.shape == (1, 1)

    def test_forward_with_explicit_attention_masks(self):
        """forward() must work when explicit attention masks are provided."""
        disc = make_discriminator()
        token_ids = torch.randint(0, VOCAB, (BATCH, TOKEN_LEN))
        thought_vecs = torch.randn(BATCH, VEC_LEN, D_IN)
        token_mask = torch.ones(BATCH, TOKEN_LEN, dtype=torch.long)
        other_mask = torch.ones(BATCH, VEC_LEN, dtype=torch.long)
        logits = disc.forward(token_ids, thought_vecs, token_mask, other_mask)
        assert logits.shape == (BATCH, 1)

    def test_no_token_type_emb(self):
        """New architecture must not have token_type_emb."""
        disc = make_discriminator()
        assert not hasattr(disc, "token_type_emb"), (
            "token_type_emb must not exist in the discriminator"
        )

    def test_has_cls_token_parameter(self):
        """Must have a learnable cls_token of shape (D_MODEL,)."""
        disc = make_discriminator()
        assert hasattr(disc, "cls_token")
        assert isinstance(disc.cls_token, nn.Parameter)
        assert disc.cls_token.shape == (D_MODEL,)

    def test_has_other_norm_not_input_norm(self):
        """Must use other_norm; old input_norm must not exist."""
        disc = make_discriminator()
        assert hasattr(disc, "other_norm"), "other_norm must exist"
        assert not hasattr(disc, "input_norm"), "input_norm must NOT exist (old architecture)"

    def test_has_other_projection_not_shared_projection(self):
        """Must use other_projection; old shared_projection must not exist."""
        disc = make_discriminator()
        assert hasattr(disc, "other_projection"), "other_projection must exist"
        assert not hasattr(disc, "shared_projection"), (
            "shared_projection must NOT exist (old architecture)"
        )

    def test_output_is_scalar_per_example(self):
        """classifier_head output must reduce to a single logit per example."""
        disc = make_discriminator()
        for batch_size in (1, 3, 8):
            token_ids = torch.randint(0, VOCAB, (batch_size, TOKEN_LEN))
            thought_vecs = torch.randn(batch_size, VEC_LEN, D_IN)
            logits = disc.forward(token_ids, thought_vecs)
            assert logits.shape == (batch_size, 1), (
                f"batch_size={batch_size}: expected ({batch_size}, 1), got {logits.shape}"
            )

class TestTrainableStateDict:

    def test_full_state_dict_contains_backbone_keys(self):
        """Sanity: full state_dict() DOES contain model.* keys (backbone is a real nn.Module)."""
        disc = make_discriminator()
        full_state = disc.state_dict()
        backbone_keys = [k for k in full_state if k.startswith("model.")]
        assert len(backbone_keys) > 0, (
            "TinyFakeLLM must register as model.* in state_dict() for this test to be meaningful"
        )

    def test_trainable_state_dict_excludes_backbone(self):
        """trainable_state_dict() must NOT contain model.* keys."""
        disc = make_discriminator()
        trainable = disc.trainable_state_dict()
        backbone_keys = [k for k in trainable if k.startswith("model.")]
        assert len(backbone_keys) == 0, (
            f"Unexpected backbone keys in trainable state dict: {backbone_keys}"
        )

    def test_trainable_state_dict_is_strict_subset_of_full(self):
        """Every key in trainable state dict must also be in the full state dict."""
        disc = make_discriminator()
        full_keys = set(disc.state_dict().keys())
        trainable_keys = set(disc.trainable_state_dict().keys())
        assert trainable_keys.issubset(full_keys), (
            f"Trainable keys not in full state_dict: {trainable_keys - full_keys}"
        )

    def test_trainable_state_dict_contains_expected_components(self):
        """Must include other_projection, cls_token, separator, classifier, type_emb."""
        disc = make_discriminator()
        state = disc.trainable_state_dict()
        required_prefixes = [
            "other_projection",
            "cls_token",
            "separator_embedding",
            "classifier_head",
            "other_type_emb",
            "post_projection_norm",
        ]
        for prefix in required_prefixes:
            matching = [k for k in state if k.startswith(prefix)]
            assert matching, (
                f"Expected key with prefix '{prefix}' in trainable state dict. "
                f"Found keys: {list(state.keys())}"
            )

    def test_roundtrip_restores_projection_weights(self):
        """Save trainable dict, modify weights, reload, verify restoration."""
        disc = make_discriminator()
                                                                                    
        state = copy.deepcopy(disc.trainable_state_dict())
        original = disc.other_projection.weight.clone()

        with torch.no_grad():
            disc.other_projection.weight.fill_(0.0)
        assert not torch.allclose(disc.other_projection.weight, original)

        disc.load_trainable_state_dict(state)
        assert torch.allclose(disc.other_projection.weight, original), (
            "other_projection.weight not restored after load_trainable_state_dict"
        )

    def test_roundtrip_restores_cls_token(self):
        """cls_token must be correctly saved and restored."""
        disc = make_discriminator()
                                                                                    
        state = copy.deepcopy(disc.trainable_state_dict())
        original_cls = disc.cls_token.data.clone()

        with torch.no_grad():
            disc.cls_token.fill_(999.0)

        disc.load_trainable_state_dict(state)
        assert torch.allclose(disc.cls_token.data, original_cls), (
            "cls_token not restored after load_trainable_state_dict"
        )

    def test_load_without_backbone_keys_no_error(self):
        """load_trainable_state_dict() must not raise even with backbone keys absent."""
        disc = make_discriminator()
        state = disc.trainable_state_dict()
        disc.load_trainable_state_dict(state)                  

    def test_num_trainable_parameters_positive(self):
        """num_trainable_parameters() must return a positive count."""
        disc = make_discriminator()
        count = disc.num_trainable_parameters()
        assert count > 0, f"Expected > 0 trainable parameters, got {count}"

    def test_backbone_excluded_reduces_checkpoint_size(self):
        """Trainable state dict must be smaller (fewer keys) than full state dict."""
        disc = make_discriminator()
        full_size = len(disc.state_dict())
        trainable_size = len(disc.trainable_state_dict())
        assert trainable_size < full_size, (
            f"Trainable dict ({trainable_size} keys) should be smaller "
            f"than full dict ({full_size} keys)"
        )
