from typing import Dict, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.model.base import BaseModel
from src.utils.config import ProjectType
from src.utils.logging import Logger

class LlamaDiscriminator(nn.Module, BaseModel):
    """Discriminator based on Llama model for binary classification."""

    def __init__(
        self,
        logger: Logger,
        model_name: str = "meta-llama/Llama-3.2-1B",
        other_vector_dim: int = 4096,
        device: str = "cuda",
        load_in_8bit: bool = False,
        load_in_4bit: bool = False,
        torch_dtype: Union[str, torch.dtype] = torch.bfloat16,
        trust_remote_code: bool = True,
        max_memory: Optional[Dict[str, str]] = None,
        freeze_base_model: bool = True,
        dropout_rate: float = 0.1,
        projection_type: ProjectType = ProjectType.SHARED,
        num_rep_spaces: int = 33,
        use_deep_projection: bool = False,
        unfreeze_last_n_layers: int = 0,
        **kwargs,
    ):
        nn.Module.__init__(self)
        BaseModel.__init__(
            self,
            logger=logger,
            model_name=model_name,
            device=device,
            load_in_8bit=load_in_8bit,
            load_in_4bit=load_in_4bit,
            torch_dtype=torch_dtype,
            trust_remote_code=trust_remote_code,
            max_memory=max_memory,
            **kwargs,
        )

        self.other_vector_dim = other_vector_dim
        self.freeze_base_model = freeze_base_model
        self.dropout_rate = dropout_rate
        self.projection_type = projection_type
        self.num_rep_spaces = num_rep_spaces
        self.use_deep_projection = use_deep_projection
        self.unfreeze_last_n_layers = unfreeze_last_n_layers
        self.logger.info(f"Got num_rep_spaces: {self.num_rep_spaces}")
        self.logger.info(f"Got use_deep_projection: {self.use_deep_projection}")
        self.logger.info(f"Got unfreeze_last_n_layers: {self.unfreeze_last_n_layers}")
                                             
        self.model_embedding_dim: Optional[int] = None
        self.other_projection: Optional[nn.Linear] = None
        self.other_norm: Optional[nn.LayerNorm] = None
        self.other_type_emb: Optional[nn.Parameter] = None
        self.separator_embedding: Optional[nn.Parameter] = None
        self.cls_token: Optional[nn.Parameter] = None
        self.classifier_head: Optional[nn.Sequential] = None

    def train(self, mode: bool = True):
                                                        
        nn.Module.train(self, mode)
        return self

    def eval(self):
                                                       
        nn.Module.eval(self)
        return self

    def load(self) -> None:
        """Load model and initialize projection matrices and classifier."""
        self.logger.info(f"Loading base model '{self.model_name}'")

        self.model = (
            AutoModelForCausalLM.from_pretrained(
                self.model_name,
                **self.model_kwargs,
            )
            .to(self.device)
            .eval()
        )

        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name,
            trust_remote_code=self.model_kwargs.get("trust_remote_code", False),
        )

        if self.tokenizer.pad_token is None:
            if self.tokenizer.eos_token is not None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            elif self.tokenizer.unk_token is not None:
                self.tokenizer.pad_token = self.tokenizer.unk_token
            else:
                self.tokenizer.add_special_tokens({"pad_token": "[PAD]"})
                self.model.resize_token_embeddings(len(self.tokenizer))

        self.model.config.pad_token_id = self.tokenizer.pad_token_id
        if hasattr(self.model, "generation_config") and self.model.generation_config is not None:
            self.model.generation_config.pad_token_id = self.tokenizer.pad_token_id

        self.model.config.use_cache = False
        self.model.gradient_checkpointing_enable()
        self.model.enable_input_require_grads()

        dtype = self.torch_dtype
        if isinstance(dtype, str):
            dtype = next(self.model.parameters()).dtype
        self.logger.info(f"Model dtype: {dtype}")

        self.model_embedding_dim = self.model.config.hidden_size
        self.logger.info(f"Model embedding dimension: {self.model_embedding_dim}")

        if self.freeze_base_model:
            self.logger.info("Freezing base model parameters")
            for param in self.model.parameters():
                param.requires_grad = False
            if self.unfreeze_last_n_layers > 0:
                                                                                
                backbone = self.model.model if hasattr(self.model, "model") else self.model
                layers = getattr(backbone, "layers", None)
                if layers is None:
                    raise RuntimeError(
                        "Cannot find backbone.layers for partial unfreeze; "
                        "architecture may not match expected Llama/decoder layout."
                    )
                n = min(self.unfreeze_last_n_layers, len(layers))
                unfrozen = 0
                for layer in layers[-n:]:
                    for p in layer.parameters():
                        p.requires_grad = True
                        unfrozen += p.numel()
                                                                                       
                final_norm = getattr(backbone, "norm", None)
                if final_norm is not None:
                    for p in final_norm.parameters():
                        p.requires_grad = True
                        unfrozen += p.numel()
                self.logger.info(
                    f"Partial unfreeze: last {n} layers + final norm, "
                    f"{unfrozen:,} additional trainable params"
                )

        self.logger.info("Initializing layers")
                                                                                               
        if self.use_deep_projection:
                                                                            
            inner_dim = 2 * self.model_embedding_dim
            self.other_projection = nn.Sequential(
                nn.Linear(self.other_vector_dim, inner_dim, bias=True, dtype=dtype),
                nn.GELU(),
                nn.LayerNorm(inner_dim, dtype=dtype),
                nn.Linear(inner_dim, self.model_embedding_dim, bias=True, dtype=dtype),
            )
            self.other_norm = nn.LayerNorm(
                self.other_vector_dim, elementwise_affine=True, dtype=dtype
            )
        else:
            self.other_projection = nn.Linear(
                self.other_vector_dim, self.model_embedding_dim, bias=True, dtype=dtype
            )
            self.other_norm = nn.LayerNorm(
                self.other_vector_dim, elementwise_affine=False, dtype=dtype
            )
        self.post_projection_norm = nn.LayerNorm(self.model_embedding_dim, dtype=dtype)

        self.other_type_emb = nn.Parameter(
            torch.empty(self.num_rep_spaces, self.model_embedding_dim, dtype=dtype)
        )
        nn.init.normal_(self.other_type_emb, std=0.02)

        self.separator_embedding = nn.Parameter(
            torch.empty(self.model_embedding_dim, dtype=dtype)
        )
        nn.init.normal_(self.separator_embedding, std=0.02)

        self.cls_token = nn.Parameter(
            torch.empty(self.model_embedding_dim, dtype=dtype)
        )
        nn.init.normal_(self.cls_token, std=0.02)

        self.classifier_head = nn.Sequential(
            nn.Linear(
                self.model_embedding_dim, self.model_embedding_dim // 2, dtype=dtype
            ),
            nn.LayerNorm(self.model_embedding_dim // 2, dtype=dtype),
            nn.ReLU(),
            nn.Dropout(self.dropout_rate),
            nn.Linear(self.model_embedding_dim // 2, 1, dtype=dtype),
        )

        target_device = self.device
        for c in [self.other_norm, self.other_projection, self.post_projection_norm, self.classifier_head]:
            c.to(target_device)

        self.other_type_emb.data = self.other_type_emb.data.to(target_device)
        self.separator_embedding.data = self.separator_embedding.data.to(target_device)
        self.cls_token.data = self.cls_token.data.to(target_device)

        self.logger.info("Discriminator loaded successfully")

    def forward(
        self,
        token_ids: torch.Tensor,
        thought_vecs: torch.Tensor,
        token_attention_mask: Optional[torch.Tensor] = None,
        other_attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if self.model is None:
            raise RuntimeError("Model not loaded. Call load() first.")

        device = token_ids.device
        batch_size = token_ids.size(0)

        token_embeddings = self.model.get_input_embeddings()(token_ids)

        thought_vecs = self.other_norm(thought_vecs)
        projected_others = self.other_projection(thought_vecs)
        projected_others = self.post_projection_norm(projected_others)

        if self.num_rep_spaces > 1:                                           
            seq_len = projected_others.size(1)
            num_repeats = (seq_len + self.num_rep_spaces - 1) // self.num_rep_spaces
            repeated_other_type_emb = self.other_type_emb.repeat(num_repeats, 1)[
                :seq_len
            ].to(device)
            projected_others = projected_others + repeated_other_type_emb
        else:
            projected_others = projected_others + self.other_type_emb[
                : projected_others.size(1)
            ].to(device)

        separator = self.separator_embedding.to(device).expand(batch_size, 1, -1)
        cls = self.cls_token.to(device).expand(batch_size, 1, -1)

        combined_embeddings = torch.cat(
            [token_embeddings, separator, projected_others, cls], dim=1
        )

        if token_attention_mask is None:
            token_attention_mask = torch.ones(
                batch_size, token_embeddings.size(1), dtype=torch.long, device=device
            )
        if other_attention_mask is None:
            other_attention_mask = torch.ones(
                batch_size, projected_others.size(1), dtype=torch.long, device=device
            )

        ones = torch.ones(batch_size, 1, dtype=torch.long, device=device)
        combined_attention_mask = torch.cat(
            [token_attention_mask, ones, other_attention_mask, ones], dim=1
        )

        backbone = self.model.model if hasattr(self.model, "model") else self.model
        outputs = backbone(
            inputs_embeds=combined_embeddings,
            attention_mask=combined_attention_mask,
        )

        pooled_output = outputs.last_hidden_state[:, -1, :]

        logits = self.classifier_head(pooled_output)

        return logits

    def compute_loss(
        self,
        token_ids: torch.Tensor,
        thought_vecs: torch.Tensor,
        labels: torch.Tensor,
        token_attention_mask: Optional[torch.Tensor] = None,
        other_attention_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        logits = self.forward(
            token_ids,
            thought_vecs,
            token_attention_mask,
            other_attention_mask,
        )
        labels = labels.to(logits.device).to(logits.dtype)
        loss = F.binary_cross_entropy_with_logits(logits.squeeze(-1), labels)
        return loss, logits

    def to(self, *args, **kwargs):
        nn.Module.to(self, *args, **kwargs)
                                                          
        if args and isinstance(args[0], (str, torch.device)):
            self.device = str(args[0])
        elif "device" in kwargs:
            self.device = str(kwargs["device"])
        return self

    def predict(
        self,
        token_ids: torch.Tensor,
        thought_vecs: torch.Tensor,
        token_attention_mask: Optional[torch.Tensor] = None,
        other_attention_mask: Optional[torch.Tensor] = None,
        threshold: float = 0.5,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Make binary predictions.

        Args:
            token_ids: Token IDs of shape (batch_size, seq_len_1)
            thought_vecs: Thought vectors of shape (batch_size, seq_len_2, other_vector_dim)
            token_attention_mask: Attention mask for token embeddings (batch_size, seq_len_1)
            other_attention_mask: Attention mask for thought vectors (batch_size, seq_len_2)
            threshold: Classification threshold (default: 0.5)

        Returns:
            Tuple of (predictions, probabilities)
            - predictions: Binary predictions of shape (batch_size,)
            - probabilities: Sigmoid probabilities of shape (batch_size,)
        """
        with torch.no_grad():
            logits = self.forward(
                token_ids,
                thought_vecs,
                token_attention_mask,
                other_attention_mask,
            )                   

            logits = logits.squeeze(-1)                 
            probabilities = torch.sigmoid(logits)                 
            predictions = (probabilities >= threshold).long()                 

        return predictions, probabilities

    def get_trainable_parameters(self):
        """Get trainable parameters for the optimizer.

        Returns:
            Iterator of trainable parameters
        """
        params = []

        params.extend(list(self.other_projection.parameters()))
        params.extend(list(self.other_norm.parameters()))                                   
        params.extend(list(self.post_projection_norm.parameters()))
        params.append(self.other_type_emb)
        params.append(self.separator_embedding)
        params.append(self.cls_token)

        params.extend(list(self.classifier_head.parameters()))

        if not self.freeze_base_model:
            params.extend(list(self.model.parameters()))
        elif self.unfreeze_last_n_layers > 0:
                                                                            
            params.extend([p for p in self.model.parameters() if p.requires_grad])

        return params

    def num_trainable_parameters(self) -> int:
        """Get number of trainable model parameters.

        Returns:
            Number of trainable parameters
        """
        count = 0

        count += sum(
            p.numel() for p in self.other_projection.parameters() if p.requires_grad
        )
        count += sum(
            p.numel() for p in self.other_norm.parameters() if p.requires_grad
        )
        count += sum(
            p.numel() for p in self.post_projection_norm.parameters() if p.requires_grad
        )

        if self.other_type_emb.requires_grad:
            count += self.other_type_emb.numel()
        if self.separator_embedding.requires_grad:
            count += self.separator_embedding.numel()
        if self.cls_token.requires_grad:
            count += self.cls_token.numel()

        count += sum(
            p.numel() for p in self.classifier_head.parameters() if p.requires_grad
        )

        if not self.freeze_base_model or self.unfreeze_last_n_layers > 0:
            count += sum(p.numel() for p in self.model.parameters() if p.requires_grad)

        return count

    def trainable_state_dict(self) -> dict:
        """Return state dict containing only trainable (non-backbone) parameters.

        Excludes self.model (the frozen LLaMA-1B backbone) to keep checkpoints small.
        """
        return {k: v for k, v in self.state_dict().items() if not k.startswith("model.")}

    def load_trainable_state_dict(self, state_dict: dict) -> None:
        """Load a trainable-only state dict produced by trainable_state_dict().

        Uses strict=False so the frozen backbone keys (model.*) are left untouched.
        """
        missing, unexpected = self.load_state_dict(state_dict, strict=False)
        backbone_missing = [k for k in missing if k.startswith("model.")]
        non_backbone_missing = [k for k in missing if not k.startswith("model.")]
        if non_backbone_missing:
            self.logger.warning(f"Missing non-backbone keys: {non_backbone_missing}")
        if unexpected:
            self.logger.warning(f"Unexpected keys in checkpoint: {unexpected}")
        if backbone_missing:
            self.logger.info(
                f"Backbone keys not in checkpoint (expected — backbone loaded separately): "
                f"{len(backbone_missing)} keys"
            )
