from typing import Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForCausalLM

from src.model.base import BaseModel
from src.utils.config import ProjectType
from src.utils.logging import Logger

class ThoughtDescriptor(nn.Module, BaseModel):
    """Probe model for estimating CE(X|T) or CE(Y|T) from thought vectors."""

    def __init__(
        self,
        logger: Logger,
        model_name: str = "meta-llama/Llama-3.2-1B",
        vector_dim: int = 4096,
        device: str = "cuda",
        load_in_8bit: bool = False,
        load_in_4bit: bool = False,
        torch_dtype: Union[str, torch.dtype] = torch.bfloat16,
        trust_remote_code: bool = True,
        max_memory: Optional[Dict[str, str]] = None,
        freeze_base_model: bool = True,
        dropout_rate: float = 0.1,
        projection_type: ProjectType = ProjectType.SHARED,
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

        self.vec_dim = vector_dim
        self.freeze_base_model = freeze_base_model
        self.dropout_rate = dropout_rate
        if isinstance(projection_type, str):
            projection_type = ProjectType(projection_type.lower())
        self.projection_type = projection_type
                                             
        self.model_embedding_dim: Optional[int] = None
        self.input_norm: Optional[nn.LayerNorm] = None
        self.shared_projection: Optional[nn.Linear] = None
        self.vec_emb: Optional[nn.Parameter] = None

    def train(self, mode: bool = True):
                                                        
        nn.Module.train(self, mode)
        return self

    def eval(self):
                                                       
        nn.Module.eval(self)
        return self

    def load(self) -> None:
        """Load model and initialize projection matrices and classifier."""
        if self.projection_type != ProjectType.SHARED:
            raise NotImplementedError(
                f"projection_type='{self.projection_type.value}' is not implemented in "
                "ThoughtDescriptor. Use 'shared'."
            )

        self.logger.info(f"Loading base model '{self.model_name}'")

        self.model = (
            AutoModelForCausalLM.from_pretrained(
                self.model_name,
                **self.model_kwargs,
            )
            .to(self.device)
            .eval()
        )

        self.model.config.use_cache = False
        self.model.gradient_checkpointing_enable()
        self.model.enable_input_require_grads()

        dtype = self.torch_dtype
        self.logger.info(f"Model dtype: {dtype}")

        self.model_embedding_dim = self.model.config.hidden_size
        self.logger.info(f"Model embedding dimension: {self.model_embedding_dim}")

        if self.freeze_base_model:
            self.logger.info("Freezing base model parameters")
            for param in self.model.parameters():
                param.requires_grad = False

        self.logger.info("Initializing layers")
        self.input_norm = nn.LayerNorm(
            self.vec_dim, elementwise_affine=False, dtype=dtype
        )

        self.shared_projection = nn.Linear(
            self.vec_dim, self.model_embedding_dim, bias=True, dtype=dtype
        )
        self.post_projection_norm = nn.LayerNorm(self.model_embedding_dim, dtype=dtype)

        self.vec_emb = nn.Parameter(torch.empty(self.model_embedding_dim, dtype=dtype))
        nn.init.normal_(self.vec_emb, std=0.02)

        target_device = self.device
        components = [
            self.input_norm,
            self.shared_projection,
            self.post_projection_norm,
        ]
        for c in components:
            c.to(target_device)

        self.vec_emb.data = self.vec_emb.data.to(target_device)

        self.logger.info("Discriminator loaded successfully")

    def forward(
        self,
        vecs: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        target_token_ids: Optional[torch.Tensor] = None,
        target_attention_mask: Optional[torch.Tensor] = None,
        prefix_token_ids: Optional[torch.Tensor] = None,
        prefix_attention_mask: Optional[torch.Tensor] = None,
        use_thought: bool = True,
    ) -> torch.Tensor:
        if self.model is None:
            raise RuntimeError("Model not loaded. Call load() first.")

        if target_token_ids is not None:
            loss, logits = self.compute_loss(
                vecs=vecs,
                target_token_ids=target_token_ids,
                attention_mask=attention_mask,
                target_attention_mask=target_attention_mask,
                prefix_token_ids=prefix_token_ids,
                prefix_attention_mask=prefix_attention_mask,
                use_thought=use_thought,
            )
                                                                                  
            from transformers.modeling_outputs import CausalLMOutputWithPast
            return CausalLMOutputWithPast(loss=loss, logits=logits)

        device = vecs.device
        batch_size = vecs.size(0)

        vecs = self.input_norm(vecs)
        projected_vecs = self.shared_projection(vecs)
        projected_vecs = self.post_projection_norm(projected_vecs)
        projected_vecs = projected_vecs + self.vec_emb.to(device)

        if attention_mask is None:
            attention_mask = torch.ones(
                batch_size, projected_vecs.size(1), dtype=torch.long, device=device
            )

        return self.model(
            inputs_embeds=projected_vecs,
            attention_mask=attention_mask,
        )

    def generate(
        self,
        vecs: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        max_new_tokens: int = 100,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        top_k: Optional[int] = None,
        repetition_penalty: Optional[float] = None,
        no_repeat_ngram_size: Optional[int] = None,
        do_sample: Optional[bool] = None,
        num_return_sequences: Optional[int] = None,
        eos_token_id: Optional[Union[int, List[int]]] = None,
        pad_token_id: Optional[int] = None,
        **kwargs,
    ) -> torch.Tensor:
        """Generate tokens autoregressively using KV cache.

        Uses HuggingFace's model.generate() for efficient generation with built-in
        sampling strategies (top-k, top-p, beam search, etc.)

        Args:
            vecs: Input vectors [batch_size, seq_len, vec_dim]
            attention_mask: Attention mask for input [batch_size, seq_len]
            max_new_tokens: Maximum number of tokens to generate
            temperature: Sampling temperature
            top_p: Nucleus sampling probability threshold
            top_k: Top-k sampling parameter
            repetition_penalty: Repetition penalty
            no_repeat_ngram_size: Size of ngrams to avoid repeating
            do_sample: Whether to sample or use greedy decoding
            num_return_sequences: Number of sequences to generate
            eos_token_id: End of sequence token ID(s)
            pad_token_id: Padding token ID
            **kwargs: Additional generation arguments passed to model.generate()

        Returns:
            generated_tokens: Only newly generated tokens [batch_size, num_generated]
        """
        if self.model is None:
            raise RuntimeError("Model not loaded. Call load() first.")

        device = vecs.device
        batch_size = vecs.size(0)

        vecs = self.input_norm(vecs)
        projected_vecs = self.shared_projection(vecs)
        projected_vecs = self.post_projection_norm(projected_vecs)
        projected_vecs = projected_vecs + self.vec_emb.to(device)

        if attention_mask is None:
            attention_mask = torch.ones(
                batch_size, projected_vecs.size(1), dtype=torch.long, device=device
            )

        gen_kwargs = {
            "max_new_tokens": max_new_tokens,
            "return_dict_in_generate": True,
            "output_scores": False,
            "use_cache": True,                                           
            **kwargs,
        }

        if eos_token_id is None:
            eos_token_id = self.model.config.eos_token_id
        if pad_token_id is None:
            pad_token_id = self.model.config.pad_token_id or eos_token_id

        gen_kwargs["eos_token_id"] = eos_token_id
        gen_kwargs["pad_token_id"] = pad_token_id

        if temperature is not None:
            gen_kwargs["temperature"] = temperature
        if top_p is not None:
            gen_kwargs["top_p"] = top_p
        if top_k is not None:
            gen_kwargs["top_k"] = top_k
        if repetition_penalty is not None:
            gen_kwargs["repetition_penalty"] = repetition_penalty
        if no_repeat_ngram_size is not None:
            gen_kwargs["no_repeat_ngram_size"] = no_repeat_ngram_size
        if do_sample is not None:
            gen_kwargs["do_sample"] = do_sample
        if num_return_sequences is not None:
            gen_kwargs["num_return_sequences"] = num_return_sequences

        with torch.no_grad():
            outputs = self.model.generate(
                inputs_embeds=projected_vecs,
                attention_mask=attention_mask,
                **gen_kwargs,
            )

        return outputs.sequences

    def compute_loss(
        self,
        vecs: torch.Tensor,
        target_token_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        target_attention_mask: Optional[torch.Tensor] = None,
        prefix_token_ids: Optional[torch.Tensor] = None,
        prefix_attention_mask: Optional[torch.Tensor] = None,
        use_thought: bool = True,
        tile_to_length: Optional[int] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute cross entropy loss for teacher-forced generation.

        Args:
            vecs: Input vectors [batch_size, seq_len, vec_dim] (ignored when use_thought=False).
            target_token_ids: Target token IDs to predict [batch_size, target_len].
            attention_mask: Attention mask for input vectors [batch_size, seq_len].
            target_attention_mask: Attention mask for targets [batch_size, target_len].
            prefix_token_ids: Optional token-id prefix [batch_size, prefix_len].
                When supplied, it is embedded via the model's input embedding
                table (no learned projection) and prepended after the projected
                T (or used alone when use_thought=False).
            prefix_attention_mask: Attention mask for the prefix.
            use_thought: When False the projection path is skipped entirely,
                so vecs/attention_mask/vec_emb do not enter the graph. Used by
                the CE(X|Y) baseline of the IB-residual minimality estimator.

        Returns:
            Tuple of (loss, logits):
                - loss: Cross entropy loss (scalar)
                - logits: Model logits [batch_size, target_len, vocab_size]
        """
        if self.model is None:
            raise RuntimeError("Model not loaded. Call load() first.")

        device = vecs.device
        batch_size = vecs.size(0)
        target_len = target_token_ids.size(1)

        segments_embeds: list[torch.Tensor] = []
        segments_masks: list[torch.Tensor] = []

        if use_thought:
            if tile_to_length is not None:
                L_target = int(tile_to_length)
                if L_target <= 0:
                    raise ValueError(f"tile_to_length must be positive, got {L_target}")
                L_cur = vecs.size(1)
                n_rep = (L_target + L_cur - 1) // L_cur
                vecs = vecs.repeat(1, n_rep, 1)[:, :L_target, :]
                if attention_mask is not None:
                    attention_mask = attention_mask.repeat(1, n_rep)[:, :L_target]
            proj = self.input_norm(vecs)
            proj = self.shared_projection(proj)
            proj = self.post_projection_norm(proj)
            proj = proj + self.vec_emb.to(device)
            if attention_mask is None:
                attention_mask = torch.ones(
                    batch_size, proj.size(1), dtype=torch.long, device=device
                )
            segments_embeds.append(proj)
            segments_masks.append(attention_mask)

        if prefix_token_ids is not None:
            prefix_embeds = self.model.get_input_embeddings()(prefix_token_ids)
            if prefix_attention_mask is None:
                prefix_attention_mask = torch.ones(
                    batch_size, prefix_token_ids.size(1), dtype=torch.long, device=device
                )
            segments_embeds.append(prefix_embeds)
            segments_masks.append(prefix_attention_mask)

        target_embeds = self.model.get_input_embeddings()(target_token_ids)
        if target_attention_mask is None:
            target_attention_mask = torch.ones(
                batch_size, target_len, dtype=torch.long, device=device
            )
        segments_embeds.append(target_embeds)
        segments_masks.append(target_attention_mask)

        if len(segments_embeds) < 2:
            raise ValueError(
                "compute_loss requires at least one of (use_thought=True, prefix_token_ids)"
                " in addition to the target tokens."
            )

        full_embeds = torch.cat(segments_embeds, dim=1)
        full_attention_mask = torch.cat(segments_masks, dim=1)

        outputs = self.model(
            inputs_embeds=full_embeds,
            attention_mask=full_attention_mask,
        )

        input_len = sum(s.size(1) for s in segments_embeds[:-1])
        logits = outputs.logits[
            :, input_len - 1 : input_len + target_len - 1, :
        ]                                        

        logits_flat = logits.reshape(
            -1, logits.size(-1)
        )                                         
        targets_flat = target_token_ids.reshape(-1)                             

        if target_attention_mask is not None:
                                                      
            mask_flat = target_attention_mask.reshape(
                -1
            ).bool()                             

            if mask_flat.sum() > 0:
                loss = F.cross_entropy(
                    logits_flat[mask_flat], targets_flat[mask_flat], reduction="mean"
                )
            else:
                loss = torch.tensor(0.0, device=device, requires_grad=True)
        else:
            loss = F.cross_entropy(logits_flat, targets_flat, reduction="mean")

        return loss, logits

    def to(self, *args, **kwargs):
        nn.Module.to(self, *args, **kwargs)
                                                          
        if args and isinstance(args[0], (str, torch.device)):
            self.device = str(args[0])
        elif "device" in kwargs:
            self.device = str(kwargs["device"])
        return self

    def get_trainable_parameters(self):
        """Get trainable parameters for the optimizer.

        Returns:
            Iterator of trainable parameters
        """
        params = []

        params.extend(list(self.shared_projection.parameters()))
        params.extend(list(self.post_projection_norm.parameters()))
                             
        params.append(self.vec_emb)

        if not self.freeze_base_model:
            params.extend(list(self.model.parameters()))

        return params

    def num_trainable_parameters(self) -> int:
        """Get number of trainable model parameters.

        Returns:
            Number of trainable parameters
        """
        return sum(
            p.numel() for p in self.get_trainable_parameters() if p.requires_grad
        )
