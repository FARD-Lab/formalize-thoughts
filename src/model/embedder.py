"""Embedder model wrapper for generating text embeddings."""

from typing import Dict, List, Optional, Union

import torch
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer

from src.model.base import BaseModel
from src.utils.config import EmbedderConfig
from src.utils.logging import Logger

class Embedder(BaseModel):
    """Wrapper for embedder models."""

    def __init__(
        self,
        logger: Logger,
        model_name: str,
        device: str = "cuda",
        load_in_8bit: bool = False,
        load_in_4bit: bool = False,
        torch_dtype: str | torch.dtype = "auto",
        trust_remote_code: bool = False,
        max_memory: Optional[Dict[str, str]] = None,
        seed: Optional[int] = None,
        **kwargs,
    ):
        """Initialize embedder.

        Args:
            **kwargs: Additional arguments passed to BaseModel
        """
        super().__init__(
            logger=logger,
            model_name=model_name,
            device=device,
            load_in_8bit=load_in_8bit,
            load_in_4bit=load_in_4bit,
            torch_dtype=torch_dtype,
            trust_remote_code=trust_remote_code,
            max_memory=max_memory,
            seed=seed,
            **kwargs,
        )

    @staticmethod
    def from_config(config: EmbedderConfig, logger: Logger) -> "Embedder":
        return Embedder(
            logger=logger,
            model_name=config.name,
            device=config.device,
            load_in_8bit=config.load_in_8bit,
            load_in_4bit=config.load_in_4bit,
            torch_dtype=config.torch_dtype,
            trust_remote_code=config.trust_remote_code,
            max_memory=config.max_memory,
        )

    def load(self) -> None:
        """Load model and tokenizer."""
        self.model = AutoModel.from_pretrained(
            self.model_name,
            **self.model_kwargs,
        ).eval()

        if self.model_kwargs.get("device_map") != "auto":
            if self.device:
                self.model = self.model.to(self.device)

        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name,
            trust_remote_code=self.model_kwargs.get("trust_remote_code", False),
            padding_side="left",
        )

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

    def _pool_embeddings(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Pool embeddings using specified strategy.

        Args:
            hidden_states: Hidden states from model [batch, seq_len, hidden_dim]
            attention_mask: Attention mask [batch, seq_len]

        Returns:
            Pooled embeddings [batch, hidden_dim]
        """
                                                              
        hidden_states = hidden_states.to(torch.float32)
        hidden_states_masked = hidden_states.masked_fill(
            ~attention_mask[..., None].bool(), 0.0
        )
        embeddings = (
            hidden_states_masked.sum(dim=1) / attention_mask.sum(dim=1)[..., None]
        )
        return F.normalize(embeddings, dim=-1)

    def embed(
        self,
        text: Union[str, List[str]],
        max_length: int = 4096,
        **kwargs,
    ) -> torch.Tensor:
        """Generate embeddings for text.

        Args:
            text: Input text(s)
            max_length: Maximum token length for inputs

        Returns:
            Embeddings tensor [num_texts, hidden_dim]
        """
        if self.model is None or self.tokenizer is None:
            raise RuntimeError("Model not loaded. Call load() first.")

        if isinstance(text, str):
            text = [text]

        inputs = self.tokenizer(
            text,
            max_length=max_length,
            padding=True,
            truncation=True,
            return_tensors="pt",
        ).to(self.device)
        attention_mask = inputs["attention_mask"]

        with torch.no_grad():
            outputs = self.model(**inputs, **kwargs)

        return self._pool_embeddings(outputs.last_hidden_state, attention_mask)

if __name__ == "__main__":
    from src.utils import set_seed_all

    set_seed_all(42)

    logger = Logger("EmbedderTest")
    embedder = Embedder(
        logger=logger,
        model_name="nvidia/llama-embed-nemotron-8b",
        device="cuda" if torch.cuda.is_available() else "cpu",
        load_in_8bit=False,
        load_in_4bit=False,
        torch_dtype="auto",
        trust_remote_code=True,
        max_memory=None,                                       
    )
    embedder.load()

    doc_embeddings = embedder.embed(
        [
            "The quick brown fox jumps over the lazy dog. This classic sentence is often used to test typing and fonts.",
            "Quantum chromodynamics describes the strong interaction, a fundamental force governing quark behavior in particle physics.",
            "A fast brown fox leaps over a sleepy dog. This well-known pangram appears in typing tests and font demos.",
        ]
    )
               
    doc_embeddings = F.normalize(doc_embeddings, dim=-1)
                                   
    cosine_similarities = F.cosine_similarity(
        doc_embeddings.unsqueeze(1), doc_embeddings.unsqueeze(0), dim=-1
    )
    print("Cosine similarities:", cosine_similarities)
