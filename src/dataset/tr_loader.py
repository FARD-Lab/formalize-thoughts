from pathlib import Path

import torch

from src.utils.config import ThoughtRepresentation
from src.utils.logging import Logger

class ThoughtRepresentationLoader:
    """
    Generic loader for cached thought representation data.

    Handles loading and caching of generated_ids and thought_vecs based on tr_type.
    """

    def __init__(
        self,
        logger: Logger,
        tr_type: ThoughtRepresentation,
        cached_data_dir: str,
        split_name: str,
        vec_dim: int = 4096,
        think_steps: int = 64,
    ):
        self.logger = logger
        self.tr_type = (
            tr_type
            if isinstance(tr_type, ThoughtRepresentation)
            else ThoughtRepresentation(tr_type)
        )
        self.vec_dim = vec_dim
        self.think_steps = think_steps
        self.generated_texts_cache = {}
        self.thought_vecs_cache = {}

        if not cached_data_dir:
            raise ValueError(
                "cached_data_dir must be provided. "
                "Please run pregenerate_disc_data.py first to generate cached data."
            )

        cached_dir = Path(cached_data_dir)
        gen_ids_file = cached_dir / f"{split_name}_generated_texts.pt"

        if gen_ids_file.exists():
            self.logger.info(f"Loading generated_ids from {gen_ids_file}...")
            self.generated_texts_cache = torch.load(gen_ids_file, map_location="cpu")
            self.logger.info(f"Loaded {len(self.generated_texts_cache)} groups")
        else:
            raise FileNotFoundError(
                f"Cached file not found: {gen_ids_file}\n"
                f"Please run pregenerate_disc_data.py first."
            )

        if self.tr_type == ThoughtRepresentation.RANDOM_VECTOR:
            self.logger.info("RANDOM_VECTOR will be generated, skipping cache load.")
            return

        if self.tr_type == ThoughtRepresentation.LAST_INPUT_HIDDEN_STATE:
            vectors_file = (
                cached_dir
                / f"{split_name}_other_vectors_{ThoughtRepresentation.LAST_INPUT_TOKEN.value}.pt"
            )
        else:
            vectors_file = (
                cached_dir / f"{split_name}_other_vectors_{self.tr_type.value}.pt"
            )

        if vectors_file.exists():
            self.logger.info(
                f"Loading thought_vecs for {self.tr_type.value} from {vectors_file}..."
            )
            self.thought_vecs_cache = torch.load(vectors_file, map_location="cpu")
            self.logger.info(f"Loaded {len(self.thought_vecs_cache)} groups")
        else:
            raise FileNotFoundError(
                f"Cached file not found: {vectors_file}\n"
                f"Please run pregenerate_disc_data.py first."
            )

    def get_group_data(self, g_idx):
        """Get a group's data from in-memory cache."""
        if g_idx not in self.generated_texts_cache:
            raise KeyError(f"Group {g_idx} not found in generated_texts_cache")
        generated_texts = self.generated_texts_cache[g_idx]

        if self.tr_type == ThoughtRepresentation.RANDOM_VECTOR:
            thought_vec = torch.randn(
                (len(generated_texts), 1, self.vec_dim),
                dtype=torch.bfloat16,
            )
            return {
                "thought_vec": thought_vec,
                "generated_texts": generated_texts,
            }

        if g_idx not in self.thought_vecs_cache:
            raise KeyError(f"Group {g_idx} not found in thought_vecs_cache")

        thought_vec = self.thought_vecs_cache[g_idx]

        if self.tr_type == ThoughtRepresentation.LAST_INPUT_HIDDEN_STATE:
            thought_vec = thought_vec[-1:, :]
        elif self.tr_type in [
            ThoughtRepresentation.SOFT_THINKING,
            ThoughtRepresentation.SOFT_THINKING_NOISE,
            ThoughtRepresentation.LATENT_THINKING,
        ]:                        
            thought_vec = thought_vec[: self.think_steps]

        return {
            "thought_vec": thought_vec,
            "generated_texts": generated_texts,
        }
