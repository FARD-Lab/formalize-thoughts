"""Pre-generate discriminator dataset for faster training."""

import json
from pathlib import Path
from typing import Dict, Optional, Set

import hydra
import torch
from omegaconf import OmegaConf
from tqdm import tqdm

from src.dataset.disc_loader import ExampleLoader
from src.model.embedder import Embedder
from src.model.llm import BaseLLM
from src.utils import set_seed_all
from src.utils.config import (
    DiscDataConfig,
    ThoughtRepresentation,
    register_configs,
)
from src.utils.logging import Logger

register_configs()

EMBEDDER_TR_TYPES = {
    ThoughtRepresentation.EMBEDDING_POOLING,
    ThoughtRepresentation.EMBEDDING_NO_POOLING,
    ThoughtRepresentation.EMBEDDING_ALL,
    ThoughtRepresentation.INPUT_EMBEDDING,
}
LLM_TR_TYPES = {
    ThoughtRepresentation.SOFT_THINKING,
    ThoughtRepresentation.SOFT_THINKING_NOISE,
    ThoughtRepresentation.LATENT_THINKING,
}
FILE_TR_TYPES = {
    ThoughtRepresentation.LAST_INPUT_TOKEN,
}
                                                                        
class DataPreGenerator:
    """Pre-generates and caches discriminator dataset."""

    def __init__(self, config: DiscDataConfig, logger: Logger):
        self.config = config
        self.logger = logger

        self.loader = ExampleLoader(
            directory=self.config.llm_data_output_dir,
            num_return_sequences=self.config.num_return_sequences,
            shard_size=self.config.shard_size,
        )

        self.base_llm: Optional[BaseLLM] = None
        self.embedder: Optional[Embedder] = None

        self.disc_index_dir = Path(self.config.disc_index_output_dir)
        self.output_dir = Path(self.config.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.think_steps = self.config.think_steps
        self.source_hidden_size = self.config.base_llm.hidden_size
        self.checkpoint_interval = 100                                  

        if self.config.tr_types is None:
            self.tr_types_to_process = set(ThoughtRepresentation)
        else:
            self.tr_types_to_process = {
                ThoughtRepresentation(tr_type) for tr_type in self.config.tr_types
            }
        self.logger.info(
            f"Processing thought representations: {[tr.value for tr in self.tr_types_to_process]}"
        )

    def _compute_thought_vec(
        self, example: dict, tr_type: ThoughtRepresentation, add_noise: bool = True
    ) -> torch.Tensor:
        """Compute thought_vec for a specific tr_type."""
        generated_texts = example["generated_texts"]

        if tr_type == ThoughtRepresentation.LAST_INPUT_TOKEN:
            first_hs_path = Path(example["first_hs_path"])
            if not first_hs_path.exists():
                raise FileNotFoundError(f"first_hs not found: {first_hs_path}")
            thought_vec = torch.load(first_hs_path, map_location="cpu")

        elif tr_type == ThoughtRepresentation.LAST_INPUT_HIDDEN_STATE:
            raise ValueError(
                "LAST_INPUT_HIDDEN_STATE should share the same file as LAST_INPUT_TOKEN"
            )

        elif tr_type in [
            ThoughtRepresentation.EMBEDDING_POOLING,
            ThoughtRepresentation.EMBEDDING_NO_POOLING,
            ThoughtRepresentation.EMBEDDING_ALL,
        ]:
            with torch.no_grad():
                embeddings = self.embedder.embed(generated_texts).to(
                    dtype=torch.bfloat16
                )
            if tr_type == ThoughtRepresentation.EMBEDDING_POOLING:
                thought_vec = torch.nn.functional.normalize(
                    torch.mean(embeddings, dim=0, keepdim=True), p=2, dim=1
                )
            elif tr_type == ThoughtRepresentation.EMBEDDING_NO_POOLING:
                thought_vec = embeddings.unsqueeze(1)
            elif tr_type == ThoughtRepresentation.EMBEDDING_ALL:
                thought_vec = embeddings

        elif tr_type == ThoughtRepresentation.SOFT_THINKING:
            thought_vec = self.base_llm.get_soft_token(
                text=example["input_text"],
                instruction=example["instruction"],
                add_noise=add_noise,
                steps=self.think_steps,
            )[0]                              

        elif tr_type == ThoughtRepresentation.LATENT_THINKING:
            thought_vec = self.base_llm.get_latent_thinking(
                text=example["input_text"],
                instruction=example["instruction"],
                steps=self.think_steps,
            )[0]                              

        elif tr_type == ThoughtRepresentation.INPUT_EMBEDDING:
            input_text = example["input_text"]
            with torch.no_grad():
                thought_vec = self.embedder.embed([input_text]).to(
                    dtype=torch.bfloat16
                )

        elif tr_type == ThoughtRepresentation.RANDOM_VECTOR:
            thought_vec = torch.randn(
                (len(generated_texts), 1, self.source_hidden_size),
                dtype=torch.bfloat16,
            )
        else:
            raise ValueError(f"Unsupported representation: {tr_type.value}")

        return thought_vec

    def _get_checkpoint_path(self, split_name: str) -> Path:
        """Get the checkpoint file path for a split."""
        return self.output_dir / f"{split_name}_checkpoint.pt"

    def _save_checkpoint(
        self,
        split_name: str,
        processed_groups: set,
        generated_texts_cache: dict,
        thought_vecs_caches: dict,
    ):
        """Save checkpoint to disk."""
        checkpoint_path = self._get_checkpoint_path(split_name)
        checkpoint = {
            "processed_groups": processed_groups,
            "generated_texts_cache": generated_texts_cache,
            "thought_vecs_caches": thought_vecs_caches,
        }
        torch.save(checkpoint, checkpoint_path)
        self.logger.debug(
            f"Saved checkpoint with {len(processed_groups)} processed groups"
        )

    def _load_checkpoint(
        self, split_name: str, tr_types: Set[ThoughtRepresentation]
    ) -> tuple:
        """Load checkpoint if it exists.

        Returns:
            tuple: (processed_groups, generated_texts_cache, thought_vecs_caches)
                   or (set(), {}, {tr_type: {} for ...}) if no checkpoint exists
        """
        checkpoint_path = self._get_checkpoint_path(split_name)
        if checkpoint_path.exists():
            self.logger.info(f"Found checkpoint at {checkpoint_path}, resuming...")
                                                                                      
            checkpoint = torch.load(
                checkpoint_path, map_location="cpu", weights_only=False
            )
            processed_groups = checkpoint["processed_groups"]
            generated_texts_cache = checkpoint["generated_texts_cache"]
            loaded_caches = checkpoint["thought_vecs_caches"]

            thought_vecs_caches = {
                tr_type: loaded_caches.get(tr_type, {}) for tr_type in tr_types
            }

            self.logger.info(
                f"Resuming from checkpoint with {len(processed_groups)} already processed groups"
            )
            return processed_groups, generated_texts_cache, thought_vecs_caches
        else:
            return set(), {}, {tr_type: {} for tr_type in tr_types}

    def _delete_checkpoint(self, split_name: str):
        """Delete checkpoint file after successful completion."""
        checkpoint_path = self._get_checkpoint_path(split_name)
        if checkpoint_path.exists():
            checkpoint_path.unlink()
            self.logger.info(f"Deleted checkpoint file: {checkpoint_path}")

    def _process_split(
        self,
        split_name: str,
        pairs_filename: str,
        tr_types: Set[ThoughtRepresentation],
        save_generated_texts: bool = True,
    ) -> tuple:
        """Process a single split for the given TR types.

        Returns:
            tuple: (generated_texts_cache, thought_vecs_caches)
        """
        pairs_file = self.disc_index_dir / pairs_filename

        self.logger.info(
            f"Processing {split_name} split from {pairs_file} "
            f"for TR types: {[t.value for t in tr_types]}..."
        )

        pairs = []
        with open(pairs_file, "r") as f:
            for line in f:
                if line.strip():
                    pairs.append(json.loads(line.strip()))

        unique_groups = set()
        for pair in pairs:
            unique_groups.add(pair["group_1_idx"])
            unique_groups.add(pair["group_2_idx"])

        self.logger.info(f"Found {len(unique_groups)} unique groups")

        (
            processed_groups,
            generated_texts_cache,
            thought_vecs_caches,
        ) = self._load_checkpoint(split_name, tr_types)

        remaining_groups = [g for g in unique_groups if g not in processed_groups]
        self.logger.info(
            f"Processing {len(remaining_groups)} remaining groups "
            f"(already processed: {len(processed_groups)})"
        )

        groups_since_checkpoint = 0

        combinable_types = {
            ThoughtRepresentation.SOFT_THINKING,
            ThoughtRepresentation.SOFT_THINKING_NOISE,
            ThoughtRepresentation.LATENT_THINKING,
        }
        use_combined = combinable_types.issubset(tr_types) and self.base_llm is not None
        remaining_tr_types = tr_types - combinable_types if use_combined else tr_types

        if use_combined:
            self.logger.info(
                "Using combined thinking (shared prefill + batched decode) "
                "for soft_thinking, soft_thinking_noise, and latent_thinking"
            )

        self.logger.info(
            f"Loading groups and computing TR types for {split_name}..."
        )
        for group_idx in tqdm(
            remaining_groups, desc=f"Processing groups for {split_name}"
        ):
            try:
                example = self.loader.load_example(group_idx)
                generated_texts = example["generated_texts"]

                if save_generated_texts:
                    generated_texts_cache[group_idx] = generated_texts

                if use_combined:
                    soft, noise, latent = self.base_llm.get_combined_thinking(
                        text=example["input_text"],
                        instruction=example["instruction"],
                        steps=self.think_steps,
                    )
                    thought_vecs_caches[ThoughtRepresentation.SOFT_THINKING][group_idx] = soft[0]
                    thought_vecs_caches[ThoughtRepresentation.SOFT_THINKING_NOISE][group_idx] = noise[0]
                    thought_vecs_caches[ThoughtRepresentation.LATENT_THINKING][group_idx] = latent[0]

                for tr_type in remaining_tr_types:
                    if tr_type in [
                        ThoughtRepresentation.LAST_INPUT_HIDDEN_STATE,
                        ThoughtRepresentation.RANDOM_VECTOR,
                    ]:
                        continue
                    elif tr_type == ThoughtRepresentation.SOFT_THINKING:
                        thought_vec = self._compute_thought_vec(
                            example, tr_type, add_noise=False
                        )
                        thought_vecs_caches[tr_type][group_idx] = thought_vec
                    elif tr_type == ThoughtRepresentation.SOFT_THINKING_NOISE:
                        thought_vec = self._compute_thought_vec(
                            example,
                            ThoughtRepresentation.SOFT_THINKING,
                            add_noise=True,
                        )
                        thought_vecs_caches[tr_type][group_idx] = thought_vec
                    else:
                        thought_vec = self._compute_thought_vec(example, tr_type)
                        thought_vecs_caches[tr_type][group_idx] = thought_vec

                processed_groups.add(group_idx)
                groups_since_checkpoint += 1

                if groups_since_checkpoint >= self.checkpoint_interval:
                    self._save_checkpoint(
                        split_name,
                        processed_groups,
                        generated_texts_cache,
                        thought_vecs_caches,
                    )
                    groups_since_checkpoint = 0

            except Exception as e:
                self.logger.warning(
                    f"Error processing group {group_idx}, saving checkpoint before exit..."
                )
                self._save_checkpoint(
                    split_name,
                    processed_groups,
                    generated_texts_cache,
                    thought_vecs_caches,
                )
                self.logger.exception(f"Error processing group {group_idx}: {e}")
                raise e

        if groups_since_checkpoint > 0:
            self._save_checkpoint(
                split_name,
                processed_groups,
                generated_texts_cache,
                thought_vecs_caches,
            )

        return generated_texts_cache, thought_vecs_caches

    def _save_split_outputs(
        self,
        split_name: str,
        generated_texts_cache: Dict,
        thought_vecs_caches: Dict,
        save_generated_texts: bool,
    ):
        """Save output files for a split."""
        if save_generated_texts and generated_texts_cache:
            gen_texts_file = self.output_dir / f"{split_name}_generated_texts.pt"
            self.logger.info(
                f"Saving generated_texts with {len(generated_texts_cache)} groups "
                f"to {gen_texts_file}..."
            )
            torch.save(generated_texts_cache, gen_texts_file)
            gen_texts_size_mb = gen_texts_file.stat().st_size / (1024 * 1024)
            self.logger.info(f"Generated texts file size: {gen_texts_size_mb:.2f} MB")

        for tr_type, thought_vecs_cache in thought_vecs_caches.items():
            if not thought_vecs_cache:
                self.logger.info(f"Skipping empty cache for {tr_type.value}")
                continue

            vectors_file = (
                self.output_dir / f"{split_name}_other_vectors_{tr_type.value}.pt"
            )
            self.logger.info(
                f"Saving thought_vecs for {tr_type.value} with "
                f"{len(thought_vecs_cache)} groups to {vectors_file}..."
            )
            torch.save(thought_vecs_cache, vectors_file)
            vectors_size_mb = vectors_file.stat().st_size / (1024 * 1024)
            self.logger.info(
                f"{tr_type.value} vectors file size: {vectors_size_mb:.2f} MB"
            )

    def _run_pass(
        self,
        tr_types: Set[ThoughtRepresentation],
        save_generated_texts: bool,
    ):
        """Run one pass over all splits for the given TR types."""
        splits = {
            "train": "train_discriminator_pairs.jsonl",
            "val": "val_discriminator_pairs.jsonl",
            "test": "test_discriminator_pairs.jsonl",
        }

        for split_name, pairs_filename in splits.items():
            pairs_file = self.disc_index_dir / pairs_filename
            if not pairs_file.exists():
                self.logger.warning(
                    f"Pairs file not found: {pairs_file}, skipping {split_name}"
                )
                continue

            generated_texts_cache, thought_vecs_caches = self._process_split(
                split_name, pairs_filename, tr_types, save_generated_texts
            )

            self._save_split_outputs(
                split_name,
                generated_texts_cache,
                thought_vecs_caches,
                save_generated_texts,
            )

            self._delete_checkpoint(split_name)

            self.logger.info(f"Saved {split_name} split successfully")

    def _get_expected_groups(self, split_name: str) -> Optional[int]:
        """Get the expected number of unique groups for a split from the pairs file."""
        pairs_file = self.disc_index_dir / f"{split_name}_discriminator_pairs.jsonl"
        if not pairs_file.exists():
            return None
        unique_groups = set()
        with open(pairs_file, "r") as f:
            for line in f:
                if line.strip():
                    pair = json.loads(line.strip())
                    unique_groups.add(pair["group_1_idx"])
                    unique_groups.add(pair["group_2_idx"])
        return len(unique_groups)

    def _pass_outputs_complete(
        self, tr_types: Set[ThoughtRepresentation], check_generated_texts: bool
    ) -> bool:
        """Check if all output files for a pass exist with the correct group count.

        Loads each output file and verifies it contains the expected number of
        groups (derived from the pairs file). This prevents skipping a pass
        whose outputs are incomplete due to a prior interruption.
        """
        splits = ["train", "val", "test"]
        for split_name in splits:
            expected = self._get_expected_groups(split_name)
            if expected is None:
                continue

            if check_generated_texts:
                gen_file = self.output_dir / f"{split_name}_generated_texts.pt"
                if not gen_file.exists():
                    return False
                data = torch.load(gen_file, map_location="cpu")
                if len(data) != expected:
                    self.logger.info(
                        f"{gen_file.name}: {len(data)}/{expected} groups, incomplete"
                    )
                    return False

            for tr_type in tr_types:
                if tr_type in [
                    ThoughtRepresentation.LAST_INPUT_HIDDEN_STATE,
                    ThoughtRepresentation.RANDOM_VECTOR,
                ]:
                    continue
                vec_file = (
                    self.output_dir
                    / f"{split_name}_other_vectors_{tr_type.value}.pt"
                )
                if not vec_file.exists():
                    return False
                data = torch.load(vec_file, map_location="cpu")
                if len(data) != expected:
                    self.logger.info(
                        f"{vec_file.name}: {len(data)}/{expected} groups, incomplete"
                    )
                    return False
        return True

    def run(self):
        """Run the two-pass pre-generation pipeline.

        Pass 1: Load embedder, compute embedding-based TR types, unload.
        Pass 2: Load LLM, compute thinking-based TR types + file-based, unload.

        Each model gets full GPU memory since only one is loaded at a time.
        On resubmit, completed passes are skipped (detected by output file
        existence) to avoid re-computation and checkpoint cross-contamination.
        """
                                                         
        embedder_types = self.tr_types_to_process & EMBEDDER_TR_TYPES
        llm_types = self.tr_types_to_process & LLM_TR_TYPES
        file_types = self.tr_types_to_process & FILE_TR_TYPES

        need_embedder = bool(embedder_types)
        need_llm = bool(llm_types)
        need_files = bool(file_types)

        if need_embedder:
            if self._pass_outputs_complete(embedder_types, check_generated_texts=True):
                self.logger.info(
                    "=== Pass 1: Embedder phase === SKIPPED (outputs already exist)"
                )
            else:
                self.logger.info(
                    "=== Pass 1: Embedder phase ==="
                    f" TR types: {[t.value for t in embedder_types]}"
                )
                self.embedder = Embedder.from_config(
                    config=self.config.embedder,
                    logger=self.logger,
                )
                self.embedder.load()
                self.embedder.eval()

                self._run_pass(embedder_types, save_generated_texts=True)

                del self.embedder
                self.embedder = None
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                self.logger.info("Embedder unloaded, CUDA cache cleared.")
        else:
            self.logger.info("No embedding TR types requested, skipping embedder pass.")

        llm_pass_types = llm_types | file_types
        if llm_pass_types:
            if self._pass_outputs_complete(
                llm_pass_types, check_generated_texts=not need_embedder
            ):
                self.logger.info(
                    "=== Pass 2: LLM phase === SKIPPED (outputs already exist)"
                )
            else:
                self.logger.info(
                    "=== Pass 2: LLM phase ==="
                    f" TR types: {[t.value for t in llm_pass_types]}"
                )
                self.base_llm = BaseLLM.from_config(
                    config=self.config.base_llm,
                    logger=self.logger,
                )
                self.base_llm.load()
                self.base_llm.eval()

                self._run_pass(llm_pass_types, save_generated_texts=not need_embedder)

                del self.base_llm
                self.base_llm = None
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                self.logger.info("LLM unloaded, CUDA cache cleared.")
        else:
            self.logger.info("No LLM TR types requested, skipping LLM pass.")

        self.logger.info("Pre-generation completed successfully!")

@hydra.main(version_base=None, config_path="../configs", config_name="disc_data")
def main(cfg: DiscDataConfig) -> None:
    """Main function for pre-generating discriminator data."""
                  
    logger = Logger.from_config(cfg.logging)

    logger.info("Pre-generation Configuration:")
    logger.info(OmegaConf.to_yaml(cfg))

    if cfg.seed is not None:
        set_seed_all(cfg.seed)
        logger.info(f"Set random seed to {cfg.seed}")

    generator = DataPreGenerator(config=cfg, logger=logger)
    generator.run()

if __name__ == "__main__":
    main()
