"""Discriminator dataset with support for pre-generated cached data."""

import json
from pathlib import Path
from typing import Dict

import torch
from torch.utils.data import Dataset

from src.dataset.tr_loader import ThoughtRepresentationLoader
from src.utils.config import ThoughtRepresentation
from src.utils.logging import Logger

class DiscriminatorDataset(Dataset):
    """
    Dataset that loads from pre-generated cached data.

    Requires separate cached files for generated_ids (shared) and thought_vecs (per tr_type).
    Run pregenerate_disc_data.py first to generate the required cache files.
    """

    def __init__(
        self,
        logger: Logger,
        pairs_file: str,
        tr_type: ThoughtRepresentation,
        cached_data_dir: str,
        vec_dim: int = 4096,
        think_steps: int = 64,
        expand_dim: int = 128,
    ):
        self.logger = logger
        self.expand_dim = expand_dim
        self.pairs = []

        tr_type_normalized = (
            tr_type
            if isinstance(tr_type, ThoughtRepresentation)
            else ThoughtRepresentation(tr_type)
        )

        path = Path(pairs_file)
        if not path.exists():
            raise FileNotFoundError(f"Pairs file not found: {pairs_file}")

        self.logger.info(f"Thought representation type: {tr_type_normalized.value}")

        self.logger.info(f"Loading pairs from {pairs_file}...")
        with open(path, "r") as f:
            for line in f:
                if line.strip():
                    self.pairs.append(json.loads(line.strip()))

        pairs_filename = Path(pairs_file).name
        if "train" in pairs_filename:
            split_name = "train"
        elif "val" in pairs_filename:
            split_name = "val"
        elif "test" in pairs_filename:
            split_name = "test"
        else:
            raise ValueError(f"Cannot determine split from pairs_file: {pairs_file}")

        self.tr_loader = ThoughtRepresentationLoader(
            logger=logger,
            tr_type=tr_type_normalized,
            cached_data_dir=cached_data_dir,
            split_name=split_name,
            vec_dim=vec_dim,
            think_steps=think_steps,
        )

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor | float]:
                                           
        pair = self.pairs[idx]
        group_1_idx = pair["group_1_idx"]
        group_2_idx = pair["group_2_idx"]
        feat_1_idx = pair["feat_1_idx"]
        feat_2_idx = pair["feat_2_idx"]
        label = pair["label"]

        group_1_data = self.tr_loader.get_group_data(group_1_idx)
        group_2_data = self.tr_loader.get_group_data(group_2_idx)
        all_gens = group_2_data["generated_texts"]
        gen_idx = feat_2_idx % len(all_gens)

        if self.tr_loader.tr_type in [
            ThoughtRepresentation.EMBEDDING_NO_POOLING,
            ThoughtRepresentation.RANDOM_VECTOR,
        ]:
            loc_idx = feat_1_idx % len(group_1_data["thought_vec"])
            thought_vec: torch.Tensor = group_1_data["thought_vec"][loc_idx]
        else:                                     
            thought_vec = group_1_data["thought_vec"]

        assert thought_vec.dim() == 2, "thought_vec must be 2D tensor"
        if thought_vec.shape[0] == 1:                            
            assert self.tr_loader.tr_type in [
                ThoughtRepresentation.EMBEDDING_POOLING,
                ThoughtRepresentation.EMBEDDING_NO_POOLING,
                ThoughtRepresentation.LAST_INPUT_HIDDEN_STATE,
                ThoughtRepresentation.SOFT_THINKING,
                ThoughtRepresentation.SOFT_THINKING_NOISE,
                ThoughtRepresentation.LATENT_THINKING,
                ThoughtRepresentation.INPUT_EMBEDDING,
                ThoughtRepresentation.RANDOM_VECTOR,
            ], "Invalid thought_vec shape for representation type."
            thought_vec = thought_vec.expand(self.expand_dim, -1)
        elif thought_vec.shape[0] < self.expand_dim:
                                        
            rep_needed = (
                self.expand_dim + thought_vec.shape[0] - 1
            ) // thought_vec.shape[0]
            thought_vec = thought_vec.repeat(rep_needed, 1)

        text = all_gens[gen_idx]
        if text is None:
            raise ValueError(
                f"Generated text missing for group {group_2_idx}, feat idx {feat_2_idx}"
            )

        return {
            "text": text,
            "thought_vec": thought_vec,
            "label": float(label),
        }
