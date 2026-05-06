"""Minimality probe dataset combining ExampleLoader and ThoughtRepresentationLoader."""

from typing import Dict

import torch
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizer

from src.dataset.disc_loader import ExampleLoader
from src.dataset.tr_loader import ThoughtRepresentationLoader
from src.utils.config import ThoughtRepresentation
from src.utils.logging import Logger

_INPUT_DERIVED_TR_TYPES = {
    ThoughtRepresentation.LAST_INPUT_TOKEN,
    ThoughtRepresentation.LAST_INPUT_HIDDEN_STATE,
    ThoughtRepresentation.INPUT_EMBEDDING,
    ThoughtRepresentation.SOFT_THINKING,
    ThoughtRepresentation.SOFT_THINKING_NOISE,
    ThoughtRepresentation.LATENT_THINKING,
    ThoughtRepresentation.RANDOM_VECTOR,
    ThoughtRepresentation.EMBEDDING_POOLING,                                   
}
                                                                                 
class MinimalityDataset(Dataset):
    """
    Dataset for training minimality and output-reconstruction probes.

    target_source="input":  target = input text X  (minimality probe)
    target_source="output": target = generated Y   (output reconstruction probe)

    For target_source="output" the dataset length is num_examples * num_return_sequences.
    - Input-derived TR types (see _INPUT_DERIVED_TR_TYPES): same T for all K beams.
    - EMBEDDING_NO_POOLING: T_k = thought_vec[k] paired with y_k.
    """

    def __init__(
        self,
        logger: Logger,
        tokenizer: PreTrainedTokenizer,
        llm_data_dir: str,
        tr_data_dir: str,
        tr_type: ThoughtRepresentation,
        split_name: str,                             
        num_return_sequences: int = 8,
        shard_size: int = 1024,
        vec_dim: int = 4096,
        think_steps: int = 128,
        max_input_length: int = 512,
        target_source: str = "input",                       
        prefix_source: str | None = None,                                                     
        max_prefix_length: int | None = None,                                        
    ):
        self.logger = logger
        self.tokenizer = tokenizer
        self.split_name = split_name
        self.max_input_length = max_input_length
        self.target_source = target_source
        self.prefix_source = prefix_source
        self.max_prefix_length = max_prefix_length if max_prefix_length is not None else max_input_length
        self.num_return_sequences = num_return_sequences

        if self.prefix_source not in (None, "output"):
            raise ValueError(
                f"Unknown prefix_source: {self.prefix_source!r}. Use None or 'output'."
            )

        self.example_loader = ExampleLoader(
            directory=llm_data_dir,
            num_return_sequences=num_return_sequences,
            shard_size=shard_size,
        )

        self.tr_type = (
            tr_type
            if isinstance(tr_type, ThoughtRepresentation)
            else ThoughtRepresentation(tr_type)
        )

        self.tr_loader = ThoughtRepresentationLoader(
            logger=logger,
            tr_type=self.tr_type,
            cached_data_dir=tr_data_dir,
            split_name=split_name,
            vec_dim=vec_dim,
            think_steps=think_steps,
        )

        self.indices = sorted(list(self.tr_loader.generated_texts_cache.keys()))

        self.logger.info(
            f"Initialized MinimalityDataset for {split_name} split with {len(self.indices)} examples"
        )
        self.logger.info(f"TR type: {self.tr_type.value}, target_source: {self.target_source}")

    def __len__(self):
        n = len(self.indices)
                                                                           
        if self.target_source == "output" or self.prefix_source == "output":
            return n * self.num_return_sequences
        return n

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Returns a dict with keys:
          - input_vecs:            [seq_len, vec_dim]
          - target_token_ids:      [target_len]
          - target_attention_mask: [target_len]
        Optional (when prefix_source="output"):
          - prefix_token_ids:      [prefix_len]
          - prefix_attention_mask: [prefix_len]

        K-fold expansion (idx -> example_idx, beam_idx) is triggered when either
        target_source or prefix_source is "output".
        """
        k_fold = self.target_source == "output" or self.prefix_source == "output"
        if k_fold:
            example_idx = idx // self.num_return_sequences
            beam_idx = idx % self.num_return_sequences
        else:
            example_idx = idx
            beam_idx = 0

        global_idx = self.indices[example_idx]
        group_data = self.tr_loader.get_group_data(global_idx)

        if self.target_source == "output":
            text = group_data["generated_texts"][beam_idx]
            tokenized = self.tokenizer(
                text,
                max_length=self.max_input_length,
                truncation=True,
                padding=False,
                return_tensors="pt",
            )
            target_token_ids = tokenized["input_ids"].squeeze(0)
            target_attention_mask = tokenized["attention_mask"].squeeze(0)
        else:           
            example = self.example_loader.load_example(global_idx)
            tokenized = self.tokenizer(
                example["input_text"],
                max_length=self.max_input_length,
                truncation=True,
                padding=False,
                return_tensors="pt",
            )
            target_token_ids = tokenized["input_ids"].squeeze(0)
            target_attention_mask = tokenized["attention_mask"].squeeze(0)

        tr_type = self.tr_loader.tr_type

        if tr_type == ThoughtRepresentation.RANDOM_VECTOR:
                                                                           
            thought_vec = group_data["thought_vec"][0]

        elif tr_type == ThoughtRepresentation.EMBEDDING_NO_POOLING and k_fold:
                                                                                   
            thought_vec = group_data["thought_vec"][beam_idx]

        elif tr_type == ThoughtRepresentation.EMBEDDING_NO_POOLING:
                                                                             
            thought_vec = group_data["thought_vec"].squeeze(1)

        else:
                                                                                 
            thought_vec = group_data["thought_vec"]

        if thought_vec.dim() != 2:
            raise ValueError(
                f"Expected 2D tensor for thought_vec, got shape {thought_vec.shape}"
            )

        item = {
            "input_vecs": thought_vec,                      
            "target_token_ids": target_token_ids,                
            "target_attention_mask": target_attention_mask,                
        }

        if self.prefix_source == "output":
            prefix_text = group_data["generated_texts"][beam_idx]
            prefix_tok = self.tokenizer(
                prefix_text,
                max_length=self.max_prefix_length,
                truncation=True,
                padding=False,
                return_tensors="pt",
            )
            item["prefix_token_ids"] = prefix_tok["input_ids"].squeeze(0)
            item["prefix_attention_mask"] = prefix_tok["attention_mask"].squeeze(0)

        return item

def create_minimality_datasets(
    logger: Logger,
    tokenizer: PreTrainedTokenizer,
    llm_data_dir: str,
    tr_data_dir: str,
    tr_type: ThoughtRepresentation,
    num_return_sequences: int = 8,
    shard_size: int = 1024,
    vec_dim: int = 4096,
    think_steps: int = 128,
    max_input_length: int = 512,
    target_source: str = "input",
    prefix_source: str | None = None,
    max_prefix_length: int | None = None,
) -> tuple[MinimalityDataset, MinimalityDataset, MinimalityDataset]:
    """
    Create train, validation, and test datasets for minimality / output-reconstruction probe.
    """
    kwargs = dict(
        logger=logger,
        tokenizer=tokenizer,
        llm_data_dir=llm_data_dir,
        tr_data_dir=tr_data_dir,
        tr_type=tr_type,
        num_return_sequences=num_return_sequences,
        shard_size=shard_size,
        vec_dim=vec_dim,
        think_steps=think_steps,
        max_input_length=max_input_length,
        target_source=target_source,
        prefix_source=prefix_source,
        max_prefix_length=max_prefix_length,
    )

    train_dataset = MinimalityDataset(split_name="train", **kwargs)
    val_dataset = MinimalityDataset(split_name="val", **kwargs)
    test_dataset = MinimalityDataset(split_name="test", **kwargs)

    logger.info(
        f"Dataset sizes - Train: {len(train_dataset)}, "
        f"Val: {len(val_dataset)}, Test: {len(test_dataset)}"
    )

    return train_dataset, val_dataset, test_dataset
