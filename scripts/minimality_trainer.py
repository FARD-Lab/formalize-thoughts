"""Minimality / sufficiency probe trainer using HuggingFace Trainer and Hydra configuration."""

import json
from pathlib import Path
from typing import Optional

import hydra
import torch
from omegaconf import OmegaConf
from transformers import (
    AutoTokenizer,
    Trainer,
    TrainingArguments,
)

from src.dataset.minimality_dataset import create_minimality_datasets
from src.model.minimality_probe import ThoughtDescriptor
from src.utils import set_seed_all
from src.utils.config import (
    MinimalityTrainerConfig,
    ThoughtRepresentation,
    register_configs,
)
from src.utils.logging import Logger

register_configs()

class MinimalityDataCollator:
    """
    Custom data collator for minimality / sufficiency probe training.

    Handles padding of input vectors and target sequences.
    """

    def __init__(self, tokenizer, pad_to_multiple_of=None):
        self.tokenizer = tokenizer
        self.pad_to_multiple_of = pad_to_multiple_of
        self.pad_token_id = tokenizer.pad_token_id

    def _pad_token_batch(self, ids_list, mask_list):
        """Right-pad a batch of variable-length token tensors to a common length."""
        max_len = max(len(ids) for ids in ids_list)
        if self.pad_to_multiple_of is not None:
            max_len = (
                (max_len + self.pad_to_multiple_of - 1)
                // self.pad_to_multiple_of
                * self.pad_to_multiple_of
            )
        padded_ids = []
        padded_masks = []
        for ids, mask in zip(ids_list, mask_list):
            pad_len = max_len - len(ids)
            if pad_len > 0:
                ids = torch.cat(
                    [ids, torch.full((pad_len,), self.pad_token_id, dtype=ids.dtype)]
                )
                mask = torch.cat([mask, torch.zeros(pad_len, dtype=mask.dtype)])
            padded_ids.append(ids)
            padded_masks.append(mask)
        return torch.stack(padded_ids), torch.stack(padded_masks)

    def __call__(self, features):
        """
        Collate a batch of examples.

        Args:
            features: List of dictionaries with keys:
                - input_vecs: [len, vec_dim]
                - target_token_ids: [target_len]
                - target_attention_mask: [target_len]
            Optional keys (when prefix_source is set in the dataset):
                - prefix_token_ids: [prefix_len]
                - prefix_attention_mask: [prefix_len]

        Returns:
            Dictionary with batched tensors:
                - vecs: [batch_size, len, vec_dim]
                - target_token_ids: [batch_size, max_target_len]
                - target_attention_mask: [batch_size, max_target_len]
            Plus, when prefix is present:
                - prefix_token_ids: [batch_size, max_prefix_len]
                - prefix_attention_mask: [batch_size, max_prefix_len]
        """
        vecs = torch.stack([f["input_vecs"] for f in features])

        target_token_ids, target_attention_mask = self._pad_token_batch(
            [f["target_token_ids"] for f in features],
            [f["target_attention_mask"] for f in features],
        )

        batch = {
            "vecs": vecs,
            "target_token_ids": target_token_ids,
            "target_attention_mask": target_attention_mask,
        }

        if "prefix_token_ids" in features[0]:
            prefix_token_ids, prefix_attention_mask = self._pad_token_batch(
                [f["prefix_token_ids"] for f in features],
                [f["prefix_attention_mask"] for f in features],
            )
            batch["prefix_token_ids"] = prefix_token_ids
            batch["prefix_attention_mask"] = prefix_attention_mask

        return batch

class MinimalityTrainer(Trainer):
    """
    Custom Trainer for minimality / sufficiency probe.
    Metric: cross-entropy loss only (no generation-based metrics).
    """

    def __init__(self, *args, tokenizer=None, use_thought: bool = True,
                 tile_to_length: Optional[int] = None, **kwargs):
        super().__init__(*args, tokenizer=tokenizer, **kwargs)
        self._use_thought = use_thought
        self._tile_to_length = tile_to_length

    def prediction_step(self, model, inputs, prediction_loss_only, ignore_keys=None):
        inputs = self._prepare_inputs(inputs)
        with torch.no_grad():
            if prediction_loss_only:
                loss = self.compute_loss(model, inputs, return_outputs=False)
                return (loss, None, None)
            else:
                loss, outputs = self.compute_loss(model, inputs, return_outputs=True)
                logits = outputs.get("logits")
                labels = inputs.get("target_token_ids")
                return (loss, logits, labels)

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        vecs = inputs["vecs"]
        target_token_ids = inputs["target_token_ids"]
        target_attention_mask = inputs["target_attention_mask"]
        prefix_token_ids = inputs.get("prefix_token_ids")
        prefix_attention_mask = inputs.get("prefix_attention_mask")

        loss, logits = model.compute_loss(
            vecs=vecs,
            target_token_ids=target_token_ids,
            attention_mask=None,
            target_attention_mask=target_attention_mask,
            prefix_token_ids=prefix_token_ids,
            prefix_attention_mask=prefix_attention_mask,
            use_thought=getattr(self, "_use_thought", True),
            tile_to_length=getattr(self, "_tile_to_length", None),
        )

        if return_outputs:
            return loss, {"logits": logits}
        return loss

@hydra.main(version_base=None, config_path="../configs", config_name="minimality_train")
def main(cfg: MinimalityTrainerConfig) -> None:
    """Main training function."""
    logger = Logger.from_config(cfg.logging)
    logger.info("=" * 80)
    logger.info("Minimality / Sufficiency Probe Trainer")
    logger.info("=" * 80)
    logger.info(f"Experiment: {cfg.experiment_name}")
    logger.info(f"Run: {cfg.run_name or 'default'}")
    logger.info(f"Output directory: {cfg.output_dir}")
    logger.info(f"Target source: {cfg.target_source}")

    logger.info("\nConfiguration:")
    logger.info(OmegaConf.to_yaml(cfg))

    if cfg.seed is not None:
        set_seed_all(cfg.seed)
        logger.info(f"Random seed set to {cfg.seed}")

    output_dir = Path(cfg.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Loading tokenizer from {cfg.probe.model_name}")
    tokenizer = AutoTokenizer.from_pretrained(cfg.probe.model_name)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        logger.info(f"Set pad_token to eos_token: {tokenizer.eos_token}")

    tr_type = ThoughtRepresentation(cfg.tr_type)
    logger.info(f"Thought representation type: {tr_type.value}")

    logger.info("Creating datasets from cached split files...")
    prefix_source = getattr(cfg, "prefix_source", None)
    max_prefix_length = getattr(cfg, "max_prefix_length", None)
    use_thought = bool(getattr(cfg, "use_thought", True))
    tile_to_length = getattr(cfg, "tile_to_length", None)
    if tile_to_length is not None:
        tile_to_length = int(tile_to_length)
    train_dataset, val_dataset, test_dataset = create_minimality_datasets(
        logger=logger,
        tokenizer=tokenizer,
        llm_data_dir=cfg.llm_data_output_dir,
        tr_data_dir=cfg.tr_data_output_dir,
        tr_type=tr_type,
        num_return_sequences=cfg.num_return_sequences,
        shard_size=cfg.shard_size,
        vec_dim=cfg.probe.vector_dim,
        think_steps=cfg.think_steps,
        max_input_length=cfg.max_input_length,
        target_source=cfg.target_source,
        prefix_source=prefix_source,
        max_prefix_length=max_prefix_length,
    )

    logger.info("Initializing minimality probe model...")

    if cfg.probe.torch_dtype == "auto":
        torch_dtype = (
            torch.bfloat16
            if cfg.bf16
            else (torch.float16 if cfg.fp16 else torch.float32)
        )
    elif cfg.probe.torch_dtype == "bfloat16":
        torch_dtype = torch.bfloat16
    elif cfg.probe.torch_dtype == "float16":
        torch_dtype = torch.float16
    else:
        torch_dtype = torch.float32

    model = ThoughtDescriptor(
        logger=logger,
        model_name=cfg.probe.model_name,
        vector_dim=cfg.probe.vector_dim,
        device=cfg.probe.device,
        load_in_8bit=cfg.probe.load_in_8bit,
        load_in_4bit=cfg.probe.load_in_4bit,
        torch_dtype=torch_dtype,
        trust_remote_code=cfg.probe.trust_remote_code,
        max_memory=cfg.probe.max_memory,
        freeze_base_model=cfg.probe.freeze_base_model,
        dropout_rate=cfg.probe.dropout_rate,
        projection_type=cfg.probe.projection_type,
    )
    model.load()

    logger.info(
        f"Model loaded with {model.num_trainable_parameters()} trainable parameters"
    )

    data_collator = MinimalityDataCollator(tokenizer=tokenizer)

    eval_disabled = cfg.eval_every_n_steps is None or cfg.eval_every_n_steps <= 0
    eval_kwargs: dict = (
        {"eval_strategy": "no"}
        if eval_disabled
        else {"eval_strategy": "steps", "eval_steps": cfg.eval_every_n_steps}
    )

    training_args = TrainingArguments(
        output_dir=str(output_dir),
        overwrite_output_dir=True,
        num_train_epochs=cfg.num_epochs,
        per_device_train_batch_size=cfg.batch_size,
        per_device_eval_batch_size=cfg.batch_size,
        gradient_accumulation_steps=cfg.gradient_accumulation_steps,
        learning_rate=cfg.learning_rate,
        weight_decay=cfg.weight_decay,
        warmup_steps=cfg.warmup_steps,
        max_grad_norm=cfg.max_grad_norm,
        logging_dir=str(output_dir / "logs"),
        logging_steps=cfg.log_every_n_steps,
        **eval_kwargs,
                                                                         
        save_strategy="no",
        lr_scheduler_type="cosine",
        optim="adamw_torch_fused" if torch.cuda.is_available() else "adamw",
        fp16=cfg.fp16,
        bf16=cfg.bf16,
        dataloader_num_workers=cfg.dataloader_num_workers,
        dataloader_pin_memory=cfg.dataloader_pin_memory,
        remove_unused_columns=cfg.remove_unused_columns,
        label_smoothing_factor=cfg.label_smoothing_factor,
        report_to="wandb" if cfg.wandb.use_wandb else "none",
        run_name=cfg.wandb.name or cfg.experiment_name,
        seed=cfg.seed,
        data_seed=cfg.seed,
        save_safetensors=False,
    )

    if cfg.wandb.use_wandb:
        import wandb

        wandb.init(
            project=cfg.wandb.project,
            entity=cfg.wandb.entity,
            name=cfg.wandb.name or cfg.experiment_name,
            tags=cfg.wandb.tags,
            notes=cfg.wandb.notes,
            config=OmegaConf.to_container(cfg, resolve=True),
            mode=cfg.wandb.mode,
            resume=cfg.wandb.resume,
        )

    logger.info("Initializing trainer...")
    trainer = MinimalityTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=data_collator,
        tokenizer=tokenizer,
        use_thought=use_thought,
        tile_to_length=tile_to_length,
    )

    if eval_disabled:
        logger.info("Pre-training validation eval (start baseline)...")
        start_results = trainer.evaluate(val_dataset, metric_key_prefix="val_start")
        logger.info(f"Start val results: {start_results}")

    logger.info("Starting training...")
    trainer.train(resume_from_checkpoint=cfg.resume_from_checkpoint)

    logger.info("Evaluating on test set...")
    test_results = trainer.evaluate(test_dataset, metric_key_prefix="test")
    logger.info(f"Test results: {test_results}")

    metrics_path = output_dir / "test_metrics.json"
    metric_payload = {
        "experiment_name": cfg.experiment_name,
        "tr_type": cfg.tr_type,
        "think_steps": cfg.think_steps,
        "target_source": cfg.target_source,
        **test_results,
    }
    metrics_path.write_text(json.dumps(metric_payload, indent=2))
    logger.info(f"Saved test metrics to {metrics_path}")

    logger.info("Running per-example test evaluation...")
    model.eval()
    per_example_records: list[dict] = []
    eval_device = next(model.parameters()).device if hasattr(model, "parameters") else cfg.probe.device
    K = test_dataset.num_return_sequences
    target_source = test_dataset.target_source
    test_prefix_source = getattr(test_dataset, "prefix_source", None)
    k_fold = target_source == "output" or test_prefix_source == "output"

    for idx in range(len(test_dataset)):
        if k_fold:
            example_idx = idx // K
            beam_idx = idx % K
        else:
            example_idx = idx
            beam_idx = 0
        global_idx = test_dataset.indices[example_idx]

        item = test_dataset[idx]
        vecs = item["input_vecs"].unsqueeze(0).to(eval_device)
        target_token_ids = item["target_token_ids"].unsqueeze(0).to(eval_device)
        target_attention_mask = item["target_attention_mask"].unsqueeze(0).to(eval_device)
        num_tokens = int(target_attention_mask.sum().item())

        prefix_token_ids = None
        prefix_attention_mask = None
        prefix_len = 0
        if "prefix_token_ids" in item:
            prefix_token_ids = item["prefix_token_ids"].unsqueeze(0).to(eval_device)
            prefix_attention_mask = item["prefix_attention_mask"].unsqueeze(0).to(eval_device)
            prefix_len = int(prefix_attention_mask.sum().item())

        with torch.no_grad():
            loss, _ = model.compute_loss(
                vecs=vecs,
                target_token_ids=target_token_ids,
                attention_mask=None,
                target_attention_mask=target_attention_mask,
                prefix_token_ids=prefix_token_ids,
                prefix_attention_mask=prefix_attention_mask,
                use_thought=use_thought,
                tile_to_length=tile_to_length,
            )
        ce = float(loss.item()) if num_tokens > 0 else None

        rec = {
            "idx": int(idx),
            "example_idx": int(example_idx),
            "beam_idx": int(beam_idx),
            "global_idx": int(global_idx),
            "ce": ce,
            "num_tokens": num_tokens,
        }
        if "prefix_token_ids" in item:
            rec["prefix_len"] = prefix_len
        per_example_records.append(rec)

        if (idx + 1) % 200 == 0:
            logger.info(f"  per-example eval {idx + 1}/{len(test_dataset)}")

    per_example_file = output_dir / "test_per_example.jsonl"
    with per_example_file.open("w") as fh:
        for rec in per_example_records:
            fh.write(json.dumps(rec) + "\n")
    logger.info(f"Saved per-example test CE to {per_example_file}")

    trainable_keys = {name for name, p in model.named_parameters() if p.requires_grad}
    full_state = model.state_dict()
    trainable_state = {k: v.detach().cpu() for k, v in full_state.items() if k in trainable_keys}
    trainable_path = output_dir / "trainable_params.pt"
    torch.save(trainable_state, trainable_path)
    total_params = sum(v.numel() for v in trainable_state.values())
    logger.info(
        f"Saved {len(trainable_state)} trainable tensors "
        f"({total_params:,} params) to {trainable_path}"
    )

    logger.info("Training complete!")

    if cfg.wandb.use_wandb:
        wandb.finish()

if __name__ == "__main__":
    main()
