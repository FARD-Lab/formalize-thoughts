"""Discriminator trainer using Hydra configuration."""

import gc
import json
from pathlib import Path
from typing import Any, Dict, List

import hydra
import torch
import torch.nn as nn
from omegaconf import OmegaConf
from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import get_cosine_schedule_with_warmup

from src.dataset.disc_dataset import DiscriminatorDataset
from src.model.discriminator import LlamaDiscriminator
from src.utils import set_seed_all
from src.utils.config import (
    DiscriminatorTrainerConfig,
    ThoughtRepresentation,
    register_configs,
    resolve_other_vector_dim,
)
from src.utils.logging import Logger

register_configs()

class DiscriminatorTrainer:
    def __init__(self, config: DiscriminatorTrainerConfig, logger: Logger):
        self.config = config
        self.logger = logger

        self.output_dir = Path(config.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.checkpoints_dir = self.output_dir / "checkpoints"
        self.checkpoints_dir.mkdir(parents=True, exist_ok=True)

        self.logger.info("Initializing components...")
        self.global_step = 0                                         
        self.batch_count = 0                                  
        self.current_epoch = 0
        self.best_val_accuracy = 0.0
        self.best_val_loss = float("inf")

        self.tr_type = ThoughtRepresentation(self.config.tr_type)
        self.sanity_check = getattr(self.config, "sanity_check", False)
        self.sanity_check_steps = getattr(self.config, "sanity_check_steps", 100)

        if self.config.seed is not None:
            self.rng = torch.Generator().manual_seed(self.config.seed)
        else:
            self.rng = None

        self.use_wandb = config.wandb.use_wandb
        if self.use_wandb:
            import wandb

            wandb.init(
                project=config.wandb.project,
                entity=config.wandb.entity,
                name=config.wandb.name or config.experiment_name,
                tags=config.wandb.tags,
                notes=config.wandb.notes,
                config=OmegaConf.to_container(config, resolve=True),
                mode=config.wandb.mode,
                resume=config.wandb.resume,
            )
            self.wandb = wandb

        self._init_components()

    def _init_components(self):
                                                                                                   
        set_seed_all(self.config.seed)

        other_vector_dim = resolve_other_vector_dim(self.tr_type, self.config.source_hidden_size)

        self.discriminator = LlamaDiscriminator(
            logger=self.logger,
            model_name=self.config.discriminator.model_name,
            load_in_4bit=self.config.discriminator.load_in_4bit,
            load_in_8bit=self.config.discriminator.load_in_8bit,
            trust_remote_code=self.config.discriminator.trust_remote_code,
            max_memory=self.config.discriminator.max_memory,
            other_vector_dim=other_vector_dim,
            device=self.config.discriminator.device,
            torch_dtype=self.config.discriminator.torch_dtype,
            freeze_base_model=self.config.discriminator.freeze_base_model,
            dropout_rate=self.config.discriminator.dropout_rate,
            num_rep_spaces=self.config.source_num_layers if self.tr_type == ThoughtRepresentation.LAST_INPUT_TOKEN else self.tr_type.num_features,
            use_deep_projection=getattr(self.config.discriminator, "use_deep_projection", False),
            unfreeze_last_n_layers=getattr(self.config.discriminator, "unfreeze_last_n_layers", 0),
        )
        self.discriminator.load()

        self.backbone_tokenizer = self.discriminator.tokenizer
        self.pad_token_id = self.backbone_tokenizer.pad_token_id
        self.dtype = next(self.discriminator.parameters()).dtype
        self.logger.info(
            f"Training parameters={self.discriminator.num_trainable_parameters()}"
        )

        disc_index_dir = Path(self.config.disc_index_output_dir)
        disc_data_dir = Path(self.config.disc_data_output_dir)

        self.logger.info(f"Loading datasets from {disc_index_dir}")

        self.train_dataset = DiscriminatorDataset(
            logger=self.logger,
            pairs_file=str(disc_index_dir / "train_discriminator_pairs.jsonl"),
            tr_type=self.tr_type,
            think_steps=self.config.think_steps,
            vec_dim=other_vector_dim,
            cached_data_dir=str(disc_data_dir),
            expand_dim=self.config.expand_dim,
        )
        self.val_dataset = DiscriminatorDataset(
            logger=self.logger,
            pairs_file=str(disc_index_dir / "val_discriminator_pairs.jsonl"),
            tr_type=self.tr_type,
            think_steps=self.config.think_steps,
            vec_dim=other_vector_dim,
            cached_data_dir=str(disc_data_dir),
            expand_dim=self.config.expand_dim,
        )
        self.test_dataset = DiscriminatorDataset(
            logger=self.logger,
            pairs_file=str(disc_index_dir / "test_discriminator_pairs.jsonl"),
            tr_type=self.tr_type,
            think_steps=self.config.think_steps,
            vec_dim=other_vector_dim,
            cached_data_dir=str(disc_data_dir),
            expand_dim=self.config.expand_dim,
        )

        self.logger.info(
            f"Dataset sizes - Train: {len(self.train_dataset)}, "
            f"Val: {len(self.val_dataset)}, Test: {len(self.test_dataset)}"
        )

        self.optimizer = AdamW(
            self.discriminator.get_trainable_parameters(),
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay,
            foreach=False,
        )

        num_batches_per_epoch = (
            len(self.train_dataset) + self.config.batch_size - 1
        ) // self.config.batch_size
        num_update_steps_per_epoch = (
            num_batches_per_epoch + self.config.gradient_accumulation_steps - 1
        ) // self.config.gradient_accumulation_steps
        max_steps = self.config.num_epochs * num_update_steps_per_epoch

        self.logger.info(
            f"Training: {num_batches_per_epoch} batches/epoch, "
            f"{num_update_steps_per_epoch} optimizer steps/epoch, "
            f"{max_steps} total optimizer steps"
        )

        self.scheduler = get_cosine_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=int(self.config.warmup_steps_percent * max_steps),
            num_training_steps=max_steps,
        )

        self.criterion = nn.BCEWithLogitsLoss()

    def load_saved_model(self, model_path: str):
        """Load a saved model for evaluation.

        Args:
            model_path: Path to checkpoint file (.pt) containing model_state_dict.
                       Should typically be from checkpoints/checkpoint_best.pt or checkpoint_final.pt
        """
        model_path = Path(model_path)

        if not model_path.exists():
            raise FileNotFoundError(f"Model file not found: {model_path}")

        self.logger.info(f"Loading saved model from {model_path}")
        state_dict = torch.load(model_path, map_location=self.discriminator.device)
        self.discriminator.load_trainable_state_dict(state_dict)
        self.logger.info(f"Loaded model from {model_path}")

        self.discriminator.eval()

    def _collate_fn(self, batch: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        texts: List[str] = [item["text"] for item in batch]
        thought_vecs: List[torch.Tensor] = [item["thought_vec"] for item in batch]
        labels: List[float] = [item["label"] for item in batch]

        encoded = self.backbone_tokenizer(
            texts, padding=True, return_tensors="pt", truncation=True
        )

        return {
            "token_ids": encoded["input_ids"],
            "attention_masks": encoded["attention_mask"].float(),
            "thought_vecs": torch.stack(thought_vecs, dim=0),
            "labels": torch.tensor(labels, dtype=self.dtype),
        }

    def evaluate(
        self, dataloader: DataLoader, desc: str = "Evaluating"
    ) -> Dict[str, float]:
        self.discriminator.eval()
        total_loss = 0.0
        correct_preds = 0
        total_preds = 0

        with torch.no_grad():
            for batch in tqdm(dataloader, desc=desc, leave=False):
                token_ids = batch["token_ids"].to(self.discriminator.device)
                attention_masks = batch["attention_masks"].to(self.discriminator.device)
                thought_vecs = batch["thought_vecs"].to(self.discriminator.device)
                labels = batch["labels"].to(self.discriminator.device)

                logits = self.discriminator.forward(
                    token_ids=token_ids,
                    thought_vecs=thought_vecs,
                    token_attention_mask=attention_masks,
                )

                loss = self.criterion(logits.squeeze(), labels)
                total_loss += loss.item()

                preds = (logits.squeeze() > 0).float()
                correct_preds += (preds == labels).sum().item()
                total_preds += labels.size(0)

        avg_loss = total_loss / len(dataloader) if len(dataloader) > 0 else 0.0
        accuracy = correct_preds / total_preds if total_preds > 0 else 0.0
        return {"loss": avg_loss, "accuracy": accuracy}

    def evaluate_with_per_example(
        self,
        dataset: DiscriminatorDataset,
        desc: str,
        output_path: Path,
    ) -> Dict[str, float]:
        """Evaluate and dump per-example predictions for later clustered bootstrap CIs.

        Each record includes pair metadata (group_{1,2}_idx, feat_{1,2}_idx),
        ground-truth label, logit, binary prediction, and correctness. The
        dataloader runs with shuffle=False so that `dataset.pairs[offset + i]`
        aligns with the i-th example of the current batch.
        """
        self.discriminator.eval()
        dataloader = DataLoader(
            dataset,
            batch_size=self.config.batch_size,
            shuffle=False,
            collate_fn=self._collate_fn,
            generator=self.rng,
            num_workers=2,
            pin_memory=True,
        )

        total_loss = 0.0
        correct_preds = 0
        total_preds = 0
        records: List[Dict[str, Any]] = []
        offset = 0

        with torch.no_grad():
            for batch in tqdm(dataloader, desc=desc, leave=False):
                token_ids = batch["token_ids"].to(self.discriminator.device)
                attention_masks = batch["attention_masks"].to(self.discriminator.device)
                thought_vecs = batch["thought_vecs"].to(self.discriminator.device)
                labels = batch["labels"].to(self.discriminator.device)

                logits = self.discriminator.forward(
                    token_ids=token_ids,
                    thought_vecs=thought_vecs,
                    token_attention_mask=attention_masks,
                )

                loss = self.criterion(logits.squeeze(), labels)
                total_loss += loss.item()

                logits_1d = logits.view(-1).detach().float().cpu()
                labels_cpu = labels.detach().float().cpu()
                preds = (logits_1d > 0).to(labels_cpu.dtype)
                correct = (preds == labels_cpu).to(torch.int)
                correct_preds += int(correct.sum().item())
                total_preds += labels.size(0)

                bs = labels.size(0)
                for i in range(bs):
                    pair = dataset.pairs[offset + i]
                    records.append(
                        {
                            "group_1_idx": pair["group_1_idx"],
                            "group_2_idx": pair["group_2_idx"],
                            "feat_1_idx": pair["feat_1_idx"],
                            "feat_2_idx": pair["feat_2_idx"],
                            "label": int(labels_cpu[i].item()),
                            "logit": float(logits_1d[i].item()),
                            "pred": int(preds[i].item()),
                            "correct": int(correct[i].item()),
                        }
                    )
                offset += bs

        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            for rec in records:
                f.write(json.dumps(rec) + "\n")
        self.logger.info(
            f"Wrote {len(records)} per-example records to {output_path}"
        )

        avg_loss = total_loss / len(dataloader) if len(dataloader) > 0 else 0.0
        accuracy = correct_preds / total_preds if total_preds > 0 else 0.0
        return {"loss": avg_loss, "accuracy": accuracy}

    def train_step(
        self, batch: Dict[str, torch.Tensor]
    ) -> tuple[float, float | None, bool]:
        self.discriminator.train()
        token_ids = batch["token_ids"].to(self.discriminator.device)
        attention_masks = batch["attention_masks"].to(self.discriminator.device)
        thought_vecs = batch["thought_vecs"].to(self.discriminator.device)
        labels = batch["labels"].to(self.discriminator.device)

        logits = self.discriminator.forward(
            token_ids=token_ids,
            thought_vecs=thought_vecs,
            token_attention_mask=attention_masks,
        )

        loss = self.criterion(logits.squeeze(), labels)

        loss_scaled = loss / self.config.gradient_accumulation_steps
        loss_scaled.backward()

        self.batch_count += 1

        grad_norm = None
        optimizer_stepped = False
        if self.batch_count % self.config.gradient_accumulation_steps == 0:
            grad_norm = torch.nn.utils.clip_grad_norm_(
                self.discriminator.get_trainable_parameters(),
                self.config.max_grad_norm,
            )
            self.optimizer.step()
            self.scheduler.step()
            self.optimizer.zero_grad()
            self.global_step += 1
            optimizer_stepped = True

        loss_value = loss.item()

        return (
            loss_value,
            grad_norm.item() if grad_norm is not None else None,
            optimizer_stepped,
        )

    def train(self) -> None:
                                 
        if self.config.eval_only:
            if not self.config.saved_model_dir:
                raise ValueError(
                    "eval_only=True requires saved_model_dir to be specified"
                )

            self.logger.info("Running in evaluation-only mode")
            self.load_saved_model(self.config.saved_model_dir)

            self.logger.info("Running evaluation on test set...")
            per_example_path = self.output_dir / "test_per_example.jsonl"
            test_metrics = self.evaluate_with_per_example(
                self.test_dataset, desc="Testing", output_path=per_example_path
            )
            self.logger.info(f"Test results: {test_metrics}")

            if self.use_wandb:
                self.wandb.log(
                    {
                        "test_loss": test_metrics["loss"],
                        "test_accuracy": test_metrics["accuracy"],
                    }
                )
                self.wandb.finish()

            return

        train_dataloader = DataLoader(
            self.train_dataset,
            batch_size=self.config.batch_size,
            shuffle=True,
            collate_fn=self._collate_fn,
            generator=self.rng,
            num_workers=2,
            pin_memory=True,
            persistent_workers=True,
        )
        val_dataloader = DataLoader(
            self.val_dataset,
            batch_size=self.config.batch_size,
            shuffle=False,
            collate_fn=self._collate_fn,
            generator=self.rng,
            num_workers=2,
            pin_memory=True,
            persistent_workers=True,
        )

        self.logger.info(f"Starting training for {self.config.num_epochs} epochs")

        if self.sanity_check:
            self.logger.info(
                f"SANITY CHECK MODE: Overfitting on a single batch for {self.sanity_check_steps} steps per epoch"
            )
                                                                             
            pos_examples = []
            neg_examples = []
            target_per_class = self.config.batch_size // 2

            for batch in train_dataloader:
                labels = batch["labels"].tolist()
                for i, label in enumerate(labels):
                    example = {
                        "token_ids": batch["token_ids"][i : i + 1],
                        "attention_masks": batch["attention_masks"][i : i + 1],
                        "thought_vecs": batch["thought_vecs"][i : i + 1],
                        "labels": torch.tensor([label], dtype=batch["labels"].dtype),
                    }
                    if label == 1.0 and len(pos_examples) < target_per_class:
                        pos_examples.append(example)
                    elif label == 0.0 and len(neg_examples) < target_per_class:
                        neg_examples.append(example)

                    if (
                        len(pos_examples) >= target_per_class
                        and len(neg_examples) >= target_per_class
                    ):
                        break
                if (
                    len(pos_examples) >= target_per_class
                    and len(neg_examples) >= target_per_class
                ):
                    break

            all_examples = pos_examples + neg_examples
            sanity_batch = {
                "token_ids": torch.cat([ex["token_ids"] for ex in all_examples], dim=0),
                "attention_masks": torch.cat(
                    [ex["attention_masks"] for ex in all_examples], dim=0
                ),
                "thought_vecs": torch.cat(
                    [ex["thought_vecs"] for ex in all_examples], dim=0
                ),
                "labels": torch.cat([ex["labels"] for ex in all_examples], dim=0),
            }
            self.logger.info(
                f"Created balanced sanity batch: {len(pos_examples)} positive, {len(neg_examples)} negative samples"
            )

        self.optimizer.zero_grad()

        for epoch in range(self.config.num_epochs):
            self.current_epoch = epoch
            self.discriminator.train()
            epoch_loss = 0.0
            num_batches = 0

            if self.sanity_check:
                                                                           
                progress_bar = tqdm(
                    [sanity_batch] * self.sanity_check_steps,
                    desc=f"Epoch {epoch + 1}/{self.config.num_epochs} [SANITY CHECK]",
                )
            else:
                progress_bar = tqdm(
                    train_dataloader, desc=f"Epoch {epoch + 1}/{self.config.num_epochs}"
                )

            for batch in progress_bar:
                try:
                    loss, grad_norm, optimizer_stepped = self.train_step(batch)
                    epoch_loss += loss
                    num_batches += 1

                    postfix = {"loss": f"{loss:.4f}", "step": self.global_step}
                    if grad_norm is not None:
                        postfix["grad_norm"] = f"{grad_norm:.4f}"
                    progress_bar.set_postfix(postfix)

                    if optimizer_stepped:
                                 
                        if self.global_step % self.config.log_every_n_steps == 0:
                            if self.use_wandb:
                                log_dict = {
                                    "train/loss": loss,
                                    "train/lr": self.scheduler.get_last_lr()[0],
                                    "train/epoch": epoch + 1,
                                    "train/step": self.global_step,
                                    "train/batch_count": self.batch_count,
                                }
                                if grad_norm is not None:
                                    log_dict["train/grad_norm"] = grad_norm
                                self.wandb.log(log_dict, step=self.global_step)

                        if (
                            not self.sanity_check
                            and self.config.eval_every_n_steps > 0
                            and self.global_step % self.config.eval_every_n_steps == 0
                        ):
                            self.logger.info(
                                f"\nRunning evaluation at step {self.global_step}..."
                            )
                            val_metrics = self.evaluate(
                                val_dataloader, desc="Validation"
                            )
                            self.logger.info(f"Validation results: {val_metrics}")

                            if self.use_wandb:
                                self.wandb.log(
                                    {
                                        "val/loss": val_metrics["loss"],
                                        "val/accuracy": val_metrics["accuracy"],
                                    },
                                    step=self.global_step,
                                )

                            if val_metrics["accuracy"] > self.best_val_accuracy:
                                self.best_val_accuracy = val_metrics["accuracy"]
                                self.save_checkpoint(is_best=True)

                        if (
                            self.config.checkpoint_every_n_steps > 0
                            and self.global_step > 0
                            and self.global_step % self.config.checkpoint_every_n_steps == 0
                        ):
                            self.save_checkpoint()
                except Exception as e:
                    self.logger.exception(f"Error during training step: {e}")
                    raise e

            avg_epoch_loss = epoch_loss / max(num_batches, 1)
            self.logger.info(
                f"Epoch {epoch + 1} finished. Average Loss = {avg_epoch_loss:.4f}"
            )

            if self.use_wandb:
                self.wandb.log(
                    {"train/epoch_loss": avg_epoch_loss, "train/epoch": epoch + 1},
                    step=self.global_step,
                )

            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        self.save_checkpoint(final=True)

        if self.sanity_check:
            self.logger.info("Sanity check training completed.")
            return
        self.logger.info("Running final evaluation on test set...")
        per_example_path = self.output_dir / "test_per_example.jsonl"
        test_metrics = self.evaluate_with_per_example(
            self.test_dataset, desc="Testing", output_path=per_example_path
        )
        self.logger.info(f"Test results: {test_metrics}")
        if self.use_wandb:
            self.wandb.log(
                {
                    "final/test_loss": test_metrics["loss"],
                    "final/test_accuracy": test_metrics["accuracy"],
                    "final/best_val_accuracy": self.best_val_accuracy,
                },
                step=self.global_step,
            )

    def save_checkpoint(self, final: bool = False, is_best: bool = False):
        """Save model checkpoint (only model_state_dict)."""

        if self.sanity_check:
            return

        if final:
            checkpoint_path = self.checkpoints_dir / "checkpoint_final.pt"
        elif is_best:
            checkpoint_path = self.checkpoints_dir / "checkpoint_best.pt"
        else:
            checkpoint_path = (
                self.checkpoints_dir / f"checkpoint_step_{self.global_step}.pt"
            )

        torch.save(self.discriminator.trainable_state_dict(), checkpoint_path)
        self.logger.info(f"Saved checkpoint to {checkpoint_path}")

        if final:
            for checkpoint_file in self.checkpoints_dir.glob("checkpoint_*.pt"):
                if checkpoint_file.name != "checkpoint_final.pt":
                    try:
                        checkpoint_file.unlink()
                        self.logger.info(f"Removed checkpoint: {checkpoint_file}")
                    except OSError as e:
                        self.logger.warning(
                            f"Error removing checkpoint {checkpoint_file}: {e}"
                        )
                                                                         
        elif getattr(self.config, "save_total_limit", 0) > 0:
            self._cleanup_checkpoints()

    def _cleanup_checkpoints(self):
        """Remove old checkpoints to maintain save_total_limit."""
        if self.sanity_check:
            return

        step_checkpoints = sorted(
            [p for p in self.checkpoints_dir.glob("checkpoint_step_*.pt")],
            key=lambda p: int(p.stem.split("_")[-1]),
        )

        limit = getattr(self.config, "save_total_limit", 3)

        reserved_count = 0
        if (self.checkpoints_dir / "checkpoint_best.pt").exists():
            reserved_count += 1
        if (self.checkpoints_dir / "checkpoint_final.pt").exists():
            reserved_count += 1

        max_step_checkpoints = max(0, limit - reserved_count)

        while len(step_checkpoints) > max_step_checkpoints:
            oldest = step_checkpoints.pop(0)
            try:
                oldest.unlink()
                self.logger.info(f"Removed old checkpoint: {oldest}")
            except OSError as e:
                self.logger.warning(f"Error removing checkpoint {oldest}: {e}")

@hydra.main(version_base=None, config_path="../configs", config_name="disc_train")
def main(cfg: DiscriminatorTrainerConfig) -> None:
    """Main function for training the discriminator."""
                  
    logger = Logger.from_config(cfg.logging)

    logger.info("Configuration:")
    logger.info(OmegaConf.to_yaml(cfg))

    if cfg.seed is not None:
        set_seed_all(cfg.seed)
        logger.info(f"Set random seed to {cfg.seed}")

    trainer = DiscriminatorTrainer(config=cfg, logger=logger)

    trainer.train()

    if trainer.use_wandb:
        trainer.wandb.finish()

if __name__ == "__main__":
    main()
