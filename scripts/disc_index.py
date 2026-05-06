"""Main script for generating discriminator training pairs (Phase 2)."""

import hydra
from omegaconf import OmegaConf

from src.dataset.curator import DataDiscIndex
from src.utils import set_seed_all
from src.utils.config import DataDiscIndexConfig, register_configs
from src.utils.logging import Logger, setup_wandb

register_configs()

@hydra.main(version_base=None, config_path="../configs", config_name="disc_index")
def main(cfg: DataDiscIndexConfig) -> None:
    """Main function for generating discriminator training pairs.

    Args:
        cfg: Hydra configuration
    """
                         
    print("=" * 20)
    print("Configuration:")
    print("=" * 20)
    print(OmegaConf.to_yaml(cfg))
    print("=" * 20)

    cfg.logging.name = cfg.run_name
                  
    logger = Logger.from_config(
        config=OmegaConf.to_container(cfg.logging, resolve=True),
    )

    logger.info("Starting Phase 2 - Discriminator Pair Generation")
    logger.info(f"Experiment: {cfg.experiment_name}")
    logger.info(f"Run name: {cfg.run_name or 'auto-generated'}")

    wandb_run = None
    if cfg.wandb.use_wandb:
        wandb_run = setup_wandb(
            project=cfg.wandb.project,
            entity=cfg.wandb.entity,
            name=cfg.run_name or cfg.experiment_name,
            tags=cfg.wandb.tags,
            notes=cfg.wandb.notes,
            config=OmegaConf.to_container(cfg, resolve=True),
            mode=cfg.wandb.mode,
            resume=cfg.resume_from_checkpoint,
        )
        logger.info(f"WandB initialized: {wandb_run.url if wandb_run else 'disabled'}")

    logger.info("Creating Phase 2 curator")
    curator = DataDiscIndex.from_config(cfg=cfg, logger=logger)

    logger.info("Running discriminator pair generation")
    set_seed_all(cfg.seed)
    curator.run()

    logger.info("Phase 2 complete!")

    if wandb_run:
        wandb_run.finish()

if __name__ == "__main__":
    main()
