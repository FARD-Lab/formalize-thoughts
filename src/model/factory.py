"""Model factory for creating model instances from config."""

from typing import Union

from omegaconf import DictConfig

from src.model.base import BaseModel
from src.model.embedder import Embedder
from src.model.llm import BaseLLM

def create_model(config: Union[dict, DictConfig]) -> BaseModel:
    """Create model instance from configuration.

    Args:
        config: Model configuration dictionary or DictConfig

    Returns:
        Model instance

    Raises:
        ValueError: If model_class is unknown
    """
                                          
    if isinstance(config, DictConfig):
        from omegaconf import OmegaConf

        config = OmegaConf.to_container(config, resolve=True)

    model_class = config.get("model_class", "base_llm")

    if model_class == "base_llm":
        return BaseLLM(**config)
    elif model_class == "embedder":
        return Embedder(**config)
    else:
        raise ValueError(f"Unknown model_class: {model_class}")

__all__ = [
    "BaseModel",
    "BaseLLM",
    "Embedder",
    "create_model",
]
