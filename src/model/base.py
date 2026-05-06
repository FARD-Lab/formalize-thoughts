from abc import ABC, abstractmethod
from typing import Any, Dict, Optional

import torch
from transformers import BitsAndBytesConfig, PreTrainedModel, PreTrainedTokenizer
from transformers.configuration_utils import PretrainedConfig
from transformers.quantizers.auto import AutoHfQuantizer

from src.utils import get_attn_impl
from src.utils.logging import Logger

def _patch_transformers_quantization_config() -> None:
    """Patch transformers bug where quantization_config=None causes AttributeError.

    When a model's config.json contains an unknown quant_method (e.g. 'mxfp4'),
    transformers parses it as None instead of raising an error. Subsequent calls
    to to_dict() and to_diff_dict() then try None.to_dict() and crash. This patch
    guards against that by temporarily removing the None quantization_config in
    both methods.
    """
    original_to_dict = PretrainedConfig.to_dict
    original_to_diff_dict = PretrainedConfig.to_diff_dict

    def _without_none_qc(self: PretrainedConfig, fn):  # type: ignore[type-arg]
        qc = self.__dict__.get("quantization_config")
        if qc is None and "quantization_config" in self.__dict__:
            del self.__dict__["quantization_config"]
            try:
                return fn(self)
            finally:
                self.__dict__["quantization_config"] = None
        return fn(self)

    def _safe_to_dict(self: PretrainedConfig) -> Dict:
        return _without_none_qc(self, original_to_dict)

    def _safe_to_diff_dict(self: PretrainedConfig) -> Dict:
        return _without_none_qc(self, original_to_diff_dict)

    PretrainedConfig.to_dict = _safe_to_dict  # type: ignore[method-assign]
    PretrainedConfig.to_diff_dict = _safe_to_diff_dict  # type: ignore[method-assign]

    _orig_supports_quant_method = AutoHfQuantizer.supports_quant_method                      

    @staticmethod
    def _safe_supports_quant_method(quantization_config_dict: Any) -> bool:  # type: ignore[override]
        if quantization_config_dict is None:
            return False
        return _orig_supports_quant_method(quantization_config_dict)

    AutoHfQuantizer.supports_quant_method = _safe_supports_quant_method  # type: ignore[method-assign]

_patch_transformers_quantization_config()

class BaseModel(ABC):
    """Abstract base class for all model wrappers."""

    logger: Logger
    device: str
    model_name: str
    torch_dtype: torch.dtype | str
    model: Optional[PreTrainedModel]
    tokenizer: Optional[PreTrainedTokenizer]
    model_kwargs: Dict[str, Any]

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
        model_type: Any = None,
        **kwargs,
    ):
        """Initialize base model.

        Args:
            model_name: HuggingFace model identifier or local path
            device: Device to load model on
            load_in_8bit: Whether to load model in 8-bit precision
            load_in_4bit: Whether to load model in 4-bit precision
            torch_dtype: Torch dtype for model weights
            trust_remote_code: Whether to trust remote code
            max_memory: Max memory per device
            **kwargs: Additional model-specific arguments
        """
        self.logger = logger
        self.device = device
        self.model_name = model_name
        self.torch_dtype: torch.dtype | str = torch_dtype
                                                   
        if torch_dtype == "float16":
            self.torch_dtype = torch.float16
        elif torch_dtype == "bfloat16":
            self.torch_dtype = torch.bfloat16
        elif torch_dtype == "float32":
            self.torch_dtype = torch.float32

        if self.device.startswith("cuda") and not torch.cuda.is_available():
            raise RuntimeError(
                f"CUDA requested (device={self.device}) but torch.cuda.is_available() is False. "
                f"This node likely has a faulty GPU (check nvidia-smi for ERR! state). "
                f"Aborting so SLURM can requeue onto a healthy node."
            )

        quantization_config = None
        if load_in_8bit or load_in_4bit:
            quantization_config = BitsAndBytesConfig(
                load_in_8bit=load_in_8bit,
                load_in_4bit=load_in_4bit,
                bnb_4bit_compute_dtype=self.torch_dtype
                if isinstance(self.torch_dtype, torch.dtype)
                else torch.bfloat16,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
            )

        self.model: Optional[PreTrainedModel] = None
        self.tokenizer: Optional[PreTrainedTokenizer] = None

        model_kwargs = {
            "trust_remote_code": trust_remote_code,
            "dtype": self.torch_dtype,
        }
        if quantization_config is not None:
            model_kwargs["quantization_config"] = quantization_config

        if self.device.startswith("cuda"):
            if ":" not in self.device:
                model_kwargs["device_map"] = "auto"
            else:
                model_kwargs["device_map"] = {"": self.device}
        else:
            model_kwargs["low_cpu_mem_usage"] = True

        attn_impl = get_attn_impl(self.torch_dtype, model_type=model_type)
        if attn_impl is not None:
            model_kwargs["attn_implementation"] = attn_impl

        if max_memory:
            model_kwargs["max_memory"] = max_memory
        self.max_memory = max_memory
        self.logger.info(
            "Resolved loading config: "
            f"device={self.device}, "
            f"device_map={model_kwargs.get('device_map')}, "
            f"attn_implementation={model_kwargs.get('attn_implementation')}"
        )
        self.logger.info(f"Loading with model kwargs: {model_kwargs}")
        self.model_kwargs = model_kwargs

    @abstractmethod
    def load(self) -> None:
        """Load model and tokenizer."""
        pass

    def to(self, device: str) -> "BaseModel":
        """Move model to device.

        Args:
            device: Device to move model to

        Returns:
            Self for chaining
        """
        if self.model is not None:
            self.model = self.model.to(device)
        self.device = device
        return self

    def eval(self) -> "BaseModel":
        """Set model to evaluation mode.

        Returns:
            Self for chaining
        """
        if self.model is not None:
            self.model.eval()
        return self

    def train(self, mode: bool = True) -> "BaseModel":
        """Set model to training mode.

        Args:
            mode: Whether to set training mode or evaluation mode

        Returns:
            Self for chaining
        """
        if self.model is not None:
            self.model.train(mode)
        return self

    def get_model_config(self) -> Dict[str, Any]:
        """Get model configuration.

        Returns:
            Model configuration dictionary
        """
        if self.model is not None:
            return self.model.config.to_dict()
        return {}

    def num_parameters(self) -> int:
        """Get number of model parameters.

        Returns:
            Number of parameters
        """
        if self.model is not None:
            return sum(p.numel() for p in self.model.parameters())
        return 0

    def num_trainable_parameters(self) -> int:
        """Get number of trainable model parameters.

        Returns:
            Number of trainable parameters
        """
        if self.model is not None:
            return sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        return 0
