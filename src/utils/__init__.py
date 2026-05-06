import random
from pathlib import Path
from typing import Any, List

import numpy as np
import torch

flash_attention_2_available = None
flash_attention_3_available = None

def is_flash_attn_2_available() -> bool:
    """Check if Flash Attention 2 is available."""
    global flash_attention_2_available

    if flash_attention_2_available is not None:
        return flash_attention_2_available
    try:
        import flash_attn  # type: ignore # noqa: F401

        flash_attention_2_available = True
    except ImportError:
        flash_attention_2_available = False
    print(f"Flash Attention 2 available: {flash_attention_2_available}")
    return flash_attention_2_available

def is_flash_attn_3_available() -> bool:
    """Check if Flash Attention 3 is available."""
    global flash_attention_3_available

    if flash_attention_3_available is not None:
        return flash_attention_3_available
    try:
        import flash_attn_3  # type: ignore # noqa: F401

        flash_attention_3_available = True
    except ImportError:
        flash_attention_3_available = False
    print(f"Flash Attention 3 available: {flash_attention_3_available}")
    return flash_attention_3_available

def get_attn_impl(
    dtype: torch.dtype | str = "auto", model_type: Any = None
) -> str | None:
    """Get the best available attention implementation based on dtype and hardware.

    Args:
        dtype: The torch dtype or "auto". If "auto", will check hardware and use
               flash attention if available (assuming fp16/bf16 will be used).

    Returns:
        Attention implementation string when explicitly selecting Flash Attention.
        Returns None to leave transformers default backend selection unchanged.
    """
                                                        
    if not torch.cuda.is_available():
        return None

    model_type_str = str(model_type).lower() if model_type is not None else ""

    if "gpt_oss" in model_type_str:
        return None

    device_type = None
    try:
        device_type = torch.cuda.get_device_properties(0).name.lower()
    except Exception:
        for dev_idx in range(torch.cuda.device_count()):
            try:
                device_type = torch.cuda.get_device_properties(dev_idx).name.lower()
                break
            except Exception:
                pass

    if (
        device_type is not None
        and is_flash_attn_3_available()
        and "h100" in device_type
        and dtype in ["auto", torch.float16, torch.bfloat16]
    ):
        return "flash_attention_3"

    if is_flash_attn_2_available():
        return "flash_attention_2"

    return None

def set_seed_all(seed: int) -> None:
    """Set random seed for reproducibility."""
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def get_subdirs_llm_data(base_dir: str | Path) -> List[Path]:
    """Get sorted list of subdirectories in the given base directory."""

    base_path = Path(base_dir)
    return sorted(
        [d for d in base_path.iterdir() if d.is_dir() and not d.name.startswith(".")]
    )

__all__: List[str] = [
    "get_subdirs_llm_data",
    "get_attn_impl",
    "set_seed_all",
]
