import json
import inspect
import os
import re
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
import torch.nn.functional as F
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
)

from src.model.base import BaseModel
from src.utils.config import BaseLLMConfig, ModelType
from src.utils.logging import Logger

def strip_thinking_tokens(text: str) -> str:
    """Strip <think>...</think> blocks from Qwen thinking model outputs."""
    return re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()

def _is_enabled(value: Optional[str]) -> bool:
    """Interpret common boolean-like environment values."""
    if value is None:
        return False
    return value.strip().lower() in {"1", "true", "yes", "on"}

class BaseLLM(BaseModel):
    """Wrapper for base LLM models with generation capabilities."""

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
        model_type: Optional[ModelType] = None,
        inference_backend: str = "transformers",
        vllm_tensor_parallel_size: int = 1,
        vllm_pipeline_parallel_size: int = 1,
        vllm_gpu_memory_utilization: float = 0.9,
        vllm_max_model_len: Optional[int] = None,
        vllm_enforce_eager: bool = False,
        vllm_disable_log_stats: bool = True,
        **kwargs,
    ):
        """Initialize base LLM.

        Args:
            model_name: HuggingFace model identifier or local path
            max_new_tokens: Maximum number of tokens to generate
            temperature: Sampling temperature
            top_p: Nucleus sampling parameter
            do_sample: Whether to use sampling (vs greedy)
            num_return_sequences: Number of sequences to generate
            **kwargs: Additional arguments passed to BaseModel
        """
        super().__init__(
            logger=logger,
            model_name=model_name,
            device=device,
            load_in_8bit=load_in_8bit,
            load_in_4bit=load_in_4bit,
            torch_dtype=torch_dtype,
            trust_remote_code=trust_remote_code,
            max_memory=max_memory,
            model_type=model_type,
            **kwargs,
        )
        self.model_type = model_type
        self.inference_backend = inference_backend.lower()
        self.vllm_tensor_parallel_size = vllm_tensor_parallel_size
        self.vllm_pipeline_parallel_size = vllm_pipeline_parallel_size
        self.vllm_gpu_memory_utilization = vllm_gpu_memory_utilization
        self.vllm_max_model_len = vllm_max_model_len
        self.vllm_enforce_eager = vllm_enforce_eager
        self.vllm_disable_log_stats = vllm_disable_log_stats
        self.vllm_model = None
        self._latent_realign_matrices: Dict[int, Tuple[torch.Tensor, torch.Tensor]] = {}
        self._input_embeddings: Optional[torch.nn.Module] = None

    @staticmethod
    def from_config(config: BaseLLMConfig, logger: Logger) -> "BaseLLM":
        def _cfg_get(cfg: Any, key: str, default: Any) -> Any:
            if isinstance(cfg, dict):
                return cfg.get(key, default)
            return getattr(cfg, key, default)

        model_type_value = _cfg_get(config, "model_type", None)
        if isinstance(model_type_value, str):
            model_type_value = ModelType(model_type_value.lower())

        return BaseLLM(
            logger=logger,
            model_name=_cfg_get(config, "name", ""),
            device=_cfg_get(config, "device", "cuda"),
            load_in_8bit=_cfg_get(config, "load_in_8bit", False),
            load_in_4bit=_cfg_get(config, "load_in_4bit", False),
            torch_dtype=_cfg_get(config, "torch_dtype", "auto"),
            trust_remote_code=_cfg_get(config, "trust_remote_code", False),
            max_memory=_cfg_get(config, "max_memory", None),
            model_type=model_type_value,
            inference_backend=_cfg_get(config, "inference_backend", "transformers"),
            vllm_tensor_parallel_size=_cfg_get(config, "vllm_tensor_parallel_size", 1),
            vllm_pipeline_parallel_size=_cfg_get(config, "vllm_pipeline_parallel_size", 1),
            vllm_gpu_memory_utilization=_cfg_get(config, "vllm_gpu_memory_utilization", 0.9),
            vllm_max_model_len=_cfg_get(config, "vllm_max_model_len", None),
            vllm_enforce_eager=_cfg_get(config, "vllm_enforce_eager", False),
            vllm_disable_log_stats=_cfg_get(config, "vllm_disable_log_stats", True),
        )

    def load(self, tokenizer_only: bool = False, embeddings_only: bool = False) -> None:
        """Load model and tokenizer.

        Args:
            tokenizer_only: If True, only load the tokenizer
            embeddings_only: If True, only load the model to extract embeddings, then unload it
        """
        self.validate_offline_cache()

        self.logger.info(f"Loading tokenizer for model '{self.model_name}'")
        offline_mode = self._offline_mode_enabled()
        cache_dir = self._resolve_hf_hub_cache_dir()
        common_load_kwargs: Dict[str, Any] = {
            "trust_remote_code": self.model_kwargs.get("trust_remote_code", False),
            "local_files_only": offline_mode,
        }
        if cache_dir:
            common_load_kwargs["cache_dir"] = cache_dir

        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name,
            padding_side="left",
            **common_load_kwargs,
        )

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        if tokenizer_only:
            return

        if self.inference_backend == "vllm":
            cuda_visible = os.getenv("CUDA_VISIBLE_DEVICES", "")
                                                                                               
            if "MIG-" in cuda_visible:
                raise RuntimeError(
                    "vLLM backend is running on MIG UUID devices, which is unsupported in this environment. "
                    "Use full H100 GPUs (e.g., --gpus=h100:N) for vLLM, or switch to "
                    "base_llm.inference_backend=transformers for MIG runs. "
                    f"CUDA_VISIBLE_DEVICES={cuda_visible}"
                )
            self._load_vllm_engine(cache_dir=cache_dir, offline_mode=offline_mode)
            return

        self.logger.info(
            f"Model load start: CUDA_VISIBLE_DEVICES={os.getenv('CUDA_VISIBLE_DEVICES', 'unset')}"
        )
        if torch.cuda.is_available():
            try:
                props = torch.cuda.get_device_properties(0)
                self.logger.info(
                    "Detected CUDA device[0]: "
                    f"name={props.name}, total_memory={props.total_memory / (1024 ** 3):.1f}GiB"
                )
            except Exception as exc:
                self.logger.warning(f"Could not read CUDA device properties: {exc}")

        if self.max_memory:
            actual = torch.cuda.device_count()
            bad = [k for k in self.max_memory if int(k) >= actual]
            if bad:
                raise RuntimeError(
                    f"max_memory specifies CUDA device(s) {bad} but only {actual} "
                    f"device(s) are available "
                    f"(CUDA_VISIBLE_DEVICES={os.getenv('CUDA_VISIBLE_DEVICES', 'unset')}). "
                    "A GPU on this node may be in ERR! state — check nvidia-smi."
                )

        self.logger.info(f"Loading model '{self.model_name}'")
        load_start = time.monotonic()
        model_load_kwargs = dict(self.model_kwargs)
        model_load_kwargs.update(common_load_kwargs)
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            **model_load_kwargs,
        )
        load_elapsed = time.monotonic() - load_start
        self.logger.info(f"Model '{self.model_name}' loaded in {load_elapsed:.1f}s")

        if self.model_kwargs.get("device_map") != "auto":
            self.model = self.model.to(self.device)

        if hasattr(self.model, "gradient_checkpointing_disable"):
            self.model.gradient_checkpointing_disable()

        self.model.eval()

        if embeddings_only:
            self.logger.info("Extracting input embeddings and unloading model...")
            self._input_embeddings = self.get_input_embeddings()
            self.unload_model()
            self.logger.info("Model unloaded, embeddings retained")

    def _load_vllm_engine(self, cache_dir: Optional[str], offline_mode: bool) -> None:
        """Load a vLLM engine for generation-only inference."""
        try:
            from vllm import LLM
        except ImportError as exc:
            raise ImportError(
                "vLLM backend requested but package 'vllm' is not installed. "
                "Install it in your environment (and pre-cache for offline runs) "
                "or set base_llm.inference_backend=transformers."
            ) from exc

        dtype = self.torch_dtype
        if isinstance(dtype, torch.dtype):
            if dtype == torch.float16:
                dtype = "float16"
            elif dtype == torch.bfloat16:
                dtype = "bfloat16"
            elif dtype == torch.float32:
                dtype = "float32"
            else:
                dtype = "auto"

        llm_kwargs: Dict[str, Any] = {
            "model": self.model_name,
            "tokenizer": self.model_name,
            "trust_remote_code": self.model_kwargs.get("trust_remote_code", False),
            "dtype": dtype,
            "tensor_parallel_size": self.vllm_tensor_parallel_size,
            "pipeline_parallel_size": self.vllm_pipeline_parallel_size,
            "gpu_memory_utilization": self.vllm_gpu_memory_utilization,
            "enforce_eager": self.vllm_enforce_eager,
            "disable_log_stats": self.vllm_disable_log_stats,
        }
        if cache_dir:
            llm_kwargs["download_dir"] = cache_dir
        if self.vllm_max_model_len is not None:
            llm_kwargs["max_model_len"] = self.vllm_max_model_len

        if "VLLM_WORKER_MULTIPROC_METHOD" not in os.environ:
            os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"

        self.logger.info(
            "Loading vLLM engine with args: "
            f"tensor_parallel_size={self.vllm_tensor_parallel_size}, "
            f"pipeline_parallel_size={self.vllm_pipeline_parallel_size}, "
            f"gpu_memory_utilization={self.vllm_gpu_memory_utilization}, "
            f"offline_mode={offline_mode}, cache_dir={cache_dir}"
        )
        self.vllm_model = LLM(**llm_kwargs)
        self.logger.info(f"vLLM engine for '{self.model_name}' loaded")

    def _offline_mode_enabled(self) -> bool:
        """Return True when Hugging Face/Transformers offline flags are enabled."""
        return _is_enabled(os.getenv("HF_HUB_OFFLINE")) or _is_enabled(os.getenv("TRANSFORMERS_OFFLINE"))

    def num_parameters(self) -> int:
        """Get number of model parameters.

        For vLLM backend, this value is not directly exposed through the runtime API.
        """
        if self.model is not None:
            return super().num_parameters()
        if self.inference_backend == "vllm" and self.vllm_model is not None:
            self.logger.info(
                "Parameter count is not available from vLLM runtime; returning 0."
            )
        return 0

    def _resolve_hf_hub_cache_dir(self) -> Optional[str]:
        """Resolve the active Hugging Face hub cache directory from environment."""
        for key in ("HF_HUB_CACHE", "HUGGINGFACE_HUB_CACHE", "TRANSFORMERS_CACHE"):
            value = os.getenv(key)
            if value:
                return value

        hf_home = os.getenv("HF_HOME")
        if hf_home:
            return str(Path(hf_home) / "hub")
        return None

    def validate_offline_cache(self) -> None:
        """Fail fast when offline mode is enabled and model cache is missing."""
        if not self._offline_mode_enabled():
            return

        model_path = Path(self.model_name)
        if model_path.exists():
            return

        if "/" not in self.model_name:
            return

        cache_dir = self._resolve_hf_hub_cache_dir()
        if not cache_dir:
            raise RuntimeError(
                "Offline mode is enabled but no Hugging Face cache path is set. "
                "Set HF_HUB_CACHE (or HF_HOME) before loading model "
                f"'{self.model_name}'."
            )

        repo_dir = Path(cache_dir) / f"models--{self.model_name.replace('/', '--')}"
        if not repo_dir.exists():
            raise FileNotFoundError(
                "Offline mode is enabled and model cache is missing. "
                f"Expected repo cache directory: {repo_dir}. "
                "Prewarm the cache with slurm/prep_offline_cache.sh or "
                "huggingface_hub.snapshot_download using the same HF_HOME/HF_HUB_CACHE."
            )

    def get_input_embeddings(self) -> torch.nn.Module:
        """Extract and return the input embeddings layer from the model.

        Returns:
            The input embeddings layer (typically an nn.Embedding module)
        """
        if self.model is None:
            raise RuntimeError("Model not loaded. Call load() first.")

        embeddings = self.model.get_input_embeddings()

        new_embeddings = torch.nn.Embedding(
            num_embeddings=embeddings.num_embeddings,
            embedding_dim=embeddings.embedding_dim,
            padding_idx=embeddings.padding_idx,
            dtype=embeddings.weight.dtype,
            device=embeddings.weight.device,
        )

        with torch.no_grad():
            new_embeddings.weight.copy_(embeddings.weight)

        new_embeddings.eval()
        new_embeddings.requires_grad_(False)

        return new_embeddings

    def unload_model(self) -> None:
        """Unload the model from memory while keeping the tokenizer."""
        if self.model is not None:
                                                        
            if self.device.startswith("cuda"):
                self.model.cpu()

            del self.model
            self.model = None

            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            import gc

            gc.collect()

            self.logger.info("Model unloaded from memory")

    def drop_weight_page_cache(self) -> None:
        """Advise the OS to evict page-cache pages for this model's safetensor shards.

        After unload_model(), Python references and GPU memory are freed, but the OS
        still holds mmap pages for the safetensor files in page cache.  SLURM counts
        that page cache toward the job's --mem cgroup limit.  In the two-pass pipeline
        the Phase 1 pages would otherwise overlap with Phase 2 vLLM pages and push
        the job over 32G.  Calling madvise(MADV_DONTNEED) on each shard tells the
        kernel it can immediately reclaim those pages.
        """
        import glob
        import mmap
        import os

        cache_root = os.environ.get(
            "HF_HUB_CACHE",
            os.path.join(os.environ.get("HF_HOME", os.path.expanduser("~/.cache/huggingface")), "hub"),
        )
        model_dir = os.path.join(cache_root, f"models--{self.model_name.replace('/', '--')}")
        shards = glob.glob(os.path.join(model_dir, "snapshots", "*", "*.safetensors"))
        if not shards:
                                                                                    
            shards = glob.glob(os.path.join(model_dir, "blobs", "*.safetensors"))

        freed = 0
        for shard in shards:
            try:
                size = os.path.getsize(shard)
                if size == 0:
                    continue
                with open(shard, "rb") as f:
                    mm = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)
                    mm.madvise(mmap.MADV_DONTNEED)
                    mm.close()
                freed += size
            except Exception:
                pass                                                      

        if freed:
            self.logger.info(
                f"Advised OS to evict {len(shards)} safetensor shards "
                f"({freed / 1e9:.1f} GB) from page cache"
            )

    def load_for_hidden_states(self) -> None:
        """Load the transformers model for prefill hidden-state extraction.

        Forces the transformers load path regardless of inference_backend.
        After this call self.model is populated; self.vllm_model is NOT loaded.
        Call unload_model() + torch.cuda.empty_cache() before loading vLLM.
        """
        saved_backend = self.inference_backend
        self.inference_backend = "transformers"
        try:
            self.load()
        finally:
            self.inference_backend = saved_backend

    def forward_prefill(
        self,
        prompts: Union[str, List[str]],
        instruction: Optional[str] = None,
    ) -> List[torch.Tensor]:
        """Run a single forward pass and return first_hs for each prompt.

        Args:
            prompts: One or more raw user-text prompts (before chat formatting).
            instruction: Optional system instruction passed to _get_formatted.

        Returns:
            List of CPU tensors, one per prompt, each shape (num_layers, hidden_size).
            The hidden state is sampled at the last input-token position.
            With left-padding the final position (index -1) is always the last
            real token, never a pad token.
        """
        if self.model is None:
            raise RuntimeError(
                "Transformers model not loaded. Call load_for_hidden_states() first."
            )

        prompts_list = [prompts] if isinstance(prompts, str) else list(prompts)
        tokenized_inputs = self._get_formatted(prompts_list, instruction=instruction)
                                                                                               
        inputs = self.tokenizer(
            tokenized_inputs,
            return_tensors="pt",
            padding=True,
        ).to(self.device)

        with torch.no_grad():
            forward_outputs = self.model(
                **inputs,
                use_cache=False,
                output_hidden_states=True,
                return_dict=True,
            )

        results: List[torch.Tensor] = []
        for i in range(inputs["input_ids"].shape[0]):
            hs_i = torch.stack(
                [h[i, -1, :].cpu() for h in forward_outputs.hidden_states],
                dim=0,
            )                             
            results.append(hs_i)

        del forward_outputs
        if self.device.startswith("cuda"):
            torch.cuda.empty_cache()

        return results

    def _get_formatted(
        self, prompts_list: Union[str, List[str]], instruction: Optional[str] = None
    ) -> List[str]:
                                              
        if isinstance(prompts_list, str):
            prompts_list = [prompts_list]

        template_kwargs: dict = {}
        fold_instruction_into_user = self.model_type in (ModelType.QWEN, ModelType.DEEPSEEK)

        tokenized_inputs = []
        for p in prompts_list:
            if fold_instruction_into_user and instruction:
                user_content = f"{instruction}\n\n{p}"
                messages = [{"role": "user", "content": user_content}]
                system_messages = []
            else:
                messages = [{"role": "user", "content": p}]
                system_messages = [{"role": "system", "content": instruction}] if instruction else []

            formatted_prompt = self.tokenizer.apply_chat_template(
                system_messages + messages,
                add_generation_prompt=True,
                tokenize=False,
                **template_kwargs,
            )
            if isinstance(formatted_prompt, list):
                formatted_prompt = formatted_prompt[0] if formatted_prompt else ""

            if self.model_type == ModelType.QWEN:
                formatted_prompt = formatted_prompt + "<think>\n"

            tokenized_inputs.append(formatted_prompt)
        return tokenized_inputs

    def generate(
        self,
        prompt: Union[str, List[str]],
        instruction: Optional[str] = None,
        max_new_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        top_k: Optional[int] = None,
        repetition_penalty: Optional[float] = None,
        no_repeat_ngram_size: Optional[int] = None,
        do_sample: Optional[bool] = None,
        num_return_sequences: Optional[int] = None,
        return_hidden_states: bool = False,
        return_logits: bool = False,
        **kwargs,
    ) -> Dict[str, Any]:
        """Generate text from prompt.

        Args:
            prompt: Input prompt(s)
            max_new_tokens: Override max_new_tokens
            temperature: Override temperature
            top_p: Override top_p
            top_k: Override top_k
            repetition_penalty: Override repetition_penalty
            no_repeat_ngram_size: Size of ngrams to avoid repeating
            do_sample: Override do_sample
            num_return_sequences: Override num_return_sequences
            return_hidden_states: Whether to return hidden states (only first step)
            return_logits: Whether to return logits
            seed: Override seed for this generation
            **kwargs: Additional generation arguments

        Returns:
            Dictionary containing generated text and optionally hidden states/logits
        """
        if self.tokenizer is None:
            raise RuntimeError("Model not loaded. Call load() first.")

        if self.inference_backend == "vllm":
            return self._generate_vllm(
                prompt=prompt,
                instruction=instruction,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                repetition_penalty=repetition_penalty,
                no_repeat_ngram_size=no_repeat_ngram_size,
                do_sample=do_sample,
                num_return_sequences=num_return_sequences,
                return_hidden_states=return_hidden_states,
                return_logits=return_logits,
                **kwargs,
            )

        if self.model is None:
            raise RuntimeError("Model not loaded. Call load() first.")

        prompts_list = [prompt] if isinstance(prompt, str) else prompt

        tokenized_inputs = self._get_formatted(prompts_list, instruction=instruction)
        self.logger.info(f"Tokenized {len(tokenized_inputs)} prompts for generation")
                                                 
        inputs = self.tokenizer(
            tokenized_inputs,
            return_tensors="pt",
            padding=True,
            truncation=True,
        ).to(self.device)

        first_step_hidden_states = None
        if return_hidden_states:
            self.logger.info("Extracting hidden states from forward pass")
            with torch.no_grad():
                forward_outputs = self.model(
                    **inputs,
                    use_cache=False,
                    output_hidden_states=True,
                    return_dict=True,
                )
                                                                 
                num_layers = len(forward_outputs.hidden_states)
                self.logger.info(
                    f"Transferring hidden states to CPU: {num_layers} layers"
                )
                first_step_hidden_states = [
                    h.cpu() for h in forward_outputs.hidden_states
                ]
                self.logger.info("Hidden states transfer complete")
                                 
                if self.device.startswith("cuda"):
                    torch.cuda.empty_cache()

        gen_kwargs = {
            "output_hidden_states": False,
            "output_scores": return_logits,
            "return_dict_in_generate": True,
            "eos_token_id": self.tokenizer.eos_token_id,
            "pad_token_id": self.tokenizer.pad_token_id,
            **kwargs,
        }

        if max_new_tokens is not None:
            gen_kwargs["max_new_tokens"] = max_new_tokens
        if do_sample is not None:
            gen_kwargs["do_sample"] = do_sample
        if temperature is not None:
            gen_kwargs["temperature"] = temperature
        if top_p is not None:
            gen_kwargs["top_p"] = top_p
        if top_k is not None:
            gen_kwargs["top_k"] = top_k
        if no_repeat_ngram_size is not None:
            gen_kwargs["no_repeat_ngram_size"] = no_repeat_ngram_size
        if repetition_penalty is not None:
            gen_kwargs["repetition_penalty"] = repetition_penalty
        if num_return_sequences is not None:
            gen_kwargs["num_return_sequences"] = num_return_sequences

        gen_kwargs["use_cache"] = True
        self.logger.info(f"Running generation with args: {gen_kwargs}")
                  
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                **gen_kwargs,
            )
        self.logger.info("Generation complete")
                                                                    
        generated_ids = outputs.sequences
        input_length = inputs["input_ids"].shape[1]
        generated_only_ids = generated_ids[:, input_length:]
        generated_text = self.tokenizer.batch_decode(
            generated_only_ids,
            skip_special_tokens=True,
        )
                                                                                    
        if self.model_type == ModelType.QWEN:
            generated_text = [strip_thinking_tokens(t) for t in generated_text]
        self.logger.info("Decoded generated text")
                      
        result = {
            "input_ids": inputs["input_ids"].cpu(),
            "generated_text": generated_text,
            "generated_ids": generated_only_ids.cpu(),
        }

        if first_step_hidden_states is not None:
            result["hidden_states"] = [first_step_hidden_states]

        if return_logits and hasattr(outputs, "scores"):
                                                        
            result["logits"] = [score.cpu() for score in outputs.scores]
            if self.device.startswith("cuda"):
                torch.cuda.empty_cache()

        return result

    def _generate_vllm(
        self,
        prompt: Union[str, List[str]],
        instruction: Optional[str] = None,
        max_new_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        top_k: Optional[int] = None,
        repetition_penalty: Optional[float] = None,
        no_repeat_ngram_size: Optional[int] = None,
        do_sample: Optional[bool] = None,
        num_return_sequences: Optional[int] = None,
        return_hidden_states: bool = False,
        return_logits: bool = False,
        **kwargs,
    ) -> Dict[str, Any]:
        """Generate text with vLLM while preserving llm_data output contract."""
        if self.vllm_model is None:
            raise RuntimeError(
                "vLLM backend selected but engine is not loaded. Call load() first."
            )

        try:
            from vllm import SamplingParams
        except ImportError as exc:
            raise ImportError(
                "vLLM backend requested but package 'vllm' is not installed."
            ) from exc

        prompts_list = [prompt] if isinstance(prompt, str) else prompt
        tokenized_inputs = self._get_formatted(prompts_list, instruction=instruction)
        input_ids = self.tokenizer(
            tokenized_inputs,
            return_tensors="pt",
            padding=True,
            truncation=True,
        )["input_ids"].cpu()

        if return_hidden_states:
            self.logger.warning(
                "vLLM backend currently does not expose hidden states in this pipeline. "
                "Ignoring return_hidden_states=true."
            )
        if return_logits:
            self.logger.warning(
                "vLLM backend currently does not expose per-step logits in this pipeline. "
                "Ignoring return_logits=true."
            )
        if no_repeat_ngram_size is not None:
            self.logger.warning(
                "vLLM backend does not support no_repeat_ngram_size directly; ignoring this setting."
            )

        thinking_token_budget = kwargs.pop("thinking_token_budget", None)
        seed = kwargs.pop("seed", None)

        n = num_return_sequences if num_return_sequences is not None else 1
        num_beams = kwargs.pop("num_beams", 1)
        use_beam_search = num_beams is not None and num_beams > 1
        best_of = max(int(num_beams), int(n)) if use_beam_search else None

        sp_kwargs: Dict[str, Any] = {
            "n": int(n),
            "max_tokens": int(max_new_tokens) if max_new_tokens is not None else 16,
            "ignore_eos": False,
            "skip_special_tokens": True,
            "early_stopping": kwargs.pop("early_stopping", False),
        }
        if use_beam_search:
            sp_kwargs["use_beam_search"] = True
            sp_kwargs["best_of"] = int(best_of)
        if temperature is not None:
            sp_kwargs["temperature"] = temperature
        if top_p is not None:
            sp_kwargs["top_p"] = top_p
        if top_k is not None:
            sp_kwargs["top_k"] = top_k
        if repetition_penalty is not None:
            sp_kwargs["repetition_penalty"] = repetition_penalty
        if thinking_token_budget is not None:
            sp_kwargs["thinking_token_budget"] = int(thinking_token_budget)
        if seed is not None:
            sp_kwargs["seed"] = int(seed)

        if kwargs:
            self.logger.warning(
                "Ignoring unsupported vLLM generation kwargs: "
                f"{sorted(kwargs.keys())}"
            )

        supported_sp_kwargs = set(inspect.signature(SamplingParams.__init__).parameters)
        supported_sp_kwargs.discard("self")
        unsupported_sp_kwargs = sorted(set(sp_kwargs) - supported_sp_kwargs)
        if unsupported_sp_kwargs:
            self.logger.warning(
                "Installed vLLM version does not support SamplingParams args: "
                f"{unsupported_sp_kwargs}. They will be ignored."
            )
        sampling_params = SamplingParams(
            **{k: v for k, v in sp_kwargs.items() if k in supported_sp_kwargs}
        )

        use_new_beam_api = use_beam_search and "use_beam_search" not in supported_sp_kwargs

        self.logger.info(
            "Running vLLM generation with args: "
            f"n={n}, best_of={best_of}, use_beam_search={use_beam_search}, "
            f"max_tokens={max_new_tokens}, use_new_beam_api={use_new_beam_api}"
        )

        eos_token_id = self.tokenizer.eos_token_id
        if eos_token_id is None:
            raise RuntimeError("Tokenizer eos_token_id is required for llm_data output packing")

        flat_texts: List[str] = []
        flat_token_ids: List[List[int]] = []
        expected = len(tokenized_inputs) * int(n)

        if use_new_beam_api:
                                                                      
            try:
                from vllm.sampling_params import BeamSearchParams
                from vllm.inputs import TokensPrompt
            except ImportError as exc:
                raise RuntimeError(
                    "vLLM beam_search API (BeamSearchParams / TokensPrompt) not available; "
                    "cannot run beam search with this vLLM version."
                ) from exc
            beam_params = BeamSearchParams(
                beam_width=int(n),
                max_tokens=int(max_new_tokens) if max_new_tokens is not None else 16,
            )
                                                                                   
            prompt_token_ids_list = [
                self.tokenizer.encode(p, add_special_tokens=False)
                for p in tokenized_inputs
            ]
            tokens_prompts = [
                TokensPrompt(prompt_token_ids=ids)
                for ids in prompt_token_ids_list
            ]
            beam_outputs = self.vllm_model.beam_search(
                prompts=tokens_prompts,
                params=beam_params,
            )
            for i, beam_out in enumerate(beam_outputs):
                seqs = beam_out.sequences
                if len(seqs) != int(n):
                    raise RuntimeError(
                        "vLLM beam_search returned unexpected number of sequences. "
                        f"Expected {n}, got {len(seqs)}."
                    )
                prompt_len = len(prompt_token_ids_list[i])
                for seq in seqs:
                    generated_token_ids = list(seq.tokens[prompt_len:])
                    text = self.tokenizer.decode(generated_token_ids, skip_special_tokens=True)
                    if self.model_type == ModelType.QWEN:
                        text = strip_thinking_tokens(text)
                    flat_texts.append(text)
                    flat_token_ids.append(generated_token_ids)
        else:
            if do_sample is not None and not do_sample and not use_beam_search:
                sampling_params.temperature = 0.0
            request_outputs = self.vllm_model.generate(
                prompts=tokenized_inputs,
                sampling_params=sampling_params,
                use_tqdm=False,
            )
            for req in request_outputs:
                if all(hasattr(o, "index") for o in req.outputs):
                    ordered_outputs = sorted(req.outputs, key=lambda o: o.index)
                else:
                                                                                
                    ordered_outputs = list(req.outputs)
                if len(ordered_outputs) != int(n):
                    raise RuntimeError(
                        "vLLM returned an unexpected number of outputs for a prompt. "
                        f"Expected {n}, got {len(ordered_outputs)}."
                    )
                for out in ordered_outputs:
                    text = out.text
                    if self.model_type == ModelType.QWEN:
                        text = strip_thinking_tokens(text)
                    flat_texts.append(text)
                    token_ids = getattr(out, "token_ids", None)
                    if token_ids is None:
                        token_ids = self.tokenizer.encode(text, add_special_tokens=False)
                    flat_token_ids.append(list(token_ids))

        if len(flat_texts) != expected:
            raise RuntimeError(
                f"vLLM output count mismatch: expected {expected}, got {len(flat_texts)}"
            )

        max_len = max((len(ids) for ids in flat_token_ids), default=0)
        generated_only_ids = torch.full(
            (len(flat_token_ids), max_len),
            fill_value=eos_token_id,
            dtype=torch.long,
        )
        for i, ids in enumerate(flat_token_ids):
            if ids:
                generated_only_ids[i, : len(ids)] = torch.tensor(ids, dtype=torch.long)

        result = {
            "input_ids": input_ids,
            "generated_text": flat_texts,
            "generated_ids": generated_only_ids,
        }

        return result

    def tokenize(
        self,
        texts: Union[str, List[str]],
        return_tensors: str = "pt",
        padding: bool = True,
        truncation: bool = True,
        max_length: Optional[int] = None,
        **kwargs,
    ) -> Dict[str, Any]:
        """Tokenize input texts.

        Args:
            texts: Input text(s)
            return_tensors: Type of tensors to return ("pt", "tf", "np", or None)
            padding: Whether to pad sequences
            truncation: Whether to truncate sequences
            max_length: Maximum sequence length
            **kwargs: Additional tokenizer arguments
        Returns:
            Tokenized inputs
        """
        if self.tokenizer is None:
            raise RuntimeError("Tokenizer not loaded. Call load() first.")

        return self.tokenizer(
            texts,
            return_tensors=return_tensors,
            padding=padding,
            truncation=truncation,
            max_length=max_length,
            **kwargs,
        )

    def get_soft_token(
        self,
        text: str | List[str],
        instruction: str,
        steps: int = 1,
        add_noise: bool = True,
                                    
        gumbel_temperature: float = 0.5,
        top_k: Optional[int] = 30,
        top_p: Optional[float] = 0.95,
    ) -> torch.Tensor:
        """Iteratively get soft token embeddings using Soft Thinking.
        Following Wu et al., "DEMYSTIFYING THE WORKING MECHANISM OF SOFT THINKING"
        When add_noise=False, uses pure soft thinking with concept tokens as probability
        distributions. When add_noise=True, applies Gumbel-Softmax for exploration.

        Args:
            text: Input text or list of texts
            instruction: Optional instruction to prepend as system message
            steps: Number of iterative soft-token steps
            add_noise: If True, apply Gumbel-Softmax reweighting; if False, use pure concept tokens
            gumbel_temperature: Temperature for Gumbel-Softmax reweighting
            top_k: Filter to top-k tokens before computing embedding (for efficiency)
            top_p: Filter to nucleus (top-p) tokens before computing embedding (for efficiency)

        Returns:
            Tensor of shape (batch_size, steps, embedding_dim)
        """
        if steps <= 0:
            raise ValueError("steps must be >= 1")

        formatted = (
            self._get_formatted([text], instruction)
            if isinstance(text, str)
            else self._get_formatted(text, instruction)
        )
        tokenized = self.tokenizer(
            formatted, return_tensors="pt", padding=True, truncation=True
        ).to(self.device)

        embedding_layer = self.model.get_input_embeddings()

        current_inputs_embeds = embedding_layer(tokenized.input_ids)
        current_attention_mask = tokenized.attention_mask
        past_key_values = None

        soft_tokens_list: List[torch.Tensor] = []

        with torch.no_grad():
            for _ in range(steps):
                              
                outputs = self.model(
                    inputs_embeds=current_inputs_embeds,
                    attention_mask=current_attention_mask,
                    past_key_values=past_key_values,
                    use_cache=True,
                    return_dict=True,
                )
                past_key_values = outputs.past_key_values
                next_token_logits = outputs.logits[:, -1, :]

                probs = torch.softmax(next_token_logits, dim=-1)

                if top_k is not None and top_k > 0:
                    probs = self._top_k_renorm_prob(probs, top_k)
                if top_p is not None and 0.0 < top_p < 1.0:
                    probs = self._top_p_renorm_prob(probs, top_p)

                actual_k = min(top_k or probs.size(-1), probs.size(-1))
                topk_probs, topk_indices = torch.topk(probs, k=actual_k, dim=-1)
                topk_probs = topk_probs / topk_probs.sum(dim=-1, keepdim=True)

                if add_noise:
                    topk_logits = torch.log(topk_probs.clamp(min=1e-12))
                    gumbel_noise = -torch.log(
                        -torch.log(torch.rand_like(topk_logits).clamp(min=1e-12))
                    )
                    topk_probs = torch.softmax(
                        (topk_logits + gumbel_noise) / gumbel_temperature, dim=-1
                    )
                                                     
                    topk_probs, sorted_idx = torch.sort(
                        topk_probs, dim=-1, descending=True
                    )
                    topk_indices = torch.gather(topk_indices, dim=1, index=sorted_idx)

                topk_indices_device = topk_indices.to(embedding_layer.weight.device)
                topk_embeddings = embedding_layer.weight[topk_indices_device]
                                                                                         
                soft_token = torch.sum(
                    topk_probs.to(topk_embeddings.device).unsqueeze(-1)
                    * topk_embeddings,
                    dim=1,
                )
                soft_tokens_list.append(soft_token)

                current_inputs_embeds = soft_token.unsqueeze(1)

                batch_size = current_inputs_embeds.shape[0]
                past_length = past_key_values[0][0].shape[2]
                current_attention_mask = torch.ones(
                    (batch_size, past_length + 1), dtype=torch.long, device=self.device
                )

        return torch.stack(soft_tokens_list, dim=1)

    def _compute_soft_token_from_logits(
        self,
        logits: torch.Tensor,
        embedding_layer,
        top_k: Optional[int],
        top_p: Optional[float],
        add_noise: bool,
        gumbel_temperature: float,
    ) -> torch.Tensor:
        """Compute a soft token embedding from logits.

        Extracted from get_soft_token to allow reuse in batched methods.

        Args:
            logits: Shape (batch_size, vocab_size)
            embedding_layer: Model's input embedding layer
            top_k: Top-k filtering
            top_p: Top-p filtering
            add_noise: Whether to apply Gumbel-Softmax
            gumbel_temperature: Temperature for Gumbel-Softmax

        Returns:
            Soft token embedding of shape (batch_size, embedding_dim)
        """
        probs = torch.softmax(logits, dim=-1)

        if top_k is not None and top_k > 0:
            probs = self._top_k_renorm_prob(probs, top_k)
        if top_p is not None and 0.0 < top_p < 1.0:
            probs = self._top_p_renorm_prob(probs, top_p)

        actual_k = min(top_k or probs.size(-1), probs.size(-1))
        topk_probs, topk_indices = torch.topk(probs, k=actual_k, dim=-1)
        topk_probs = topk_probs / topk_probs.sum(dim=-1, keepdim=True)

        if add_noise:
            topk_logits = torch.log(topk_probs.clamp(min=1e-12))
            gumbel_noise = -torch.log(
                -torch.log(torch.rand_like(topk_logits).clamp(min=1e-12))
            )
            topk_probs = torch.softmax(
                (topk_logits + gumbel_noise) / gumbel_temperature, dim=-1
            )
            topk_probs, sorted_idx = torch.sort(
                topk_probs, dim=-1, descending=True
            )
            topk_indices = torch.gather(topk_indices, dim=1, index=sorted_idx)

        topk_indices_device = topk_indices.to(embedding_layer.weight.device)
        topk_embeddings = embedding_layer.weight[topk_indices_device]
        soft_token = torch.sum(
            topk_probs.to(topk_embeddings.device).unsqueeze(-1)
            * topk_embeddings,
            dim=1,
        )
        return soft_token

    @staticmethod
    def _expand_past_key_values(past_key_values, target_batch_size: int):
        """Expand KV cache from batch_size=1 to target_batch_size.

        Creates contiguous copies so each batch element's cache is independent
        for subsequent autoregressive steps.
        """
        from transformers.cache_utils import DynamicCache, DynamicLayer

        if isinstance(past_key_values, DynamicCache):
            expanded = DynamicCache()
                                                                             
            for layer in past_key_values.layers:
                new_layer = DynamicLayer()
                new_layer.dtype = layer.keys.dtype
                new_layer.device = layer.keys.device
                new_layer.is_initialized = True
                new_layer.keys = (
                    layer.keys
                    .expand(target_batch_size, -1, -1, -1)
                    .contiguous()
                )
                new_layer.values = (
                    layer.values
                    .expand(target_batch_size, -1, -1, -1)
                    .contiguous()
                )
                expanded.layers.append(new_layer)
            return expanded

        return tuple(
            (
                layer[0].expand(target_batch_size, -1, -1, -1).contiguous(),
                layer[1].expand(target_batch_size, -1, -1, -1).contiguous(),
            )
            for layer in past_key_values
        )

    def get_combined_thinking(
        self,
        text: str,
        instruction: str,
        steps: int = 128,
        gumbel_temperature: float = 0.5,
        top_k: Optional[int] = 30,
        top_p: Optional[float] = 0.95,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Compute soft_thinking, soft_thinking_noise, and latent_thinking with
        shared prefill and batched decode for ~3x speedup.

        Instead of three separate calls (each tokenizing and prefilling the same
        input independently), this method:
        1. Runs a single shared prefill (the expensive part for long prompts)
        2. Forks into 3 branches with cloned KV caches
        3. Runs batched decode steps (batch_size=3) for remaining iterations

        The decode phase is memory-bandwidth-bound on the 70B model, so
        batch_size=3 has nearly the same cost as batch_size=1.

        Args:
            text: Single input text string
            instruction: Instruction to prepend as system message
            steps: Number of iterative thinking steps
            gumbel_temperature: Temperature for Gumbel-Softmax (soft_thinking_noise)
            top_k: Top-k filtering for soft thinking
            top_p: Top-p filtering for soft thinking

        Returns:
            Tuple of (soft_thinking, soft_thinking_noise, latent_thinking),
            each of shape (1, steps, embedding_dim)
        """
        if steps <= 0:
            raise ValueError("steps must be >= 1")

        formatted = self._get_formatted([text], instruction)
        tokenized = self.tokenizer(
            formatted, return_tensors="pt", padding=True, truncation=True
        ).to(self.device)

        embedding_layer = self.model.get_input_embeddings()
        inputs_embeds = embedding_layer(tokenized.input_ids)
        attention_mask = tokenized.attention_mask

        with torch.no_grad():
                                           
            outputs = self.model(
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                use_cache=True,
                return_dict=True,
                output_hidden_states=True,
            )
            prefill_logits = outputs.logits[:, -1, :]                   
            prefill_hidden = outputs.hidden_states[-1][:, -1, :]          
            prefill_past = outputs.past_key_values

            soft_0 = self._compute_soft_token_from_logits(
                prefill_logits, embedding_layer, top_k, top_p,
                add_noise=False, gumbel_temperature=gumbel_temperature,
            )          
            noise_0 = self._compute_soft_token_from_logits(
                prefill_logits, embedding_layer, top_k, top_p,
                add_noise=True, gumbel_temperature=gumbel_temperature,
            )          
            latent_0 = self._apply_latent_realignment(prefill_hidden)          

            embed_device = embedding_layer.weight.device
            soft_tokens = [soft_0]
            noise_tokens = [noise_0]
            latent_tokens = [latent_0.detach().clone().to(embed_device)]

            if steps == 1:
                return (
                    torch.stack(soft_tokens, dim=1),
                    torch.stack(noise_tokens, dim=1),
                    torch.stack(latent_tokens, dim=1),
                )

            batched_past = self._expand_past_key_values(prefill_past, 3)

            for _ in range(steps - 1):
                                                         
                batched_embeds = torch.cat([
                    soft_tokens[-1].unsqueeze(1),
                    noise_tokens[-1].unsqueeze(1),
                    latent_tokens[-1].unsqueeze(1),
                ], dim=0)

                past_len = batched_past[0][0].shape[2]
                batched_mask = torch.ones(
                    (3, past_len + 1), dtype=torch.long, device=self.device
                )

                outputs = self.model(
                    inputs_embeds=batched_embeds,
                    attention_mask=batched_mask,
                    past_key_values=batched_past,
                    use_cache=True,
                    return_dict=True,
                    output_hidden_states=True,
                )
                batched_past = outputs.past_key_values
                batched_logits = outputs.logits[:, -1, :]                   
                batched_hidden = outputs.hidden_states[-1][:, -1, :]          

                soft_tokens.append(self._compute_soft_token_from_logits(
                    batched_logits[0:1], embedding_layer, top_k, top_p,
                    add_noise=False, gumbel_temperature=gumbel_temperature,
                ))
                                               
                noise_tokens.append(self._compute_soft_token_from_logits(
                    batched_logits[1:2], embedding_layer, top_k, top_p,
                    add_noise=True, gumbel_temperature=gumbel_temperature,
                ))
                                           
                latent_vec = self._apply_latent_realignment(batched_hidden[2:3])
                latent_tokens.append(latent_vec.detach().clone().to(embed_device))

        return (
            torch.stack(soft_tokens, dim=1),                   
            torch.stack(noise_tokens, dim=1),                  
            torch.stack(latent_tokens, dim=1),                 
        )

    def _top_k_renorm_prob(
        self,
        probs: torch.Tensor,
        top_k: int,
    ) -> torch.Tensor:
        """Apply top-k filtering and renormalization to probability distribution.

        Sets probabilities outside top-k to 0 and renormalizes the remaining.

        Args:
            probs: Probability distribution of shape (batch_size, vocab_size)
            top_k: Number of top tokens to keep

        Returns:
            Renormalized probability distribution
        """
        if top_k <= 0:
            return probs

        k = min(top_k, probs.size(-1))

        top_k_probs, top_k_indices = torch.topk(probs, k=k, dim=-1)

        probs_filtered = torch.zeros_like(probs)
        probs_filtered.scatter_(dim=-1, index=top_k_indices, src=top_k_probs)

        probs_sum = probs_filtered.sum(dim=-1, keepdim=True).clamp(min=1e-12)
        probs_filtered = probs_filtered / probs_sum

        return probs_filtered

    def _top_p_renorm_prob(
        self,
        probs: torch.Tensor,
        top_p: float,
    ) -> torch.Tensor:
        """Apply top-p (nucleus) filtering and renormalization to probability distribution.

        Sets probabilities outside the nucleus (cumulative probability > top_p) to 0
        and renormalizes the remaining.

        Args:
            probs: Probability distribution of shape (batch_size, vocab_size)
            top_p: Cumulative probability threshold (0.0 < top_p < 1.0)

        Returns:
            Renormalized probability distribution
        """
        if not (0.0 < top_p < 1.0):
            return probs

        sorted_probs, sorted_indices = torch.sort(probs, descending=True, dim=-1)

        cumulative_probs = torch.cumsum(sorted_probs, dim=-1)

        sorted_indices_to_remove = cumulative_probs > top_p
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = False

        sorted_probs[sorted_indices_to_remove] = 0.0

        probs_filtered = torch.zeros_like(probs)
        probs_filtered.scatter_(dim=-1, index=sorted_indices, src=sorted_probs)

        probs_sum = probs_filtered.sum(dim=-1, keepdim=True).clamp(min=1e-12)
        probs_filtered = probs_filtered / probs_sum

        return probs_filtered

    def _build_latent_realign_matrix(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        From Zou et al. "Latent Collaboration in Multi-Agent System"
        """
        input_embeds = (
            self.model.get_input_embeddings()
            if hasattr(self.model, "get_input_embeddings")
            else None
        )
        output_embeds = (
            self.model.get_output_embeddings()
            if hasattr(self.model, "get_output_embeddings")
            else None
        )
        if output_embeds is None:
            output_embeds = getattr(self.model, "lm_head", None)
        if (
            input_embeds is None
            or output_embeds is None
            or not hasattr(input_embeds, "weight")
            or not hasattr(output_embeds, "weight")
        ):
            raise RuntimeError(
                "Cannot build latent realignment matrix: embedding weights not accessible."
            )
        input_weight = input_embeds.weight.detach().to(
            device=self.device, dtype=torch.float32
        )
        output_weight = output_embeds.weight.detach().to(
            device=self.device, dtype=torch.float32
        )
        gram = torch.matmul(output_weight.T, output_weight)
        reg = 1e-5 * torch.eye(gram.shape[0], device=gram.device, dtype=gram.dtype)
        gram = gram + reg
        rhs = torch.matmul(output_weight.T, input_weight)
        realign_matrix = torch.linalg.solve(gram, rhs)
        target_norm = input_weight.norm(dim=1).mean().detach()

        return realign_matrix, target_norm

    def _ensure_latent_realign_matrix(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        From Zou et al. "Latent Collaboration in Multi-Agent System"
        """
        info = self._latent_realign_matrices.get(0, None)
        target_device = torch.device(self.device)

        if info is None:
            matrix, target_norm = self._build_latent_realign_matrix()
        else:
            matrix, target_norm = info
            if matrix.device != target_device:
                matrix = matrix.to(target_device)

        target_norm = (
            target_norm.to(device=target_device, dtype=matrix.dtype)
            if isinstance(target_norm, torch.Tensor)
            else torch.as_tensor(target_norm, device=target_device, dtype=matrix.dtype)
        )
        self._latent_realign_matrices[0] = (matrix, target_norm)

        return matrix, target_norm

    def _apply_latent_realignment(self, hidden: torch.Tensor) -> torch.Tensor:
        """
        From Zou et al. "Latent Collaboration in Multi-Agent System"
        """
        matrix, target_norm = self._ensure_latent_realign_matrix()
                                                                                        
        matrix = matrix.to(hidden.device)
        target_norm = target_norm.to(hidden.device)
        hidden_fp32 = hidden.to(torch.float32)
        aligned = torch.matmul(hidden_fp32, matrix)

        aligned_norm = aligned.norm(dim=-1, keepdim=True).clamp_min(1e-6)
        pre_aligned = aligned.detach().clone()
        self.pre_aligned = pre_aligned
        aligned = aligned * (target_norm / aligned_norm)
        return aligned.to(hidden.dtype)

    def get_latent_thinking(
        self,
        text: Union[str, List[str]],
        instruction: Optional[str] = None,
        steps: int = 1,
    ):
        """
        Following the work of Zou et al. "Latent Collaboration in Multi-Agent System"
        Feed back the last hidden state iteratively after applying the projection matrix
        """
        if steps <= 0:
            raise ValueError("steps must be >= 1")

        formatted = (
            self._get_formatted([text], instruction)
            if isinstance(text, str)
            else self._get_formatted(text, instruction)
        )
        tokenized = self.tokenizer(
            formatted,
            return_tensors="pt",
            padding=True,
            truncation=True,
        ).to(self.device)

        with torch.no_grad():
                                                                        
            embedding_layer = self.model.get_input_embeddings()
            inputs_embeds = embedding_layer(tokenized.input_ids)
            attention_mask = tokenized.attention_mask
            outputs = self.model(
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                return_dict=True,
                output_hidden_states=True,
                use_cache=True,
            )
            past = outputs.past_key_values
            last_hidden = outputs.hidden_states[-1][:, -1, :]          

            latent_vecs_all: List[torch.Tensor] = []
                                                           
            latent_vec = self._apply_latent_realignment(last_hidden)          
            latent_vecs_all.append(latent_vec.detach().clone())
            for _ in range(steps - 1):
                latent_embed = latent_vec.unsqueeze(1)             

                past_len = past[0][0].shape[2]
                latent_mask = torch.ones(
                    (latent_embed.shape[0], past_len + 1),
                    dtype=torch.long,
                    device=self.device,
                )
                outputs = self.model(
                    inputs_embeds=latent_embed,
                    attention_mask=latent_mask,
                    past_key_values=past,
                    use_cache=True,
                    output_hidden_states=True,
                    return_dict=True,
                )
                past = outputs.past_key_values
                last_hidden = outputs.hidden_states[-1][:, -1, :]
                latent_vec = self._apply_latent_realignment(last_hidden)          
                latent_vecs_all.append(latent_vec.detach().clone())

        return torch.stack(latent_vecs_all, dim=1)                 

    def compute_log_probs(
        self,
        prompt: Union[str, List[str]],
        output_sequence: Union[str, List[str]],
        instruction: Optional[str] = None,
        normalize: bool = True,
    ) -> Union[float, List[float]]:
        """
        Compute log probabilities of output sequence(s) given prompt(s).
        Uses tokenizer offset mapping (preferred) to locate the output token span
        inside the full tokenization of `prompt + output`.

        Note: This method is batched for efficiency. It processes all prompts in a single
        forward pass through the model.

        Important: tokenize(A) + tokenize(B) != tokenize(A+B) due to tokenizer context
        dependencies. We always tokenize the full prompt+output together.

        Token boundary handling: When locating output tokens using character offsets, we
        include any token whose character span overlaps the output span (token_end > output_start
        AND token_start < output_end). This correctly handles tokens that straddle the
        prompt/output boundary (e.g., "hello" token spanning prompt ending with "he" and
        output starting with "llo").

        Multiple occurrences: If output_text appears multiple times in full_text, we use
        rfind() to match the last occurrence (assuming it's the actual output).

        Args:
            prompt: Input prompt(s)
            output_sequence: Output sequence(s) to score
            instruction: Optional instruction to prepend
            normalize: If True, return average log prob per token; if False, return sum

        Returns:
            Log probability (or list of log probabilities for batch)
        """
        if self.model is None or self.tokenizer is None:
            raise RuntimeError("Model not loaded. Call load() first.")

        single_input = isinstance(prompt, str)
        prompts = [prompt] if single_input else list(prompt)
        outputs = (
            [output_sequence]
            if isinstance(output_sequence, str)
            else list(output_sequence)
        )
        if len(prompts) != len(outputs):
            raise ValueError("Number of prompts must match number of output sequences")

        formatted_prompts = self._get_formatted(prompts, instruction=instruction)
        device = getattr(self, "device", torch.device("cpu"))
        self.model.eval()

        full_texts = []
        for prompt_text, output_text in zip(formatted_prompts, outputs):
                                  
            if not prompt_text and not output_text:
                full_texts.append("")
                continue
            if not output_text:
                full_texts.append(prompt_text)
                continue
            if not prompt_text:
                full_texts.append(output_text)
                continue

            if not prompt_text.endswith((" ", "\n")) and not output_text.startswith(
                (" ", "\n")
            ):
                full_texts.append(prompt_text + " " + output_text)
            else:
                full_texts.append(prompt_text + output_text)

        enc_full = self.tokenizer(
            full_texts,
            return_tensors="pt",
            add_special_tokens=True,
            padding=True,
            truncation=True,
            return_offsets_mapping=True,
        )

        input_ids = enc_full["input_ids"].to(device)                
        attention_mask = enc_full.get("attention_mask")
        if attention_mask is not None:
            attention_mask = attention_mask.to(device)

        batch_size = input_ids.shape[0]
        token_spans = []                                                   
        offsets_available = enc_full.get("offset_mapping") is not None

        for batch_idx in range(batch_size):
            prompt_text = formatted_prompts[batch_idx]
            output_text = outputs[batch_idx]
            full_text = full_texts[batch_idx]

            if not output_text:
                token_spans.append(None)
                continue

            start_token_idx = None
            end_token_idx = None

            if offsets_available:
                offsets = enc_full["offset_mapping"][batch_idx]                
                                                                  
                start_char = full_text.rfind(output_text)
                if start_char == -1:
                                               
                    start_char = len(prompt_text)

                end_char = start_char + len(output_text)

                for i, (s, e) in enumerate(offsets):
                    try:
                        s_val = int(s) if not isinstance(s, int) else s
                        e_val = int(e) if not isinstance(e, int) else e
                    except Exception:
                        s_val = int(s.item())
                        e_val = int(e.item())

                    if s_val == e_val == 0:
                        continue

                    if start_token_idx is None and e_val > start_char:
                        start_token_idx = i

                    if s_val < end_char:
                        end_token_idx = i

            if start_token_idx is None or end_token_idx is None:
                                                                     
                enc_output = self.tokenizer(
                    output_text, return_tensors="pt", add_special_tokens=False
                )
                out_ids = enc_output["input_ids"][0].tolist()
                full_ids = input_ids[batch_idx].tolist()

                if len(out_ids) > 0:
                                                                      
                    max_start = len(full_ids) - len(out_ids)
                    for j in range(max_start, -1, -1):
                        if full_ids[j : j + len(out_ids)] == out_ids:
                            start_token_idx = j
                            end_token_idx = j + len(out_ids) - 1
                            break

            token_spans.append(
                (start_token_idx, end_token_idx)
                if start_token_idx is not None
                else None
            )

        with torch.no_grad():
            model_outputs = self.model(
                input_ids=input_ids, attention_mask=attention_mask
            )
            logits = model_outputs.logits                   

        shift_logits = logits[:, :-1, :].contiguous()                     
        shift_labels = input_ids[:, 1:].contiguous()                  

        results: List[float] = []

        for batch_idx in range(batch_size):
            token_span = token_spans[batch_idx]

            if token_span is None:
                results.append(0.0)
                continue

            start_token_idx, end_token_idx = token_span

            pred_start = start_token_idx - 1
            pred_end = end_token_idx - 1             

            if pred_start < 0:
                if pred_end < 0:
                                        
                    results.append(0.0)
                    continue
                pred_start = 0

            item_logits = shift_logits[
                batch_idx, pred_start : pred_end + 1, :
            ]                
            item_labels = shift_labels[
                batch_idx, pred_start : pred_end + 1
            ]             

            if attention_mask is not None:
                                                              
                item_mask = attention_mask[
                    batch_idx, pred_start + 1 : pred_end + 2
                ]             
                                                 
                valid_mask = item_mask.bool()
                if valid_mask.sum() == 0:
                    results.append(0.0)
                    continue
                item_logits = item_logits[valid_mask]
                item_labels = item_labels[valid_mask]

            out_len = item_labels.size(0)
            if out_len == 0:
                results.append(0.0)
                continue

            log_probs = F.log_softmax(item_logits, dim=-1)                
            target_log_probs = log_probs.gather(-1, item_labels.unsqueeze(-1)).squeeze(
                -1
            )             
            seq_logprob = target_log_probs.sum().item()

            results.append(seq_logprob / out_len if normalize else seq_logprob)

        return results[0] if single_input else results

def index_hidden_states(
    hidden_states: List[List[torch.Tensor]],
    batch_return_idx: int,
    decoding_step: int,
    layer_idx: int,
    token_idx: Optional[int] = None,
) -> torch.Tensor:
    """Extract hidden states for specific indices.

    The hidden states structure from model.generate() is:
    - List of decoding steps, each containing:
      - List of layers, each containing:
        - Tensor of shape (batch_size * num_return_sequences, seq_len, hidden_size)

    Note: The first decoding step (step 0) contains hidden states for the entire input
    sequence, while subsequent steps contain hidden states for single newly generated tokens.

    Args:
        hidden_states: Hidden states from model.generate() with return_hidden_states=True
        batch_return_idx: Index for batch * num_return_sequences dimension
        decoding_step: Which decoding step (0 = first step with full input)
        layer_idx: Which model layer to extract from
        token_idx: Optional token position index within the sequence.
                  - For step 0: can be 0 to input_length-1
                  - For step > 0: should be 0 (or None) as only 1 token exists
                  - If None, returns all tokens for that step

    Returns:
        torch.Tensor: Hidden state vector(s) of shape (hidden_size,) if token_idx is specified,
                     or (seq_len, hidden_size) if token_idx is None
    """
    if not hidden_states:
        raise ValueError("hidden_states is empty")

    if decoding_step >= len(hidden_states):
        raise IndexError(
            f"decoding_step {decoding_step} out of range. "
            f"Available steps: 0 to {len(hidden_states) - 1}"
        )

    step_hidden_states = hidden_states[decoding_step]

    if layer_idx >= len(step_hidden_states):
        raise IndexError(
            f"layer_idx {layer_idx} out of range. "
            f"Available layers: 0 to {len(step_hidden_states) - 1}"
        )

    layer_hidden_states = step_hidden_states[layer_idx]

    if batch_return_idx >= layer_hidden_states.shape[0]:
        raise IndexError(
            f"batch_return_idx {batch_return_idx} out of range. "
            f"Available indices: 0 to {layer_hidden_states.shape[0] - 1}"
        )

    batch_hidden_states = layer_hidden_states[
        batch_return_idx
    ]                          

    if token_idx is not None:
        if token_idx >= batch_hidden_states.shape[0]:
            raise IndexError(
                f"token_idx {token_idx} out of range. "
                f"Available tokens at step {decoding_step}: 0 to {batch_hidden_states.shape[0] - 1}"
            )
        return batch_hidden_states[token_idx]                  

    if batch_hidden_states.shape[0] == 1:
        return batch_hidden_states.squeeze(0)                  

    return batch_hidden_states

def len_output_sequence(
    generated_ids: torch.Tensor,
    eos_token_id: int,
    num_return_sequences: int = 1,
) -> List[int]:
    """Calculate the length of generated sequences for each item in the batch.

    Args:
        generated_ids: Generated token IDs of shape (batch_size * num_return_sequences, total_seq_len)
        input_ids: Input token IDs of shape (batch_size, input_seq_len)
        eos_token_id: The EOS token ID to identify sequence end
        num_return_sequences: Number of sequences generated per input (None is treated as 1)

    Returns:
        List of integers representing the length of generated tokens for each batch item,
        counted up to and including the first EOS token (or total length if no EOS found)
    """
                      
    if num_return_sequences is None:
        num_return_sequences = 1

    lengths = []

    for i in range(generated_ids.shape[0]):
        generated_only = generated_ids[i]

        eos_mask = generated_only == eos_token_id
        eos_positions = eos_mask.nonzero(as_tuple=True)[0]

        length = (
            min(eos_positions[0].item() + 1, len(generated_only))
            if len(eos_positions) > 0
            else len(generated_only)
        )
        lengths.append(length)

    return lengths

def index_logits(
    logits: List[torch.Tensor],
    batch_return_idx: int,
    decoding_step: int,
) -> torch.Tensor:
    """Extract logits for specific indices.

    The logits structure from model.generate() is:
    - List of decoding steps, each containing:
      - Tensor of shape (batch_size * num_return_sequences, vocab_size)

    Note: Logits represent the model's output scores for each token in the vocabulary
    at each decoding step. Step 0 corresponds to the first generated token.

    Args:
        logits: Logits from model.generate() with return_logits=True
        batch_return_idx: Index for batch * num_return_sequences dimension
        decoding_step: Which decoding step (0 = first generated token)

    Returns:
        torch.Tensor: Logits vector of shape (vocab_size,)
    """
    if not logits:
        raise ValueError("logits is empty")

    if decoding_step >= len(logits):
        raise IndexError(
            f"decoding_step {decoding_step} out of range. "
            f"Available steps: 0 to {len(logits) - 1}"
        )

    step_logits = logits[decoding_step]

    if batch_return_idx >= step_logits.shape[0]:
        raise IndexError(
            f"batch_return_idx {batch_return_idx} out of range. "
            f"Available indices: 0 to {step_logits.shape[0] - 1}"
        )

    return step_logits[batch_return_idx]                 

