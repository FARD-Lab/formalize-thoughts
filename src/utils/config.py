"""Configuration dataclasses for the project using Hydra."""

import enum
import logging
from dataclasses import dataclass, field
from typing import List, Optional

from omegaconf import MISSING

class ProjectType(enum.Enum):
    NONE = "none"
    SHARED = "shared"
    PER_LAYER = "per_layer"

class ModelClassEnum(enum.Enum):
    BASE_LLM = "base_llm"
    EMBEDDER = "embedder"
    DISCRIMINATOR = "discriminator"

class ModelType(enum.Enum):
    LLAMA = "llama"
    DEEPSEEK = "deepseek"
    QWEN = "qwen"
    MISTRAL = "mistral"
    GEMMA = "gemma"
    GPT_OSS = "gpt_oss"                        

class ThoughtRepresentation(enum.Enum):
    """Enum for thought representation types with associated attributes.

    Attributes:
        num_features:      Number of sequence positions in the representation.
        is_embedding_based: True if the vector comes from an embedding model (e.g. Nemotron),
                            whose output dim is fixed regardless of the source LLM.
                            False if the vector comes from the source LLM's hidden states,
                            whose dim equals the source LLM's hidden_size.
    """

    num_features: Optional[int]
    is_embedding_based: bool

    def __new__(cls, value: str, num_features: Optional[int] = None, is_embedding_based: bool = False):
        obj = object.__new__(cls)
        obj._value_ = value
        obj.num_features = num_features
        obj.is_embedding_based = is_embedding_based
        return obj

    @classmethod
    def _missing_(cls, value: object):
        if isinstance(value, str):
            member = cls._value2member_map_.get(value)
            if member is not None:
                return member
        return super()._missing_(value)

    LAST_INPUT_TOKEN        = ("last_input_token",        33, False)                          
    LAST_INPUT_HIDDEN_STATE = ("last_input_hidden_state",  1, False)
    EMBEDDING_POOLING       = ("embedding_pooling",        1, True)
    EMBEDDING_NO_POOLING    = ("embedding_no_pooling",     1, True)
    EMBEDDING_ALL           = ("embedding_all",            1, True)
    SOFT_THINKING           = ("soft_thinking",            1, False)
    SOFT_THINKING_NOISE     = ("soft_thinking_noise",      1, False)
    LATENT_THINKING         = ("latent_thinking",          1, False)
    INPUT_EMBEDDING         = ("input_embedding",          1, True)
    RANDOM_VECTOR           = ("random_vector",            1, False)

@dataclass
class ModelConfig:
    """Base configuration for any model."""

    name: str = MISSING                                              
    model_class: ModelClassEnum = MISSING                                              
    model_type: ModelType = MISSING
    device: str = "cuda"
    load_in_8bit: bool = False
    load_in_4bit: bool = False
    torch_dtype: str = "auto"                                            
    trust_remote_code: bool = False
    max_memory: Optional[dict] = None                                    

@dataclass
class BaseLLMConfig(ModelConfig):
    """Configuration for base LLM models."""

    name: str = "meta-llama/Llama-3.1-8B-Instruct"
    model_class: ModelClassEnum = ModelClassEnum.BASE_LLM
    model_type: ModelType = ModelType.LLAMA
    hidden_size: int = 4096                                                                        
    num_hidden_layers: int = 32                                                                             

    max_new_tokens: int = 16
    temperature: Optional[float] = 0
    top_p: Optional[float] = 0.8
    top_k: Optional[int] = 50
    repetition_penalty: Optional[float] = 1.1
    no_repeat_ngram_size: Optional[int] = 4
    do_sample: bool = True
    num_return_sequences: Optional[int] = None
    return_hidden_states: bool = True
    return_logits: bool = True

    inference_backend: str = "transformers"                            
    vllm_tensor_parallel_size: int = 1
    vllm_pipeline_parallel_size: int = 1
    vllm_gpu_memory_utilization: float = 0.9
    vllm_max_model_len: Optional[int] = None
    vllm_enforce_eager: bool = False
    vllm_disable_log_stats: bool = True

@dataclass
class EmbedderConfig(ModelConfig):
    """Configuration for embedder model."""

    name: str = "nvidia/llama-embed-nemotron-8b"
    model_class: ModelClassEnum = ModelClassEnum.EMBEDDER
    model_type: ModelType = ModelType.LLAMA
    output_dim: int = 4096                                                                 

@dataclass
class EarlyStoppingConfig:
    """Configuration for early stopping."""

    patience: int = 5                                                             
    min_delta: float = 1e-5                                            

@dataclass
class DiscriminatorConfig(ModelConfig):
    """Configuration for the discriminator model."""

    model_name: str = "meta-llama/Llama-3.2-1B"
    model_class: ModelClassEnum = ModelClassEnum.DISCRIMINATOR
    model_type: ModelType = ModelType.LLAMA

    freeze_base_model: bool = True
    dropout_rate: float = 0.1
                                                                       
    use_deep_projection: bool = False
                                                                          
    unfreeze_last_n_layers: int = 0

@dataclass
class LoaderConfig:
    """Base configuration for data loaders."""

    pass

@dataclass
class BBEHLoaderConfig(LoaderConfig):
    """Configuration for BBEH data loader."""

    data_dir: str = "./data/bbeh"
    task_name: Optional[str] = None                                    
    num_examples: Optional[int] = 10                     
                                                                               
    example_start_idx: Optional[int] = None
    example_end_idx: Optional[int] = None

@dataclass
class LoggingConfig:
    """Configuration for logging."""

    name: str = "hidden_thoughts"
    log_level: int = logging.INFO                                         
    log_dir: Optional[str] = None
    log_to_file: bool = False
    log_to_console: bool = True
    use_rich: bool = False                           
    log_format: Optional[str] = None

@dataclass
class WandBConfig:
           
    use_wandb: bool = True
    project: str = "hidden-thoughts"
    entity: Optional[str] = None
    name: Optional[str] = None
    tags: List[str] = field(default_factory=list)
    notes: Optional[str] = None
    config: Optional[dict] = None
    mode: str = "online"                                   
    resume: Optional[str] = None

@dataclass
class ExperimentConfig:
                         
    experiment_name: str = "dataset_curation"
    run_name: Optional[str] = None
    resume_from_checkpoint: Optional[str] = None

    seed: int = 42

@dataclass
class TrainingConfig:
    """Base configuration for training."""

    batch_size: int = 8
    num_epochs: int = 3
    learning_rate: float = 1e-4
    weight_decay: float = 0.01
    warmup_steps: int = 100
    gradient_accumulation_steps: int = 1
    max_grad_norm: float = 1.0

    checkpoint_every_n_steps: int = 100
    log_every_n_steps: int = 10
    eval_every_n_steps: int = 100
    save_total_limit: int = 3

@dataclass
class DataLLMConfig(ExperimentConfig):
    """Main configuration for LLM dataset curation."""

    base_llm: BaseLLMConfig = field(default_factory=BaseLLMConfig)
    loader: LoaderConfig = field(default_factory=BBEHLoaderConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    wandb: WandBConfig = field(default_factory=WandBConfig)

    output_dir: str = "./data"
    max_input_length: int = 4096

    save_format: str = "jsonl"                      
    shard_size: int = 1_024                               

    batch_size: int = 8

@dataclass
class DataDiscIndexConfig(ExperimentConfig):
    """Main configuration for discriminator pair generation (disc index)."""

    logging: LoggingConfig = field(default_factory=LoggingConfig)
    wandb: WandBConfig = field(default_factory=WandBConfig)

    output_dir: str = "./outputs/disc_index_output"
    num_return_sequences: int = 8                       
    total_rows: int = 480                                       
    llm_data_output_dir: str = "./outputs/llm_data_8B/"
    save_format: str = "jsonl"                      

    train_ratio: float = 0.9
    val_ratio: float = 0.05
    test_ratio: float = 0.05
    cross_task: bool = False

@dataclass
class DiscriminatorTrainerConfig(ExperimentConfig, TrainingConfig):
    """Main configuration for training the discriminator."""

    discriminator: DiscriminatorConfig = field(default_factory=DiscriminatorConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    wandb: WandBConfig = field(default_factory=WandBConfig)

    tr_type: ThoughtRepresentation = field(
        default_factory=lambda: ThoughtRepresentation.RANDOM_VECTOR
    )
    think_steps: int = 128
    expand_dim: int = 128
    disc_index_output_dir: str = "./outputs/disc_index_output/"
    disc_data_output_dir: str = "./outputs/disc_data_8B/"

    source_hidden_size: int = 4096

    source_num_layers: int = 33

    eval_only: bool = False                                            
    saved_model_dir: Optional[str] = (
        None                                                                                                                    
    )

    output_dir: str = "./outputs/discriminator_training"

    num_return_sequences: int = 8                      
    shard_size: int = 1024

@dataclass
class DiscDataConfig:
    discriminator: DiscriminatorConfig = field(default_factory=DiscriminatorConfig)
    base_llm: BaseLLMConfig = field(default_factory=BaseLLMConfig)
    embedder: EmbedderConfig = field(default_factory=EmbedderConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)

    num_return_sequences: int = 8                       
    shard_size: int = 1024
    think_steps: int = 128                                          
    llm_data_output_dir: str = "./outputs/llm_data_output/"
    disc_index_output_dir: str = "./outputs/disc_index_output/"
    output_dir: str = "./outputs/disc_data_8B/"
    seed: int = 42
    tr_types: Optional[List[str]] = (
        None                                                    
    )

@dataclass
class MinimalityProbeConfig(ModelConfig):
    """Configuration for the minimality probe model (ThoughtDescriptor)."""

    model_name: str = "meta-llama/Llama-3.2-1B"
    model_class: ModelClassEnum = ModelClassEnum.BASE_LLM
    model_type: ModelType = ModelType.LLAMA

    vector_dim: int = 4096
    freeze_base_model: bool = True
    dropout_rate: float = 0.1
    projection_type: str = "shared"                                 

@dataclass
class MinimalityTrainerConfig(ExperimentConfig, TrainingConfig):
    """Main configuration for training the minimality probe."""

    probe: MinimalityProbeConfig = field(default_factory=MinimalityProbeConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    wandb: WandBConfig = field(default_factory=WandBConfig)

    llm_data_output_dir: str = "./outputs/llm_data_8B/"
    tr_data_output_dir: str = "./outputs/disc_data_8B/"                               
    tr_type: str = "last_input_token"                                              

    num_return_sequences: int = 8                       
    shard_size: int = 1024
    think_steps: int = 128                                          

    max_input_length: int = 512                                       
    target_source: str = "input"                                                                

    prefix_source: Optional[str] = None
    max_prefix_length: Optional[int] = None
    use_thought: bool = True
                                                                            
    tile_to_length: Optional[int] = None

    fp16: bool = False
    bf16: bool = True
    dataloader_num_workers: int = 4
    dataloader_pin_memory: bool = True
    remove_unused_columns: bool = False
    label_smoothing_factor: float = 0.0

    output_dir: str = "./outputs/minimality_training"

@dataclass
class CausalityEvalConfig(ExperimentConfig):
    """Configuration for causality evaluation (KL divergence metric)."""

    discriminator: DiscriminatorConfig = field(default_factory=DiscriminatorConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    wandb: WandBConfig = field(default_factory=WandBConfig)

    tr_type: str = "last_input_token"
    think_steps: int = 1                                            

    disc_dir: str = "./outputs/disc3/llama_8b/same_lit"

    tr_data_dir: str = "./outputs/disc_data_8B/"
    num_return_sequences: int = 8
    shard_size: int = 1024

    z_split_tokens: int = 50                                                               
    num_examples: int = -1                                            

    source_hidden_size: int = 4096

    source_num_layers: int = 33

    proj_source: str = "disc"
    min_proj_run_label: str = "llama_70b_disc3"
    min_proj_root: str = "./outputs/min_prob_output"

    tile_to_length: Optional[int] = None

    output_dir: str = "./outputs/causality"

@dataclass
class DCSEvalConfig(ExperimentConfig):
    """Configuration for DCS (Distributional Consistency Score) evaluation."""

    discriminator: DiscriminatorConfig = field(default_factory=DiscriminatorConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    wandb: WandBConfig = field(default_factory=WandBConfig)

    tr_type: str = "last_input_token"
    think_steps: int = 1                                            

    disc_dir: str = "./outputs/disc2/same_lit"

    tr_data_dir: str = "./outputs/disc_data_8B/"
    num_return_sequences: int = 8
    shard_size: int = 1024

    tau: float = 0.9                                              
    expand_dim: int = 128                                                
    max_token_length: int = 512                                                            

    num_examples: int = -1                                     

    source_hidden_size: int = 4096

    source_num_layers: int = 33

    output_dir: str = "./outputs/dcs"

def resolve_other_vector_dim(tr_type: ThoughtRepresentation, source_hidden_size: int, embedder_output_dim: int = 4096) -> int:
    """Return the correct other_vector_dim for a given TR type.

    Embedding-based TR types (e.g. embedding_pooling) always use the
    embedder's output dimension (fixed for a given embedder model).
    LLM hidden-state TR types use the source LLM's hidden size, which varies by model.
    """
    return embedder_output_dim if tr_type.is_embedding_based else source_hidden_size

def register_configs() -> None:
    """Register structured configs with Hydra's ConfigStore."""
                                                       
    pass
