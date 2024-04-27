import transformers

from dataclasses import dataclass, field
from typing import Optional, Union


@dataclass
class ModelArguments:
    model_name_or_path: Optional[str] = field(default="facebook/opt-125m")


@dataclass
class DataArguments:
    train_data_path: str = field(
        default=None, metadata={"help": "Path to the training data."}
    )
    eval_data_path: str = field(
        default=None, metadata={"help": "Path to the eval data."}
    )


@dataclass
class TrainingArguments(transformers.TrainingArguments):
    cache_dir: Optional[str] = field(default=None)
    optim: str = field(default="adamw_torch")
    checkpoint: Optional[str] = field(default=None)
    model_max_length: int = field(
        default=512,
        metadata={
            "help": "Maximum sequence length. Sequences will be right padded (and possibly truncated)."  # noqa: E501
        },
    )
    gradient_checkpointing_kwargs: Optional[Union[dict, str]] = field(
        default_factory={"use_reentrant": False}.copy,
        metadata={
            "help": "Gradient checkpointing key word arguments such as `use_reentrant`. Will be passed to `torch.utils.checkpoint.checkpoint` through `model.gradient_checkpointing_enable`."
        },
    )
