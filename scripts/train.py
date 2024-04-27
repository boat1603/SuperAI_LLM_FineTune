import transformers
from transformers import Trainer

from llm_finetune.arguments import (
    ModelArguments,
    DataArguments,
    TrainingArguments,
)
from llm_finetune.dataset import make_supervised_data_module


def train():
    parser = transformers.HfArgumentParser(
        (ModelArguments, DataArguments, TrainingArguments)
    )
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    model = transformers.AutoModelForCausalLM.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=training_args.cache_dir,
    )

    tokenizer = transformers.AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=training_args.cache_dir,
        model_max_length=training_args.model_max_length,
        padding_side="right",
        use_fast=False,
    )
    tokenizer.pad_token = tokenizer.eos_token

    data_module = make_supervised_data_module(
        tokenizer=tokenizer,
        data_args=data_args,
    )
    trainer = Trainer(
        model=model, tokenizer=tokenizer, args=training_args, **data_module
    )
    trainer.train(training_args.checkpoint)
    trainer.save_state()
    trainer.save_model(output_dir=training_args.output_dir)


if __name__ == "__main__":
    train()
