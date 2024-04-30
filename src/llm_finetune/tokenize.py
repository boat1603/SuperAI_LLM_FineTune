import transformers
from typing import Dict


def _tokenize_fn(text: str, tokenizer: transformers.PreTrainedTokenizer) -> Dict:
    """Tokenize a string."""
    text_tokenized = tokenizer(
        text,
        return_tensors="pt",
        padding="longest",
        max_length=tokenizer.model_max_length,
        truncation=True,
    )
    input_ids = labels = text_tokenized.input_ids[0]
    input_ids_lens = labels_lens = (
        text_tokenized.input_ids.ne(tokenizer.pad_token_id).sum().item()
    )
    return dict(
        input_ids=input_ids,
        labels=labels,
        input_ids_lens=input_ids_lens,
        labels_lens=labels_lens,
    )
