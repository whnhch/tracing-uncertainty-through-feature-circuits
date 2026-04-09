"""
Dataset loading and prompt construction.

Loads TriviaQA (rc.nocontext split) and formats each example as a simple
Q/A prompt. Returns tokenized prompts alongside the token id of the first
token of the correct answer, which is used as ground truth for top-k
coherence checks in Phase 3.
"""

import torch
from datasets import load_dataset
from transformer_lens import HookedTransformer
from typing import NamedTuple


class PromptBatch(NamedTuple):
    # Raw question strings
    questions: list[str]
    # Formatted "Q: ... \nA:" strings
    prompts: list[str]
    # Correct answer strings (first alias)
    answers: list[str]
    # Tokenized prompts: [n, seq_len] (left-padded to same length)
    input_ids: torch.Tensor
    # Attention mask: [n, seq_len]
    attention_mask: torch.Tensor
    # Token id of the first token of the correct answer: [n]
    correct_token_ids: torch.Tensor


def format_prompt(question: str) -> str:
    return f"Q: {question}\nA:"


def load_trivia_qa(
    model: HookedTransformer,
    n_prompts: int,
    device: str,
    seed: int = 42,
    dataset_name: str = "trivia_qa",
    dataset_config: str = "rc.nocontext",
) -> PromptBatch:
    """
    Load TriviaQA, format as Q/A prompts, tokenize, and return a PromptBatch.

    Args:
        model:          loaded HookedTransformer (provides tokenizer)
        n_prompts:      how many examples to load
        device:         torch device string
        seed:           random seed for shuffling
        dataset_name:   HuggingFace dataset identifier
        dataset_config: dataset configuration name

    Returns:
        PromptBatch with all fields populated
    """
    raw = load_dataset(dataset_name, dataset_config, split="validation")
    raw = raw.shuffle(seed=seed).select(range(min(n_prompts, len(raw))))

    questions = []
    prompts = []
    answers = []

    for example in raw:
        question = example["question"]
        # Use the first listed answer alias as ground truth
        answer = example["answer"]["value"]

        questions.append(question)
        prompts.append(format_prompt(question))
        answers.append(answer)

    # Tokenize prompts
    tokenizer = model.tokenizer
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    encoded = tokenizer(
        prompts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=128,
    )
    input_ids = encoded["input_ids"].to(device)
    attention_mask = encoded["attention_mask"].to(device)

    # Get the first token of each correct answer (used for top-k coherence check)
    correct_token_ids = torch.tensor(
        [tokenizer.encode(" " + ans, add_special_tokens=False)[0] for ans in answers],
        dtype=torch.long,
        device=device,
    )

    return PromptBatch(
        questions=questions,
        prompts=prompts,
        answers=answers,
        input_ids=input_ids,
        attention_mask=attention_mask,
        correct_token_ids=correct_token_ids,
    )


def get_batches(batch: PromptBatch, batch_size: int):
    """
    Yield sub-batches of a PromptBatch for memory-efficient processing.
    """
    n = len(batch.questions)
    for start in range(0, n, batch_size):
        end = min(start + batch_size, n)
        yield PromptBatch(
            questions=batch.questions[start:end],
            prompts=batch.prompts[start:end],
            answers=batch.answers[start:end],
            input_ids=batch.input_ids[start:end],
            attention_mask=batch.attention_mask[start:end],
            correct_token_ids=batch.correct_token_ids[start:end],
        )
