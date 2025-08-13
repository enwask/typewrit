from transformers import pipeline

from typewrit.utils import config


_completions = pipeline(
    task='text-generation',
    model=config.llm.completion_model,
)


def get_completions(
        prompt: str,
        max_length: int = 50,
        num_return_sequences: int = 3
) -> list[str]:
    return [
        res['generated_text'] for res in _completions(
            prompt,
            max_length=max_length,
            num_return_sequences=num_return_sequences
        )
    ]
