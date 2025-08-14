from functools import cached_property
from transformers import pipeline

from typewrit.utils import config


_completions = pipeline(
    task='text-generation',
    model=config.llm.completion_model,
    device_map=config.llm.device_map,
)


class Completion:
    """
    Describes a processed completion from a language model.

    'Word' here refers to a sequence of characters separated by spaces;
    that is, a word may include punctuation or other characters.
    """

    def __init__(self, prompt: str, text: str):
        self.prompt_text = prompt
        self.completed_text = text

    @cached_property
    def prefix(self) -> str:
        """
        The shared prefix of the prompt and completion, up to the last
        word in the prompt. If the completion deviates partway through
        the last word, the prefix will not include part of that word.
        """
        common_len = 0
        while self.prompt_text[:common_len] \
                == self.completed_text[:common_len]:
            common_len += 1

        # If no words differed, go back a word for the pivot
        if common_len == len(self.prompt_text):
            common_len -= 1
            while common_len > 0 and self.prompt_text[common_len] != ' ':
                common_len -= 1

        # Check if we are within a word
        within_word = self.prompt_text[common_len] != ' '

        if within_word:
            while common_len > 0 and self.prompt_text[common_len] != ' ':
                common_len -= 1

        return self.prompt_text[:common_len]

    @cached_property
    def pivot(self) -> str:
        """
        The pivot point in the completion, which is the first word that
        (fully or partially) deviates from the prompt. If the completion
        deviates partway through a word, that word is the pivot.
        """
        common_len = len(self.prefix)
        assert common_len < len(self.completed_text)

        left = right = common_len + 1
        while right < len(self.completed_text) \
                and self.completed_text[right] != ' ':
            right += 1

        return self.completed_text[left:right]

    @cached_property
    def suffix(self) -> str:
        """
        The suffix of the completion, which is the part that follows
        the pivot point.
        """
        suffix_start = len(self.prefix) + 2 + len(self.pivot)  # +2 for spaces
        return self.completed_text[suffix_start:]

    def __str__(self) -> str:
        return f"{self.prefix}|{self.pivot}|{self.suffix}".strip()


def get_completions(
        prompt: str,
        max_new_tokens: int = 20,
        num_return_sequences: int = 3
) -> list[Completion]:
    return [
        Completion(prompt, res['generated_text']) for res in _completions(
            prompt,
            max_new_tokens=max_new_tokens,
            num_return_sequences=num_return_sequences
        )
    ]
